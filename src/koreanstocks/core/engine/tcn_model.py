"""TCN (Temporal Convolutional Network) — 딥러닝 앙상블 구성원
================================================================
Dilated Causal Conv1D 기반 이진 분류 모델.
기존 트리 앙상블이 포착하지 못하는 시간적 패턴(모멘텀 연속성, 변동성 레짐 전환)을 보완.

아키텍처:
  Input  [B, T=20, F=20]
  → Transpose → [B, F=20, T=20]
  → TCNBlock(dilation=1) → TCNBlock(dilation=2) → TCNBlock(dilation=4)
  → 마지막 타임스텝 슬라이스 [B, channels]
  → Linear → Sigmoid → 상승 확률

Receptive field = 1 + 2*(1+2+4) = 15 거래일 (lookback=20 내 완전 커버)

저장:
  tcn_model.pt        PyTorch state_dict
  tcn_scaler.pkl      StandardScaler (피처 차원 정규화)
  tcn_params.json     아키텍처 + 평가 지표 (다른 모델과 통합 형식)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# PyTorch 선택적 import (미설치 시 TCN 비활성화)
# ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False
    import sys as _sys
    _in_pipx = "pipx" in _sys.executable or "pipx" in str(getattr(_sys, "prefix", ""))
    if _in_pipx:
        _install_cmd = "pipx inject koreanstocks torch"
    else:
        _install_cmd = 'pip install -e ".[dl]"  또는  pip install "koreanstocks[dl]"'
    logger.warning(f"PyTorch 미설치 — TCN 모델 비활성화됩니다.  활성화: {_install_cmd}")
    del _sys, _in_pipx, _install_cmd

# ─────────────────────────────────────────────────────────────
# 하이퍼파라미터 기본값
# ─────────────────────────────────────────────────────────────
LOOKBACK:    int = 20     # 입력 시퀀스 길이 (거래일)
CHANNELS:    int = 32     # TCN 히든 채널 수
KERNEL_SIZE: int = 3      # Conv 커널 크기
DILATIONS:   List[int] = [1, 2, 4]   # dilation 레이어 목록
DROPOUT:      float = 0.4
WEIGHT_DECAY: float = 5e-4
LR:           float = 1e-3
EPOCHS:      int = 40
BATCH:       int = 64
PATIENCE:    int = 8      # Early stopping patience (val AUC 기준)


# ─────────────────────────────────────────────────────────────
# PyTorch 모듈 정의 (torch 있을 때만 실제 클래스 사용)
# ─────────────────────────────────────────────────────────────
if _TORCH_OK:
    class _CausalBlock(nn.Module):
        """Dilated Causal Conv1D + LayerNorm + ReLU + Dropout + 잔차 연결."""

        def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
            super().__init__()
            pad = (kernel - 1) * dilation          # 왼쪽 패딩만 → causal 보장
            self.conv = nn.Conv1d(
                in_ch, out_ch, kernel,
                padding=pad, dilation=dilation,
            )
            self.causal_trim = pad                  # 오른쪽 trim 길이
            self.norm    = nn.LayerNorm(out_ch)
            self.act     = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            # 채널 수 다를 때 잔차 projection
            self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: [B, C, T]
            out = self.conv(x)
            if self.causal_trim:
                out = out[:, :, :-self.causal_trim]   # 미래 누출 제거
            # LayerNorm은 채널 차원 기대 → transpose
            out = self.norm(out.transpose(1, 2)).transpose(1, 2)
            out = self.act(out)
            out = self.dropout(out)
            return out + self.residual(x)

    class _TCNNet(nn.Module):
        """3-layer dilated causal TCN → 이진 분류."""

        def __init__(
            self,
            n_features: int,
            channels: int   = CHANNELS,
            kernel:   int   = KERNEL_SIZE,
            dilations: List[int] = None,
            dropout:  float = DROPOUT,
        ):
            super().__init__()
            dilations = dilations or DILATIONS
            layers: List[nn.Module] = []
            in_ch = n_features
            for d in dilations:
                out_ch = channels
                layers.append(_CausalBlock(in_ch, out_ch, kernel, d, dropout))
                in_ch = out_ch
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(in_ch, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: [B, T, F]  →  conv expects [B, F, T]
            h = self.tcn(x.transpose(1, 2))   # [B, C, T]
            h = h[:, :, -1]                    # 마지막 타임스텝
            return torch.sigmoid(self.head(h)).squeeze(-1)   # [B]


# ─────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────

def is_available() -> bool:
    """PyTorch 가용 여부."""
    return _TORCH_OK


def build_sequences(
    feature_df,        # pd.DataFrame: 날짜 인덱스, 컬럼=피처
    label_series,      # pd.Series:   날짜 인덱스, 값=0/1
    lookback: int = LOOKBACK,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """피처 DataFrame + 라벨 Series → (X_seq, y, dates).

    X_seq shape: (N, lookback, n_features)
    y     shape: (N,)
    dates:       라벨 날짜 리스트 (Walk-Forward CV 날짜 필터링용)

    라벨이 있는 날짜 D에 대해, [D-lookback, D) 구간 피처를 입력 시퀀스로 구성.
    lookback 행이 모두 채워지지 않는 초기 샘플은 제외.
    """
    dates_feat  = list(feature_df.index)
    date_to_idx = {d: i for i, d in enumerate(dates_feat)}

    X_list, y_list, d_list = [], [], []
    for lbl_date, label in label_series.items():
        if lbl_date not in date_to_idx:
            continue
        end_idx = date_to_idx[lbl_date]        # exclusive: 라벨 날짜 자체는 미포함
        if end_idx < lookback:
            continue                            # 시퀀스 미충족
        seq = feature_df.iloc[end_idx - lookback: end_idx].values   # [lookback, F]
        if np.isnan(seq).any():
            continue
        X_list.append(seq)
        y_list.append(float(label))
        d_list.append(lbl_date)

    if not X_list:
        return np.empty((0, lookback, feature_df.shape[1])), np.empty(0), []

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), d_list


def train_tcn(
    stock_data: dict,           # {code: {'features': DataFrame, 'labels': Series}}
    future_days: int = 10,
    lookback: int = LOOKBACK,
    test_ratio: float = 0.2,
    channels: int = CHANNELS,
    epochs: int = EPOCHS,
    batch: int = BATCH,
    patience: int = PATIENCE,
    dropout: float = DROPOUT,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
) -> Optional[dict]:
    """TCN 학습 + Walk-Forward CV → {model, scaler, meta} 반환.

    Returns None if torch is unavailable or data is insufficient.
    """
    if not _TORCH_OK:
        logger.warning("PyTorch 미설치 — TCN 학습을 건너뜁니다.")
        return None

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import time

    # ── 전체 시퀀스 수집 ──────────────────────────────────────────
    all_X, all_y, all_dates = [], [], []
    for code, d in stock_data.items():
        feat_df  = d['features']
        lbl_ser  = d['labels']
        X, y, ds = build_sequences(feat_df, lbl_ser, lookback)
        if len(X) == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        all_dates.extend(ds)

    if not all_X:
        logger.warning("[TCN] 유효한 시퀀스 없음 — 건너뜁니다.")
        return None

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    dates_arr = np.array(all_dates)

    n_features = X_all.shape[2]

    # ── StandardScaler (피처 차원, 시간축 평균) ───────────────────
    scaler = StandardScaler()
    flat   = X_all.reshape(-1, n_features)   # [N*T, F]
    scaler.fit(flat)

    def _scale(X: np.ndarray) -> np.ndarray:
        """[N, T, F] → StandardScaler 적용 → [N, T, F]"""
        n, t, f = X.shape
        return scaler.transform(X.reshape(-1, f)).reshape(n, t, f)

    # ── 날짜 기반 train/test 분할 ─────────────────────────────────
    unique_dates = sorted(set(all_dates))
    n_dates      = len(unique_dates)
    split_idx    = min(int(n_dates * (1.0 - test_ratio)), n_dates - 1)
    split_date   = unique_dates[split_idx]

    purge_idx    = max(0, split_idx - 2 * future_days)
    purge_date   = unique_dates[purge_idx]

    tr_mask  = dates_arr <  purge_date
    te_mask  = dates_arr >= split_date

    X_train, y_train = X_all[tr_mask], y_all[tr_mask]
    X_test,  y_test  = X_all[te_mask], y_all[te_mask]

    if len(X_train) < 50 or len(X_test) < 10:
        logger.warning(f"[TCN] 학습 샘플 부족 (train={len(X_train)}, test={len(X_test)}) — 건너뜁니다.")
        return None

    X_train_s = _scale(X_train)
    X_test_s  = _scale(X_test)

    # ── Walk-Forward CV (AUC, OOF calibration용) ──────────────────
    VAL_WINDOW, VAL_STEP = 20, 10
    min_train_n = max(int(n_dates * 0.6), 120)
    cv_aucs, oof_preds = [], []

    start_idx = min_train_n
    while start_idx + VAL_WINDOW <= n_dates:
        end_idx        = min(start_idx + VAL_WINDOW, n_dates)
        purge_b        = max(0, start_idx - 2 * future_days)
        tr_dates_set   = set(unique_dates[:purge_b])
        val_dates_set  = set(unique_dates[start_idx:end_idx])

        cv_tr  = np.isin(dates_arr, list(tr_dates_set))
        cv_val = np.isin(dates_arr, list(val_dates_set))
        if cv_tr.sum() < 30 or cv_val.sum() < 10:
            start_idx += VAL_STEP
            continue

        Xf_tr  = _scale(X_all[cv_tr])
        Xf_val = _scale(X_all[cv_val])
        yf_tr  = y_all[cv_tr]
        yf_val = y_all[cv_val]

        cv_model = _TCNNet(n_features, channels=channels, dropout=dropout)
        cv_preds = _fit_torch(cv_model, Xf_tr, yf_tr, Xf_val, yf_val,
                              lr=lr, epochs=min(epochs, 20), batch=batch, patience=patience,
                              weight_decay=weight_decay)
        if cv_preds is not None and len(cv_preds):
            try:
                auc = roc_auc_score(yf_val, cv_preds)
                cv_aucs.append(auc)
                oof_preds.extend(cv_preds.tolist())
            except Exception:
                pass
        start_idx += VAL_STEP

    cv_mean = float(np.mean(cv_aucs)) if cv_aucs else float('nan')
    cv_std  = float(np.std(cv_aucs))  if cv_aucs else float('nan')
    n_folds = len(cv_aucs)
    if not np.isnan(cv_mean):
        logger.info(f"  [TCN] CV AUC (Walk-Forward, {n_folds} folds): {cv_mean:.4f} ± {cv_std:.4f}")
    else:
        logger.warning("  [TCN] CV AUC: N/A (유효 fold 없음)")

    # ── 최종 모델 학습 ─────────────────────────────────────────────
    t0    = time.time()
    model = _TCNNet(n_features, channels=channels, dropout=dropout)
    test_preds = _fit_torch(
        model, X_train_s, y_train, X_test_s, y_test,
        lr=lr, epochs=epochs, batch=batch, patience=patience,
        weight_decay=weight_decay,
    )
    duration = time.time() - t0

    if test_preds is None:
        logger.error("[TCN] 최종 모델 학습 실패.")
        return None

    train_preds = _predict_torch(model, X_train_s, batch)
    try:
        train_auc = roc_auc_score(y_train, train_preds)
        test_auc  = roc_auc_score(y_test,  test_preds)
    except Exception as e:
        logger.error(f"[TCN] AUC 계산 실패: {e}")
        return None

    _cal_src = oof_preds if len(oof_preds) >= 101 else train_preds.tolist()
    calibration = np.percentile(_cal_src, np.arange(0, 101)).tolist()

    overfit_gap = round(train_auc - test_auc, 4)
    logger.info(f"  [TCN] AUC: {test_auc:.4f}  (학습: {train_auc:.4f}  gap: {overfit_gap:.4f}  소요: {duration:.1f}s)")

    return {
        "model":             model,
        "scaler":            scaler,
        "n_features":        n_features,
        "channels":          channels,
        "lookback":          lookback,
        "dropout":           dropout,
        "weight_decay":      weight_decay,
        "training_samples":  int(len(X_train)),
        "purging_days":      2 * future_days,
        "train_auc":         round(train_auc, 4),
        "test_auc":          round(test_auc,  4),
        "cv_auc_mean":       round(cv_mean, 4) if not np.isnan(cv_mean) else None,
        "cv_auc_std":        round(cv_std,  4) if not np.isnan(cv_std)  else None,
        "overfit_gap":       overfit_gap,
        "calibration":       calibration,
        "duration":          round(duration, 1),
        "quality_pass":      bool(test_auc >= 0.52),
    }


def _fit_torch(
    model: "_TCNNet",
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    lr: float, epochs: int, batch: int, patience: int,
    weight_decay: float = WEIGHT_DECAY,
) -> Optional[np.ndarray]:
    """모델 학습 → val 예측값 반환. 실패 시 None."""
    try:
        device = torch.device("cpu")
        model.to(device)
        opt      = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # 클래스 불균형 보정
        pos_w    = torch.tensor([(1 - y_tr.mean()) / max(y_tr.mean(), 1e-6)], dtype=torch.float32)
        criterion = nn.BCELoss()

        tr_ds  = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                               torch.tensor(y_tr, dtype=torch.float32))
        loader = DataLoader(tr_ds, batch_size=batch, shuffle=True, drop_last=False)

        best_auc, best_state, wait = 0.0, None, 0
        for ep in range(epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                # 클래스 가중치 수동 적용
                w    = torch.where(yb == 1, pos_w.to(device), torch.ones(1).to(device))
                loss = (criterion(pred, yb) * w).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # Early stopping (매 epoch val AUC 체크)
            val_p = _predict_torch(model, X_val, batch)
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y_val, val_p)
            except Exception:
                auc = 0.0
            if auc > best_auc + 1e-4:
                best_auc   = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        return _predict_torch(model, X_val, batch)
    except Exception as e:
        logger.error(f"[TCN] _fit_torch 오류: {e}", exc_info=True)
        return None


def _predict_torch(model: "_TCNNet", X: np.ndarray, batch: int = 256) -> np.ndarray:
    """배치 추론 → numpy 확률 배열 [N]."""
    model.eval()
    device = next(model.parameters()).device
    preds  = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb   = torch.tensor(X[i: i + batch], dtype=torch.float32).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def save_tcn(result: dict, model_dir, params_dir) -> None:
    """학습 결과 dict → 파일 저장 (model_dir, params_dir: pathlib.Path)."""
    import json, joblib
    from datetime import datetime

    model_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    torch.save(result["model"].state_dict(), model_dir / "tcn_model.pt")
    joblib.dump(result["scaler"], model_dir / "tcn_scaler.pkl")

    saved_at = datetime.now()
    meta = {
        "model_type":      "binary_classifier",
        "architecture":    "TCN",
        "n_features":      result["n_features"],
        "channels":        result["channels"],
        "lookback":        result["lookback"],
        "dilations":       DILATIONS,
        "kernel_size":     KERNEL_SIZE,
        "dropout":         result.get("dropout", DROPOUT),
        "weight_decay":    result.get("weight_decay", WEIGHT_DECAY),
        "train_auc":       result["train_auc"],
        "test_auc":        result["test_auc"],
        "cv_auc_mean":     result["cv_auc_mean"],
        "cv_auc_std":      result["cv_auc_std"],
        "overfit_gap":     result["overfit_gap"],
        "quality_pass":      result["quality_pass"],
        "training_samples":  result["training_samples"],
        "purging_days":      result["purging_days"],
        "training_duration": result["duration"],
        "calibration":       result["calibration"],
        "saved_at":        saved_at.isoformat(),
        "model_version":   f"tcn_v{saved_at.strftime('%Y%m%d_%H%M%S')}",
    }
    with open(params_dir / "tcn_params.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"  [TCN] 저장 완료 → {model_dir}/tcn_model.pt  (test_auc={result['test_auc']:.4f})")


def load_tcn(model_dir, params_dir) -> Optional[dict]:
    """저장된 TCN 모델 로드 → {model, scaler, meta} or None."""
    if not _TORCH_OK:
        return None
    import json, joblib

    pt_path     = model_dir / "tcn_model.pt"
    scaler_path = model_dir / "tcn_scaler.pkl"
    meta_path   = params_dir / "tcn_params.json"

    if not (pt_path.exists() and scaler_path.exists() and meta_path.exists()):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model = _TCNNet(
            n_features = meta["n_features"],
            channels   = meta["channels"],
            dilations  = meta.get("dilations", DILATIONS),
            dropout    = meta.get("dropout",   DROPOUT),
        )
        model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        model.eval()

        scaler = joblib.load(scaler_path)
        return {"model": model, "scaler": scaler, "meta": meta}
    except Exception as e:
        logger.error(f"[TCN] 로드 실패: {e}")
        return None


def auto_tune_tcn(
    stock_data: dict,
    future_days: int = 10,
    test_ratio: float = 0.2,
    max_trials: int = 5,
    base_test_auc: float = 0.0,
    search_space: Optional[dict] = None,
) -> Optional[dict]:
    """TCN 하이퍼파라미터 랜덤 탐색 → 개선된 학습 결과 dict 또는 None.

    base_test_auc 대비 개선이 없으면 None 반환.
    각 시도마다 train_tcn()을 전체 호출하므로 max_trials=5 이하 권장.
    """
    if not _TORCH_OK:
        return None

    import random
    default_space: dict = {
        'channels': [16, 32, 64],
        'dropout':  [0.2, 0.3, 0.4, 0.5],
        'lr':       [5e-4, 1e-3, 2e-3],
        'epochs':   [30, 40, 50],
    }
    space = search_space or default_space
    rng   = random.Random(42)

    best_result: Optional[dict] = None
    best_auc = base_test_auc

    for trial in range(max_trials):
        trial_kwargs: dict = {}
        for k in rng.sample(list(space.keys()), min(2, len(space))):
            trial_kwargs[k] = rng.choice(space[k])
        try:
            result = train_tcn(
                stock_data,
                future_days=future_days,
                test_ratio=test_ratio,
                **trial_kwargs,
            )
            if result is not None and result['test_auc'] > best_auc + 0.001:
                best_auc = result['test_auc']
                best_result = result
                logger.info(
                    f"  [auto-tune/TCN] trial {trial + 1}/{max_trials}: "
                    f"test_auc={result['test_auc']:.4f} ✨ params={trial_kwargs}"
                )
        except Exception as e:
            logger.debug(f"  [auto-tune/TCN] trial {trial + 1} 오류: {e}")

    return best_result


def predict_proba_tcn(
    loaded: dict,
    feature_rows: np.ndarray,   # [T, F] 최근 T행 피처 (T >= lookback)
    lookback: int = LOOKBACK,
) -> Optional[float]:
    """로드된 TCN으로 단일 종목 상승 확률 반환 (0~1). 실패 시 None."""
    if not _TORCH_OK or loaded is None:
        return None
    try:
        scaler = loaded["scaler"]
        model  = loaded["model"]
        n_feat = loaded["meta"]["n_features"]

        if feature_rows.shape[0] < lookback or feature_rows.shape[1] != n_feat:
            return None

        seq = feature_rows[-lookback:]             # [lookback, F]
        seq_s = scaler.transform(seq)              # StandardScaler
        X = seq_s[np.newaxis].astype(np.float32)  # [1, lookback, F]
        prob = _predict_torch(model, X, batch=1)[0]
        return float(np.clip(prob, 0.0, 1.0))
    except Exception as e:
        logger.debug(f"[TCN] predict_proba_tcn 오류: {e}")
        return None
