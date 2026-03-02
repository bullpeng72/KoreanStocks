"""모델 신뢰도 라우터 — GET /api/model_health"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])

# params.json 파일이 저장된 디렉토리 (절대 경로)
PARAMS_DIR = Path(__file__).parents[4] / "models" / "saved" / "model_params"

_MODEL_CONFIGS = [
    ("random_forest",       "랜덤 포레스트",      "random_forest_params.json"),
    ("gradient_boosting",   "그래디언트 부스팅",   "gradient_boosting_params.json"),
    ("xgboost",             "XGBoost",            "xgboost_params.json"),
]

_MIN_AUC_THRESHOLD = 0.52


def _days_since(saved_at: str) -> int:
    """저장 시각(ISO 문자열) → 현재까지 경과 일수."""
    try:
        dt = datetime.fromisoformat(saved_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        return max(0, (now - dt).days)
    except Exception:
        return -1


def _load_model_info(name: str, label: str, filename: str) -> dict | None:
    """단일 모델 params.json 로드 및 집계."""
    path = PARAMS_DIR / filename
    if not path.exists():
        logger.warning(f"model params not found: {path}")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
    except Exception as e:
        logger.error(f"params 로드 실패 [{filename}]: {e}")
        return None

    test_auc  = p.get("test_auc", 0.0)
    cv_mean   = p.get("cv_auc_mean", 0.0)
    saved_at  = p.get("saved_at", "")

    return {
        "name":               name,
        "label":              label,
        "test_auc":           test_auc,
        "train_auc":          p.get("train_auc", 0.0),
        "cv_auc_mean":        cv_mean,
        "cv_auc_std":         p.get("cv_auc_std", 0.0),
        "overfit_gap":        p.get("overfit_gap", 0.0),
        "regime_gap":         round(test_auc - cv_mean, 4),
        "test_logloss":       p.get("test_logloss", 0.0),
        "quality_pass":       p.get("quality_pass", False),
        "training_samples":   p.get("training_samples", 0),
        "purging_days":       p.get("purging_days", 0),
        "saved_at":           saved_at,
        "days_since_training": _days_since(saved_at),
        "training_duration":  p.get("training_duration", 0.0),
        "feature_importances": p.get("feature_importances", []),
    }


def _compute_ensemble(models: list[dict]) -> dict:
    """앙상블 집계 및 드리프트 등급 산출."""
    active = [m for m in models if m is not None]
    n = len(active)
    if n == 0:
        return {
            "active_count": 0,
            "mean_test_auc": 0.0,
            "mean_overfit_gap": 0.0,
            "mean_regime_gap": 0.0,
            "all_quality_pass": False,
            "min_auc_threshold": _MIN_AUC_THRESHOLD,
            "days_since_training": -1,
            "drift_level": "HIGH",
            "drift_factors": ["모델 파일 없음"],
            "retrain_recommended": True,
        }

    mean_test_auc    = round(sum(m["test_auc"]    for m in active) / n, 4)
    mean_overfit_gap = round(sum(m["overfit_gap"] for m in active) / n, 4)
    mean_regime_gap  = round(sum(m["regime_gap"]  for m in active) / n, 4)
    all_quality_pass = all(m["quality_pass"] for m in active)
    days_since       = max(m["days_since_training"] for m in active)

    # 드리프트 등급 결정
    factors: list[str] = []
    retrain = False

    if days_since > 30:
        factors.append(f"마지막 학습 {days_since}일 경과 (권장: 30일 이내)")
        retrain = True
    if mean_overfit_gap > 0.10:
        factors.append(f"평균 과적합 갭 {mean_overfit_gap:.4f} (임계: 0.10)")
        retrain = True
    if not all_quality_pass:
        factors.append("품질 기준 미달 모델 존재 (AUC < 0.52)")
        retrain = True

    if retrain:
        drift_level = "HIGH"
    elif days_since > 14 or mean_regime_gap > 0.07:
        drift_level = "MEDIUM"
        if days_since > 14:
            factors.append(f"마지막 학습 {days_since}일 경과 (권장: 14일 이내)")
        if mean_regime_gap > 0.07:
            factors.append(f"레짐 갭 {mean_regime_gap:.4f} > 0.07 (시장 의존도 주의)")
    else:
        drift_level = "LOW"

    return {
        "active_count":       n,
        "mean_test_auc":      mean_test_auc,
        "mean_overfit_gap":   mean_overfit_gap,
        "mean_regime_gap":    mean_regime_gap,
        "all_quality_pass":   all_quality_pass,
        "min_auc_threshold":  _MIN_AUC_THRESHOLD,
        "days_since_training": days_since,
        "drift_level":        drift_level,
        "drift_factors":      factors,
        "retrain_recommended": retrain,
    }


@router.get("/model_health")
def get_model_health():
    """ML 모델 신뢰도 및 앙상블 상태 반환."""
    models = []
    for name, label, filename in _MODEL_CONFIGS:
        info = _load_model_info(name, label, filename)
        if info:
            models.append(info)

    ensemble = _compute_ensemble(models)

    return {
        "models": models,
        "ensemble": ensemble,
        "scoring_formula": {
            "with_ml":    "tech×0.40 + ml×0.35 + sentiment_norm×0.25",
            "without_ml": "tech×0.65 + sentiment_norm×0.35",
        },
    }
