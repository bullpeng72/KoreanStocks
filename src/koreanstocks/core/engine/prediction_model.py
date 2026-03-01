import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import json

from koreanstocks.core.config import config
from koreanstocks.core.engine.indicators import indicators
from koreanstocks.core.data.database import db_manager

logger = logging.getLogger(__name__)

# 예측 시 사용할 피처 목록 — trainer.py BASE_FEATURE_COLS와 동기화 필수 (25개)
_FEATURE_COLS = [
    'atr_ratio', 'adx', 'adx_di_diff', 'bb_width',
    'rs_vs_mkt_3m', 'rs_vs_mkt_1m', 'high_52w_ratio', 'mom_accel',
    'macd_diff', 'macd_diff_change', 'macd_slope_5d', 'price_sma_5_ratio',
    'fisher', 'bullish_fractal_5d',
    'cmf', 'vzo', 'obv_change', 'vol_ratio', 'vol_change', 'rsi_mfi_div',
    'sqzmi',
    'candle_body',
    'vix_level', 'vix_change_5d', 'sp500_1m',
]
# 구버전(22/34/37/23) 구분용 문서 상수 (코드 로직에서는 _FEATURE_COLS 컬럼명 선택 사용)
_BASE_FEATURE_COUNT = 25
# AUC 가중치 하한선: AUC 기반 가중치 = (AUC - 0.5) / 상수
# AUC=0.55 → weight=0.05, AUC=0.60 → weight=0.10 (최대 2배 차이 허용)
_AUC_WEIGHT_FLOOR = 0.50   # AUC - 0.5가 이 값 이하면 weight=0 처리
# 모델 품질 게이트: test_auc 가 이 값 미만이면 모델 로드를 거부하고 tech_score 폴백 사용
# AUC < 0.52 = 랜덤(0.5)과 사실상 동등 → 예측력 없음
_MIN_MODEL_AUC = 0.52
# 하위 호환: 구버전 R² 기반 모델 로드 시 fallback용 (사용 안 함)
_MIN_MODEL_R2 = 0.0


class StockPredictionModel:
    """머신러닝 기반 주가 예측 모델 클래스 (앙상블)"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_weights = {}   # name → 1/RMSE 가중치 (성능 기반 앙상블용)
        self.calibrations: Dict[str, list] = {}  # name → 101분위수 배열 (predict_proba → 0~100)
        # 절대 경로 설정
        self.model_dir = os.path.join(config.BASE_DIR, "models", "saved", "prediction_models")
        self.params_dir = os.path.join(config.BASE_DIR, "models", "saved", "model_params")
        # 시장 지수 당일 캐시 (KS11/KQ11 별도 캐싱, 상대강도 피처용)
        self._market_cache: Dict[str, Any] = {}  # symbol → {'df': DataFrame, 'date': str}
        self._load_existing_models()

    def _load_existing_models(self):
        """저장된 모델 및 스케일러 로드 (한 쌍이 모두 존재할 때만 활성화)"""
        model_names = ['random_forest', 'gradient_boosting', 'xgboost']
        
        if not os.path.exists(self.model_dir):
            logger.error(f"Model directory not found: {self.model_dir}")
            return

        for name in model_names:
            model_path = os.path.join(self.model_dir, f"{name}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{name}_scaler.pkl")
            
            # 모델과 스케일러가 모두 존재해야 로드 (정합성 유지)
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    loaded_model = joblib.load(model_path)
                    loaded_scaler = joblib.load(scaler_path)

                    # params JSON에서 품질 지표 확인 — 기준 미달 모델은 로드 거부
                    params_path = os.path.join(self.params_dir, f"{name}_params.json")
                    if os.path.exists(params_path):
                        with open(params_path, 'r', encoding='utf-8') as pf:
                            meta = json.load(pf)
                        model_type = meta.get("model_type", "regression")
                        if model_type == "binary_classifier":
                            auc = float(meta.get("test_auc", 0.0))
                            if auc < _MIN_MODEL_AUC:
                                logger.warning(
                                    f"⚠️  {name} 품질 기준 미달 (test_auc={auc:.4f} < {_MIN_MODEL_AUC}) — "
                                    f"로드 건너뜀. tech_score 폴백으로 동작합니다."
                                )
                                continue
                            # AUC 기반 가중치: (AUC - 0.5) 에 비례
                            self.model_weights[name] = max(auc - _AUC_WEIGHT_FLOOR, 1e-6)
                            # 캘리브레이션 배열 로드 (101분위수)
                            cal = meta.get("calibration")
                            if cal and len(cal) == 101:
                                self.calibrations[name] = cal
                            logger.info(f"✅ Loaded classifier: {name} (auc={auc:.4f}, weight={self.model_weights[name]:.4f})")
                        else:
                            # 구버전 regression 모델 — R² 기준
                            r2   = float(meta.get("test_r2",   0.0))
                            rmse = float(meta.get("test_rmse", 30.0))
                            if r2 < _MIN_MODEL_R2:
                                logger.warning(f"⚠️  {name} 구버전 R² 미달 ({r2:.4f}) — 건너뜀.")
                                continue
                            self.model_weights[name] = 1.0 / max(rmse, 5.0)
                            logger.info(f"✅ Loaded regressor: {name} (r2={r2:.4f})")
                    else:
                        self.model_weights[name] = 0.05  # 파라미터 없으면 기본 가중치

                    self.models[name]  = loaded_model
                    self.scalers[name] = loaded_scaler
                except Exception as e:
                    logger.error(f"❌ Error loading {name} package: {e}")
            else:
                missing = []
                if not os.path.exists(model_path): missing.append("model.pkl")
                if not os.path.exists(scaler_path): missing.append("scaler.pkl")
                logger.warning(f"⚠️ Skipping {name}: Missing {', '.join(missing)}")


    def _get_market_df(self, index_symbol: str = 'KS11') -> pd.DataFrame:
        """시장 지수 수익률 DataFrame 반환 (KS11=KOSPI, KQ11=KOSDAQ, 당일 캐싱).

        컬럼: return_1m (20d), return_3m (60d) — 인덱스: 날짜
        FDR 실패 시 yfinance(^KS11/^KQ11) 폴백.
        """
        from datetime import date as _date
        from koreanstocks.core.data.provider import data_provider as _dp
        today = _date.today().isoformat()
        cached = self._market_cache.get(index_symbol, {})
        if cached.get('date') == today and not cached.get('df', pd.DataFrame()).empty:
            return cached['df']
        # FDR 1차 시도
        try:
            raw = _dp.get_ohlcv(index_symbol, period='2y')
            if not raw.empty:
                mkt = pd.DataFrame(index=raw.index)
                mkt['return_1m'] = raw['close'].pct_change(20)
                mkt['return_3m'] = raw['close'].pct_change(60)
                self._market_cache[index_symbol] = {'df': mkt, 'date': today}
                return mkt
        except Exception as e:
            logger.warning(f"Failed to fetch {index_symbol} via FDR: {e}")
        # yfinance 폴백
        yf_map = {'KS11': '^KS11', 'KQ11': '^KQ11'}
        yf_sym = yf_map.get(index_symbol)
        if yf_sym:
            try:
                import yfinance as yf
                raw = yf.download(yf_sym, period='2y', progress=False)
                if not raw.empty:
                    close = raw.xs('Close', level=0, axis=1).iloc[:, 0] \
                        if isinstance(raw.columns, pd.MultiIndex) else raw['Close']
                    close.index = pd.to_datetime(close.index).tz_localize(None)
                    mkt = pd.DataFrame(index=close.index)
                    mkt['return_1m'] = close.pct_change(20)
                    mkt['return_3m'] = close.pct_change(60)
                    self._market_cache[index_symbol] = {'df': mkt, 'date': today}
                    logger.debug(f"Market data ({index_symbol}) loaded via yfinance fallback.")
                    return mkt
            except Exception as e2:
                logger.warning(f"yfinance fallback for {yf_sym} also failed: {e2}")
        return pd.DataFrame()

    def _get_macro_df(self) -> pd.DataFrame:
        """VIX·S&P500 거시경제 데이터 반환 (당일 캐싱).

        컬럼: vix_level, vix_change_5d, sp500_1m — 인덱스: 날짜(tz-naive)
        """
        from datetime import date as _date
        today = _date.today().isoformat()
        cached = self._market_cache.get('__macro__', {})
        if cached.get('date') == today and not cached.get('df', pd.DataFrame()).empty:
            return cached['df']
        try:
            import yfinance as yf
            raw = yf.download(['^VIX', '^GSPC'], period='2y', progress=False)
            if not raw.empty:
                close = raw.xs('Close', level=0, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw['Close']
                macro = pd.DataFrame(index=close.index)
                macro.index = pd.to_datetime(macro.index).tz_localize(None)
                macro['vix_level']     = close['^VIX'].values
                macro['vix_change_5d'] = close['^VIX'].pct_change(5).values
                macro['sp500_1m']      = close['^GSPC'].pct_change(20).values
                macro = macro.ffill()
                self._market_cache['__macro__'] = {'df': macro, 'date': today}
                return macro
        except Exception as e:
            logger.warning(f"Macro data fetch failed: {e}")
        return pd.DataFrame()

    def prepare_features(self, df: pd.DataFrame,
                         market_df: pd.DataFrame = None,
                         macro_df: pd.DataFrame = None) -> pd.DataFrame:
        """원본 OHLCV에서 지표를 계산한 뒤 특성(Feature) 생성"""
        if df.empty: return df
        df_ind = indicators.calculate_all(df)
        return self._extract_features(df_ind, market_df=market_df, macro_df=macro_df)

    def _extract_features(self, df: pd.DataFrame,
                          market_df: pd.DataFrame = None,
                          macro_df: pd.DataFrame = None) -> pd.DataFrame:
        """이미 지표가 계산된 데이터프레임에서 특성(Feature)만 추출"""
        if df.empty: return df
        feat = pd.DataFrame(index=df.index)

        # ── 기존 4개 ──────────────────────────────────
        feat['rsi']               = df['rsi']
        feat['macd_diff']         = df['macd_diff']
        feat['price_sma_20_ratio'] = df['close'] / df['sma_20']
        feat['vol_change']        = df['volume'].pct_change()

        # ── 추세 (multi-timeframe) ────────────────────
        feat['price_sma_5_ratio'] = df['close'] / df['sma_5']
        feat['rsi_change']        = df['rsi'].diff()
        feat['macd_diff_change']  = df['macd_diff'].diff()
        feat['macd_slope_5d']     = df['macd_diff'].diff(5)

        # ── 볼린저 밴드 ───────────────────────────────
        bb_range = (df['bb_high'] - df['bb_low']).replace(0, np.nan)
        feat['bb_position']       = (df['close'] - df['bb_low']) / bb_range
        feat['bb_width']          = bb_range / df['bb_mid']

        # ── 거래량 ────────────────────────────────────
        feat['vol_ratio']         = df['volume'] / df['vol_sma_20'].replace(0, np.nan)

        # ── 모멘텀 (오실레이터) ───────────────────────
        if 'stoch_k' in df.columns:
            feat['stoch_k']       = df['stoch_k']
        if 'stoch_d' in df.columns:
            feat['stoch_d']       = df['stoch_d']
        if 'cci' in df.columns:
            feat['cci']           = df['cci']

        # ── 변동성 ────────────────────────────────────
        if 'atr' in df.columns:
            feat['atr_ratio']     = df['atr'] / df['close']

        # ── OBV 변화율 ────────────────────────────────
        if 'obv' in df.columns:
            feat['obv_change']    = df['obv'].pct_change().clip(-1, 1)

        # ── 당일 캔들 ────────────────────────────────
        feat['candle_body']       = (df['close'] - df['open']) / df['open']

        # ── 모멘텀 팩터 (신규) ────────────────────────
        feat['return_1m']         = df['close'].pct_change(20)
        feat['return_3m']         = df['close'].pct_change(60)
        feat['high_52w_ratio']    = df['close'] / df['close'].rolling(config.TRADING_DAYS_PER_YEAR, min_periods=60).max()
        feat['mom_accel']         = feat['return_1m'] - feat['return_3m'] / 3.0

        # ── 시장 상대강도 ─────────────────────────────
        if market_df is not None and not market_df.empty:
            aligned = market_df.reindex(feat.index).ffill()
            feat['rs_vs_mkt_1m'] = (feat['return_1m'] - aligned.get('return_1m', 0)).fillna(0)
            feat['rs_vs_mkt_3m'] = (feat['return_3m'] - aligned.get('return_3m', 0)).fillna(0)
        else:
            feat['rs_vs_mkt_1m'] = 0.0
            feat['rs_vs_mkt_3m'] = 0.0

        # ── 신규 12개 피처 ────────────────────────────

        # ADX 추세 강도 + DI 방향
        if 'adx' in df.columns:
            feat['adx'] = df['adx']
        if 'adx_pos' in df.columns and 'adx_neg' in df.columns:
            feat['adx_di_diff'] = df['adx_pos'] - df['adx_neg']

        # VWAP 대비 현재가 위치
        if 'vwap' in df.columns:
            feat['vwap_ratio'] = (
                df['close'] / df['vwap'].replace(0, np.nan)
            ).clip(0.5, 2.0)

        # Donchian Channel 위치
        if 'dc_high' in df.columns and 'dc_low' in df.columns:
            dc_range = (df['dc_high'] - df['dc_low']).replace(0, np.nan)
            feat['dc_position'] = (
                (df['close'] - df['dc_low']) / dc_range
            ).clip(0, 1)

        # 거래량 방향성 지표
        if 'cmf' in df.columns:
            feat['cmf'] = df['cmf']
        if 'mfi' in df.columns:
            feat['mfi'] = df['mfi']
        if 'rsi' in df.columns and 'mfi' in df.columns:
            feat['rsi_mfi_div'] = df['rsi'] - df['mfi']

        # Squeeze Momentum (finta)
        if 'sqzmi' in df.columns:
            feat['sqzmi']       = df['sqzmi']
            feat['sqzmi_accel'] = df['sqzmi'].diff().fillna(0)

        # Volume Zone Oscillator (finta)
        if 'vzo' in df.columns:
            feat['vzo'] = df['vzo']

        # Fisher Transform (finta)
        if 'fisher' in df.columns:
            feat['fisher'] = df['fisher']

        # Williams Fractal 5일 내 강세 프랙탈 (finta)
        if 'bullish_fractal' in df.columns:
            feat['bullish_fractal_5d'] = (
                df['bullish_fractal'].rolling(5, min_periods=1).max()
            )

        # ── 거시경제 3개 피처 ────────────────────────────────────
        if macro_df is not None and not macro_df.empty:
            aligned = macro_df.reindex(feat.index, method='ffill')
            feat['vix_level']     = aligned.get('vix_level',     pd.Series(dtype=float))
            feat['vix_change_5d'] = aligned.get('vix_change_5d', pd.Series(dtype=float))
            feat['sp500_1m']      = aligned.get('sp500_1m',      pd.Series(dtype=float))
        else:
            feat['vix_level']     = 20.0
            feat['vix_change_5d'] = 0.0
            feat['sp500_1m']      = 0.0

        # inf → NaN 치환 후 제거
        return feat.replace([np.inf, -np.inf], np.nan).dropna()

    def predict(self, code: str, df: pd.DataFrame,
                df_with_indicators: pd.DataFrame = None,
                fallback_score: float = None) -> Dict[str, Any]:
        """앙상블 예측 수행. 순수 ML 점수만 반환 (sentiment 블렌딩은 호출 측에서 처리).

        Parameters
        ----------
        df_with_indicators : 이미 지표가 계산된 DataFrame (전달 시 재계산 생략)
        fallback_score     : ML 모델 없을 때 대체할 tech_score
        """
        # 종목 시장에 맞는 벤치마크 지수 선택 (KOSDAQ → KQ11, 그 외 → KS11)
        index_symbol = 'KS11'
        try:
            from koreanstocks.core.data.provider import data_provider as _dp
            stock_list = _dp.get_stock_list()
            matched = stock_list[stock_list['code'] == code]
            if not matched.empty and matched.iloc[0].get('market') == 'KOSDAQ':
                index_symbol = 'KQ11'
        except Exception:
            pass
        market_df = self._get_market_df(index_symbol)
        macro_df  = self._get_macro_df()
        if df_with_indicators is not None:
            features = self._extract_features(df_with_indicators, market_df=market_df, macro_df=macro_df)
        else:
            features = self.prepare_features(df, market_df=market_df, macro_df=macro_df)
        if features.empty:
            return {"error": "Insufficient data for ML prediction"}

        # _FEATURE_COLS 순서로 피처 선택 (학습 시 FEATURE_COLS와 동일 순서)
        feat_cols = [c for c in _FEATURE_COLS if c in features.columns]
        latest_x = features[feat_cols].iloc[-1:].values  # shape (1, n_features)

        # AUC 기반 가중 앙상블: w_i = (AUC_i - 0.5)
        # predict_proba()[:, 1] → P(top 30%) → 0~1 → ×100 → 0~100 점수
        weighted_sum = 0.0
        total_weight = 0.0
        model_count  = 0
        for name, model in self.models.items():
            try:
                scaler = self.scalers.get(name)
                x = latest_x.copy()
                if scaler is not None:
                    x = scaler.transform(x)
                # 분류기: predict_proba → 캘리브레이션(percentile rank) → 0~100
                if hasattr(model, 'predict_proba'):
                    p_raw = float(model.predict_proba(x)[0, 1])
                    cal = self.calibrations.get(name)
                    if cal:
                        # np.searchsorted: p_raw가 101분위수 배열 중 몇 번째 분위인지 → 0~100
                        p = float(np.clip(np.searchsorted(cal, p_raw), 0, 100))
                    else:
                        p = p_raw * 100.0
                else:
                    p = float(model.predict(x)[0])
                w = self.model_weights.get(name, 0.05)
                weighted_sum += p * w
                total_weight += w
                model_count  += 1
            except Exception as e:
                logger.debug(f"[{name}] predict failed: {e}")
                continue

        if model_count == 0:
            # 저장된 모델이 없을 때: tech_score 폴백
            if fallback_score is not None:
                score = round(float(np.clip(fallback_score, 0.0, 100.0)), 2)
                logger.warning(f"No ML models loaded for {code}. Using tech_score fallback: {score}")
                return {"ensemble_score": score, "model_count": 0, "note": "fallback_to_tech_score"}
            else:
                latest = features.iloc[-1]
                macd_diff = float(latest.get('macd_diff', 0))
                atr_ratio = float(latest.get('atr_ratio', 0.02))
                heuristic = 50.0 + (10.0 if macd_diff > 0 else -10.0) - atr_ratio * 200
                score = round(float(np.clip(heuristic, 0.0, 100.0)), 2)
                logger.warning(f"No ML models loaded for {code}. Using feature heuristic fallback: {score}")
                return {"ensemble_score": score, "model_count": 0, "note": "fallback_heuristic"}

        # AUC 가중 앙상블 점수 (0~100): P(top 30%) 확률의 가중평균
        ensemble_score = float(np.clip(weighted_sum / total_weight, 0.0, 100.0))
        return {
            "ensemble_score":     round(ensemble_score, 2),
            "model_count":        model_count,
            "prediction_date":    datetime.now().strftime('%Y-%m-%d'),
        }

prediction_model = StockPredictionModel()
