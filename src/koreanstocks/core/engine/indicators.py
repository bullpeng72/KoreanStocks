import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from finta import TA as _FTA
    _FINTA_AVAILABLE = True
except ImportError:
    _FINTA_AVAILABLE = False
    logger.warning("finta 미설치 — SQZMI, VZO, Fisher, Williams Fractal 지표 비활성화")

class IndicatorCalculator:
    """기술적 지표 계산 및 분석을 담당하는 클래스"""

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 주요 기술적 지표를 계산하여 데이터프레임에 추가"""
        if df.empty or len(df) < 30:
            return df
        
        df = df.copy()
        data_len = len(df)
        
        try:
            # 1. 이동평균 (Trend) - 데이터 길이에 따라 선택적 계산
            df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5, fillna=False)
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20, fillna=False)
            
            if data_len >= 60:
                df['sma_60'] = ta.trend.sma_indicator(df['close'], window=60, fillna=False)
            if data_len >= 120:
                df['sma_120'] = ta.trend.sma_indicator(df['close'], window=120, fillna=False)
            
            # 2. MACD (Trend) - 기본적으로 26일 이상이면 가능
            df['macd'] = ta.trend.macd(df['close'], fillna=False)
            df['macd_signal'] = ta.trend.macd_signal(df['close'], fillna=False)
            df['macd_diff'] = ta.trend.macd_diff(df['close'], fillna=False)
            
            # 3. RSI (Momentum) - 14일 이상이면 가능
            df['rsi'] = ta.momentum.rsi(df['close'], window=14, fillna=False)
            
            # 4. 볼린저 밴드 (Volatility) - 20일 이상이면 가능
            indicator_bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2, fillna=False)
            df['bb_high'] = indicator_bb.bollinger_hband()
            df['bb_mid'] = indicator_bb.bollinger_mavg()
            df['bb_low'] = indicator_bb.bollinger_lband()
            
            # 5. 거래량 지표 (Volume)
            df['vol_sma_20'] = ta.trend.sma_indicator(df['volume'], window=20, fillna=False)
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'], fillna=False)

            # 6. 스토캐스틱 (Momentum)
            stoch = ta.momentum.StochasticOscillator(
                df['high'], df['low'], df['close'], window=14, smooth_window=3, fillna=False
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

            # 7. CCI (Commodity Channel Index)
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20, fillna=False)

            # 8. ATR (Average True Range)
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=14, fillna=False
            )

            # 9. ADX (Average Directional Index) — 추세 강도 + 방향
            adx_ind = ta.trend.ADXIndicator(
                df['high'], df['low'], df['close'], window=14, fillna=False
            )
            df['adx']     = adx_ind.adx()
            df['adx_pos'] = adx_ind.adx_pos()   # DI+
            df['adx_neg'] = adx_ind.adx_neg()   # DI-

            # 10. VWAP (거래량 가중 평균가, 14일 롤링)
            df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
                df['high'], df['low'], df['close'], df['volume'], window=14, fillna=False
            ).volume_weighted_average_price()

            # 11. Donchian Channel (20일 고가/저가 채널)
            dc_ind = ta.volatility.DonchianChannel(
                df['high'], df['low'], df['close'], window=20, fillna=False
            )
            df['dc_high'] = dc_ind.donchian_channel_hband()
            df['dc_low']  = dc_ind.donchian_channel_lband()

            # 12. CMF (Chaikin Money Flow) — 매수/매도 압력 -1~+1
            df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
                df['high'], df['low'], df['close'], df['volume'], window=20, fillna=False
            ).chaikin_money_flow()

            # 13. MFI (Money Flow Index) — 거래량 가중 RSI
            df['mfi'] = ta.volume.MFIIndicator(
                df['high'], df['low'], df['close'], df['volume'], window=14, fillna=False
            ).money_flow_index()

            # 14. finta 지표 (SQZMI, VZO, Fisher Transform, Williams Fractal)
            if _FINTA_AVAILABLE:
                df_f = df[['open', 'high', 'low', 'close', 'volume']].copy()
                try:
                    df['sqzmi'] = _FTA.SQZMI(df_f).fillna(0)
                except Exception as e:
                    logger.debug(f"SQZMI 계산 실패: {e}")
                    df['sqzmi'] = 0.0
                try:
                    df['vzo'] = _FTA.VZO(df_f).fillna(0)
                except Exception as e:
                    logger.debug(f"VZO 계산 실패: {e}")
                    df['vzo'] = 0.0
                try:
                    df['fisher'] = _FTA.FISH(df_f).fillna(0).clip(-5, 5)
                except Exception as e:
                    logger.debug(f"Fisher Transform 계산 실패: {e}")
                    df['fisher'] = 0.0
                try:
                    wf = _FTA.WILLIAMS_FRACTAL(df_f)
                    df['bullish_fractal'] = wf['BullishFractal'].fillna(0)
                except Exception as e:
                    logger.debug(f"Williams Fractal 계산 실패: {e}")
                    df['bullish_fractal'] = 0.0
            else:
                df['sqzmi']          = 0.0
                df['vzo']            = 0.0
                df['fisher']         = 0.0
                df['bullish_fractal'] = 0.0

            # 전략 수립에 필수적인 핵심 지표(RSI, MACD)가 생성되는 시점부터 데이터 유지
            # 장기 SMA가 NaN이더라도 dropna()로 인해 데이터가 통째로 날아가는 것을 방지하기 위해 
            # 필수 지표 컬럼들만 기준으로 dropna 수행
            essential_cols = ['rsi', 'macd', 'macd_signal', 'bb_mid']
            return df.dropna(subset=essential_cols)
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

    def get_composite_score(self, df: pd.DataFrame) -> float:
        """기술적 지표들을 종합하여 0~100 사이의 점수 산출

        구성 (최대 100pt):
          추세 (40pt)  : 단기SMA + 중기SMA60 + MACD
          모멘텀 (30pt): RSI 구간별 차등 (추세 맥락 반영, BB폭으로 신뢰도 보정)
          위치+거래량 (30pt): BB 위치(25pt) + 거래량 확인(5pt)
        """
        if df.empty or 'rsi' not in df.columns:
            return 50.0

        latest = df.iloc[-1]

        # ── 1. 추세 점수 (40pt max) ─────────────────────────────────
        trend_score = 0
        if latest['close'] > latest['sma_20']: trend_score += 10
        if latest['sma_5'] > latest['sma_20']: trend_score += 10

        try:
            sma_60_valid = pd.notna(latest['sma_60'])
        except KeyError:
            sma_60_valid = False

        if sma_60_valid:
            if latest['macd'] > latest['macd_signal']: trend_score += 15
            if latest['close'] > latest['sma_60']:     trend_score += 5
        else:
            if latest['macd'] > latest['macd_signal']: trend_score += 20

        # ADX DI 방향성 보너스: DI+ > DI- 이면 추세 방향 확인
        try:
            if pd.notna(latest.get('adx_pos')) and pd.notna(latest.get('adx_neg')):
                if latest['adx_pos'] > latest['adx_neg']:
                    trend_score = min(40, trend_score + 3)
        except (KeyError, TypeError):
            pass

        # ── 2. 모멘텀 점수 (30pt max) ───────────────────────────────
        # RSI 구간별 점수 — 추세 맥락(MACD 방향) 반영
        # - 상승 추세(MACD↑): 강한 RSI가 긍정 신호 (75 이상도 패널티 없음)
        # - 하락/중립(MACD↓): 과매도 반등 구간이 최적
        mom_score = 0
        rsi = latest['rsi']
        is_uptrend = latest['macd'] > latest['macd_signal']

        if is_uptrend:
            # 상승 추세: RSI 높을수록 추세 강도 확인 → 과매수 패널티 최소화
            if 55 <= rsi <= 75:    mom_score += 30  # 핵심 상승 구간 (최적)
            elif rsi > 75:         mom_score += 24  # 강한 과매수 — 모멘텀 강함
            elif 45 <= rsi < 55:   mom_score += 20  # 추세 초입
            elif 35 <= rsi < 45:   mom_score += 12  # 추세 약화 경고
            else:                  mom_score += 6   # RSI < 35: 상승 추세인데 하락 — 신뢰 저하
        else:
            # 하락/중립 추세: 과매도 반등 구간이 최적
            if 35 <= rsi <= 50:    mom_score += 30  # 과매도 탈출, 반등 준비 (최적)
            elif 30 <= rsi < 35:   mom_score += 24  # 깊은 과매도, 반등 기대
            elif rsi < 30:         mom_score += 18  # 심한 과매도, 단기 반등 가능
            elif 50 < rsi <= 65:   mom_score += 14  # 중립 ~ 완만한 상승
            elif 65 < rsi <= 75:   mom_score += 8   # 하락 추세인데 RSI 높음 — 약한 신호
            else:                  mom_score += 4   # RSI > 75 + 하락 추세 — 과열 경고

        # BB폭 신뢰도 보정: 밴드가 매우 좁으면(스퀴즈) 돌파 방향 불확실 → ±3pt 조정
        try:
            bb_range = latest['bb_high'] - latest['bb_low']
            bb_width_ratio = bb_range / latest['bb_mid'] if latest['bb_mid'] != 0 else 0.05
            if bb_width_ratio < 0.03:   # 극단적 스퀴즈 — 아직 방향 미결정
                mom_score = max(0, mom_score - 3)
            elif bb_width_ratio > 0.12: # 밴드 확장 — 추세 명확
                mom_score = min(30, mom_score + 2)
        except (KeyError, TypeError, ZeroDivisionError):
            pass

        # ── 3. 가격 위치 + 거래량 확인 (30pt max) ───────────────────
        vol_score = 0
        bb_range = latest['bb_high'] - latest['bb_low']
        bb_pos   = (latest['close'] - latest['bb_low']) / bb_range if bb_range != 0 else 0.5

        # BB 위치 (20pt): MACD 방향에 따라 최적 구간 이동
        if is_uptrend:
            if 0.4 <= bb_pos <= 0.75:   vol_score += 20
            elif 0.75 < bb_pos <= 0.9:  vol_score += 14
            elif 0.2 <= bb_pos < 0.4:   vol_score += 11
            elif bb_pos > 0.9:          vol_score += 6
            else:                        vol_score += 2
        else:
            if 0.2 <= bb_pos <= 0.5:    vol_score += 20
            elif 0.5 < bb_pos <= 0.7:   vol_score += 14
            elif 0.1 <= bb_pos < 0.2:   vol_score += 10
            elif 0.7 < bb_pos < 0.9:    vol_score += 6
            else:                        vol_score += 2

        # CMF 자금 흐름 (5pt)
        try:
            cmf_val = float(latest['cmf'])
            if cmf_val > 0.05:   vol_score += 5
            elif cmf_val > 0:    vol_score += 3
        except (KeyError, TypeError):
            pass

        # 거래량 확인 (5pt)
        try:
            if latest['vol_sma_20'] > 0 and latest['volume'] / latest['vol_sma_20'] >= 1.5:
                vol_score += 5
        except (KeyError, TypeError, ZeroDivisionError):
            pass

        return float(trend_score + mom_score + vol_score)

indicators = IndicatorCalculator()
