"""
ML 모델 학습 엔진
=================
koreanstocks train 명령어 및 train_models.py 양쪽에서 호출되는 공통 학습 로직.
패키지에 포함되므로 pip/pipx 전역설치 환경에서도 동작합니다.
"""

import os
import json
import time
import pickle
import hashlib
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
import joblib
import xgboost as xgb

from koreanstocks.core.config import config
from koreanstocks.core.data.provider import data_provider
from koreanstocks.core.engine.indicators import indicators

logger = logging.getLogger("koreanstocks.trainer")

# ───────────────────────────── 경로 설정 ─────────────────────────────

MODEL_DIR       = os.path.join(config.BASE_DIR, "models", "saved", "prediction_models")
PARAMS_DIR      = os.path.join(config.BASE_DIR, "models", "saved", "model_params")
PYKRX_CACHE_DIR = os.path.join(config.BASE_DIR, "data", "storage")

# ───────────────────────────── 학습 종목 목록 ─────────────────────────────

DEFAULT_TRAINING_STOCKS: List[str] = [
    # ── KOSPI 대형주 (20개) ──────────────────────────────────
    '005930',  # 삼성전자          반도체
    '000660',  # SK하이닉스        반도체
    '035420',  # NAVER             인터넷
    '005380',  # 현대차            자동차
    '051910',  # LG화학            화학/2차전지
    '006400',  # 삼성SDI           2차전지
    '035720',  # 카카오            인터넷/플랫폼
    '068270',  # 셀트리온          바이오(KOSPI)
    '105560',  # KB금융            은행
    '055550',  # 신한지주          은행
    '003550',  # LG                지주
    '096770',  # SK이노베이션      에너지/2차전지
    '028260',  # 삼성물산          건설/무역
    '066570',  # LG전자            가전/부품
    '017670',  # SK텔레콤          통신
    '030200',  # KT                통신
    '032830',  # 삼성생명          보험
    '000270',  # 기아              자동차
    '012330',  # 현대모비스        자동차부품
    '011170',  # 롯데케미칼        화학

    # ── KOSPI 미학습 섹터 (13개) ────────────────────────────
    '207940',  # 삼성바이오로직스  바이오 대형
    '326030',  # SK바이오팜        바이오 중형
    '259960',  # 크래프톤          게임
    '005490',  # POSCO홀딩스       철강
    '004020',  # 현대제철          철강
    '010140',  # 삼성중공업        조선
    '329180',  # HD현대중공업      조선
    '000720',  # 현대건설          건설
    '139480',  # 이마트            유통
    '097950',  # CJ제일제당        식품
    '006800',  # 미래에셋증권      증권
    '000810',  # 삼성화재          보험(손해)
    '032640',  # LG유플러스        통신

    # ── KOSPI 금융 (7개) ─────────────────────────────────────
    '086790',  # 하나금융지주      은행
    '316140',  # 우리금융지주      은행
    '138040',  # 메리츠금융지주    금융
    '016360',  # 삼성증권          증권
    '039490',  # 키움증권          증권
    '001450',  # 현대해상          보험(손해)
    '005830',  # DB손해보험        보험(손해)

    # ── KOSPI 방산/항공 (4개) ────────────────────────────────
    '012450',  # 한화에어로스페이스  방산
    '047810',  # 한국항공우주       방산/항공
    '079550',  # LIG넥스원          방산
    '003490',  # 대한항공            항공

    # ── KOSPI 화장품 (2개) ───────────────────────────────────
    '090430',  # 아모레퍼시픽       화장품
    '051900',  # LG생활건강         화장품

    # ── KOSPI 제약 (5개) ─────────────────────────────────────
    '128940',  # 한미약품           제약
    '000100',  # 유한양행           제약
    '006280',  # 녹십자             제약
    '185750',  # 종근당             제약
    '069620',  # 대웅제약           제약

    # ── KOSPI IT/게임/전자부품 (6개) ─────────────────────────
    '036570',  # 엔씨소프트         게임
    '251270',  # 넷마블             게임
    '009150',  # 삼성전기           전자부품
    '011070',  # LG이노텍           전자부품
    '018260',  # 삼성SDS            IT서비스
    '323410',  # 카카오뱅크         인터넷은행

    # ── KOSPI 에너지/화학/소재 (6개) ─────────────────────────
    '010950',  # S-Oil              정유
    '011780',  # 금호석유           화학
    '003670',  # 포스코퓨처엠       이차전지소재
    '020150',  # 일진머티리얼즈     이차전지소재
    '010060',  # OCI홀딩스          화학
    '028300',  # 에이치엘비         바이오

    # ── KOSPI 건설/중공업 (7개) ──────────────────────────────
    '006360',  # GS건설             건설
    '028050',  # 삼성엔지니어링     엔지니어링
    '034020',  # 두산에너빌리티     에너지/원전
    '241560',  # 두산밥캣           기계
    '298040',  # 효성중공업         중공업/전력기기
    '010120',  # LS일렉트릭         전력기기
    '042660',  # 한화오션           조선

    # ── KOSPI 유통/물류/식품/해운 (6개) ─────────────────────
    '023530',  # 롯데쇼핑           유통
    '086280',  # 현대글로비스       물류
    '000120',  # CJ대한통운         물류
    '004370',  # 농심               식품
    '271560',  # 오리온             식품
    '011200',  # HMM                해운

    # ── KOSPI 지주/기타 (8개) ────────────────────────────────
    '034730',  # SK                 지주
    '078930',  # GS                 에너지/유통
    '000880',  # 한화               방산/화학
    '103140',  # 풍산               비철금속
    '001040',  # CJ                 지주
    '112610',  # 씨에스윈드         풍력타워
    '298050',  # 효성첨단소재       소재
    '180640',  # 한진칼             항공지주

    # ── KOSDAQ 대표주 (7개) ──────────────────────────────────
    '247540',  # 에코프로비엠       2차전지 소재
    '086520',  # 에코프로           2차전지 지주
    '196170',  # 알테오젠           바이오
    '068760',  # 셀트리온제약       바이오
    '145020',  # 휴젤               바이오/뷰티
    '293490',  # 카카오게임즈       게임
    '112040',  # 위메이드           게임/블록체인

    # ── KOSDAQ 이차전지소재 (5개) ────────────────────────────
    '066970',  # L&F               이차전지소재
    '278280',  # 천보               이차전지소재
    '348370',  # 엔켐               이차전지소재
    '121600',  # 나노신소재         이차전지소재
    '336370',  # 솔루스첨단소재     이차전지소재

    # ── KOSDAQ 바이오/제약 (8개) ─────────────────────────────
    '096530',  # 씨젠               바이오/진단
    '095700',  # 제넥신             바이오
    '328130',  # 루닛               AI의료
    '338220',  # 뷰노               AI의료
    '086900',  # 메디톡스           바이오/뷰티
    '008930',  # 한미사이언스       제약
    '086450',  # 동국제약           제약
    '085670',  # 케어젠             바이오

    # ── KOSDAQ 의료기기 (4개) ────────────────────────────────
    '214150',  # 클래시스           의료기기/뷰티
    '214450',  # 파마리서치         제약/뷰티
    '048260',  # 오스템임플란트     의료기기
    '145720',  # 덴티움             의료기기

    # ── KOSDAQ 뷰티/생활 (5개) ───────────────────────────────
    '278470',  # 에이피알           뷰티
    '237880',  # 클리오             뷰티
    '161890',  # 한국콜마           뷰티
    '192820',  # 코스맥스           뷰티
    '018290',  # 브이티             뷰티

    # ── KOSDAQ 반도체장비/소재 (12개) ────────────────────────
    '036930',  # 주성엔지니어링     반도체장비
    '240810',  # 원익IPS            반도체장비
    '042700',  # 한미반도체         반도체장비
    '058470',  # 리노공업           반도체
    '095340',  # ISC                반도체
    '222800',  # 심텍               반도체PCB
    '067310',  # 하나마이크론       반도체패키징
    '319660',  # 피에스케이         반도체장비
    '090460',  # 비에이치           FPCB
    '183300',  # 코미코             반도체부품
    '102710',  # 이엔에프테크놀로지 반도체소재
    '029460',  # 케이씨텍           반도체장비

    # ── KOSDAQ 엔터/미디어 (5개) ─────────────────────────────
    '041510',  # SM엔터테인먼트     엔터
    '035900',  # JYP엔터테인먼트   엔터
    '122870',  # YG엔터테인먼트    엔터
    '253450',  # 스튜디오드래곤    드라마제작
    '067160',  # 아프리카TV         미디어

    # ── KOSDAQ 게임 (6개) ────────────────────────────────────
    '263750',  # 펄어비스           게임
    '192080',  # 더블유게임즈       게임
    '069080',  # 웹젠               게임
    '078340',  # 컴투스             게임
    '225570',  # 넥슨게임즈         게임
    '095660',  # 네오위즈           게임

    # ── KOSDAQ IT서비스/장비 (5개) ───────────────────────────
    '030190',  # NICE평가정보       IT서비스
    '056190',  # 에스에프에이       장비
    '079940',  # 가비아             IT/도메인
    '054040',  # 포스코DX           IT/자동화
    '196180',  # 파크시스템스       반도체/측정

    # ── KOSDAQ 화학/소재/기타 (5개) ──────────────────────────
    '025900',  # 동화기업           화학
    '005680',  # 코스모화학         화학
    '357780',  # 솔브레인홀딩스     반도체소재
    '091810',  # 두산테스나         반도체
    '101490',  # 에스앤에스텍       반도체부품
]

# ───────────────────────────── 모델 설정 ─────────────────────────────

# ─────────────────────────────────────────────────────────────────────────
# 이진 분류 설정
# 타깃: 10거래일 후 수익률이 상위 30%이면 1 (매수 신호), 나머지 0
# 지표: AUC-ROC (랜덤 = 0.5, 목표 ≥ 0.55)
# ─────────────────────────────────────────────────────────────────────────
TOP_K_PERCENTILE    = 0.75   # 상위 25% = 1 (rank pct ≥ 0.75)
BOTTOM_K_PERCENTILE = 0.25   # 하위 25% = 0 (rank pct ≤ 0.25), 중간 50% 제외

MODEL_CONFIGS: Dict[str, dict] = {
    'random_forest': {
        'class': RandomForestClassifier,
        'params': dict(
            n_estimators=300, max_depth=4, min_samples_split=20,
            min_samples_leaf=20, max_features=0.5,
            class_weight='balanced', random_state=42, n_jobs=-1,
        ),
    },
    'gradient_boosting': {
        'class': GradientBoostingClassifier,
        'params': dict(
            n_estimators=200, learning_rate=0.05, max_depth=2,
            min_samples_leaf=25, subsample=0.7, random_state=42,
        ),
    },
    'xgboost': {
        'class': xgb.XGBClassifier,
        'params': dict(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=30,
            reg_alpha=1.0, reg_lambda=3.0,
            scale_pos_weight=1.0,   # 중립 구간 제거로 클래스 균형 (50:50)
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0,
        ),
    },
}

BASE_FEATURE_COLS = [
    # ── 변동성 / 추세 강도 ────────────────────────────────────
    'atr_ratio',            # 변동성 (ATR/가격) — importance 1위
    'adx',                  # 추세 강도 (방향 무관)
    'adx_di_diff',          # DI+ - DI- (추세 방향성 — ADX 보완)
    'bb_width',             # 볼린저 밴드 폭 (변동성 압축 감지)
    # ── 중기 모멘텀 / 상대강도 ────────────────────────────────
    'rs_vs_mkt_3m',         # KOSPI 대비 3개월 초과수익
    'rs_vs_mkt_1m',         # KOSPI 대비 1개월 초과수익
    'high_52w_ratio',       # 52주 고가 대비 현재가 (추세 위치)
    'mom_accel',            # 모멘텀 가속도 (1m - 3m/3)
    # ── 추세 / 가격 모멘텀 ────────────────────────────────────
    'macd_diff',            # MACD 다이버전스 (추세 전환)
    'macd_diff_change',     # MACD 다이버전스 변화율
    'macd_slope_5d',        # MACD 다이버전스 5일 기울기 (모멘텀 가속)
    'price_sma_5_ratio',    # 단기 추세 (가격/SMA5)
    # ── 반전/패턴 신호 ────────────────────────────────────────
    'fisher',               # Fisher Transform (극값=반전)
    'bullish_fractal_5d',   # Williams 강세 프랙탈 5일
    # ── 거래량 방향성 ─────────────────────────────────────────
    'cmf',                  # Chaikin Money Flow (매수/매도 압력)
    'vzo',                  # Volume Zone Oscillator
    'obv_change',           # OBV 변화율
    'vol_ratio',            # 거래량 / 20일 평균 비율
    'vol_change',           # 거래량 전일 대비 변화율
    'rsi_mfi_div',          # RSI-MFI 다이버전스 (가격강도 vs 거래량강도)
    # ── 스퀴즈 모멘텀 ─────────────────────────────────────────
    'sqzmi',                # Squeeze Momentum (BB 내 MACD 압축 돌파)
    # ── 캔들 / 단기 신호 ──────────────────────────────────────
    'candle_body',          # 캔들 실체 (상승/하락 강도)
    # ── 거시경제 ──────────────────────────────────────────────
    'vix_level',            # VIX 공포지수
    'vix_change_5d',        # VIX 5일 변화율
    'sp500_1m',             # S&P500 1개월 수익률
]  # 25개 피처

PYKRX_FEATURE_COLS = [
    'pbr', 'per', 'div', 'pbr_xs', 'per_xs',
    'foreign_5d_ratio', 'inst_5d_ratio', 'foreign_xs', 'inst_xs',
]

FEATURE_COLS = BASE_FEATURE_COLS  # PyKrx 제외 — 현재 수집 불가로 노이즈만 추가

MIN_STOCKS_PER_DATE = 5

# ───────────────────────────── 데이터 수집 ─────────────────────────────

def build_features(df_ind: pd.DataFrame,
                   market_df: pd.DataFrame = None,
                   macro_df: pd.DataFrame = None) -> pd.DataFrame:
    """지표 DataFrame에서 모델 입력 특성을 추출 (prediction_model._extract_features와 동일)."""
    feat = pd.DataFrame(index=df_ind.index)

    feat['rsi']                = df_ind['rsi']
    feat['macd_diff']          = df_ind['macd_diff']
    feat['price_sma_20_ratio'] = df_ind['close'] / df_ind['sma_20']
    feat['vol_change']         = df_ind['volume'].pct_change()
    feat['price_sma_5_ratio']  = df_ind['close'] / df_ind['sma_5']
    feat['rsi_change']         = df_ind['rsi'].diff()
    feat['macd_diff_change']   = df_ind['macd_diff'].diff()
    feat['macd_slope_5d']      = df_ind['macd_diff'].diff(5)

    bb_range = (df_ind['bb_high'] - df_ind['bb_low']).replace(0, np.nan)
    feat['bb_position'] = (df_ind['close'] - df_ind['bb_low']) / bb_range
    feat['bb_width']    = bb_range / df_ind['bb_mid']
    feat['vol_ratio']   = df_ind['volume'] / df_ind['vol_sma_20'].replace(0, np.nan)

    feat['stoch_k']   = df_ind['stoch_k']
    feat['stoch_d']   = df_ind['stoch_d']
    feat['cci']       = df_ind['cci']
    feat['atr_ratio'] = df_ind['atr'] / df_ind['close']
    feat['candle_body'] = (df_ind['close'] - df_ind['open']) / df_ind['open']
    feat['obv_change']  = df_ind['obv'].pct_change().clip(-1, 1)

    feat['return_1m']      = df_ind['close'].pct_change(20)
    feat['return_3m']      = df_ind['close'].pct_change(60)
    feat['high_52w_ratio'] = df_ind['close'] / df_ind['close'].rolling(252, min_periods=60).max()
    feat['mom_accel']      = feat['return_1m'] - feat['return_3m'] / 3.0

    if market_df is not None and not market_df.empty:
        aligned = market_df.reindex(feat.index, method='ffill')
        feat['rs_vs_mkt_1m'] = (feat['return_1m'] - aligned.get('return_1m', 0)).fillna(0)
        feat['rs_vs_mkt_3m'] = (feat['return_3m'] - aligned.get('return_3m', 0)).fillna(0)
    else:
        feat['rs_vs_mkt_1m'] = 0.0
        feat['rs_vs_mkt_3m'] = 0.0

    # ── 신규 12개 피처 ────────────────────────────────────────

    # ADX 추세 강도 + DI 방향 차이
    feat['adx']         = df_ind['adx']
    feat['adx_di_diff'] = df_ind['adx_pos'] - df_ind['adx_neg']

    # VWAP 대비 현재가 위치 (기관 평균매수가 기준선)
    feat['vwap_ratio'] = (
        df_ind['close'] / df_ind['vwap'].replace(0, np.nan)
    ).clip(0.5, 2.0)

    # Donchian Channel 위치 — 0=20일 최저가, 1=20일 최고가 (신고가 돌파 모멘텀)
    dc_range = (df_ind['dc_high'] - df_ind['dc_low']).replace(0, np.nan)
    feat['dc_position'] = (
        (df_ind['close'] - df_ind['dc_low']) / dc_range
    ).clip(0, 1)

    # 거래량 방향성 지표
    feat['cmf']         = df_ind['cmf']
    feat['mfi']         = df_ind['mfi']
    feat['rsi_mfi_div'] = df_ind['rsi'] - df_ind['mfi']   # 양수=가격 강도>거래량 강도

    # Squeeze Momentum (finta) — 스퀴즈 돌파 신호
    feat['sqzmi']       = df_ind['sqzmi']
    feat['sqzmi_accel'] = df_ind['sqzmi'].diff().fillna(0)  # 모멘텀 가속도

    # Volume Zone Oscillator (finta)
    feat['vzo']    = df_ind['vzo']

    # Fisher Transform (finta) — 극값=반전 신호, -5~+5 클리핑 적용됨
    feat['fisher'] = df_ind['fisher']

    # Williams Fractal 5일 내 강세 프랙탈 여부 (finta)
    feat['bullish_fractal_5d'] = (
        df_ind['bullish_fractal'].rolling(5, min_periods=1).max()
    )

    # ── 거시경제 3개 피처 ────────────────────────────────────
    if macro_df is not None and not macro_df.empty:
        aligned = macro_df.reindex(feat.index, method='ffill')
        feat['vix_level']     = aligned.get('vix_level',     pd.Series(dtype=float))
        feat['vix_change_5d'] = aligned.get('vix_change_5d', pd.Series(dtype=float))
        feat['sp500_1m']      = aligned.get('sp500_1m',      pd.Series(dtype=float))
    else:
        feat['vix_level']     = 20.0   # VIX 역사적 평균 근처
        feat['vix_change_5d'] = 0.0
        feat['sp500_1m']      = 0.0

    return feat.replace([np.inf, -np.inf], np.nan).dropna()


def _fetch_macro_data(period: str) -> pd.DataFrame:
    """yfinance로 VIX·S&P500 거시경제 데이터 수집 (학습 전 1회 호출).

    Returns: DataFrame(vix_level, vix_change_5d, sp500_1m), 인덱스=날짜(tz-naive)
    """
    try:
        import yfinance as yf
        raw = yf.download(['^VIX', '^GSPC'], period=period, progress=False)
        if raw.empty:
            return pd.DataFrame()
        close = raw.xs('Close', level=0, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw['Close']
        macro = pd.DataFrame(index=close.index)
        macro.index = pd.to_datetime(macro.index).tz_localize(None)
        macro['vix_level']     = close['^VIX'].values
        macro['vix_change_5d'] = close['^VIX'].pct_change(5).values
        macro['sp500_1m']      = close['^GSPC'].pct_change(20).values
        macro = macro.ffill()
        logger.info(f"  [거시경제] VIX·S&P500 {len(macro)}개 데이터 로드 완료")
        return macro
    except Exception as e:
        logger.warning(f"  [거시경제] VIX·S&P500 수집 실패: {e} — macro 피처는 0으로 채워집니다.")
        return pd.DataFrame()


def _fetch_market_returns(symbol: str, period: str) -> pd.DataFrame:
    """시장 지수의 롤링 수익률 DataFrame 반환 (시장 상대강도 피처 계산용).

    FDR(KS11) 실패 시 yfinance ^KS11 로 폴백.
    """
    try:
        raw = data_provider.get_ohlcv(symbol, period=period)
        if not raw.empty:
            mkt = pd.DataFrame(index=raw.index)
            mkt['return_1m'] = raw['close'].pct_change(20)
            mkt['return_3m'] = raw['close'].pct_change(60)
            logger.info(f"  [시장] {symbol} 수익률 데이터 {len(mkt)}개 로드 완료")
            return mkt
    except Exception as e:
        logger.warning(f"  [시장] {symbol} FDR 로드 실패: {e}")

    # ── yfinance 폴백 ─────────────────────────────────────────
    yf_map = {'KS11': '^KS11', 'KQ11': '^KQ11'}
    yf_sym = yf_map.get(symbol)
    if yf_sym:
        try:
            import yfinance as yf
            raw = yf.download(yf_sym, period=period, progress=False)
            if not raw.empty:
                close = raw.xs('Close', level=0, axis=1).iloc[:, 0] \
                    if isinstance(raw.columns, pd.MultiIndex) else raw['Close']
                close.index = pd.to_datetime(close.index).tz_localize(None)
                mkt = pd.DataFrame(index=close.index)
                mkt['return_1m'] = close.pct_change(20)
                mkt['return_3m'] = close.pct_change(60)
                logger.info(f"  [시장] {yf_sym} (yfinance 폴백) {len(mkt)}개 로드 완료")
                return mkt
        except Exception as e2:
            logger.warning(f"  [시장] {yf_sym} yfinance 폴백도 실패: {e2}")
    return pd.DataFrame()


def _fetch_pykrx_all_stocks(codes: List[str], period: str) -> Dict[str, Dict]:
    """PyKrx로 전 종목의 펀더멘털·외국인/기관 데이터를 일괄 로드 (파일 캐시 사용)."""
    cache_key  = hashlib.md5((','.join(sorted(codes)) + period).encode()).hexdigest()[:8]
    cache_file = os.path.join(PYKRX_CACHE_DIR, f"pykrx_cache_{cache_key}.pkl")

    _CACHE_TTL_DAYS = 7
    if os.path.exists(cache_file):
        age_days = (datetime.now().timestamp() - os.path.getmtime(cache_file)) / 86400
        if age_days <= _CACHE_TTL_DAYS:
            try:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                logger.info(f"[PyKrx] 캐시 로드: {cache_file} ({len(cached)}종목, {age_days:.1f}일 전)")
                return cached
            except Exception as e:
                logger.warning(f"[PyKrx] 캐시 로드 실패: {e}, 재수집합니다.")
        else:
            logger.info(f"[PyKrx] 캐시 만료 ({age_days:.1f}일 > {_CACHE_TTL_DAYS}일) — 재수집합니다.")

    try:
        from pykrx import stock as pykrx_stock
    except ImportError:
        logger.warning("[PyKrx] pykrx 미설치 — 펀더멘털/투자자 피처 건너뜀")
        return {}

    period_map = {'1y': 365, '2y': 730, '3y': 1095, '3m': 90, '6m': 180}
    days     = period_map.get(period, 730)
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days + 30)
    start_str, end_str = start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d')

    result: Dict[str, Dict] = {}
    for i, code in enumerate(codes):
        try:
            fund_df  = pykrx_stock.get_market_fundamental_by_date(start_str, end_str, code)
            trade_df = pykrx_stock.get_market_trading_value_by_date(start_str, end_str, code)
            fund_df.index  = pd.to_datetime(fund_df.index)
            trade_df.index = pd.to_datetime(trade_df.index)
            fund_cols  = [c for c in ['PBR', 'PER', 'DIV'] if c in fund_df.columns]
            trade_cols = [c for c in ['외국인합계', '기관합계'] if c in trade_df.columns]
            result[code] = {
                'fundamental': fund_df[fund_cols]   if fund_cols  else pd.DataFrame(),
                'trading':     trade_df[trade_cols] if trade_cols else pd.DataFrame(),
            }
        except Exception as e:
            logger.debug(f"[PyKrx] {code} 수집 실패: {e}")
            result[code] = {'fundamental': pd.DataFrame(), 'trading': pd.DataFrame()}
        if (i + 1) % 20 == 0:
            logger.info(f"[PyKrx] {i+1}/{len(codes)} 종목 수집 완료")

    try:
        os.makedirs(PYKRX_CACHE_DIR, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"[PyKrx] 캐시 저장 완료: {cache_file}")
    except Exception as e:
        logger.warning(f"[PyKrx] 캐시 저장 실패: {e}")

    return result


def _collect_stock_features(code: str, period: str, future_days: int,
                             market_df: pd.DataFrame = None,
                             pykrx_cache: Optional[Dict] = None,
                             macro_df: pd.DataFrame = None) -> pd.DataFrame:
    """단일 종목의 (날짜, 특성, 미래수익률) DataFrame 반환."""
    try:
        df = data_provider.get_ohlcv(code, period=period)
        if df is None or df.empty or len(df) < 60:
            logger.warning(f"  [{code}] 데이터 부족 ({len(df) if df is not None else 0}행) — 건너뜀")
            return pd.DataFrame()

        df_ind = indicators.calculate_all(df)
        if df_ind.empty:
            return pd.DataFrame()

        feat = build_features(df_ind, market_df=market_df, macro_df=macro_df)
        if len(feat) <= future_days:
            return pd.DataFrame()

        close      = df_ind['close'].reindex(feat.index)
        future_ret = (close.shift(-future_days) - close) / close
        valid_idx  = feat.index[:-future_days]
        result     = feat.loc[valid_idx].copy()
        result['raw_return'] = future_ret.loc[valid_idx]

        if pykrx_cache and code in pykrx_cache:
            pkx     = pykrx_cache[code]
            fund_df = pkx.get('fundamental', pd.DataFrame())
            if not fund_df.empty:
                for src_col, dst_col in [('PBR', 'pbr'), ('PER', 'per'), ('DIV', 'div')]:
                    if src_col in fund_df.columns:
                        result[dst_col] = fund_df[src_col].reindex(result.index, method='ffill')

            trade_df = pkx.get('trading', pd.DataFrame())
            if not trade_df.empty:
                close_s   = df_ind['close'].reindex(result.index)
                vol_s     = df_ind['volume'].reindex(result.index)
                turnover  = (close_s * vol_s).rolling(5, min_periods=1).sum().replace(0, np.nan)
                for src_col, dst_col in [('외국인합계', 'foreign_5d_ratio'), ('기관합계', 'inst_5d_ratio')]:
                    if src_col in trade_df.columns:
                        aligned = trade_df[src_col].reindex(result.index, method='ffill').fillna(0)
                        net_5d  = aligned.rolling(5, min_periods=1).sum()
                        result[dst_col] = (net_5d / turnover).clip(-0.5, 0.5).fillna(0)

        base_subset = [c for c in BASE_FEATURE_COLS + ['raw_return'] if c in result.columns]
        valid = result.dropna(subset=base_subset)
        logger.info(f"  [{code}] {len(valid)}개 샘플 수집")
        return valid

    except Exception as exc:
        logger.error(f"  [{code}] 처리 오류: {exc}")
        return pd.DataFrame()


def fetch_train_test_samples(
    codes: List[str], period: str, future_days: int, test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """크로스섹셔널 상대 강도 순위를 타깃으로 하는 학습/검증 세트 수집."""
    market_df = _fetch_market_returns('KS11', period)
    if market_df.empty:
        logger.warning("KS11 시장 데이터 미수신 — rs_vs_mkt 피처는 0으로 채워집니다.")
        market_df = None

    logger.info("[거시경제] VIX·S&P500 데이터 수집 중...")
    macro_df = _fetch_macro_data(period)
    if macro_df.empty:
        macro_df = None

    logger.info("[PyKrx] 펀더멘털/투자자 데이터 수집 중...")
    pykrx_cache = _fetch_pykrx_all_stocks(codes, period)
    if not pykrx_cache:
        logger.warning("[PyKrx] 데이터 없음 — 펀더멘털/투자자 피처는 중립값으로 처리됩니다.")

    frames = [_collect_stock_features(c, period, future_days, market_df, pykrx_cache, macro_df)
              for c in codes]
    frames = [f for f in frames if not f.empty]

    if not frames:
        raise RuntimeError("수집된 학습 샘플이 없습니다. 네트워크 또는 종목 코드를 확인하세요.")

    df_all = pd.concat(frames)

    for raw_col, xs_col in [('pbr', 'pbr_xs'), ('per', 'per_xs'),
                             ('foreign_5d_ratio', 'foreign_xs'), ('inst_5d_ratio', 'inst_xs')]:
        if raw_col in df_all.columns:
            df_all[xs_col] = df_all.groupby(df_all.index)[raw_col].rank(pct=True) * 100.0
        else:
            df_all[xs_col] = 50.0

    for col in PYKRX_FEATURE_COLS:
        if col in df_all.columns:
            median_val = df_all[col].median()
            df_all[col] = df_all[col].fillna(median_val if pd.notna(median_val) else 50.0)
        else:
            df_all[col] = 50.0

    # 이진 타깃 (중립 구간 제거): 상위 25% = 1, 하위 25% = 0, 중간 50% 제외
    rank_pct = df_all.groupby(df_all.index)['raw_return'].rank(pct=True)
    df_all['target'] = np.nan
    df_all.loc[rank_pct >= TOP_K_PERCENTILE,    'target'] = 1
    df_all.loc[rank_pct <= BOTTOM_K_PERCENTILE, 'target'] = 0
    df_all = df_all.dropna(subset=['target'])
    df_all['target'] = df_all['target'].astype(int)

    stocks_per_date = df_all.groupby(df_all.index)['raw_return'].count()
    valid_dates     = stocks_per_date[stocks_per_date >= MIN_STOCKS_PER_DATE].index
    df_all          = df_all[df_all.index.isin(valid_dates)]

    all_dates  = sorted(df_all.index.unique())
    n_dates    = len(all_dates)
    split_idx  = int(n_dates * (1.0 - test_ratio))
    split_date = all_dates[split_idx]

    keep_cols = [c for c in FEATURE_COLS if c in df_all.columns] + ['target']
    df_train  = df_all[df_all.index <  split_date][keep_cols].dropna()
    df_test   = df_all[df_all.index >= split_date][keep_cols].dropna()

    pos_rate = df_train['target'].mean()
    logger.info(
        f"\n타깃: {future_days}거래일 후 수익률 상위 25% = 1 / 하위 25% = 0 (중간 50% 제외)"
        f"\n분할 기준일: {split_date.date()}"
        f"\n총 날짜: {n_dates} (학습 {split_idx}일 / 검증 {n_dates - split_idx}일)"
        f"\n총 샘플: 학습 {len(df_train)} / 검증 {len(df_test)}"
        f"\n날짜별 평균 레이블 종목 수: {stocks_per_date[valid_dates].mean():.1f}"
        f"\n학습 양성 비율: {pos_rate:.1%} (≈50%)"
    )
    return df_train, df_test


def train_and_save(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """모델 학습(이진 분류) → 평가 → 모델/스케일러/파라미터 저장.

    타깃: 10거래일 후 수익률 상위 25% = 1 / 하위 25% = 0 (중간 50% 제외, 이진 분류)
    지표: AUC-ROC (랜덤 기준선 0.5, 목표 ≥ 0.52)
    교차검증: 5-fold TimeSeriesSplit (날짜 기준 분할, cross-sectional 무결성 보장)
    """
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(PARAMS_DIR, exist_ok=True)

    feat_names = [c for c in FEATURE_COLS if c in df_train.columns]
    X_train = df_train[feat_names].values
    y_train = df_train['target'].values

    if df_test.empty:
        logger.warning("검증 세트가 없습니다. 학습 세트 성능만 기록됩니다.")
        X_test, y_test = X_train, y_train
    else:
        X_test = df_test[feat_names].values
        y_test = df_test['target'].values

    pos_rate = y_train.mean()
    logger.info(f"\n기준선 AUC: 0.5000  (랜덤 분류기)")
    logger.info(f"학습 샘플: {len(X_train)}, 검증 샘플: {len(X_test)}, 양성 비율: {pos_rate:.1%}\n")

    # 날짜 목록 (TimeSeriesSplit 기준)
    unique_dates = sorted(df_train.index.unique())

    results = []
    for name, cfg in MODEL_CONFIGS.items():
        logger.info(f"{'─'*40}")
        logger.info(f"  학습 중: {name}")
        t0 = time.time()

        # ── H: 5-fold TimeSeriesSplit 교차검증 (날짜 기준 분할) ──────────────
        tscv     = TimeSeriesSplit(n_splits=5)
        cv_aucs  = []
        for tr_d_idx, val_d_idx in tscv.split(unique_dates):
            tr_dates  = {unique_dates[i] for i in tr_d_idx}
            val_dates = {unique_dates[i] for i in val_d_idx}
            tr_mask   = df_train.index.isin(tr_dates)
            val_mask  = df_train.index.isin(val_dates)
            if tr_mask.sum() < 10 or val_mask.sum() < 10:
                continue
            cv_sc = StandardScaler()
            cv_m  = cfg['class'](**cfg['params'])
            cv_m.fit(cv_sc.fit_transform(X_train[tr_mask]), y_train[tr_mask])
            cv_p  = cv_m.predict_proba(cv_sc.transform(X_train[val_mask]))[:, 1]
            cv_aucs.append(roc_auc_score(y_train[val_mask], cv_p))
        cv_mean = float(np.mean(cv_aucs)) if cv_aucs else float('nan')
        cv_std  = float(np.std(cv_aucs))  if cv_aucs else float('nan')
        logger.info(f"  CV AUC (5-fold TS): {cv_mean:.4f} ± {cv_std:.4f}")

        # ── 최종 모델 학습 (전체 학습 세트 사용) ─────────────────────────────
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_train)
        X_te   = scaler.transform(X_test)

        model = cfg['class'](**cfg['params'])
        model.fit(X_tr, y_train)
        duration = time.time() - t0

        # AUC-ROC 평가
        train_proba = model.predict_proba(X_tr)[:, 1]
        test_proba  = model.predict_proba(X_te)[:, 1]
        # 캘리브레이션: 테스트 확률의 101분위수 저장 → predict 시 0~100 균등 스케일로 변환
        # train_proba 기준이면 과적합 분포(0.13~0.87)가 되어 실제 예측 범위를 반영 못 함
        calibration_points = np.percentile(test_proba, np.arange(0, 101)).tolist()
        train_auc   = roc_auc_score(y_train, train_proba)
        test_auc    = roc_auc_score(y_test,  test_proba)
        test_logloss = log_loss(y_test, test_proba)
        overfit_gap  = round(train_auc - test_auc, 4)

        logger.info(f"  AUC : {test_auc:.4f}  (학습 AUC: {train_auc:.4f}  과적합 gap: {overfit_gap:.4f})")
        logger.info(f"  LogLoss: {test_logloss:.4f}")
        logger.info(f"  소요: {duration:.1f}초")

        model_path  = os.path.join(MODEL_DIR, f"{name}_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{name}_scaler.pkl")
        joblib.dump(model,  model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"  저장: {model_path}")
        logger.info(f"  저장: {scaler_path}")

        from koreanstocks.core.engine.prediction_model import _MIN_MODEL_AUC
        quality_pass = test_auc >= _MIN_MODEL_AUC
        if quality_pass:
            logger.info(f"  ✅ 품질 게이트 통과 (test_auc={test_auc:.4f} ≥ {_MIN_MODEL_AUC})")
        else:
            logger.warning(
                f"  ⚠️  품질 게이트 미달 (test_auc={test_auc:.4f} < {_MIN_MODEL_AUC}) "
                f"— 저장은 하지만 예측 시 tech_score 폴백이 사용됩니다."
            )

        # Feature importance 추출
        feature_importances = []
        if hasattr(model, 'feature_importances_') and len(feat_names) == len(model.feature_importances_):
            fi_pairs = sorted(
                zip(feat_names, model.feature_importances_.tolist()),
                key=lambda x: x[1], reverse=True,
            )
            feature_importances = [[n, round(v, 6)] for n, v in fi_pairs]
            top5 = ', '.join(f"{n}({v:.3f})" for n, v in fi_pairs[:5])
            logger.info(f"  Top-5 피처: {top5}")

        version = f"{name}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        meta = {
            "parameters": {k: v for k, v in cfg['params'].items()
                           if k not in ('random_state', 'n_jobs', 'verbosity',
                                        'use_label_encoder', 'eval_metric')},
            "model_type":          "binary_classifier",
            "target_definition":   (
                f"top {int((1-TOP_K_PERCENTILE)*100)}% / "
                f"bottom {int(BOTTOM_K_PERCENTILE*100)}% return in {10}d (neutral zone)"
            ),
            "training_samples":    int(len(X_train)),
            "positive_rate":       round(float(pos_rate), 4),
            "cv_auc_mean":         round(cv_mean, 4) if not np.isnan(cv_mean) else None,
            "cv_auc_std":          round(cv_std,  4) if not np.isnan(cv_std)  else None,
            "train_auc":           round(train_auc,   4),
            "test_auc":            round(test_auc,    4),
            "test_logloss":        round(test_logloss, 4),
            "overfit_gap":         overfit_gap,
            "quality_pass":        quality_pass,
            "training_duration":   round(duration, 1),
            "saved_at":            datetime.now().isoformat(),
            "model_version":       version,
            "feature_importances": feature_importances,
            "calibration":         calibration_points,
        }
        params_path = os.path.join(PARAMS_DIR, f"{name}_params.json")
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        results.append((name, test_auc, cv_mean, train_auc, quality_pass))

    logger.info(f"\n{'═'*60}")
    logger.info("  학습 완료 요약  (이진 분류 / AUC-ROC)")
    logger.info(f"{'─'*60}")
    logger.info(f"  {'모델':<22} {'test AUC':>9} {'CV AUC':>9} {'train AUC':>9} {'갭':>6} {'품질'}")
    logger.info(f"{'─'*60}")
    for name, tauc, cv_mu, trauc, qpass in results:
        gap    = trauc - tauc
        mark   = "✅" if qpass else "⚠️ "
        cv_str = f"{cv_mu:.4f}" if not np.isnan(cv_mu) else "   N/A"
        logger.info(f"  {name:<22} {tauc:>9.4f} {cv_str:>9} {trauc:>9.4f} {gap:>6.4f}  {mark}")
    logger.info(f"{'═'*40}")
    logger.info(f"✅ 모든 모델 저장 완료  →  {MODEL_DIR}")


# ───────────────────────────── 공개 진입점 ─────────────────────────────

def run_training(
    period: str = "2y",
    future_days: int = 10,
    stocks: Optional[List[str]] = None,
    test_ratio: float = 0.2,
) -> None:
    """koreanstocks train 명령어 및 train_models.py 양쪽에서 호출하는 진입점.

    Args:
        period:      학습 데이터 기간 (예: '2y', '1y')
        future_days: 예측 대상 거래일 수 (기본값 10 = 2주, 중기 노이즈 최소화)
        stocks:      학습 종목 코드 리스트 (None이면 DEFAULT_TRAINING_STOCKS 사용)
        test_ratio:  검증 세트 비율 (0~1)
    """
    if stocks is None:
        stocks = DEFAULT_TRAINING_STOCKS

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    logger.info("=" * 40)
    logger.info("  ML 모델 학습 시작")
    logger.info(f"  종목 수     : {len(stocks)}")
    logger.info(f"  데이터 기간 : {period}")
    logger.info(f"  예측 기간   : {future_days}거래일 후")
    logger.info(f"  검증 비율   : {test_ratio * 100:.0f}% (시계열 후반부)")
    logger.info(f"  타깃 변수   : {future_days}거래일 후 수익률 상위 25%/하위 25% 이진 분류 (중간 50% 제외, AUC-ROC)")
    logger.info(f"  피처 수     : {len(FEATURE_COLS)}개 (기술적+TA+거시경제)")
    logger.info("=" * 40)

    logger.info("\n[1/2] 학습 데이터 수집 중...")
    df_train, df_test = fetch_train_test_samples(
        stocks, period=period, future_days=future_days, test_ratio=test_ratio,
    )

    logger.info("\n[2/2] 모델 학습 및 저장 중...")
    train_and_save(df_train, df_test)
