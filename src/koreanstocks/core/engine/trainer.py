"""
ML 모델 학습 엔진
=================
koreanstocks train 명령어 및 train_models.py 양쪽에서 호출되는 공통 학습 로직.
패키지에 포함되므로 pip/pipx 전역설치 환경에서도 동작합니다.
"""

import json
import socket
from pathlib import Path
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeout

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from koreanstocks.core.config import config
from koreanstocks.core.constants import MIN_MODEL_AUC, AUTO_TUNE_THRESHOLDS
from koreanstocks.core.data.provider import data_provider, fetch_macro_df, fetch_market_df
from koreanstocks.core.engine.indicators import indicators
from koreanstocks.core.engine.features import build_features, BASE_FEATURE_COLS
from koreanstocks.core.engine import tcn_model as _tcn

logger = logging.getLogger("koreanstocks.trainer")

# ───────────────────────────── 경로 설정 ─────────────────────────────

MODEL_DIR  = Path(config.BASE_DIR) / "models" / "saved" / "prediction_models"
PARAMS_DIR = Path(config.BASE_DIR) / "models" / "saved" / "model_params"

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
    # '048260',  # 오스템임플란트     의료기기 — 2023년 상장폐지 (MBK 인수)
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
    # '196180',  # 파크시스템스       반도체/측정 — FDR 데이터 없음

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
# 타깃: 10거래일 후 수익률 상위 25% = 1, 하위 25% = 0, 중간 50% 제외 (neutral zone)
# 지표: AUC-ROC (랜덤 = 0.5, 목표 ≥ 0.55)
# ─────────────────────────────────────────────────────────────────────────
TOP_K_PERCENTILE    = 0.75   # 상위 25% = 1 (rank pct ≥ 0.75)
BOTTOM_K_PERCENTILE = 0.25   # 하위 25% = 0 (rank pct ≤ 0.25), 중간 50% 제외
# neutral zone 34%→25%: 유효 샘플 68%→75% (+10%), 극단 신호 강도 소폭 희석

MODEL_CONFIGS: Dict[str, dict] = {
    'random_forest': {
        'class': RandomForestClassifier,
        'params': dict(
            n_estimators=300, max_depth=4, min_samples_split=30,
            min_samples_leaf=30, max_features=0.4, max_samples=0.8,
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
    'lightgbm': {
        'class': lgb.LGBMClassifier,
        'params': dict(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            num_leaves=4, min_child_samples=100,
            subsample=0.6, subsample_freq=1, colsample_bytree=0.5,
            reg_alpha=2.0, reg_lambda=5.0,
            class_weight='balanced', random_state=42, verbose=-1,
        ),
    },
    'catboost': {
        'class': CatBoostClassifier,
        'params': dict(
            iterations=200, depth=3, learning_rate=0.05,
            l2_leaf_reg=5.0, min_data_in_leaf=40,
            bootstrap_type='Bernoulli', subsample=0.7,
            auto_class_weights='Balanced',
            random_seed=42, verbose=0,
        ),
    },
    'xgboost_ranker': {
        'class': xgb.XGBRanker,
        'is_ranker': True,
        'params': dict(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=25,
            reg_alpha=1.0, reg_lambda=3.0,
            random_state=42, verbosity=0,
        ),
    },
}

# BASE_FEATURE_COLS 는 features.py 에서 import (단일 소스, 중복 정의 제거)
# 변경 시 features.py 만 수정하면 학습/추론 양쪽 자동 반영됨

MIN_STOCKS_PER_DATE = 5

# ───────────────────────── Auto-Tune 설정 ─────────────────────────────────────

# 모델별 랜덤 탐색 공간 (각 키=파라미터명, 값=후보값 리스트)
_AT_SEARCH_SPACES: Dict[str, Dict[str, list]] = {
    'random_forest': {
        'max_depth':         [3, 4, 5],
        'min_samples_leaf':  [15, 20, 30, 40, 50],
        'min_samples_split': [20, 25, 30, 40, 50],
        'max_features':      [0.3, 0.4, 0.5, 0.6],
        'max_samples':       [0.7, 0.8, 0.9],
    },
    'gradient_boosting': {
        'n_estimators':     [200, 300, 400, 500],
        'max_depth':        [2, 3, 4],
        'learning_rate':    [0.02, 0.03, 0.05, 0.08],
        'min_samples_leaf': [15, 20, 25, 30, 40],
        'subsample':        [0.5, 0.6, 0.7, 0.8],
    },
    'lightgbm': {
        'max_depth':         [2, 3],
        'min_child_samples': [50, 75, 100, 150, 200],
        'reg_alpha':         [1.0, 2.0, 3.0, 5.0],
        'reg_lambda':        [3.0, 5.0, 8.0, 10.0],
        'colsample_bytree':  [0.4, 0.5, 0.6],
    },
    'catboost': {
        'depth':            [2, 3, 4],
        'l2_leaf_reg':      [3.0, 5.0, 7.0, 10.0],
        'min_data_in_leaf': [20, 30, 40, 60, 80],
        'subsample':        [0.6, 0.7, 0.8],
    },
    'xgboost_ranker': {
        'max_depth':        [2, 3, 4],
        'min_child_weight': [15, 20, 25, 30, 40],
        'reg_alpha':        [0.5, 1.0, 2.0, 3.0],
        'reg_lambda':       [2.0, 3.0, 5.0, 7.0],
        'colsample_bytree': [0.5, 0.6, 0.7],
    },
}

# 진단별 규칙 기반 파라미터 조정 (op: 'mul'=곱, 'add'=덧셈, 'set'=대입)
_AT_ADJUST_RULES: Dict[str, Dict[str, Dict[str, tuple]]] = {
    'OVERFIT': {
        'random_forest':     {'min_samples_leaf': ('mul', 1.5), 'min_samples_split': ('mul', 1.5), 'max_depth': ('add', -1)},
        'gradient_boosting': {'min_samples_leaf': ('mul', 1.5), 'learning_rate': ('mul', 0.6)},
        'lightgbm':          {'min_child_samples': ('mul', 1.5), 'reg_alpha': ('mul', 1.5), 'reg_lambda': ('mul', 1.5)},
        'catboost':          {'min_data_in_leaf': ('mul', 1.5), 'l2_leaf_reg': ('mul', 1.5), 'depth': ('add', -1)},
        'xgboost_ranker':    {'min_child_weight': ('mul', 1.5), 'reg_alpha': ('mul', 1.5), 'reg_lambda': ('mul', 1.5)},
    },
    'UNDERFIT': {
        'random_forest':     {'min_samples_leaf': ('mul', 0.7), 'min_samples_split': ('mul', 0.7), 'max_depth': ('add', 1)},
        'gradient_boosting': {'min_samples_leaf': ('mul', 0.7), 'max_depth': ('add', 1)},
        'lightgbm':          {'min_child_samples': ('mul', 0.7), 'reg_alpha': ('mul', 0.7), 'reg_lambda': ('mul', 0.7)},
        'catboost':          {'min_data_in_leaf': ('mul', 0.7), 'l2_leaf_reg': ('mul', 0.7), 'depth': ('add', 1)},
        'xgboost_ranker':    {'min_child_weight': ('mul', 0.7), 'reg_alpha': ('mul', 0.7), 'reg_lambda': ('mul', 0.7)},
    },
    'UNSTABLE': {
        'random_forest':     {'min_samples_leaf': ('mul', 2.0), 'min_samples_split': ('mul', 2.0)},
        'gradient_boosting': {'min_samples_leaf': ('mul', 2.0), 'learning_rate': ('mul', 0.5), 'subsample': ('mul', 0.9)},
        'lightgbm':          {'min_child_samples': ('mul', 2.0), 'reg_alpha': ('mul', 2.0), 'reg_lambda': ('mul', 2.0)},
        'catboost':          {'min_data_in_leaf': ('mul', 2.0), 'l2_leaf_reg': ('mul', 2.0)},
        'xgboost_ranker':    {'min_child_weight': ('mul', 2.0), 'reg_alpha': ('mul', 2.0), 'reg_lambda': ('mul', 2.0)},
    },
    'WEAK': {
        'random_forest':     {'min_samples_leaf': ('mul', 0.7), 'max_features': ('mul', 1.25)},
        'gradient_boosting': {'learning_rate': ('mul', 1.5), 'subsample': ('mul', 1.1)},
        'lightgbm':          {'min_child_samples': ('mul', 0.7), 'reg_alpha': ('mul', 0.6), 'reg_lambda': ('mul', 0.6)},
        'catboost':          {'min_data_in_leaf': ('mul', 0.7), 'l2_leaf_reg': ('mul', 0.7)},
        'xgboost_ranker':    {'min_child_weight': ('mul', 0.7), 'colsample_bytree': ('mul', 1.1)},
    },
}

# 오버라이드 저장 제외 파라미터 (환경/재현성용, 하이퍼파라미터 아님)
_AT_SKIP_PARAMS: frozenset = frozenset({
    'random_state', 'random_seed', 'n_jobs', 'verbosity', 'verbose',
    'use_label_encoder', 'eval_metric', 'class_weight', 'auto_class_weights',
    'bootstrap_type',
})

# 모델별 "depth" 역할 파라미터명 — 방향 제약 검사에 사용
_DEPTH_PARAMS: Dict[str, str] = {
    'random_forest':     'max_depth',
    'gradient_boosting': 'max_depth',
    'lightgbm':          'max_depth',
    'catboost':          'depth',
    'xgboost_ranker':    'max_depth',
}

# Phase3 채택 기준: 원본 test_auc 대비 허용 하락폭 (이 범위 초과 하락 시 결과 거부)
_AT_TEST_AUC_MARGIN: float = 0.005

# ───────────────────────────── 데이터 수집 ─────────────────────────────

# provider.py 의 단일 소스 구현으로 위임 (심볼 상수도 provider.MACRO_SYMBOLS 참조)
def _fetch_macro_data(period: str) -> pd.DataFrame:
    """거시경제 데이터 수집 — provider.fetch_macro_df 에 위임."""
    return fetch_macro_df(period=period)


def _fetch_market_returns(symbol: str, period: str) -> pd.DataFrame:
    """시장 지수 롤링 수익률 — provider.fetch_market_df 에 위임."""
    return fetch_market_df(symbol=symbol, period=period)



def _fetch_stock_base(
    code: str, period: str, future_days: int,
    min_len: int = 60,
    market_df: pd.DataFrame = None,
    macro_df: pd.DataFrame = None,
) -> Optional[tuple]:
    """공통 베이스: OHLCV 수집 → 지표 계산 → 피처 빌드 → 미래 수익률.

    Returns
    -------
    (feat_valid, ret_valid, df_ind) 또는 None (데이터 부족 / 오류 시)
        feat_valid : DataFrame — 인덱스 = feat.index[:-future_days]
        ret_valid  : Series   — 동일 인덱스, 미래 수익률
        df_ind     : DataFrame — indicators.calculate_all() 결과 (TCN 불필요, 참고용)
    """
    df = data_provider.get_ohlcv(code, period=period)
    if df is None or df.empty or len(df) < min_len:
        logger.warning(f"  [{code}] 데이터 부족 ({len(df) if df is not None else 0}행) — 건너뜀")
        return None
    df_ind = indicators.calculate_all(df)
    if df_ind.empty:
        return None
    feat = build_features(df_ind, market_df=market_df, macro_df=macro_df)
    if len(feat) <= future_days:
        return None
    close      = df_ind['close'].reindex(feat.index)
    future_ret = (close.shift(-future_days) - close) / close
    valid_idx  = feat.index[:-future_days]
    return feat.loc[valid_idx], future_ret.loc[valid_idx], df_ind


def _collect_stock_features(code: str, period: str, future_days: int,
                             market_df: pd.DataFrame = None,
                             macro_df: pd.DataFrame = None) -> pd.DataFrame:
    """단일 종목의 (날짜, 특성, 미래수익률) DataFrame 반환."""
    try:
        base = _fetch_stock_base(code, period, future_days,
                                 market_df=market_df, macro_df=macro_df)
        if base is None:
            return pd.DataFrame()
        feat_valid, ret_valid, _ = base

        result = feat_valid.copy()
        result['raw_return'] = ret_valid

        base_subset = [c for c in BASE_FEATURE_COLS + ['raw_return'] if c in result.columns]
        valid       = result.dropna(subset=base_subset)
        logger.info(f"  [{code}] {len(valid)}개 샘플 수집")
        return valid

    except Exception as exc:
        logger.error(f"  [{code}] 처리 오류: {exc}")
        return pd.DataFrame()


def _collect_stock_tcn(code: str, period: str, future_days: int,
                        market_df: pd.DataFrame = None,
                        macro_df: pd.DataFrame = None) -> Optional[dict]:
    """TCN용: 단일 종목의 전체 피처 시계열 + 이진 라벨 반환.

    Returns:
        {'features': DataFrame(날짜×피처), 'raw_return': Series}
        또는 None (데이터 부족 시)
    """
    try:
        base = _fetch_stock_base(code, period, future_days,
                                 min_len=60 + _tcn.LOOKBACK,
                                 market_df=market_df, macro_df=macro_df)
        if base is None:
            return None
        feat_valid, ret_valid, _ = base

        if len(feat_valid) <= _tcn.LOOKBACK:
            return None

        feat_cols  = [c for c in BASE_FEATURE_COLS if c in feat_valid.columns]
        feat_clean = feat_valid[feat_cols].dropna()
        ret_align  = ret_valid.reindex(feat_clean.index).dropna()
        feat_clean = feat_clean.reindex(ret_align.index)

        if len(ret_align) < 30:
            return None

        # 크로스섹셔널 라벨은 fetch_train_test_samples 가 처리하므로
        # 여기서는 raw_return만 담아 반환 → 호출 측에서 rank 기반 라벨 변환
        return {'features': feat_clean, 'raw_return': ret_align}

    except Exception as exc:
        logger.error(f"  [{code}] TCN 수집 오류: {exc}")
        return None


def fetch_train_test_samples(
    codes: List[str], period: str, future_days: int, test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """크로스섹셔널 상대 강도 순위를 타깃으로 하는 학습/검증 세트 수집.

    Returns:
        (df_train, df_test, tcn_stock_data)
        tcn_stock_data: {code: {'features': DataFrame, 'labels': Series}} — TCN 학습용
    """
    market_df = _fetch_market_returns('KS11', period)
    if market_df.empty:
        logger.warning("KS11 시장 데이터 미수신 — rs_vs_mkt 피처는 0으로 채워집니다.")
        market_df = None

    logger.info("[거시경제] VIX·S&P500 데이터 수집 중...")
    macro_df = _fetch_macro_data(period)
    if macro_df.empty:
        macro_df = None

    # ── 소켓 타임아웃 설정 ────────────────────────────────────────────────
    # FDR DataReader는 내부적으로 소켓을 사용. 전역 소켓 타임아웃을 30초로 설정해
    # Naver rate-limit으로 응답 없는 연결이 영구 hang하는 현상 방지.
    _prev_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)

    logger.info(f"  종목 {len(codes)}개 병렬 수집 중 (max_workers=5, socket_timeout=30s)...")
    frames = []
    tcn_raw: dict = {}   # {code: {'features': df, 'raw_return': series}}
    executor = ThreadPoolExecutor(max_workers=5)
    try:
        # 트리 모델용 + TCN용 동시 수집 (같은 데이터, 다른 형태)
        tree_futures = {
            executor.submit(_collect_stock_features, c, period, future_days, market_df, macro_df): ('tree', c)
            for c in codes
        }
        tcn_futures = {
            executor.submit(_collect_stock_tcn, c, period, future_days, market_df, macro_df): ('tcn', c)
            for c in codes
        } if _tcn.is_available() else {}

        all_futures = {**tree_futures, **tcn_futures}
        # per-call timeout은 provider.get_ohlcv 내부에서 25s 강제됨.
        # 여기서는 전체 수집에 걸리는 시간을 보수적으로 제한 (종목수 × 1.5배 마진).
        n_futures = len(all_futures)
        outer_timeout = max(300, n_futures * 3)   # 최소 5분, 최대 종목당 3s 기대
        try:
            for fut in as_completed(all_futures, timeout=outer_timeout):
                kind, c = all_futures[fut]
                try:
                    result = fut.result()
                    if kind == 'tree':
                        if result is not None and not result.empty:
                            frames.append(result)
                    else:
                        if result is not None:
                            tcn_raw[c] = result
                except Exception as e:
                    logger.error(f"  [{c}/{kind}] 병렬 수집 오류: {e}")
        except _FuturesTimeout:
            done_codes = [all_futures[f] for f in all_futures if f.done()]
            hung_codes = [all_futures[f] for f in all_futures if not f.done()]
            logger.warning(
                f"  ⚠️  수집 타임아웃: {len(done_codes)}/{len(all_futures)}개 완료 "
                f"— 미완료 {len(hung_codes)}개 건너뜀"
            )
    finally:
        executor.shutdown(wait=False)
        socket.setdefaulttimeout(_prev_timeout)

    df_all = pd.concat(frames)

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
    split_idx  = min(int(n_dates * (1.0 - test_ratio)), n_dates - 1)
    split_date = all_dates[split_idx]

    # ── Purging: 학습/테스트 경계 ─────────────────────────────────────────────
    # split_date 직전 2×future_days 거래일의 샘플은 학습에서 제거.
    # future_days 후 수익률 라벨이 테스트 기간 가격에 의존 (label leakage).
    # 2× 마진: 라벨이 경계 너머까지 간접 영향을 줄 수 있는 경우를 보수적으로 차단.
    purge_idx  = max(0, split_idx - 2 * future_days)
    purge_date = all_dates[purge_idx]

    keep_cols = [c for c in BASE_FEATURE_COLS if c in df_all.columns] + ['target']
    df_train  = df_all[df_all.index <  purge_date][keep_cols].dropna()
    df_test   = df_all[df_all.index >= split_date][keep_cols].dropna()

    purged_n = len(df_all[(df_all.index >= purge_date) & (df_all.index < split_date)])
    logger.info(
        f"[Purging] 학습/테스트 경계 제거: {purge_date.date()} ~ {split_date.date()} "
        f"({future_days}거래일 gap) → {purged_n}샘플 제거"
    )

    pos_rate = df_train['target'].mean()
    logger.info(
        f"\n타깃: {future_days}거래일 후 수익률 상위 25% = 1 / 하위 25% = 0 (중간 50% 제외)"
        f"\n분할 기준일: {split_date.date()}"
        f"\n총 날짜: {n_dates} (학습 {split_idx}일 / 검증 {n_dates - split_idx}일)"
        f"\n총 샘플: 학습 {len(df_train)} / 검증 {len(df_test)}"
        f"\n날짜별 평균 레이블 종목 수: {stocks_per_date[valid_dates].mean():.1f}"
        f"\n학습 양성 비율: {pos_rate:.1%} (≈50%)"
    )

    # ── TCN용: 크로스섹셔널 rank 기반 이진 라벨 생성 ──────────────────────
    tcn_stock_data: dict = {}
    if tcn_raw:
        # 전체 종목의 raw_return을 날짜별로 합산해 rank 계산
        combined = {}
        for code, d in tcn_raw.items():
            for dt, rv in d['raw_return'].items():
                combined.setdefault(dt, {})[code] = rv
        # 날짜별 rank → 종목별 이진 라벨 Series 생성
        code_labels: dict = {}
        for dt, code_ret in combined.items():
            if len(code_ret) < MIN_STOCKS_PER_DATE:
                continue
            rets   = pd.Series(code_ret)
            ranks  = rets.rank(pct=True)
            for code, rp in ranks.items():
                if rp >= TOP_K_PERCENTILE:
                    code_labels.setdefault(code, {})[dt] = 1
                elif rp <= BOTTOM_K_PERCENTILE:
                    code_labels.setdefault(code, {})[dt] = 0
                # 중립 구간은 라벨 없음 (TCN build_sequences 가 자동 제외)

        for code, lbl_dict in code_labels.items():
            if code in tcn_raw:
                tcn_stock_data[code] = {
                    'features': tcn_raw[code]['features'],
                    'labels':   pd.Series(lbl_dict),
                }
        logger.info(f"[TCN] 라벨 생성 완료: {len(tcn_stock_data)}개 종목")

    return df_train, df_test, tcn_stock_data


def _load_effective_configs() -> Dict[str, dict]:
    """MODEL_CONFIGS 를 deepcopy 후 PARAMS_DIR 오버라이드 파일을 병합해 반환.

    원본 MODULE 레벨 MODEL_CONFIGS 는 변경하지 않음.
    """
    import copy as _copy
    effective: Dict[str, dict] = {}
    for _name, _cfg in MODEL_CONFIGS.items():
        _merged = _copy.deepcopy(_cfg)
        _override_path = PARAMS_DIR / f"{_name}_overrides.json"
        if _override_path.exists():
            try:
                with open(_override_path, encoding="utf-8") as _f:
                    _ov = json.load(_f)
                _merged['params'].update(_ov)
                logger.info(f"[override] {_name} 파라미터 오버라이드 적용: {_ov}")
            except Exception as _e:
                logger.warning(f"[override] {_name} 오버라이드 로드 실패 — 기본값 사용: {_e}")
        effective[_name] = _merged
    return effective


def _walk_forward_cv(
    df_train: pd.DataFrame,
    feat_names: List[str],
    cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    unique_dates: list,
    future_days: int,
) -> Tuple[List[float], List[float]]:
    """Walk-Forward CV (롤링 윈도우, Purging 적용) → (cv_aucs, oof_preds).

    검증 윈도우: 20거래일(≈1개월), 스텝: 10거래일 (overlapping)
    최소 학습 기간: max(전체 날짜 60%, 120일) — 초반 fold AUC 신뢰도 확보
    VAL_STEP=10: fold 수 2배(≈24→48) — CV AUC 신뢰도 향상 (Purging으로 leakage 방지)
    """
    VAL_WINDOW  = 20
    VAL_STEP    = 10
    min_train_n = max(int(len(unique_dates) * 0.6), 120)
    is_ranker   = cfg.get('is_ranker', False)
    cv_aucs:   List[float] = []
    oof_preds: List[float] = []
    start_idx = min_train_n
    while start_idx + VAL_WINDOW <= len(unique_dates):
        end_idx        = min(start_idx + VAL_WINDOW, len(unique_dates))
        purge_boundary = start_idx - 2 * future_days
        tr_dates_set   = {unique_dates[i] for i in range(max(0, purge_boundary))}
        val_dates_set  = {unique_dates[i] for i in range(start_idx, end_idx)}
        tr_mask  = df_train.index.isin(tr_dates_set)
        val_mask = df_train.index.isin(val_dates_set)
        if tr_mask.sum() < 10 or val_mask.sum() < 10:
            start_idx += VAL_STEP
            continue
        if is_ranker:
            fold_tr  = df_train[tr_mask].sort_index()
            fold_val = df_train[val_mask].sort_index()
            cv_sc    = StandardScaler()
            cv_m     = cfg['class'](**cfg['params'])
            X_cv_tr  = cv_sc.fit_transform(fold_tr[feat_names].values)
            X_cv_val = cv_sc.transform(fold_val[feat_names].values)
            g_cv_tr  = fold_tr.groupby(fold_tr.index).size().values
            cv_m.fit(X_cv_tr, fold_tr['target'].values, group=g_cv_tr)
            cv_scores = cv_m.predict(X_cv_val)
            oof_preds.extend(cv_scores.tolist())
            cv_aucs.append(roc_auc_score(fold_val['target'].values, cv_scores))
        else:
            cv_sc    = StandardScaler()
            cv_m     = cfg['class'](**cfg['params'])
            X_cv_tr  = pd.DataFrame(cv_sc.fit_transform(X_train[tr_mask]), columns=feat_names)
            X_cv_val = pd.DataFrame(cv_sc.transform(X_train[val_mask]),    columns=feat_names)
            cv_m.fit(X_cv_tr, y_train[tr_mask])
            cv_p = cv_m.predict_proba(X_cv_val)[:, 1]
            oof_preds.extend(cv_p.tolist())
            cv_aucs.append(roc_auc_score(y_train[val_mask], cv_p))
        start_idx += VAL_STEP
    return cv_aucs, oof_preds


def _train_final_model(
    cfg: dict,
    feat_names: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_train: pd.DataFrame,
    oof_preds: List[float],
    t0: float,
) -> Tuple[Any, StandardScaler, float, float, float, List[float], list]:
    """최종 모델 학습(전체 학습 세트) 및 평가.

    Returns
    -------
    (model, scaler, train_auc, test_auc, test_logloss, calibration_points,
     feature_importances, duration)
    """
    scaler    = StandardScaler()
    is_ranker = cfg.get('is_ranker', False)
    if is_ranker:
        df_tr_sorted = df_train.sort_index()
        g_tr         = df_tr_sorted.groupby(df_tr_sorted.index).size().values
        X_tr  = scaler.fit_transform(df_tr_sorted[feat_names].values)
        X_te  = scaler.transform(X_test)
        model = cfg['class'](**cfg['params'])
        model.fit(X_tr, df_tr_sorted['target'].values, group=g_tr)
        duration     = time.time() - t0
        train_scores = model.predict(X_tr)
        test_scores  = model.predict(X_te)
        _cal_src = oof_preds if len(oof_preds) >= 101 else train_scores.tolist()
        calibration_points = np.percentile(_cal_src, np.arange(0, 101)).tolist()
        train_auc    = roc_auc_score(df_tr_sorted['target'].values, train_scores)
        test_auc     = roc_auc_score(y_test, test_scores)
        test_logloss = float('nan')
    else:
        X_tr_arr = scaler.fit_transform(X_train)
        X_te_arr = scaler.transform(X_test)
        X_tr  = pd.DataFrame(X_tr_arr, columns=feat_names)
        X_te  = pd.DataFrame(X_te_arr, columns=feat_names)
        model = cfg['class'](**cfg['params'])
        model.fit(X_tr, y_train)
        duration     = time.time() - t0
        train_proba  = model.predict_proba(X_tr)[:, 1]
        test_proba   = model.predict_proba(X_te)[:, 1]
        _cal_src = oof_preds if len(oof_preds) >= 101 else train_proba.tolist()
        calibration_points = np.percentile(_cal_src, np.arange(0, 101)).tolist()
        train_auc    = roc_auc_score(y_train, train_proba)
        test_auc     = roc_auc_score(y_test,  test_proba)
        test_logloss = log_loss(y_test, test_proba)

    feature_importances: list = []
    if hasattr(model, 'feature_importances_') and len(feat_names) == len(model.feature_importances_):
        fi_pairs = sorted(
            zip(feat_names, model.feature_importances_.tolist()),
            key=lambda x: x[1], reverse=True,
        )
        feature_importances = [[n, round(v, 6)] for n, v in fi_pairs]

    return model, scaler, train_auc, test_auc, test_logloss, calibration_points, feature_importances, duration


def _at_diagnose(test_auc: float, train_auc: float, cv_mean: float, cv_std: float) -> str:
    """모델 진단 → 'PASS' | 'OVERFIT' | 'UNDERFIT' | 'UNSTABLE' | 'WEAK'.

    우선순위: OVERFIT > UNDERFIT > UNSTABLE > WEAK > PASS
    """
    thr = AUTO_TUNE_THRESHOLDS
    overfit_gap = train_auc - test_auc
    if overfit_gap > thr['max_overfit_gap']:
        return 'OVERFIT'
    if test_auc < thr['trigger_test_auc']:
        return 'UNDERFIT'
    if not np.isnan(cv_std) and cv_std > thr['max_cv_auc_std']:
        return 'UNSTABLE'
    if not np.isnan(cv_mean) and cv_mean < thr['min_cv_auc']:
        return 'WEAK'
    return 'PASS'


def _at_apply_rules(model_name: str, base_params: dict, diagnosis: str) -> dict:
    """규칙 기반 파라미터 조정 → 조정된 params dict 반환 (base_params 깊은 복사본)."""
    import copy
    params = copy.deepcopy(base_params)
    rules = _AT_ADJUST_RULES.get(diagnosis, {}).get(model_name, {})
    for param, (op, val) in rules.items():
        if param not in params:
            continue
        cur = params[param]
        if op == 'mul':
            new_val = cur * val
            params[param] = max(1, round(new_val)) if isinstance(cur, int) else new_val
        elif op == 'add':
            new_val = cur + val
            params[param] = max(1, int(new_val)) if isinstance(cur, int) else new_val
        elif op == 'set':
            params[param] = val
    return params


def _at_write_overrides(model_name: str, tuned_params: dict, original_params: dict) -> None:
    """original_params 와 다른 값만 {name}_overrides.json 에 저장."""
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    delta = {
        k: v for k, v in tuned_params.items()
        if k not in _AT_SKIP_PARAMS and original_params.get(k) != v
    }
    if not delta:
        return
    override_path = PARAMS_DIR / f"{model_name}_overrides.json"
    with open(override_path, 'w', encoding='utf-8') as f:
        json.dump(delta, f, indent=2, ensure_ascii=False)
    logger.info(f"  [auto-tune] {model_name} 오버라이드 저장: {delta}")


def _auto_tune_model(
    name: str,
    cfg: dict,
    df_train: pd.DataFrame,
    feat_names: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    unique_dates: list,
    future_days: int,
    current_metrics: dict,
    max_trials: int = 15,
) -> Tuple[Optional[dict], dict]:
    """Phase 1 (규칙 기반) + Phase 2 (랜덤 탐색) Auto-Tune.

    Walk-Forward CV만 실행 (최종 학습은 호출측 Phase 3에서 수행).

    Returns
    -------
    (best_cfg, tune_log)
        best_cfg : 개선된 cfg dict, 개선 없으면 None
        tune_log : 진단·탐색 결과 메타 dict
    """
    import copy
    import random

    test_auc = current_metrics['test_auc']
    train_auc = current_metrics['train_auc']
    cv_mean = current_metrics.get('cv_mean', float('nan'))
    cv_std  = current_metrics.get('cv_std',  float('nan'))

    diagnosis = _at_diagnose(test_auc, train_auc, cv_mean, cv_std)
    base_cv = cv_mean if not np.isnan(cv_mean) else 0.0
    tune_log: dict = {
        'diagnosis':    diagnosis,
        'base_cv_mean': round(base_cv, 4) if base_cv else None,
        'phase1_cv':    None,
        'phase2_cv':    None,
        'best_delta':   {},
        'improvement':  0.0,
        'trials':       0,
    }

    if diagnosis == 'PASS':
        logger.info(f"  [auto-tune] {name}: PASS — 튜닝 건너뜀")
        return None, tune_log

    logger.info(f"  [auto-tune] {name}: {diagnosis} — 파라미터 탐색 시작 (max_trials={max_trials})")

    best_cv   = base_cv
    best_cfg: Optional[dict] = None
    base_params = cfg['params']
    search_space = _AT_SEARCH_SPACES.get(name, {})

    # ── 스테일 override 감지 (OVERFIT 진단인데 depth가 기본값보다 큰 경우) ──────
    depth_key = _DEPTH_PARAMS.get(name)
    if depth_key:
        baseline_depth = MODEL_CONFIGS.get(name, {}).get('params', {}).get(depth_key)
        effective_depth = base_params.get(depth_key)
        if (diagnosis == 'OVERFIT'
                and baseline_depth is not None
                and effective_depth is not None
                and effective_depth > baseline_depth):
            logger.warning(
                f"  [auto-tune] ⚠️  {name}: 기존 override가 {depth_key}를 "
                f"{baseline_depth} → {effective_depth} 로 증가시킴 — OVERFIT 주요 원인일 수 있습니다. "
                f"`train --reset-overrides` 후 재시도를 권장합니다."
            )

    def _cv_score(candidate_params: dict) -> float:
        cand_cfg = copy.deepcopy(cfg)
        cand_cfg['params'] = candidate_params
        try:
            cv_aucs, _ = _walk_forward_cv(
                df_train, feat_names, cand_cfg, X_train, y_train, unique_dates, future_days
            )
            return float(np.mean(cv_aucs)) if cv_aucs else 0.0
        except Exception as e:
            logger.debug(f"  [auto-tune] CV 오류 ({name}): {e}")
            return 0.0

    # ── Phase 1: 규칙 기반 조정 ────────────────────────────────────────────
    p1_params = _at_apply_rules(name, base_params, diagnosis)
    p1_cv = _cv_score(p1_params)
    tune_log['phase1_cv'] = round(p1_cv, 4)
    logger.info(f"  [auto-tune] Phase1 ({diagnosis} 규칙): CV={p1_cv:.4f}")
    if p1_cv > best_cv + 0.003:
        best_cv = p1_cv
        best_cfg = copy.deepcopy(cfg)
        best_cfg['params'] = p1_params

    # ── Phase 2: 랜덤 탐색 ─────────────────────────────────────────────────
    if search_space:
        rng = random.Random(int(time.time() * 1000) % (2**31))
        p2_best_cv = best_cv
        for trial in range(max_trials):
            trial_params = copy.deepcopy(best_cfg['params'] if best_cfg else base_params)
            n_change = rng.randint(1, max(1, len(search_space) // 2))
            for k in rng.sample(list(search_space.keys()), min(n_change, len(search_space))):
                trial_params[k] = rng.choice(search_space[k])

            # 방향 제약: OVERFIT이면 depth 증가 금지, UNDERFIT이면 depth 감소 금지
            if depth_key and depth_key in trial_params:
                base_depth_val = base_params.get(depth_key)
                if base_depth_val is not None:
                    if diagnosis == 'OVERFIT' and trial_params[depth_key] > base_depth_val:
                        trial_params[depth_key] = base_depth_val
                    elif diagnosis == 'UNDERFIT' and trial_params[depth_key] < base_depth_val:
                        trial_params[depth_key] = base_depth_val

            t_cv = _cv_score(trial_params)
            if t_cv > p2_best_cv + 0.003:
                p2_best_cv = t_cv
                best_cfg = copy.deepcopy(cfg)
                best_cfg['params'] = trial_params
                logger.info(f"  [auto-tune] Phase2 trial {trial + 1}/{max_trials}: CV={t_cv:.4f} ✨")
        tune_log['phase2_cv'] = round(p2_best_cv, 4)
        tune_log['trials']    = max_trials
        best_cv = p2_best_cv
    else:
        logger.info(f"  [auto-tune] {name}: 탐색 공간 없음 — Phase2 건너뜀")

    tune_log['improvement'] = round(best_cv - base_cv, 4)
    if best_cfg is not None:
        tune_log['best_delta'] = {
            k: best_cfg['params'][k]
            for k in search_space
            if k in best_cfg['params'] and best_cfg['params'].get(k) != base_params.get(k)
        }
        logger.info(f"  [auto-tune] {name}: 최적 CV={best_cv:.4f} (개선: +{tune_log['improvement']:.4f})")
    else:
        logger.info(f"  [auto-tune] {name}: 개선 없음 — 기본 파라미터 유지")

    return best_cfg, tune_log


def train_and_save(df_train: pd.DataFrame, df_test: pd.DataFrame,
                   future_days: int = 10,
                   tcn_stock_data: Optional[dict] = None,
                   auto_tune: bool = False,
                   max_trials: int = 15,
                   save_overrides: bool = True) -> None:
    """모델 학습(이진 분류) → 평가 → 모델/스케일러/파라미터 저장.

    타깃: future_days 거래일 후 수익률 상위 25% = 1 / 하위 25% = 0 (중간 50% 제외, 이진 분류)
    지표: AUC-ROC (랜덤 기준선 0.5, 목표 ≥ 0.52)
    교차검증: Walk-Forward (롤링 윈도우, 20거래일 val 스텝, Purging 적용)
    Purging:  각 val 시작 전 future_days 거래일을 학습에서 제거 (label leakage 방지).
    tcn_stock_data: TCN 전용 시계열 데이터 ({code: {features, labels}}), None이면 TCN 건너뜀.
    auto_tune: True이면 품질 미달 모델에 Phase1+2 CV 탐색 후 Phase3 전체 재학습 수행.
    max_trials: Phase2 랜덤 탐색 시도 횟수 (기본 15).
    save_overrides: True이면 Phase3 채택 파라미터를 {name}_overrides.json 에 저장.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)

    feat_names = [c for c in BASE_FEATURE_COLS if c in df_train.columns]
    if len(feat_names) < len(BASE_FEATURE_COLS):
        missing = [c for c in BASE_FEATURE_COLS if c not in df_train.columns]
        logger.warning(f"누락 피처 {len(missing)}개 — 학습에서 제외됩니다: {missing}")
    X_train = df_train[feat_names].values
    y_train = df_train['target'].values

    if df_test.empty:
        logger.warning("검증 세트가 없습니다. 학습 세트 성능만 기록됩니다.")
        X_test, y_test = X_train, y_train
    else:
        X_test = df_test[feat_names].values
        y_test = df_test['target'].values

    pos_rate     = y_train.mean()
    unique_dates = sorted(df_train.index.unique())
    logger.info(f"\n기준선 AUC: 0.5000  (랜덤 분류기)")
    logger.info(f"학습 샘플: {len(X_train)}, 검증 샘플: {len(X_test)}, 양성 비율: {pos_rate:.1%}\n")

    effective_configs = _load_effective_configs()

    results = []
    for name, cfg in effective_configs.items():
        logger.info(f"{'─'*40}")
        logger.info(f"  학습 중: {name}")
        t0 = time.time()

        # ── Walk-Forward CV ───────────────────────────────────────────────
        cv_aucs, oof_preds = _walk_forward_cv(
            df_train, feat_names, cfg, X_train, y_train, unique_dates, future_days
        )
        n_folds = len(cv_aucs)
        cv_mean = float(np.mean(cv_aucs)) if cv_aucs else float('nan')
        cv_std  = float(np.std(cv_aucs))  if cv_aucs else float('nan')
        if not (np.isnan(cv_mean) or np.isnan(cv_std)):
            logger.info(f"  CV AUC (Walk-Forward, {n_folds} folds, purged): {cv_mean:.4f} ± {cv_std:.4f}")
        else:
            logger.warning("  CV AUC (Walk-Forward): N/A (유효 fold 없음)")

        # ── 최종 모델 학습 ────────────────────────────────────────────────
        model, scaler, train_auc, test_auc, test_logloss, calibration_points, \
            feature_importances, duration = _train_final_model(
                cfg, feat_names, X_train, y_train, X_test, y_test, df_train, oof_preds, t0
            )
        overfit_gap  = round(train_auc - test_auc, 4)
        quality_pass = bool(test_auc >= MIN_MODEL_AUC)

        # ── Auto-Tune Phase 1+2 (CV 탐색) + Phase 3 (전체 재학습) ───────────
        tune_log: Optional[dict] = None
        if auto_tune:
            current_metrics = {
                'test_auc': test_auc, 'train_auc': train_auc,
                'cv_mean': cv_mean,   'cv_std':    cv_std,
            }
            best_at_cfg, tune_log = _auto_tune_model(
                name, cfg, df_train, feat_names,
                X_train, y_train, unique_dates, future_days,
                current_metrics, max_trials,
            )
            if best_at_cfg is not None and tune_log.get('improvement', 0) > 0.001:
                logger.info(f"  [auto-tune] Phase3: {name} 최적 파라미터로 전체 재학습 중...")
                # 원본 모델 아티팩트 보관 (test AUC guard 실패 시 복원)
                _at_orig = dict(
                    model=model, scaler=scaler,
                    test_auc=test_auc, train_auc=train_auc,
                    test_logloss=test_logloss,
                    calibration_points=calibration_points,
                    feature_importances=feature_importances,
                    duration=duration,
                    overfit_gap=overfit_gap,
                    quality_pass=quality_pass,
                )
                t0_at = time.time()
                model, scaler, train_auc, test_auc, test_logloss, calibration_points, \
                    feature_importances, duration = _train_final_model(
                        best_at_cfg, feat_names, X_train, y_train, X_test, y_test,
                        df_train, oof_preds, t0_at,
                    )
                overfit_gap  = round(train_auc - test_auc, 4)
                quality_pass = bool(test_auc >= MIN_MODEL_AUC)
                # Test AUC guard: 원본 대비 허용 하락폭 초과 시 Phase3 결과 거부
                if test_auc >= _at_orig['test_auc'] - _AT_TEST_AUC_MARGIN:
                    logger.info(
                        f"  [auto-tune] Phase3 채택: test_auc "
                        f"{_at_orig['test_auc']:.4f} → {test_auc:.4f}  "
                        f"gap={overfit_gap:.4f}  {'✅' if quality_pass else '⚠️'}"
                    )
                    if save_overrides:
                        _at_write_overrides(name, best_at_cfg['params'], MODEL_CONFIGS[name]['params'])
                    cfg = best_at_cfg   # meta 저장 시 실제 사용 파라미터 반영
                    tune_log['accepted'] = True
                else:
                    logger.warning(
                        f"  [auto-tune] Phase3 거부: test_auc 하락 "
                        f"{_at_orig['test_auc']:.4f} → {test_auc:.4f} "
                        f"(허용 하락폭 {_AT_TEST_AUC_MARGIN}) — 원본 파라미터 유지"
                    )
                    model, scaler         = _at_orig['model'], _at_orig['scaler']
                    test_auc, train_auc   = _at_orig['test_auc'], _at_orig['train_auc']
                    test_logloss          = _at_orig['test_logloss']
                    calibration_points    = _at_orig['calibration_points']
                    feature_importances   = _at_orig['feature_importances']
                    duration              = _at_orig['duration']
                    overfit_gap           = _at_orig['overfit_gap']
                    quality_pass          = _at_orig['quality_pass']
                    tune_log['accepted']  = False

        logger.info(f"  AUC : {test_auc:.4f}  (학습 AUC: {train_auc:.4f}  과적합 gap: {overfit_gap:.4f})")
        if not np.isnan(test_logloss):
            logger.info(f"  LogLoss: {test_logloss:.4f}")
        logger.info(f"  소요: {duration:.1f}초")
        if quality_pass:
            logger.info(f"  ✅ 품질 게이트 통과 (test_auc={test_auc:.4f} ≥ {MIN_MODEL_AUC})")
        else:
            logger.warning(
                f"  ⚠️  품질 게이트 미달 (test_auc={test_auc:.4f} < {MIN_MODEL_AUC}) "
                f"— 저장은 하지만 예측 시 tech_score 폴백이 사용됩니다."
            )
        if feature_importances:
            top5 = ', '.join(f"{n}({v:.3f})" for n, v in feature_importances[:5])
            logger.info(f"  Top-5 피처: {top5}")

        # ── 아티팩트 저장 ─────────────────────────────────────────────────
        model_path  = MODEL_DIR / f"{name}_model.pkl"
        scaler_path = MODEL_DIR / f"{name}_scaler.pkl"
        joblib.dump(model,  model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"  저장: {model_path}")
        logger.info(f"  저장: {scaler_path}")

        saved_at = datetime.now()
        version  = f"{name}_v{saved_at.strftime('%Y%m%d_%H%M%S')}"
        meta = {
            "parameters":          {k: v for k, v in cfg['params'].items()
                                    if k not in ('random_state', 'n_jobs', 'verbosity',
                                                 'use_label_encoder', 'eval_metric')},
            "model_type":          "ranker" if cfg.get('is_ranker') else "binary_classifier",
            "target_definition":   (
                f"top {int((1-TOP_K_PERCENTILE)*100)}% / "
                f"bottom {int(BOTTOM_K_PERCENTILE*100)}% return in {future_days}d (neutral zone)"
            ),
            "training_samples":    int(len(X_train)),
            "positive_rate":       round(float(pos_rate), 4),
            "cv_auc_mean":         round(cv_mean, 4) if not np.isnan(cv_mean) else None,
            "cv_auc_std":          round(cv_std,  4) if not np.isnan(cv_std)  else None,
            "purging_days":        future_days,
            "train_auc":           round(train_auc,    4),
            "test_auc":            round(test_auc,     4),
            "test_logloss":        round(test_logloss, 4) if not np.isnan(test_logloss) else None,
            "overfit_gap":         overfit_gap,
            "quality_pass":        quality_pass,
            "training_duration":   round(duration, 1),
            "saved_at":            saved_at.isoformat(),
            "model_version":       version,
            "feature_importances": feature_importances,
            "calibration":         calibration_points,
            "tune_log":            tune_log,
        }
        params_path = PARAMS_DIR / f"{name}_params.json"
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        results.append((name, test_auc, cv_mean, train_auc, quality_pass))

    # ── 요약 출력 ─────────────────────────────────────────────────────────
    logger.info(f"\n{'═'*60}")
    logger.info("  학습 완료 요약  (이진 분류 / AUC-ROC, Walk-Forward CV)")
    logger.info(f"{'─'*60}")
    logger.info(f"  {'모델':<22} {'test AUC':>9} {'CV AUC':>9} {'train AUC':>9} {'갭':>6} {'품질'}")
    logger.info(f"{'─'*60}")
    for name, tauc, cv_mu, trauc, qpass in results:
        gap    = trauc - tauc
        mark   = "✅" if qpass else "⚠️ "
        cv_str = f"{cv_mu:.4f}" if not np.isnan(cv_mu) else "N/A"
        logger.info(f"  {name:<22} {tauc:>9.4f} {cv_str:>9} {trauc:>9.4f} {gap:>6.4f}  {mark}")
    logger.info(f"{'═'*40}")
    logger.info(f"✅ 트리 앙상블 저장 완료  →  {MODEL_DIR}")

    # ── TCN 학습 ──────────────────────────────────────────────────────────
    if tcn_stock_data and _tcn.is_available():
        logger.info(f"\n{'─'*40}")
        logger.info("  학습 중: tcn  (Temporal Convolutional Network)")
        tcn_result = _tcn.train_tcn(
            tcn_stock_data,
            future_days=future_days,
            test_ratio=0.2,
        )
        if tcn_result is not None:
            if auto_tune:
                at_tcn = _tcn.auto_tune_tcn(
                    tcn_stock_data, future_days=future_days, test_ratio=0.2,
                    max_trials=min(5, max_trials),
                    base_test_auc=tcn_result['test_auc'],
                )
                if at_tcn is not None:
                    tcn_result = at_tcn
                    logger.info(f"  [auto-tune] TCN 개선: test_auc={tcn_result['test_auc']:.4f}")
            _tcn.save_tcn(tcn_result, MODEL_DIR, PARAMS_DIR)
            qmark  = "✅" if tcn_result["quality_pass"] else "⚠️ "
            cv_str = f"{tcn_result['cv_auc_mean']:.4f}" if tcn_result["cv_auc_mean"] else "N/A"
            logger.info(
                f"  {'tcn':<22} {tcn_result['test_auc']:>9.4f} {cv_str:>9} "
                f"{tcn_result['train_auc']:>9.4f} {tcn_result['overfit_gap']:>6.4f}  {qmark}"
            )
        else:
            logger.warning("  [TCN] 학습 실패 또는 건너뜀 — 앙상블에서 제외됩니다.")
    elif not _tcn.is_available():
        import sys as _sys
        _in_pipx = "pipx" in _sys.executable or "pipx" in str(getattr(_sys, "prefix", ""))
        _cmd = "pipx inject koreanstocks torch" if _in_pipx else 'pip install -e ".[dl]"  또는  pip install "koreanstocks[dl]"'
        logger.info(f"  [TCN] PyTorch 미설치 — 건너뜁니다.  활성화: {_cmd}")
        del _sys, _in_pipx, _cmd

    logger.info(f"✅ 모든 모델 저장 완료  →  {MODEL_DIR}")


# ───────────────────────────── 공개 진입점 ─────────────────────────────

def run_training(
    period: str = "2y",
    future_days: int = 10,
    stocks: Optional[List[str]] = None,
    test_ratio: float = 0.2,
    auto_tune: bool = False,
    max_trials: int = 15,
    save_overrides: bool = True,
    reset_overrides: bool = False,
) -> None:
    """koreanstocks train 명령어 및 train_models.py 양쪽에서 호출하는 진입점.

    Args:
        period:          학습 데이터 기간 (예: '2y', '1y')
        future_days:     예측 대상 거래일 수 (기본값 10 = 2주, 중기 노이즈 최소화)
        stocks:          학습 종목 코드 리스트 (None이면 DEFAULT_TRAINING_STOCKS 사용)
        test_ratio:      검증 세트 비율 (0~1)
        auto_tune:       True이면 품질 미달 모델 자동 파라미터 탐색·재학습
        max_trials:      Phase2 랜덤 탐색 시도 횟수
        save_overrides:  True이면 채택 파라미터를 overrides.json 에 저장
        reset_overrides: True이면 학습 전 모든 overrides.json 삭제 (기본 MODEL_CONFIGS 복원)
    """
    if stocks is None:
        stocks = DEFAULT_TRAINING_STOCKS

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    if reset_overrides:
        override_files = list(PARAMS_DIR.glob("*_overrides.json"))
        for f in override_files:
            f.unlink()
            logger.info(f"[reset] 오버라이드 초기화: {f.name}")
        if override_files:
            logger.info(f"[reset] {len(override_files)}개 오버라이드 파일 삭제 → 기본 MODEL_CONFIGS 복원")
        else:
            logger.info("[reset] 초기화할 오버라이드 파일 없음")

    logger.info("=" * 40)
    logger.info("  ML 모델 학습 시작")
    logger.info(f"  종목 수     : {len(stocks)}")
    logger.info(f"  데이터 기간 : {period}")
    logger.info(f"  예측 기간   : {future_days}거래일 후")
    logger.info(f"  검증 비율   : {test_ratio * 100:.0f}% (시계열 후반부)")
    logger.info(f"  타깃 변수   : {future_days}거래일 후 수익률 상위 25%/하위 25% 이진 분류 (중간 50% 제외, AUC-ROC)")
    logger.info(f"  피처 수     : {len(BASE_FEATURE_COLS)}개 (기술적+TA+거시경제)")
    if auto_tune:
        logger.info(f"  Auto-Tune   : 활성화 (max_trials={max_trials}, save_overrides={save_overrides})")
    logger.info("=" * 40)

    logger.info("\n[1/2] 학습 데이터 수집 중...")
    df_train, df_test, tcn_stock_data = fetch_train_test_samples(
        stocks, period=period, future_days=future_days, test_ratio=test_ratio,
    )

    logger.info("\n[2/2] 모델 학습 및 저장 중...")
    train_and_save(df_train, df_test, future_days=future_days,
                   tcn_stock_data=tcn_stock_data,
                   auto_tune=auto_tune, max_trials=max_trials,
                   save_overrides=save_overrides)
