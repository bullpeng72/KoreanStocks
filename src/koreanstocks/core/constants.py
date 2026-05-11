"""프로젝트 공통 상수 모음"""
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

# ── ML 모델 품질 게이트 ─────────────────────────────────────────────────────

# AUC < 이 값이면 모델 로드 거부 및 신뢰도 UI에서 미달 표시 (단일 소스)
MIN_MODEL_AUC: float = 0.52

# ── 버킷 (후보군 분류) ──────────────────────────────────────────────────────

# 버킷 기본값 (bucket 필드가 없는 종목에 배정)
BUCKET_DEFAULT: str = 'volume'

# 버킷 한국어 레이블 (대시보드·슬라이드 배지, 추천 결과 저장용)
BUCKET_LABELS: Dict[str, str] = {
    'volume':   '거래량 상위',
    'momentum': '상승 모멘텀',
    'rebound':  '반등 후보',
}

# 버킷별 후보 풀 할당 비율 (합계 = 1.0)
BUCKET_RATIOS: List[Tuple[str, float]] = [
    ('volume',   0.40),
    ('momentum', 0.35),
    ('rebound',  0.25),
]

# ── ML 앙상블 가중치 및 하이퍼파라미터 (단일 소스) ─────────────────────────────
# 분류기(RF·GB·LGB·CB·TCN) : 랭커(XGBRanker) 블렌딩 비율
ENSEMBLE_CLF_WEIGHT: float = 0.75
ENSEMBLE_RNK_WEIGHT: float = 0.25

# Softmax 온도 (AUC 기반 모델 가중치 정규화 시 사용)
# 값이 클수록 높은 AUC 모델에 가중치가 집중됨
SOFTMAX_TEMPERATURE: float = 5.0

# ── 레짐별 composite_score 최소 임계값 (recommendation_agent 단일 소스) ────────
# risk_off일수록 문턱을 높여 보수적 추천
REGIME_SCORE_THRESHOLD: Dict[str, float] = {
    "risk_on":   45.0,
    "uncertain": 50.0,
    "risk_off":  57.0,
}

# ── Auto-Tune 진단 임계값 (trainer.py 단일 소스) ──────────────────────────────
# PASS 이외 진단 시 파라미터 자동 조정·재학습 대상 (koreanstocks train --auto-tune)
AUTO_TUNE_THRESHOLDS: Dict[str, float] = {
    'trigger_test_auc': 0.545,   # 이 값 미만이면 UNDERFIT 진단 → 자동 조정 대상
    'max_overfit_gap':  0.10,    # train-test AUC 갭 상한 (초과 시 OVERFIT)
    'min_cv_auc':       0.500,   # CV AUC 최솟값 (미달 시 WEAK)
    'max_cv_auc_std':   0.050,   # CV AUC 표준편차 상한 (초과 시 UNSTABLE)
}

# ── 병렬 처리 Worker 수 (단일 소스) ──────────────────────────────────────────
MAX_ANALYSIS_WORKERS: int = 10   # recommendation_agent 종목 병렬 분석
MAX_SCREEN_WORKERS:   int = 15   # value/quality_screener 펀더멘털 배치 수집

# ── 종합 점수 가중치 (단일 소스) ─────────────────────────────────────────────
# 변경 시 모델 재학습 여부 검토 필요 (CLAUDE.md "자동 수정 금지 대상" 참조)
# Phase 2: 거시감성 포함 가중치 추가 (macro_sentiment_score 제공 시 적용)
_W_TECH_ML       = (0.35, 0.35, 0.20, 0.10)  # ML+거시: tech, ml, stock_sent, macro_sent
_W_TECH_ML_NM    = (0.40, 0.35, 0.25)         # ML만: tech, ml, stock_sent (거시감성 없을 때)
_W_TECH_NOML     = (0.65, 0.35)               # ML 없음 fallback: tech, stock_sent


# ── 가치주 스크리너 정렬 가중치 (단일 소스) ──────────────────────────────────────
# f_score(0~9)를 0~40점 구간으로 정규화한 뒤 value_score(0~100)와 복합 정렬
# 합계는 반드시 1.0: FSCORE_WEIGHT + VALUE_SCORE_WEIGHT == 1.0
FSCORE_WEIGHT: float = 0.40
VALUE_SCORE_WEIGHT: float = 0.60


def calc_composite_score(
    tech_score: float,
    ml_score: float,
    sentiment_score: float,
    ml_model_count: int,
    macro_sentiment_score: Optional[float] = None,
) -> float:
    """종합 점수 산출 (단일 소스 — analysis_agent / recommendation_agent 공용).

    Parameters
    ----------
    tech_score            : 기술적 지표 점수 (0~100)
    ml_score              : ML 앙상블 점수 (0~100)
    sentiment_score       : 종목 뉴스 감성 raw 값 (-100~100)
    ml_model_count        : 활성 ML 모델 수 (0이면 fallback 가중치 사용)
    macro_sentiment_score : 거시경제 감성 raw 값 (-100~100, None=미사용)

    Returns
    -------
    float : 종합 점수 (0~100)
    """
    sentiment_norm = max(0.0, min(100.0, (sentiment_score + 100.0) / 2.0))
    if ml_model_count > 0:
        if macro_sentiment_score is not None:
            macro_norm = max(0.0, min(100.0, (macro_sentiment_score + 100.0) / 2.0))
            wt, wm, ws, wc = _W_TECH_ML
            return wt * tech_score + wm * ml_score + ws * sentiment_norm + wc * macro_norm
        wt, wm, ws = _W_TECH_ML_NM
        return wt * tech_score + wm * ml_score + ws * sentiment_norm
    wt, ws = _W_TECH_NOML
    return wt * tech_score + ws * sentiment_norm


def calc_composite_score_from_dict(x: Dict[str, Any]) -> float:
    """dict 기반 래퍼 — recommendation_agent 정렬 key 함수용.

    예외 발생 시 0.0 반환 (sorted() key 함수로 사용 가능).
    macro_sentiment 필드가 있으면 거시감성 반영.
    """
    try:
        macro_sent = x.get('macro_sentiment')
        return calc_composite_score(
            tech_score            = float(x.get('tech_score')      or 50.0),
            ml_score              = float(x.get('ml_score')        or 50.0),
            sentiment_score       = float(x.get('sentiment_score') or 0.0),
            ml_model_count        = int(x.get('ml_model_count')    or 0),
            macro_sentiment_score = float(macro_sent) if macro_sent is not None else None,
        )
    except (TypeError, ValueError):
        return 0.0
