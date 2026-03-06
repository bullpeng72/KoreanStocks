"""가치주 스크리너 API 라우터"""
import logging
from typing import Optional

from fastapi import APIRouter, Query

from koreanstocks.core.engine.value_screener import value_screener

logger = logging.getLogger(__name__)
router = APIRouter(tags=["value"])


@router.get("/value_stocks")
async def get_value_stocks(
    market: str = Query("ALL", description="시장 필터: ALL | KOSPI | KOSDAQ"),
    per_max: float = Query(25.0, description="PER 상한"),
    pbr_max: float = Query(3.0,  description="PBR 상한"),
    roe_min: float = Query(8.0,  description="ROE 하한 (%)"),
    debt_max: float = Query(150.0, description="부채비율 상한 (%)"),
    revenue_yoy_min: float = Query(-15.0, description="매출 YoY 하한 (%)"),
    f_score_min: int = Query(4, description="Piotroski F-Score 최소값 (0~9)"),
    candidate_limit: int = Query(200, description="시가총액 상위 탐색 종목 수 (100~500)"),
):
    """
    펀더멘털 기반 가치주 스크리닝.

    중기(3~6개월) 관점으로 저PER·저PBR·고ROE·안전한 재무구조를 갖춘
    종목을 Piotroski F-Score와 가치 점수로 정렬하여 반환합니다.

    - **value_score**: PER·PBR·ROE·부채비율·영업이익YoY 종합 점수 (0~100)
    - **f_score**: 간소화 Piotroski F-Score (0~9, 높을수록 재무 건전)
    """
    try:
        results = value_screener.screen(
            market=market,
            per_max=per_max,
            pbr_max=pbr_max,
            roe_min=roe_min,
            debt_max=debt_max,
            revenue_yoy_min=revenue_yoy_min,
            f_score_min=f_score_min,
            candidate_limit=candidate_limit,
            limit=candidate_limit,  # 통과 종목 전부 반환
        )
        # f_checks는 직렬화 가능하지만 API 응답에는 요약만 포함
        for r in results:
            r.pop("fundamentals", None)   # 상세 데이터는 별도 엔드포인트로 분리
        return {"stocks": results, "total": len(results)}
    except Exception as e:
        logger.error(f"[VALUE API] 스크리닝 실패: {e}")
        return {"stocks": [], "total": 0, "error": str(e)}


@router.get("/value_stocks/filters")
async def get_value_filters():
    """현재 기본 필터 임계값 반환."""
    return value_screener.get_filter_defaults()
