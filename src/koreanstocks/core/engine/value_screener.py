"""
가치주 스크리너 (Phase 2)
===========================
펀더멘털 기반 중기(3~6개월) 가치주 발굴.

필터 체인:
  1. 영업이익 흑자 (가치함정 1차 방어)
  2. PER / PBR 상한
  3. ROE 하한 (자본 효율성)
  4. 부채비율 상한 (재무 안전성)
  5. 매출 역성장 하한 (가치함정 2차 방어)
  6. Piotroski F-Score 하한

점수:
  value_score (0~100) — PER·PBR·ROE·부채비율·영업이익YoY 가중합
  f_score     (0~9)   — 간소화 Piotroski (가용 데이터 기준)
"""

import logging
from datetime import date as _date
from typing import Dict, List, Optional, Tuple

from koreanstocks.core.data.fundamental_provider import fundamental_provider
from koreanstocks.core.data.provider import data_provider

logger = logging.getLogger(__name__)

# ── 기본 필터 임계값 ─────────────────────────────────────────────
DEFAULT_FILTERS = {
    "per_max":          25.0,   # PER 상한 (성장주 포함 여유)
    "pbr_max":           3.0,   # PBR 상한
    "roe_min":           8.0,   # ROE 하한 (%)
    "debt_max":        150.0,   # 부채비율 상한 (%)
    "revenue_yoy_min": -15.0,   # 매출 역성장 하한 (%) — 가치함정 방어
    "f_score_min":       4,     # Piotroski 최소 점수
}


# ── Piotroski F-Score ────────────────────────────────────────────

def piotroski_score(f: Dict) -> Tuple[int, Dict[str, bool]]:
    """
    간소화 Piotroski F-Score (0~9점).

    네이버 coinfo에서 가져올 수 있는 데이터로 9개 항목을 평가한다.

    수익성 (3):
      P1  순이익(영업이익) > 0
      P2  ROE > 0
      P3  ROE 전년 대비 개선
    레버리지·안전성 (3):
      L1  부채비율 < 100%
      L2  부채비율 전년 대비 감소
      L3  배당 지급 이력 있음 (배당수익률 > 0)
    성장성·효율성 (3):
      E1  영업이익 흑자
      E2  영업이익 YoY > 0
      E3  매출 역성장 없음 (YoY > -5%)
    """
    def safe(key, default=None):
        v = f.get(key)
        return v if v is not None else default

    checks: Dict[str, bool] = {
        # 수익성
        "P1_영업이익흑자":    bool(f.get("op_income_positive")),
        "P2_ROE양수":        safe("roe", -1) > 0,
        "P3_ROE개선":        bool(f.get("roe_improved")),
        # 레버리지·안전성
        "L1_부채100미만":     safe("debt_ratio", 999) < 100,
        "L2_부채감소":        bool(f.get("debt_decreased")),
        "L3_배당지급":        safe("dividend_yield", 0) > 0,
        # 성장성·효율성
        "E1_영업이익흑자확인": bool(f.get("op_income_positive")),
        "E2_영업이익성장":    safe("op_income_yoy", -999) > 0,
        "E3_매출역성장없음":  safe("revenue_yoy", 0) > -5,
    }
    score = sum(1 for v in checks.values() if v)
    return score, checks


# ── 가치 점수 ────────────────────────────────────────────────────

def value_score(f: Dict, sector_per_median: float = 12.0) -> float:
    """
    가치 점수 산출 (0~100).

    구성:
      PER   25pt  낮을수록 유리 (업종 중앙값 대비 상대 평가)
      PBR   20pt  낮을수록 유리
      ROE   20pt  높을수록 유리 (20%에서 만점)
      부채   20pt  낮을수록 유리 (200% 이상이면 0점)
      영업YoY 15pt  성장률이 높을수록 유리 (30%+ 만점)
    """
    parts: List[Tuple[float, float]] = []   # (earned, possible)

    per = f.get("per")
    if per is not None and per > 0:
        ratio = min(per / max(sector_per_median, 1), 2.5)
        parts.append((max(0.0, 25 * (1 - ratio / 2.5)), 25.0))

    pbr = f.get("pbr")
    if pbr is not None and pbr > 0:
        parts.append((max(0.0, 20 * (1 - min(pbr, 4) / 4)), 20.0))

    roe = f.get("roe")
    if roe is not None:
        parts.append((min(20.0, max(0.0, roe / 20 * 20)), 20.0))

    debt = f.get("debt_ratio")
    if debt is not None:
        parts.append((max(0.0, 20 * (1 - min(debt, 200) / 200)), 20.0))

    opi_yoy = f.get("op_income_yoy")
    if opi_yoy is not None:
        parts.append((min(15.0, max(0.0, opi_yoy / 30 * 15)), 15.0))

    if not parts:
        return 50.0

    earned   = sum(p[0] for p in parts)
    possible = sum(p[1] for p in parts)
    return round(earned / possible * 100, 1)


# ── 스크리너 ─────────────────────────────────────────────────────

class ValueScreener:
    """
    펀더멘털 기반 가치주 스크리너.

    실행 흐름:
      1. 시가총액 상위 후보군 + PER/ROE 사전 필터
      2. 펀더멘털 병렬 수집 (fundamental_provider, 당일 SQLite 캐시)
      3. 하드 필터 통과
      4. value_score + f_score 복합 정렬
      5. 상위 limit개 반환

    캐시:
      동일 필터 조합의 스크리닝 결과를 당일(자정까지) 인메모리 캐시로 보관.
      재요청 시 1~2분 소요되는 DART 수집 없이 즉시 반환한다.
    """

    def __init__(self):
        self._cache: Dict[tuple, List[Dict]] = {}
        self._cache_date: Optional[_date] = None

    def screen(
        self,
        market: str = "ALL",
        per_max: float = DEFAULT_FILTERS["per_max"],
        pbr_max: float = DEFAULT_FILTERS["pbr_max"],
        roe_min: float = DEFAULT_FILTERS["roe_min"],
        debt_max: float = DEFAULT_FILTERS["debt_max"],
        revenue_yoy_min: float = DEFAULT_FILTERS["revenue_yoy_min"],
        f_score_min: int = DEFAULT_FILTERS["f_score_min"],
        candidate_limit: int = 200,
        limit: int = 20,
    ) -> List[Dict]:
        """
        가치주 스크리닝 실행.

        Args:
            market: ALL | KOSPI | KOSDAQ
            per_max: PER 상한
            pbr_max: PBR 상한
            roe_min: ROE 하한 (%)
            debt_max: 부채비율 상한 (%)
            revenue_yoy_min: 매출 YoY 하한 (%) — 가치함정 방어
            f_score_min: Piotroski 최소 점수
            candidate_limit: 거래량 상위 몇 종목을 후보로 볼지
            limit: 최종 반환 종목 수

        Returns:
            value_score 내림차순 정렬된 Dict 리스트
        """
        # ── 당일 인메모리 캐시 확인 ───────────────────────────────────
        today = _date.today()
        if self._cache_date != today:
            self._cache.clear()
            self._cache_date = today

        cache_key = (market, per_max, pbr_max, roe_min, debt_max,
                     revenue_yoy_min, f_score_min, candidate_limit, limit)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            logger.info(f"[VALUE] 캐시 히트 — {len(cached)}종목 즉시 반환 (당일 동일 조건)")
            return [dict(r) for r in cached]  # 방어적 복사

        logger.info(f"[VALUE] 스크리닝 시작 (market={market}, limit={limit})")

        # 1. 후보 종목 (시가총액 상위 + PER/ROE 사전 필터 → 가치주 후보군)
        candidates = data_provider.get_value_candidates(
            limit=candidate_limit,
            market=market,
            per_max=per_max * 1.5,   # DART 없이 사전 필터 → 관용 상한으로 설정
            roe_min=max(0.0, roe_min - 5),  # 여유 있게 (DART 기준 재확인)
        )
        if not candidates:
            logger.warning("[VALUE] 후보 종목 없음 — 스크리닝 중단")
            return []

        candidates = candidates[:candidate_limit]
        stock_list = data_provider.get_stock_list()
        logger.info(f"[VALUE] 후보 {len(candidates)}종목 펀더멘털 수집 중...")

        # 2. 펀더멘털 병렬 수집
        fund_map = fundamental_provider.get_fundamentals_batch(candidates, max_workers=15)

        # 3. 필터 + 점수 산출
        passed: List[Dict] = []
        skipped_no_data  = 0
        skipped_filter   = 0

        for code in candidates:
            f = fund_map.get(code, {})

            per    = f.get("per")
            pbr    = f.get("pbr")
            roe    = f.get("roe")
            debt   = f.get("debt_ratio")
            revyoy = f.get("revenue_yoy")
            op_pos = f.get("op_income_positive", False)

            # 데이터 부족 → 스킵 (PER·PBR 모두 없으면 판단 불가)
            if per is None and pbr is None:
                skipped_no_data += 1
                continue

            # ── 하드 필터 ──────────────────────────────────────
            fail = False

            # 영업이익 흑자 필수
            if not op_pos:
                fail = True

            # PER: 유효한 경우에만 체크 (음수 PER = 적자 기업)
            if not fail and per is not None and (per <= 0 or per > per_max):
                fail = True

            if not fail and pbr is not None and pbr > pbr_max:
                fail = True

            if not fail and roe is not None and roe < roe_min:
                fail = True

            if not fail and debt is not None and debt > debt_max:
                fail = True

            # 매출 역성장 (데이터 없으면 통과 — 보수적 허용)
            if not fail and revyoy is not None and revyoy < revenue_yoy_min:
                fail = True

            if fail:
                skipped_filter += 1
                continue

            # Piotroski
            fscore, fchecks = piotroski_score(f)
            if fscore < f_score_min:
                skipped_filter += 1
                continue

            # 종목 메타
            row  = stock_list[stock_list["code"] == code]
            name = row.iloc[0]["name"]   if not row.empty else code
            mkt  = row.iloc[0].get("market", "") if not row.empty else ""
            sect = row.iloc[0].get("sector", "") if not row.empty else ""

            vscore = value_score(f)

            passed.append({
                "code":           code,
                "name":           name,
                "market":         mkt,
                "sector":         sect,
                "per":            per,
                "pbr":            pbr,
                "roe":            roe,
                "debt_ratio":     debt,
                "op_margin":      f.get("op_margin"),
                "revenue_yoy":    revyoy,
                "op_income_yoy":  f.get("op_income_yoy"),
                "dividend_yield": f.get("dividend_yield"),
                "f_score":        fscore,
                "f_checks":       fchecks,
                "value_score":    vscore,
                "fundamentals":   {
                    k: v for k, v in f.items()
                    if k not in ("code",)
                },
            })

        # 4. 정렬: f_score 우선, 동점이면 value_score
        passed.sort(key=lambda x: (x["f_score"], x["value_score"]), reverse=True)
        result = passed[:limit]

        logger.info(
            f"[VALUE] 완료: 후보 {len(candidates)} → "
            f"데이터없음 {skipped_no_data} → "
            f"필터탈락 {skipped_filter} → "
            f"통과 {len(passed)} → 최종 {len(result)}"
        )

        # 결과 캐시 저장 (당일 유효)
        self._cache[cache_key] = [dict(r) for r in result]

        return result

    def get_filter_defaults(self) -> Dict:
        """현재 기본 필터 임계값 반환 (API 노출용)."""
        return dict(DEFAULT_FILTERS)


value_screener = ValueScreener()
