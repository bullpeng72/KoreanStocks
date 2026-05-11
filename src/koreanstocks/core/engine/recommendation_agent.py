import json
import logging
from collections import Counter
from datetime import date
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from koreanstocks.core.data.provider import data_provider
from koreanstocks.core.engine.analysis_agent import analysis_agent
from koreanstocks.core.data.database import db_manager
from koreanstocks.core.constants import (
    BUCKET_DEFAULT, BUCKET_LABELS, BUCKET_RATIOS as _BUCKET_RATIOS,
    calc_composite_score_from_dict,
    MAX_ANALYSIS_WORKERS, REGIME_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# 하위 호환 alias (버킷 쿼터 정렬·최종 점수 계산에 사용)
_composite_score = calc_composite_score_from_dict


# ── 품질 필터 함수 (성과 분석 기반) ─────────────────────────────────────────


def _is_volume_overheated(analysis: Dict[str, Any]) -> bool:
    """거래량 폭증(평균 대비 6배+) 여부.

    성과 분석: 6x+ 종목 정답률 33%, 중앙값 -3.8% → 유입 완료 신호.
    """
    stats = analysis.get('stats') or {}
    avg_vol = stats.get('avg_vol') or 0
    cur_vol = stats.get('current_vol') or 0
    if avg_vol <= 0:
        return False
    return (cur_vol / avg_vol) >= 6.0


def _is_price_overheated(analysis: Dict[str, Any]) -> bool:
    """당일 급등(5%+) + 강긍정 감성(50+) 동시 조건.

    성과 분석: 이 조합은 재료 소진 신호 — 흥구석유·SK증권 등 대표 실패 케이스.
    반면 당일 3% 미만 + 감성 30 이하 → 정답률 78%, 중앙값 +9.1%.
    """
    change_pct = abs(float(analysis.get('change_pct') or 0))
    sentiment  = float(analysis.get('sentiment_score') or 0)
    return change_pct >= 5.0 and sentiment >= 50


def _passes_kospi_filter(analysis: Dict[str, Any]) -> bool:
    """KOSPI 종목에 황금조합 조건 강제 적용.

    성과 분석(n=61): KOSPI 황금조합 62.2% vs 비황금조합 50.0%.
    황금조합: 거래량 배율 < 3x, 감성 0~40, 당일 변동 < 3%.
    KOSDAQ은 조건 없이 통과 (전체 감성 필터는 _is_sentiment_overheated로 별도 적용).
    """
    if analysis.get('market') != 'KOSPI':
        return True
    stats = analysis.get('stats') or {}
    avg_vol    = stats.get('avg_vol') or 1
    cur_vol    = stats.get('current_vol') or 0
    vol_mult   = (cur_vol / avg_vol) if avg_vol > 0 else 0
    sentiment  = float(analysis.get('sentiment_score') or 0)
    change_pct = abs(float(analysis.get('change_pct') or 0))
    return vol_mult < 3.0 and 0 <= sentiment <= 40 and change_pct < 3.0


def _is_sentiment_overheated(analysis: Dict[str, Any]) -> bool:
    """강긍정 감성(>50) 차단 — 전체 시장(KOSPI·KOSDAQ) 적용.

    성과 분석(n=234): 감성 구간별 역U자형 패턴이 전 시장에서 확인됨.
    강긍정(60+) 정답률 33.3%, 중앙값 -11.01% — 재료 선반영 신호.
    KOSPI는 _passes_kospi_filter(sentiment≤40)로 이미 차단되므로 실질 적용 대상은 KOSDAQ.
    """
    sentiment = float(analysis.get('sentiment_score') or 0)
    return sentiment > 50


def _is_sentiment_rsi_overheated(analysis: Dict[str, Any]) -> bool:
    """감성 과열(≥25) + RSI 과매수(≥65) 복합 조건 — 전체 시장 적용.

    성과 분석(n=41): 정답률 43.9%, 평균 수익 -2.89%.
    단일 조건(감성≥25: 60.5% / RSI≥65: 49.1%)보다 조합 효과가 강함.
    """
    sentiment = float(analysis.get('sentiment_score') or 0)
    inds = analysis.get('indicators') or {}
    rsi = float(inds.get('rsi') or 0)
    return sentiment >= 25 and rsi >= 65


def _apply_bucket_quota(
    results: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """버킷별 쿼터를 보장하며 최종 종목 선정.

    1단계: 각 버킷 자체 후보에서 쿼터만큼 선정 (섹터 다양성 반영).
    2단계: 쿼터 미달 버킷은 교차 버킷 잉여 후보로 보충 + 버킷 재태깅
           → 최종 결과에 3개 버킷이 모두 대표되도록 보장.
    3단계: 전체 limit 미달이면 점수 순으로 보충.
    """
    max_per_sector = max(1, round(limit / 3))

    # 버킷별 쿼터 계산 (합계가 limit이 되도록 조정)
    quotas: Dict[str, int] = {}
    assigned = 0
    for i, (bucket_name, ratio) in enumerate(_BUCKET_RATIOS):
        if i < len(_BUCKET_RATIOS) - 1:
            q = max(1, round(limit * ratio))
            quotas[bucket_name] = q
            assigned += q
        else:
            quotas[bucket_name] = max(1, limit - assigned)

    selected: List[Dict[str, Any]] = []
    selected_codes: set[str] = set()
    sector_count: Dict[str, int] = {}

    def _pick(candidates: List[Dict[str, Any]], quota: int) -> List[Dict[str, Any]]:
        """candidates 중 쿼터만큼 섹터 다양성을 고려해 선정."""
        picked: List[Dict[str, Any]] = []
        deferred: List[Dict[str, Any]] = []
        for rec in candidates:
            if rec.get('code') in selected_codes:
                continue
            sector = (rec.get('sector') or '').strip() or '__unknown__'
            cnt = sector_count.get(sector, 0)
            if cnt < max_per_sector:
                picked.append(rec)
                sector_count[sector] = cnt + 1
            else:
                deferred.append(rec)
            if len(picked) >= quota:
                break
        # 섹터 한도로 미달이면 보충
        if len(picked) < quota:
            for rec in deferred:
                if rec.get('code') not in selected_codes:
                    picked.append(rec)
                    sector = (rec.get('sector') or '').strip() or '__unknown__'
                    sector_count[sector] = sector_count.get(sector, 0) + 1  # deferred: 섹터 한도 초과 허용
                    if len(picked) >= quota:
                        break
        return picked

    # ── 1단계: 버킷별 자체 후보 선정 ──────────────────────────────
    bucket_shortfall: Dict[str, int] = {}
    for bucket_name, quota in quotas.items():
        bucket_results = sorted(
            [r for r in results if r.get('bucket') == bucket_name],
            key=_composite_score, reverse=True,
        )
        picks = _pick(bucket_results, quota)
        selected.extend(picks)
        selected_codes.update(r.get('code') for r in picks if r.get('code'))

        shortfall = quota - len(picks)
        if shortfall > 0:
            bucket_shortfall[bucket_name] = shortfall
            logger.warning(
                f"버킷 '{bucket_name}' 후보 부족: {len(picks)}/{quota} "
                f"→ {shortfall}개 교차 버킷에서 보충 예정"
            )

    # ── 2단계: 미달 버킷 교차 보충 (잉여 후보를 재태깅) ───────────
    for bucket_name, needed in bucket_shortfall.items():
        cross_pool = sorted(
            [r for r in results if r.get('code') not in selected_codes],
            key=_composite_score, reverse=True,
        )
        cross_picks = _pick(cross_pool, needed)
        for r in cross_picks:
            r['bucket'] = bucket_name  # 미달 버킷으로 재태깅
        selected.extend(cross_picks)
        selected_codes.update(r.get('code') for r in cross_picks if r.get('code'))
        if cross_picks:
            logger.info(
                f"버킷 '{bucket_name}' 교차 보충 완료: "
                f"{len(cross_picks)}종목 (재태깅)"
            )
        else:
            logger.warning(f"버킷 '{bucket_name}' 교차 보충 불가 (전체 후보 소진)")

    # ── 3단계: 전체 limit 미달 시 잔여 종목으로 보충 (점수 순) ────
    if len(selected) < limit:
        remaining = sorted(
            [r for r in results if r.get('code') not in selected_codes],
            key=_composite_score, reverse=True,
        )
        picks = _pick(remaining, limit - len(selected))
        selected.extend(picks)
        selected_codes.update(r.get('code') for r in picks if r.get('code'))

    return selected[:limit]


class RecommendationAgent:
    """분석된 데이터를 바탕으로 투자 종목을 추천하는 에이전트"""

    def get_recommendations(
        self,
        limit: int = 5,
        market: str = 'ALL',
        theme_keywords: List[str] = None,
        theme_label: str = '전체',
    ) -> List[Dict[str, Any]]:
        """유망 종목 추천 리스트 생성 (버킷 기반 후보군 구성)

        버킷 구성:
          volume   (40%) — 거래량 상위 유동성 안정주
          momentum (35%) — 상승 모멘텀 (+2%~+15%)
          rebound  (25%) — 거래량 상위 중 하락 반등 후보

        분석 풀: min(limit * 8, 80)  →  limit=5: 40개, limit=10: 80개

        최종 추천에서 3개 버킷이 모두 대표되도록 쿼터를 보장한다.
        후보 부족 버킷은 교차 버킷 잉여 종목으로 보충 후 재태깅.
        """
        logger.info(f"Generating recommendations (Market: {market}, Theme: {theme_label})...")

        # ── 1. 후보군 코드 선정 ─────────────────────────────────────
        if theme_keywords:
            # 테마 지정 시: 테마 종목 + 거래량 랭킹 교집합 우선, 나머지 추가
            theme_df    = data_provider.get_stocks_by_theme(theme_keywords, market)
            theme_codes = set(theme_df['code'].tolist()) if 'code' in theme_df.columns else set()
            ranked      = data_provider.get_market_ranking(limit=200, market=market)
            candidate_codes = [c for c in ranked if c in theme_codes]
            existing = set(candidate_codes)
            candidate_codes += [c for c in (theme_df['code'].tolist() if 'code' in theme_df.columns else []) if c not in existing]

            # 테마 모드에서도 버킷 분류 적용
            # 거래량·등락률 기반 버킷 맵을 가져와 테마 종목에 매핑
            market_buckets = data_provider.get_market_buckets(market)
            bucket_map: Dict[str, str] = {}
            for bname, codes in market_buckets.items():
                for code in codes:
                    if code not in bucket_map:
                        bucket_map[code] = bname
            # 버킷 맵에 없는 테마 종목은 기본 버킷으로 배정
            code_bucket: Dict[str, str] = {c: bucket_map.get(c, BUCKET_DEFAULT) for c in candidate_codes}
        else:
            # 버킷 기반 후보군 구성
            buckets = data_provider.get_market_buckets(market)
            total_pool = min(limit * 8, 80)
            code_bucket = {}
            seen: set[str] = set()

            for bucket_name, ratio in _BUCKET_RATIOS:
                pool_size = max(2, round(total_pool * ratio))
                count = 0
                for code in buckets.get(bucket_name, []):
                    if code not in seen and count < pool_size:
                        code_bucket[code] = bucket_name
                        seen.add(code)
                        count += 1

            candidate_codes = list(code_bucket.keys())

        if not candidate_codes:
            return []

        # 종목명 매칭용 전체 리스트 (후보군 확정 후 1회 조회)
        stock_list = data_provider.get_stock_list()

        # 후보군 버킷 분포 로깅
        pool_dist = Counter(code_bucket[c] for c in candidate_codes)
        logger.info(
            f"분석 후보군 {len(candidate_codes)}종목 "
            f"(volume={pool_dist.get('volume',0)}, "
            f"momentum={pool_dist.get('momentum',0)}, "
            f"rebound={pool_dist.get('rebound',0)})"
        )

        # ── 2. 종목명 해석 ──────────────────────────────────────────
        name_map: Dict[str, str] = (
            dict(zip(stock_list['code'], stock_list['name']))
            if 'code' in stock_list.columns and 'name' in stock_list.columns
            else {}
        )
        candidates: List[Tuple[str, str]] = []
        for code in candidate_codes:
            _nm = name_map.get(code)
            nm = _nm if isinstance(_nm, str) and _nm else code
            candidates.append((code, nm))

        # ── 3. 병렬 분석 ────────────────────────────────────────────
        # as_completed(timeout) 으로 전체 글로벌 타임아웃 제한.
        # result(timeout=60) 은 as_completed 가 이미 완료된 future 를 yield 하므로
        # 개별 타임아웃 효과가 없다 → 글로벌 timeout 이 실질적 한계.
        # 종목 수 × 단종목 소요 추정(30s) / workers + 여유 = max(120, n*3) 초
        _global_timeout = max(120, len(candidates) * 3)
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=MAX_ANALYSIS_WORKERS) as executor:
            futures = {
                executor.submit(self._analyze_candidate, code, nm): code
                for code, nm in candidates
            }
            try:
                for future in as_completed(futures, timeout=_global_timeout):
                    code = futures[future]
                    try:
                        res = future.result()
                        if res is not None:
                            res['bucket'] = code_bucket.get(code, BUCKET_DEFAULT)
                            results.append(res)
                    except Exception as e:
                        logger.warning(f"Analysis error for {code}: {e}")
            except FuturesTimeoutError:
                done   = [futures[f] for f in futures if f.done()]
                hung   = [futures[f] for f in futures if not f.done()]
                logger.warning(
                    f"분석 글로벌 타임아웃 ({_global_timeout}s): "
                    f"{len(done)}/{len(candidates)}개 완료 — 미완료 {len(hung)}개 건너뜀: {hung}"
                )

        if not results:
            logger.warning("No successful analyses to recommend.")
            return []

        # 분석 완료 버킷 분포 로깅
        result_dist = Counter(r.get('bucket', '?') for r in results)
        logger.info(
            f"분석 완료 {len(results)}종목 "
            f"(volume={result_dist.get('volume',0)}, "
            f"momentum={result_dist.get('momentum',0)}, "
            f"rebound={result_dist.get('rebound',0)})"
        )

        # ── Phase 3: 거시 레짐 기반 필터링 ─────────────────────────
        # analysis_agent가 이미 macro_news_agent를 캐시로 호출했으므로 추가 API 호출 없음
        from koreanstocks.core.engine.macro_news_agent import macro_news_agent
        macro_ctx         = macro_news_agent.get_macro_context()
        regime            = macro_ctx.get("macro_regime", "uncertain")

        # 레짐별 composite_score 최소 임계값 (constants.py 단일 소스)
        threshold = REGIME_SCORE_THRESHOLD.get(regime, 50.0)
        pre_n             = len(results)
        pre_filter_results = results  # fallback 복원용 — 필터 전에 반드시 저장
        results           = [r for r in results if _composite_score(r) >= threshold]
        if len(results) < pre_n:
            logger.info(
                f"[레짐 필터] {regime} 임계값 {threshold} 적용: "
                f"{pre_n} → {len(results)}종목"
            )

        if not results:
            logger.warning(
                f"레짐 필터({regime}) 후 후보 없음 — 임계값 완화: 전체 {pre_n}종목으로 복원"
            )
            results = pre_filter_results

        # ── 품질 필터 [I-2][I-3][S-2][N-2][N-4] ────────────────────
        # 분석 근거: docs/6_PERFORMANCE_IMPROVEMENT.md §5·Phase2 참조.
        _pre_quality = len(results)
        _quality_filtered = [
            r for r in results
            if not _is_volume_overheated(r)        # [I-2] 거래량 6x+
            and not _is_price_overheated(r)        # [I-3] 급등+강감성
            and not _is_sentiment_overheated(r)    # [N-4] 강긍정(>50) 전시장
            and not _is_sentiment_rsi_overheated(r)# [N-2] 감성≥25+RSI≥65 복합
            and _passes_kospi_filter(r)            # [S-2] KOSPI 황금조합
        ]
        if _quality_filtered:
            _removed = _pre_quality - len(_quality_filtered)
            if _removed > 0:
                logger.info(
                    f"[품질 필터] {_pre_quality} → {len(_quality_filtered)}종목 "
                    f"({_removed}건 제외: 거래량폭증·과열·강감성·감성RSI복합·KOSPI조건)"
                )
            results = _quality_filtered
        else:
            logger.warning("[품질 필터] 결과 없음 — 필터 건너뜀 (전체 유지)")

        # ── [N-3] risk_off + rebound BUY→HOLD ───────────────────────
        # 성과 분석(n=10): risk_off 레짐 + rebound 버킷 정답률 30%, 평균 -7.45%.
        # 하락장에서 기술적 반등 신호의 신뢰도가 구조적으로 낮음.
        if regime == 'risk_off':
            _n3_converted = 0
            for r in results:
                if r.get('bucket') == 'rebound':
                    ai_op = r.get('ai_opinion') or {}
                    if ai_op.get('action') == 'BUY':
                        ai_op['action'] = 'HOLD'
                        r['action_override'] = 'risk_off 레짐 + rebound 버킷 — BUY→HOLD 전환'
                        _n3_converted += 1
            if _n3_converted:
                logger.info(f"[N-3] risk_off+rebound BUY→HOLD {_n3_converted}건 전환")

        # ── 상대 순위 계산 [S-3] ─────────────────────────────────────
        # 복합점수 절대값은 64~92점에 집중되어 변별력이 낮음.
        # 세션 내 백분위 순위를 score_percentile 필드에 추가 (표시·정렬 보조용).
        if results:
            _all_scores = sorted(_composite_score(r) for r in results)
            _n = len(_all_scores)
            for r in results:
                _cs = _composite_score(r)
                _rank = sum(1 for s in _all_scores if s <= _cs)
                r['score_percentile'] = round(_rank / _n * 100)

        # ── 4. 버킷 쿼터 + 섹터 다양성으로 최종 선정 ───────────────
        final_recs = _apply_bucket_quota(results, limit)
        for rec in final_recs:
            rec['theme']            = theme_label
            rec['analysis_market']  = market
            rec['composite_score']  = round(_composite_score(rec), 2)
            rec['bucket_label']     = BUCKET_LABELS.get(rec.get('bucket', BUCKET_DEFAULT), '')
            rec['macro_regime']     = regime
            rec['macro_regime_label'] = macro_ctx.get("macro_regime_label", "불확실")
        self._save_to_db(final_recs)

        # 최종 버킷 분포 로깅
        final_dist = Counter(r.get('bucket', '?') for r in final_recs)
        logger.info(
            f"최종 추천 {len(final_recs)}종목 버킷 분포: "
            f"volume={final_dist.get('volume',0)}, "
            f"momentum={final_dist.get('momentum',0)}, "
            f"rebound={final_dist.get('rebound',0)}"
        )

        return final_recs

    def _analyze_candidate(self, code: str, name: str) -> Optional[Dict[str, Any]]:
        """단일 종목 분석 — ThreadPoolExecutor 워커에서 호출"""
        try:
            analysis = analysis_agent.analyze_stock(code, name)
            return analysis if analysis is not None and "error" not in analysis else None
        except Exception as e:
            logger.warning(f"Analysis failed for {code} ({name}): {e}")
            return None

    def _save_to_db(self, recommendations: List[Dict]):
        """추천 결과를 날짜별로 저장 (동일 날짜+종목은 덮어쓰기)"""
        if not recommendations:
            return
        session_date = date.today().isoformat()
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            saved_count = 0
            for i, rec in enumerate(recommendations):
                assert isinstance(i, int) and 0 <= i < len(recommendations), f"잘못된 SAVEPOINT 인덱스: {i}"
                sp = f'sp_rec_{i}'
                cursor.execute(f'SAVEPOINT {sp}')
                try:
                    _cs = rec.get('composite_score')
                    composite = _cs if _cs is not None else round(_composite_score(rec), 2)
                    try:
                        detail_json = json.dumps(rec, ensure_ascii=False, default=str)
                    except Exception as e:
                        logger.warning(f"JSON serialization failed for {rec.get('code', '?')}: {e}")
                        detail_json = None
                    ai_opinion = rec.get('ai_opinion') or {}
                    try:
                        target_price = float(ai_opinion.get('target_price') or 0)
                    except (TypeError, ValueError):
                        target_price = 0.0
                    cursor.execute(
                        'DELETE FROM recommendations WHERE code = ? AND session_date = ?',
                        (rec.get('code'), session_date)
                    )
                    cursor.execute('''
                        INSERT INTO recommendations
                            (code, type, score, reason, target_price, source, detail_json, session_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        rec.get('code'),
                        ai_opinion.get('action', 'HOLD'),
                        composite,
                        ai_opinion.get('summary', ''),
                        target_price,
                        'AI_RECOMMENDER_V2',
                        detail_json,
                        session_date,
                    ))
                    cursor.execute(f'RELEASE {sp}')
                    saved_count += 1
                except Exception as e:
                    cursor.execute(f'ROLLBACK TO {sp}')
                    cursor.execute(f'RELEASE {sp}')
                    logger.warning(f"DB 저장 실패 (code={rec.get('code', '?')}): {e}, 건너뜀")
            conn.commit()
        logger.info(f"Saved {saved_count}/{len(recommendations)} recommendations for {session_date}")


recommendation_agent = RecommendationAgent()
