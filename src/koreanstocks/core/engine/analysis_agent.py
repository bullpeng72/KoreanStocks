import math
import time
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any
import openai
from openai import RateLimitError as _OpenAIRateLimitError
import json
from koreanstocks.core.config import config
from koreanstocks.core.data.provider import data_provider
from koreanstocks.core.engine.indicators import indicators
from koreanstocks.core.data.database import db_manager
from koreanstocks.core.constants import calc_composite_score
from koreanstocks.core.engine.news_agent import news_agent
from koreanstocks.core.engine.prediction_model import prediction_model
from koreanstocks.core.engine.macro_news_agent import macro_news_agent

logger = logging.getLogger(__name__)


def _safe_float(val, ndigits: int = 2, fallback=None):
    """float 변환 후 NaN/Inf 검사 — JSON 직렬화 안전값 반환."""
    try:
        v = float(val)
        return fallback if (math.isnan(v) or math.isinf(v)) else round(v, ndigits)
    except (TypeError, ValueError):
        return fallback


def _safe_int(val, fallback=None):
    """int 변환 — NaN/None/오류 시 fallback 반환 (int(nan) ValueError 방지)."""
    try:
        v = float(val)
        return fallback if (math.isnan(v) or math.isinf(v)) else int(v)
    except (TypeError, ValueError):
        return fallback


class AnalysisAgent:
    """주식 데이터 분석 및 AI 의견 생성을 담당하는 에이전트"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def analyze_stock(self, code: str, name: str = "") -> Dict[str, Any]:
        """특정 종목에 대한 심층 분석 수행"""
        logger.info(f"Analyzing stock: {code} ({name})")
        
        # 1. 데이터 수집
        df = data_provider.get_ohlcv(code, period='1y')
        if df.empty:
            return {"error": f"No data found for {code}"}

        # 2. 기술적 지표 계산
        df_with_indicators = indicators.calculate_all(df)
        if df_with_indicators.empty:
            return {"error": f"지표 계산 실패 — 데이터 부족 ({code})"}
        tech_score = float(indicators.get_composite_score(df_with_indicators) or 0)

        # 3. 뉴스 감성 분석 (ML 예측보다 먼저 수행하여 블렌딩에 활용)
        news_res = news_agent.get_sentiment_score(name or code, stock_code=code)
        sentiment_score = float(news_res.get("sentiment_score") or 0)

        # 4. 시장/섹터 정보 조회 (ML predict 에 market 전달해 중복 stock_list 호출 제거)
        stock_list   = data_provider.get_stock_list()
        _row         = stock_list[stock_list['code'] == code] if 'code' in stock_list.columns else stock_list.iloc[0:0]
        market_val   = str(_row.iloc[0]['market'])                 if not _row.empty else ''
        sector_val   = str(_row.iloc[0].get('sector',   '') or '') if not _row.empty else ''
        industry_val = str(_row.iloc[0].get('industry', '') or '') if not _row.empty else ''

        # 5. ML 예측 점수 산출 (순수 ML 앙상블; sentiment 블렌딩은 composite 단계에서 일원화)
        ml_res = prediction_model.predict(
            code, df,
            df_with_indicators=df_with_indicators,
            fallback_score=tech_score,
            market=market_val,   # 이미 조회한 market 전달 → predict() 내부 중복 호출 생략
        )
        ml_raw_score   = float(ml_res.get("ensemble_score") or tech_score)  # 순수 ML 앙상블 점수
        ml_model_count = int(ml_res.get("model_count") or 0)              # 활성 모델 수 (composite 가중치 분기용)

        # 5-b. 거시경제 컨텍스트 (Phase 2 & 3) — 일별 캐시로 종목 루프 비용 無
        macro_ctx       = macro_news_agent.get_macro_context()
        macro_sentiment = float(macro_ctx.get("macro_sentiment_score") or 0)

        # 종합 점수 (constants.calc_composite_score 단일 소스)
        composite_score = round(calc_composite_score(
            tech_score=tech_score,
            ml_score=ml_raw_score,
            sentiment_score=sentiment_score,
            ml_model_count=ml_model_count,
            macro_sentiment_score=macro_sentiment,
        ), 1)

        # 참고용: ML + 뉴스 감성 블렌딩 점수 (composite 계산에는 미사용, 표시 전용)
        _sentiment_norm = max(0.0, min(100.0, (sentiment_score + 100.0) / 2.0))
        ml_blended = round(0.65 * ml_raw_score + 0.35 * _sentiment_norm, 2)

        # 6. 시장 지수 수집 (AI 분석 컨텍스트용)
        market_indices = data_provider.get_market_indices()

        # 7. AI 분석 (최근 데이터 + 순수 ML 점수 + 뉴스 점수 + 시장/섹터 + 거시 맥락)
        current_price = _safe_float(df_with_indicators.iloc[-1]['close'], 0, fallback=0.0)
        ai_opinion = self._get_ai_opinion(
            name or code, df_with_indicators.tail(30), tech_score, ml_raw_score, news_res, current_price,
            market=market_val, sector=sector_val, market_indices=market_indices,
            composite_score=composite_score, macro_ctx=macro_ctx,
        )

        # [N-1] RSI ≥ 65 BUY→HOLD (규칙 기반).
        # [I-1]은 GPT weakness 텍스트에 '과매수'가 포함될 때만 동작하나,
        # 실제 GPT 응답에서 해당 단어 출현 빈도가 낮아 실질적으로 비활성화 상태였음.
        # 성과 분석(n=234): RSI 65~70 구간 정답률 37.8%, 중앙값 -5.25% — 전 구간 중 최악.
        _rsi_latest = _safe_float(df_with_indicators.iloc[-1].get('rsi'), 1)
        if _rsi_latest is not None and _rsi_latest >= 65 and ai_opinion.get('action') == 'BUY':
            ai_opinion['action'] = 'HOLD'
            ai_opinion['action_override'] = f'RSI {_rsi_latest:.1f} ≥ 65 과매수 구간 — BUY→HOLD 자동 조정'
            logger.info(f"[{name}] [N-1] RSI {_rsi_latest:.1f} BUY→HOLD 전환")

        # 7. 결과 정리
        latest = df_with_indicators.iloc[-1]
        _bd = _safe_float(latest['bb_high'] - latest['bb_low']) if ('bb_low' in latest and 'bb_high' in latest) else None
        _bb_pos = _safe_float((latest['close'] - latest['bb_low']) / _bd, 2) if _bd else None
        analysis_res = {
            "code": code,
            "name": name,
            "market": market_val,
            "sector": sector_val,
            "industry": industry_val,
            "current_price": current_price,
            "change_pct": (
                _safe_float(latest['change'] * 100, 2) if 'change' in latest and latest['change'] != 0
                else (
                    _safe_float(
                        (float(df['close'].iloc[-1]) - float(df['close'].iloc[-2]))
                        / float(df['close'].iloc[-2]) * 100, 2
                    )
                    if len(df) >= 2 and float(df['close'].iloc[-2]) != 0 else 0.0
                )
            ) or 0.0,
            "tech_score": tech_score,
            "ml_score":       round(ml_raw_score, 2),    # 순수 ML 앙상블 (composite 계산용)
            "ml_blended":     ml_blended,               # ML + 감성 블렌딩 참고값 (표시 전용)
            "ml_model_count": ml_model_count,           # 활성 모델 수 (0이면 fallback)
            "composite_score": composite_score,         # 종합 점수 (tech+ml+sentiment 가중합)
            "sentiment_score": sentiment_score,
            "sentiment_info": news_res,
            "stats": {
                "high_52w":   _safe_float(df['high'].max(),              0),
                "low_52w":    _safe_float(df['low'].min(),               0),
                "avg_vol":    _safe_int(df['volume'].tail(20).mean()),
                "current_vol":_safe_int(latest['volume']),
            },
            "indicators": {
                "rsi":     _safe_float(latest['rsi'],           2),
                "macd":    _safe_float(latest['macd'],          2),
                "macd_sig":_safe_float(latest['macd_signal'],   2),
                "sma_20":  _safe_float(latest['sma_20'],        0),
                "bb_pos":  _bb_pos,
            },
            "ai_opinion":      ai_opinion,
            "macro_regime":    macro_ctx.get("macro_regime",       "uncertain"),
            "macro_sentiment": macro_sentiment,
            "macro_summary":   macro_ctx.get("macro_summary",      ""),
            "analysis_date":   datetime.now().strftime('%Y-%m-%d %H:%M'),
        }

        # 8. 분석 이력 저장
        try:
            db_manager.save_analysis_history(analysis_res)
        except Exception as e:
            logger.error(f"Failed to save analysis history: {e}")

        return analysis_res

    def _get_ai_opinion(self, name: str, recent_df: pd.DataFrame, tech_score: float, ml_score: float,
                        news_res: Dict, current_price: float = 0.0,
                        market: str = '', sector: str = '', market_indices: Dict = None,
                        composite_score: float = None, macro_ctx: Dict = None) -> Dict[str, Any]:
        """GPT-4o-mini를 사용한 정성적 분석 (ML·뉴스 감성·시장/섹터·거시 맥락 반영)"""
        try:
            # 최근 가격 흐름 요약 (종가, 거래량, 주요 지표 포함)
            indicator_cols = ['close', 'volume', 'rsi', 'macd', 'macd_signal', 'bb_low', 'bb_high']
            available_cols = [c for c in indicator_cols if c in recent_df.columns]
            price_summary = recent_df[available_cols].tail(10).round(2).fillna('').to_string()

            latest = recent_df.iloc[-1]
            rsi_val      = _safe_float(latest['rsi'],          1) if 'rsi'          in latest else None
            macd_val     = _safe_float(latest['macd'],         2) if 'macd'         in latest else None
            macd_sig_val = _safe_float(latest['macd_signal'],  2) if 'macd_signal'  in latest else None
            if macd_val is not None and macd_sig_val is not None:
                macd_direction = "골든크로스(상승)" if macd_val > macd_sig_val else "데드크로스(하락)"
            else:
                macd_direction = "N/A"
            rsi_val      = rsi_val      if rsi_val      is not None else 'N/A'
            macd_val     = macd_val     if macd_val     is not None else 'N/A'
            macd_sig_val = macd_sig_val if macd_sig_val is not None else 'N/A'
            _bb_denom = _safe_float(latest['bb_high'] - latest['bb_low']) if ('bb_low' in latest and 'bb_high' in latest) else None
            bb_pos = _safe_float((latest['close'] - latest['bb_low']) / _bb_denom, 2) if _bb_denom else 'N/A'

            # 시장/섹터 맥락 문자열 구성
            mkt_lines = []
            if market or sector:
                parts = [f"소속 시장: {market}" if market else '', f"섹터: {sector}" if sector else '']
                mkt_lines.append(' | '.join(p for p in parts if p))
            if market_indices:
                for key, label in [('KOSPI', 'KOSPI 지수'), ('KOSDAQ', 'KOSDAQ 지수'), ('USD_KRW', 'USD/KRW 환율')]:
                    val = market_indices.get(key)
                    chg = market_indices.get(f"{key}_change")
                    if val is not None:
                        chg_str = f" (전일 대비 {chg:+.2f}%)" if chg is not None else ''
                        unit = '원' if key == 'USD_KRW' else ''
                        mkt_lines.append(f"{label}: {val:,.2f}{unit}{chg_str}")
            # 거시 레짐 + 요약 추가 (Phase 2 & 3)
            if macro_ctx:
                regime_label  = macro_ctx.get("macro_regime_label", "")
                macro_summary = macro_ctx.get("macro_summary", "")
                macro_sent    = macro_ctx.get("macro_sentiment_score", 0)
                if regime_label:
                    mkt_lines.append(f"거시 레짐: {regime_label} (거시감성: {macro_sent:+d})")
                if macro_summary:
                    mkt_lines.append(f"거시 요약: {macro_summary}")
            market_context = '\n            '.join(f"- {l}" for l in mkt_lines) if mkt_lines else '- (정보 없음)'

            # 레짐별 추가 지침 (Phase 3)
            regime_guidance = ""
            if macro_ctx:
                regime = macro_ctx.get("macro_regime", "uncertain")
                if regime == "risk_off":
                    regime_guidance = (
                        "\n            ⚠️ 현재 거시 레짐: 위험회피 — "
                        "BUY 판단 시 종목 고유 강점이 거시 역풍을 상쇄할 수 있는지 명시하세요."
                    )
                elif regime == "risk_on":
                    regime_guidance = (
                        "\n            ✅ 현재 거시 레짐: 위험선호 — "
                        "모멘텀·성장 신호를 긍정적으로 해석하되 과열 리스크도 점검하세요."
                    )

            prompt = f"""
            주식 종목 '{name}'에 대한 데이터와 뉴스 심리를 바탕으로 심층 분석해줘.

            [점수 해석 기준]
            - 기술적 점수: 0~39 약세, 40~59 중립, 60~79 강세, 80~100 매우 강세
            - ML 점수: 향후 10거래일 크로스섹셔널 순위 예측 (0=전체 최하위 상대강도, 50=평균, 100=전체 최상위)
            - 종목 감성: -100~-50 매우 부정, -49~-1 부정, 0 중립, 1~50 긍정, 51~100 매우 긍정
            - 거시 감성: 동일 척도, 한국 주식시장 전체에 미치는 거시경제 영향

            [시장/섹터 및 거시 맥락]{regime_guidance}
            {market_context}

            [정량 점수]
            - 종합 점수(가중합): {f'{composite_score:.1f}' if composite_score is not None else 'N/A'}/100  ← 핵심 판단 기준 (tech 35% + ML 35% + 종목감성 20% + 거시감성 10%)
            - 기술적 지표 점수: {tech_score}/100
            - 머신러닝 예측 점수: {ml_score}/100
            - 종목 뉴스 감성: {float(news_res.get('sentiment_score') or 0)} (-100~100, 양수면 호재)

            [현재 기술적 지표]
            - 현재가: {int(current_price):,}원
            - RSI(14): {rsi_val} (30 이하: 과매도, 70 이상: 과매수)
            - MACD: {macd_val} / Signal: {macd_sig_val} → {macd_direction}
            - 볼린저 밴드 위치: {bb_pos} (0=하단, 0.5=중간, 1=상단)

            [최근 뉴스 요약]
            - 근거: {news_res.get('reason', '정보 없음')}
            - 주요 이슈: {news_res.get('top_news', '정보 없음')}

            [최근 10일 가격/지표 데이터]
            {price_summary}

            위 정보를 종합하여 다음 형식의 JSON으로만 응답해줘:
            {{
                "summary": "한 줄 요약",
                "strength": "강점 (최대 2개)",
                "weakness": "약점 (최대 2개)",
                "reasoning": "기술적 지표, ML 예측, 뉴스 심리, 거시 레짐을 모두 반영한 상세 추천 사유",
                "action": "BUY, HOLD, SELL 중 하나 (반드시 영문 대문자 3종 중 하나만)",
                "target_price": "10거래일 목표가 (숫자만, 현재가 기준으로 BUY면 현재가 이상, SELL이면 현재가 이하로 설정)",
                "target_rationale": "목표가 산출의 구체적 근거"
            }}
            """

            for _attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=config.DEFAULT_MODEL,
                        messages=[
                            {"role": "system", "content": "당신은 한국 주식 시장 전문 퀀트 애널리스트입니다. 제공된 정량 데이터에 근거하여 객관적이고 일관된 투자 분석을 JSON 형식으로만 제공합니다."},
                            {"role": "user", "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1,
                        max_tokens=600,
                    )
                    result = json.loads(response.choices[0].message.content)
                    break
                except _OpenAIRateLimitError:
                    if _attempt < 2:
                        wait = 10 * (2 ** _attempt)  # 10s → 20s
                        logger.warning(f"[{name}] GPT Rate limit, {wait}초 후 재시도 ({_attempt + 1}/3)")
                        time.sleep(wait)
                    else:
                        logger.error(f"[{name}] GPT Rate limit: 재시도 한도 초과")
                        return {"summary": "AI 분석 실패 (Rate limit)", "action": "N/A", "target_price": 0}

            # action 정규화: 비표준 값(소문자·부가 텍스트 등) → BUY/HOLD/SELL
            _parts = str(result.get('action', 'HOLD')).strip().upper().split()
            raw_action = _parts[0].rstrip('.') if _parts else 'HOLD'
            result['action'] = raw_action if raw_action in ('BUY', 'HOLD', 'SELL') else 'HOLD'

            # [I-1] 과매수 경고 시 BUY → HOLD 강제 전환
            # GPT가 weakness에 과매수를 명시했음에도 BUY를 출력하는 모순을 차단.
            # 성과 분석: 과매수 경고 있음 33% 정답률 vs 경고 없음 71% (SKAI 제거 후에도 동일).
            _weakness = result.get('weakness', '')
            _weakness_text = ' '.join(str(w) for w in _weakness) if isinstance(_weakness, list) else str(_weakness)
            if '과매수' in _weakness_text and result['action'] == 'BUY':
                result['action'] = 'HOLD'
                result['action_override'] = 'RSI 과매수 구간 경고 — BUY→HOLD 자동 조정'
                logger.info(f"[{name}] 과매수 경고로 BUY→HOLD 전환")

            # 데이터 정제: target_price를 숫자로 변환
            if 'target_price' in result:
                try:
                    price_str = str(result['target_price']).replace(',', '').replace('원', '').strip()
                    result['target_price'] = int(float(price_str))
                except (ValueError, TypeError):
                    result['target_price'] = 0

            # action ↔ target_price 일관성 보정
            if current_price > 0 and result.get('target_price', 0) > 0:
                tp = result['target_price']
                action = result['action']
                if action == 'BUY' and tp < current_price * 0.98:
                    result['target_price'] = int(current_price * 1.03)
                    logger.warning(f"[{name}] BUY but target_price below current. Auto-adjusted to {result['target_price']}")
                elif action == 'HOLD' and tp < current_price * 0.92:
                    result['action'] = 'SELL'
                    logger.warning(f"[{name}] HOLD but target_price significantly below current. Changed action to SELL.")
                elif action == 'SELL' and tp > current_price * 1.02:
                    result['target_price'] = int(current_price * 0.97)
                    logger.warning(f"[{name}] SELL but target_price above current. Auto-adjusted to {result['target_price']}")

            return result
        except Exception as e:
            logger.error(f"AI Analysis Error: {e}")
            return {"summary": "AI 분석 실패", "action": "N/A", "target_price": 0}

analysis_agent = AnalysisAgent()
