import io
import math
import re
import time
import zipfile
import requests
import logging
import json
from datetime import datetime, date, timedelta, timezone
from typing import List, Dict, Any
from urllib.parse import urlparse
from xml.etree import ElementTree
import openai
from openai import RateLimitError as _OpenAIRateLimitError
from koreanstocks.core.config import config

logger = logging.getLogger(__name__)

class NewsAgent:
    """주식 관련 뉴스 수집 및 감성 분석을 담당하는 에이전트"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.naver_client_id = config.NAVER_CLIENT_ID
        self.naver_client_secret = config.NAVER_CLIENT_SECRET
        self.dart_api_key = config.DART_API_KEY
        self._cache: Dict[str, Any] = {}             # key: "{종목명}_{YYYY-MM-DD}" — 당일 TTL 캐시
        self._dart_corp_cache: Dict[str, str] = {}   # stock_code → DART corp_code (영구 캐시)

    def get_sentiment_score(self, stock_name: str, stock_code: str = '') -> Dict[str, Any]:
        """특정 종목의 최신 뉴스 + DART 공시를 분석하여 감성 점수 반환.

        캐시 우선순위:
          L1 — 프로세스 내 메모리 (동일 실행 내 중복 호출 방지)
          L2 — SQLite 영속 캐시 (GitHub Actions 재실행·앱 재시작 시 GPT 비용 절감)
          당일(YYYY-MM-DD) 키가 일치할 때만 히트; 날짜가 바뀌면 자동으로 새로 수집.
        """
        from koreanstocks.core.data.database import db_manager

        cache_key = f"{stock_name}_{date.today().isoformat()}"

        # L1: 메모리 캐시
        if cache_key in self._cache:
            logger.debug(f"[캐시 L1 히트] {stock_name}")
            return self._cache[cache_key]

        # L2: SQLite 캐시
        cached = db_manager.get_sentiment_cache(cache_key)
        if cached is not None:
            logger.debug(f"[캐시 L2 히트] {stock_name}")
            self._cache[cache_key] = cached  # L1에도 올려둠
            return cached

        # 캐시 미스: 수집 → 분석
        news_items = self._fetch_news(stock_name)
        dart_items = self._fetch_dart_disclosures(stock_code) if stock_code else []

        if not news_items and not dart_items:
            result = {"sentiment_score": 0, "sentiment_label": "Neutral", "reason": "최근 뉴스·공시 없음", "articles": []}
        else:
            result = self._analyze_sentiment_with_ai(stock_name, news_items, dart_items)
            result["articles"] = news_items

        # L1 + L2 저장
        self._cache[cache_key] = result
        db_manager.save_sentiment_cache(cache_key, result)
        return result

    def _fetch_news(self, stock_name: str) -> List[Dict[str, str]]:
        """네이버 뉴스 API를 통해 뉴스 제목 + 날짜 수집.

        Returns: [{"title": str, "pubDate": str}, ...]
        """
        if not self.naver_client_id or not self.naver_client_secret:
            logger.warning("Naver API credentials missing.")
            return []

        # " 주가" 추가로 검색 정확도 향상 (동명 계열사 뉴스 혼입 방지)
        # display=50: 계열사 필터·중복 제거 후에도 충분한 기사 수 확보
        query = f"{stock_name} 주가"
        url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display=50&sort=date"
        headers = {
            "X-Naver-Client-Id": self.naver_client_id,
            "X-Naver-Client-Secret": self.naver_client_secret,
        }

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                items = response.json().get('items', [])
                result = []
                for item in items:
                    title = (
                        item.get('title', '')
                        .replace('<b>', '').replace('</b>', '')
                        .replace('&quot;', '"').replace('&amp;', '&')
                    )
                    pub_date = item.get('pubDate', '')
                    days_int = self._parse_days_ago(pub_date)
                    result.append({
                        "title":        title,
                        "link":         item.get('link', ''),          # Naver 뉴스 페이지
                        "originallink": item.get('originallink', ''),  # 원문 URL
                        "pubDate":      pub_date,
                        "days_ago":     self._days_ago_label(pub_date),
                        "days_ago_int": days_int,
                    })
                # 계열사 혼입 제거 → 중복 제거 순으로 적용
                result = self._filter_by_stock_name(result, stock_name)
                before = len(result)
                result = self._deduplicate_news(result)
                logger.debug(f"[{stock_name}] 뉴스 전처리: 계열사 필터 후 {before}건, 중복 제거 후 {len(result)}건")
                return result
        except Exception as e:
            logger.error(f"News fetching error: {e}")

        return []

    @staticmethod
    def _filter_by_stock_name(items: List[Dict], stock_name: str) -> List[Dict]:
        """계열사 혼입 기사 제거 — 종목명이 단독으로 언급된 기사만 유지.

        판단 기준: 제목에서 종목명 다음에 오는 문자가
          (A) 한글 조사(가/는/를/의/에서 등)이고, 그 뒤가 한글이 아닌 경우  →  단독 언급
          (B) 공백·숫자·영문·문장부호 등 비한글 문자인 경우               →  단독 언급
          (C) 조사 없이 한글이 바로 이어지는 경우                          →  계열사 복합어 → 제거

        예)
          '카카오가 주가 상승'  →  '가' + 공백  →  유지 ✓
          '카카오 주가'        →  공백(비한글)  →  유지 ✓
          '카카오뱅크 실적'    →  '뱅' 바로 이어짐 → 제거 ✗
          '삼성전자서비스 공시' →  '서'(조사처럼 보이나) 뒤 '비'(한글) → 제거 ✗

        필터 후 5건 미만이면 원본 반환 (소형주 혹은 짧은 종목명 방어 fallback).
        """
        # 한국어 조사 목록: 긴 형태를 먼저 나열 (greedy 방지)
        particles = (
            '이고|이며|이나|이라|에서|로서|로부터|까지|처럼|보다|부터|마다|'
            '조차|마저|뿐|씩|이든지|이라도|이든|이|가|은|는|을|를|의|에|로|과|와|도|만|며|고|서'
        )
        # 종목명 뒤: (조사 + 비한글) 또는 (비한글 바로)
        standalone = re.compile(
            re.escape(stock_name)
            + r'(?:(?:' + particles + r')(?![가-힣])|(?![가-힣]))'
        )
        filtered = [item for item in items if standalone.search(item['title'])]

        removed = len(items) - len(filtered)
        if filtered and removed:
            logger.debug(f"[계열사 필터] {stock_name}: {removed}건 제거 ({len(filtered)}건 유지)")

        # 결과가 너무 적으면 원본 유지 (소형주·짧은 종목명 방어)
        # 3건 미만일 때만 fallback — 3건 이상이면 관련성 높은 기사만 활용하는 것이 품질상 유리
        if len(filtered) < 3:
            logger.debug(f"[계열사 필터] {stock_name}: 필터 후 {len(filtered)}건 미만 → fallback")
            return items
        return filtered

    @staticmethod
    def _deduplicate_news(items: List[Dict]) -> List[Dict]:
        """중복 기사 2단계 제거.

        1단계 — 도메인 중복 제거:
            originallink 기준으로 같은 매체(도메인)의 기사는 가장 최신 1건만 유지.
            연합뉴스발 기사를 수십 개 매체가 동시에 게재하는 패턴을 걸러낸다.

        2단계 — 제목 유사도 중복 제거:
            한글·영문 토큰 Jaccard 유사도 > 0.75이면 같은 내용으로 판단하고 제거.
            도메인이 달라도 사실상 동일한 기사를 걸러낸다.
        """
        # 1단계: 도메인 중복 제거 (API는 최신순이므로 첫 번째 = 가장 최신)
        seen_domains: set = set()
        domain_deduped: List[Dict] = []
        for item in items:
            url = item.get('originallink') or item.get('link', '')
            try:
                domain = urlparse(url).netloc
            except Exception:
                domain = ''
            if not domain:
                domain_deduped.append(item)
            elif domain not in seen_domains:
                seen_domains.add(domain)
                domain_deduped.append(item)

        # 2단계: 제목 Jaccard 유사도 중복 제거
        def tokenize(title: str) -> set:
            return set(re.split(r'[\s\W]+', title)) - {''}

        final: List[Dict] = []
        seen_token_sets: List[set] = []
        for item in domain_deduped:
            tokens = tokenize(item['title'])
            is_dup = any(
                len(tokens & seen) / len(tokens | seen) > 0.75
                for seen in seen_token_sets
                if tokens and seen
            )
            if not is_dup:
                final.append(item)
                seen_token_sets.append(tokens)

        return final

    # ── DART 공시 ────────────────────────────────────────────────────

    def _load_dart_corp_map(self) -> None:
        """DART corpCode.xml ZIP을 다운로드하여 stock_code → corp_code 전체 매핑을 캐시에 로드.

        디스크 캐시 우선 조회 ({BASE_DIR}/data/storage/dart_corp_cache.json, 당일 유효):
          - 당일 캐시 파일이 있으면 API 재호출 없이 즉시 로드 (~2 MB ZIP 다운로드 생략)
          - 없거나 날짜가 다르면 DART API에서 새로 다운로드 후 캐시 파일 갱신
        """
        from pathlib import Path
        cache_path = Path(config.BASE_DIR) / "data" / "storage" / "dart_corp_cache.json"
        today = date.today().isoformat()

        # ── 디스크 캐시 확인 ──────────────────────────────────────────
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                if cached.get('date') == today:
                    self._dart_corp_cache.update(cached.get('data', {}))
                    self._dart_corp_cache["__loaded__"] = "__loaded__"
                    logger.info(f"[DART] corpCode 캐시 로드 (디스크): {len(self._dart_corp_cache) - 1}개 기업")
                    return
            except Exception as e:
                logger.debug(f"[DART] 디스크 캐시 읽기 실패: {e}")

        # ── DART API 다운로드 ─────────────────────────────────────────
        try:
            resp = requests.get(
                "https://opendart.fss.or.kr/api/corpCode.xml",
                params={"crtfc_key": self.dart_api_key},
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning(f"[DART] corpCode.xml 다운로드 실패: HTTP {resp.status_code}")
                return
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                xml_bytes = zf.read("CORPCODE.xml")
            root = ElementTree.fromstring(xml_bytes)
            corp_map: Dict[str, str] = {}
            for item in root.findall("list"):
                sc = (item.findtext("stock_code") or "").strip()
                cc = (item.findtext("corp_code") or "").strip()
                if sc:
                    corp_map[sc] = cc
            self._dart_corp_cache.update(corp_map)
            self._dart_corp_cache["__loaded__"] = "__loaded__"
            logger.info(f"[DART] corpCode 매핑 로드 완료: {len(corp_map)}개 기업")

            # ── 디스크 캐시 저장 (다음 서버 시작 시 재사용) ──────────
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({'date': today, 'data': corp_map}, f, ensure_ascii=False)
                logger.debug(f"[DART] corpCode 캐시 저장: {cache_path}")
            except Exception as e:
                logger.debug(f"[DART] 디스크 캐시 저장 실패: {e}")
        except Exception as e:
            logger.warning(f"[DART] corpCode.xml 파싱 실패: {e}")

    def _get_dart_corp_code(self, stock_code: str) -> str:
        """주식 종목코드(6자리) → DART 고유번호(8자리) 변환 (메모리 캐시).

        최초 호출 시 corpCode.xml ZIP을 일괄 다운로드하여 전체 매핑을 캐시에 저장한다.
        이후 호출은 캐시만 조회하므로 API 호출 없음.
        """
        if "__loaded__" not in self._dart_corp_cache:
            self._load_dart_corp_map()
        return self._dart_corp_cache.get(stock_code, "")

    def _fetch_dart_disclosures(self, stock_code: str, days: int = 30) -> List[Dict]:
        """최근 N일 DART 공시 목록 반환 (정기공시·주요사항·지분공시 포함).

        DART Open API: GET /api/list.json
        반환: [{"title": str, "date": str, "category": str}, ...]

        API 키 미설정 시 빈 목록 반환 (graceful fallback).
        """
        if not self.dart_api_key or not stock_code:
            return []

        corp_code = self._get_dart_corp_code(stock_code)
        if not corp_code:
            return []

        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=days)

        # 공시 유형 분류표 (DART pblntf_ty 코드 → 한글)
        TYPE_LABEL = {
            "A": "정기공시",    # 분기·반기·사업보고서
            "B": "주요사항",    # 유상증자·전환사채·합병 등
            "C": "발행공시",    # 증권신고서
            "D": "지분공시",    # 최대주주·임원 지분 변동
            "E": "기타공시",
            "F": "외부감사",
            "I": "거래소공시",  # 계약·수주·자사주 취득 등
        }

        try:
            resp = requests.get(
                "https://opendart.fss.or.kr/api/list.json",
                params={
                    "crtfc_key":  self.dart_api_key,
                    "corp_code":  corp_code,
                    "bgn_de":     start_dt.strftime("%Y%m%d"),
                    "end_de":     end_dt.strftime("%Y%m%d"),
                    "page_count": 10,
                },
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "000":
                    result = []
                    for item in data.get("list", []):
                        ptype = item.get("pblntf_ty", "E")
                        result.append({
                            "title":    item.get("report_nm", ""),
                            "date":     item.get("rcept_dt", ""),   # YYYYMMDD
                            "category": TYPE_LABEL.get(ptype, "기타공시"),
                        })
                    logger.info(f"[DART] {stock_code} 공시 {len(result)}건 수집")
                    return result
        except Exception as e:
            logger.debug(f"[DART] 공시 조회 실패 ({stock_code}): {e}")
        return []

    # ── 시간 가중치 ──────────────────────────────────────────────────

    @staticmethod
    def _parse_days_ago(pub_date_str: str) -> int:
        """Naver pubDate 문자열 → 경과 일수(정수). 파싱 실패 시 7 반환."""
        try:
            pub_dt = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
            return max((datetime.now(timezone.utc) - pub_dt.astimezone(timezone.utc)).days, 0)
        except Exception:
            return 7

    @staticmethod
    def _time_weight(days: int) -> float:
        """지수 감쇠 시간 가중치.

        w = exp(−0.35 × days)
          오늘(0일): 1.00
          1일 전   : 0.70
          3일 전   : 0.35
          7일 전   : 0.09
        """
        return round(math.exp(-0.35 * days), 2)

    @staticmethod
    def _days_ago_label(pub_date_str: str) -> str:
        """Naver pubDate 문자열 → '오늘/N일 전' 표시용 문자열."""
        days = NewsAgent._parse_days_ago(pub_date_str)
        if days == 0:
            return "오늘"
        elif days == 1:
            return "1일 전"
        else:
            return f"{days}일 전"

    def _analyze_sentiment_with_ai(self, stock_name: str,
                                   news_items: List[Dict],
                                   dart_items: List[Dict] = None) -> Dict[str, Any]:
        """GPT-4o-mini를 사용하여 뉴스 + DART 공시의 투자 심리 분석.

        - 뉴스: 지수 감쇠 시간 가중치 (Python 계산 → GPT에 수치 전달)
        - DART 공시: 뉴스와 동일한 시간 가중치 적용 + 신뢰도 높음 명시
        - temperature=0.1 으로 응답 일관성 확보
        """
        # 뉴스 섹션: 시간 가중치 수치 포함
        news_lines = []
        for item in news_items:
            days   = item.get('days_ago_int', self._parse_days_ago(item.get('pubDate', '')))
            weight = self._time_weight(days)
            age_label = item.get('days_ago', '')
            news_lines.append(f"- [가중치 {weight:.2f} / {age_label}] {item['title']}")
        news_section = "\n".join(news_lines) if news_lines else "- (없음)"

        # DART 공시 섹션: 뉴스와 동일한 시간 가중치 적용
        dart_lines = []
        for d in (dart_items or []):
            date_str = d.get('date', '')
            date_fmt = date_str
            if len(date_str) == 8:
                date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                try:
                    dart_dt = datetime.strptime(date_str, "%Y%m%d")
                    days = max((datetime.now() - dart_dt).days, 0)
                except Exception:
                    days = 7
            else:
                days = 7
            weight = self._time_weight(days)
            dart_lines.append(f"- [가중치 {weight:.2f} / {d.get('category', '')} / {date_fmt}] {d['title']}")
        dart_section = "\n".join(dart_lines) if dart_lines else "- (없음)"

        dart_instruction = (
            "\n\n[금감원 공식 공시 — 뉴스보다 직접적이고 신뢰도 높음]\n"
            f"{dart_section}\n"
            "공시 내용(유상증자·전환사채·합병·계약·실적)은 뉴스 헤드라인보다 주가 영향이 크므로 점수 산출 시 더 높은 비중을 두세요."
            if dart_lines else ""
        )

        prompt = f"""
        다음은 주식 종목 '{stock_name}'에 대한 최신 정보입니다.
        뉴스와 공시 모두 시간 가중치(오늘=1.00, 오래될수록 감소)가 표시되어 있습니다.

        [뉴스 제목 및 시간 가중치]
        {news_section}{dart_instruction}

        위 정보를 종합하여 향후 주가에 미칠 영향의 감성 점수를 산출해주세요.

        ▶ 채점 기준 (반드시 준수):
        - 0 (중립): 특별한 재료 없음, 단순 시황, 업종 전반 언급, 루머성 기사
        - ±10~25: 목표가 소폭 조정, 임원 소규모 매매, 단기 수급 변화
        - ±25~50: 어닝 서프라이즈·쇼크(±10~20%), 유상증자, 대형 수주·계약
        - ±50~100: 어닝 대폭 서프라이즈(±30%+), M&A·공개매수, 사기·횡령 공시

        ▶ 주의: 대부분의 종목은 특별한 재료가 없으면 0 근처입니다.
        확실한 근거 없이 +50 이상을 사용하지 마세요.
        예상 분포: 부정 25% / 중립 50% / 긍정 25%

        다음 형식의 JSON으로만 응답해줘:
        {{
            "sentiment_score": 점수(숫자, -100~100),
            "sentiment_label": "Very Bullish/Bullish/Neutral/Bearish/Very Bearish",
            "reason": "점수 산출 근거 (한 문장, 공시·고가중치 뉴스 위주 요약)",
            "top_news": "가장 영향력이 큰 뉴스 또는 공시 한 줄 요약"
        }}
        """

        for _attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=config.DEFAULT_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "당신은 냉정한 퀀트 애널리스트입니다. 반드시 JSON 형식으로만 답변하세요. "
                                "감성 점수는 실제 주가 영향이 확인된 재료에만 ±30 이상을 부여하고, "
                                "모호하거나 일상적인 뉴스는 -10~10 사이로 채점합니다."
                            ),
                        },
                        {"role": "user",   "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,   # 일관된 감성 점수 산출
                    max_tokens=200,
                )
                result = json.loads(response.choices[0].message.content)
                # 감성 점수 범위 클램핑 (-100~100 보장)
                try:
                    result['sentiment_score'] = max(-100, min(100, int(float(result.get('sentiment_score', 0)))))
                except (TypeError, ValueError):
                    result['sentiment_score'] = 0
                return result
            except _OpenAIRateLimitError:
                if _attempt < 2:
                    wait = 10 * (2 ** _attempt)  # 10s → 20s
                    logger.warning(f"[뉴스감성] GPT Rate limit, {wait}초 후 재시도 ({_attempt + 1}/3)")
                    time.sleep(wait)
                else:
                    logger.error("GPT Rate limit: 재시도 한도 초과")
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
                break
        return {"sentiment_score": 0, "sentiment_label": "Neutral", "reason": "분석 실패"}

news_agent = NewsAgent()
