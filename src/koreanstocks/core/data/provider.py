import pandas as pd
import numpy as np
import requests
import FinanceDataReader as fdr
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeout
from datetime import datetime, timedelta, date as date_type
import logging
import time
from functools import lru_cache
from typing import List, Dict, Optional

from koreanstocks.core.config import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_xkrx_calendar():
    """XKRX 거래 캘린더 (프로세스당 1회만 생성)."""
    import exchange_calendars as xcals
    return xcals.get_calendar("XKRX", start="2020-01-01", end="2029-12-31")


@lru_cache(maxsize=10)
def _get_kr_holidays(year: int):
    """holidays.KR 연도별 캐시 (선거일·임시공휴일 보완)."""
    import holidays
    return holidays.country_holidays('KR', years=year)


# ── 시장 API 불가 시 정적 유동성 종목 풀 (최후 폴백) ─────────────────────
# FDR StockListing 및 KIND API 조회가 모두 실패할 때 사용.
# KOSPI200 + KOSDAQ 유동성 상위 기준 선별.
_STATIC_KOSPI_POOL: List[str] = [
    '005930', '000660', '373220', '207940', '005380', '000270', '005490',
    '051910', '068270', '035720', '035420', '006400', '012330', '105560',
    '055550', '086790', '316140', '017670', '030200', '066570', '096770',
    '010950', '015760', '033780', '090430', '051900', '028260', '000810',
    '009150', '034220', '047810', '138040', '006800', '079550', '086280',
    '128940', '000100', '006280', '003550', '009830', '034020', '139480',
    '282330', '097950', '000120', '071050', '000240', '004020', '030000',
    '004370', '271560', '326030', '352820', '259960', '036570', '251270',
    '035250', '078930', '023530', '011170', '010130', '024110', '180640',
    '003490', '016360', '018880', '042660', '011200', '000080',
]
_STATIC_KOSDAQ_POOL: List[str] = [
    '247540', '086520', '028300', '293490', '091990', '058470', '357780',
    '240810', '263750', '145020', '214150', '277810', '403870', '196180',
    '048260', '196170', '035900', '383220', '041510', '095660', '039030',
    '178920', '014680', '237690', '041830', '036930', '096530', '086900',
    '090460',
]
_STATIC_STOCK_POOL: List[str] = _STATIC_KOSPI_POOL + _STATIC_KOSDAQ_POOL


class StockDataProvider:
    """한국 시장 주식 데이터 수집을 담당하는 클래스"""

    def __init__(self):
        self._krx_cache = None
        self._krx_timestamp = None
        self._krx_fail_timestamp = None  # KRX API 실패 시각 (단기 재시도 차단)
        self._market_cache = None
        self._market_timestamp = None
        self._ohlcv_cache: Dict[str, tuple] = {}  # key: "code_period" → (timestamp, df)
        self._volume_cache: Optional[pd.DataFrame] = None  # 전종목 거래량+등락률 캐시
        self._volume_timestamp: Optional[datetime] = None  # 캐시 생성 시각

    @staticmethod
    def _normalize_market_df(df: pd.DataFrame, market_name: str) -> pd.DataFrame:
        """fdr 반환 df에서 기존 market 컬럼을 제거하고 표준 market 레이블을 추가한다."""
        df = df.drop(columns=[c for c in df.columns if c.lower() == 'market'])
        df['market'] = market_name
        return df

    def get_stock_list(self) -> pd.DataFrame:
        """KOSPI, KOSDAQ 상장 종목 리스트를 반환 (캐싱 적용).

        수집 우선순위:
          1차 — FDR StockListing('KOSPI'/'KOSDAQ')  [data.krx.co.kr]
          2차 — KIND API (kind.krx.co.kr)           [KRX 세션 인증 우회 가능]
        """
        now = datetime.now()
        if self._krx_cache is not None and self._krx_timestamp:
            if (now - self._krx_timestamp).total_seconds() < config.CACHE_EXPIRE_STOCKS:
                return self._krx_cache

        # FDR 실패 쿨다운 중: 캐시 우선, 없으면 KIND 직접 시도
        if self._krx_fail_timestamp:
            if (now - self._krx_fail_timestamp).total_seconds() < 300:
                if self._krx_cache is not None:
                    return self._krx_cache
                return self._fetch_kind_stock_list(now)

        # ── 1차: FDR StockListing (최대 3회 재시도) ──────────────────
        try:
            last_exc = None
            for _attempt in range(3):
                try:
                    kospi_df = self._normalize_market_df(fdr.StockListing('KOSPI'), 'KOSPI')
                    kosdaq_df = self._normalize_market_df(fdr.StockListing('KOSDAQ'), 'KOSDAQ')
                    last_exc = None
                    break
                except Exception as _e:
                    last_exc = _e
                    if _attempt < 2:
                        time.sleep(2 ** _attempt)  # 1s, 2s
            if last_exc is not None:
                raise last_exc

            df = pd.concat([kospi_df, kosdaq_df], ignore_index=True)
            column_mapping = {
                'Code': 'code', 'Name': 'name',
                'Sector': 'sector', 'Industry': 'industry', 'Dept': 'dept',
            }
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            for col in ['sector', 'industry', 'market']:
                if col not in df.columns:
                    df[col] = ''
            final_columns = ['code', 'name', 'market', 'sector', 'industry']
            df = df[[col for col in final_columns if col in df.columns]]
            for col in ['sector', 'industry']:
                if col not in df.columns:
                    df[col] = ''
            df = df.drop_duplicates(subset=['code'])
            self._krx_cache = df
            self._krx_timestamp = now
            return df
        except Exception as e:
            self._krx_fail_timestamp = now  # 5분간 FDR 재시도 차단
            logger.error(f"Error fetching stock list (FDR): {e}")

        # ── 2차: KIND API 폴백 ────────────────────────────────────────
        return self._fetch_kind_stock_list(now)

    def _fetch_kind_stock_list(self, now: datetime = None) -> pd.DataFrame:
        """KIND API(kind.krx.co.kr) — FDR 실패 시 폴백.

        KRX data.krx.co.kr이 세션 인증을 강화해 LOGOUT을 반환할 때,
        KIND는 세션 없이 Excel 형태로 전종목 목록을 제공합니다.
        컬럼: code, name, market(KOSPI/KOSDAQ), sector, industry
        """
        import requests
        from io import BytesIO

        if now is None:
            now = datetime.now()
        _empty = pd.DataFrame(columns=['code', 'name', 'market', 'sector', 'industry'])
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            df_raw = pd.read_html(BytesIO(r.content), encoding='euc-kr')[0]
            # 시장 구분: 유가(증권) → KOSPI, 코스닥 → KOSDAQ
            market_map = {'유가': 'KOSPI', '코스닥': 'KOSDAQ'}
            df_raw['market'] = df_raw['시장구분'].map(market_map)
            df_raw = df_raw.dropna(subset=['market'])  # KONEX 등 제외
            df_raw = df_raw.rename(columns={
                '종목코드': 'code', '회사명': 'name', '업종': 'sector',
            })
            df_raw['code'] = df_raw['code'].astype(str).str.zfill(6)
            if '주요제품' in df_raw.columns:
                df_raw['industry'] = df_raw['주요제품'].fillna('')
            else:
                df_raw['industry'] = ''
            df = df_raw[['code', 'name', 'market', 'sector', 'industry']].copy()
            df = df.drop_duplicates(subset=['code']).reset_index(drop=True)
            logger.info(f"KIND API 폴백 성공: {len(df)}종목 (KOSPI+KOSDAQ)")
            self._krx_cache = df
            self._krx_timestamp = now
            return df
        except Exception as e:
            logger.error(f"KIND API 폴백 실패: {e}")
            return self._krx_cache if self._krx_cache is not None else _empty

    # ── 거래량·등락률 수집 (KRX 차단 대응) ──────────────────────────────

    def _get_volume_change_df(self, valid_codes: set) -> pd.DataFrame:
        """당일 전종목 거래량+등락률 DataFrame 반환 (3단계 폴백, 10분 캐시).

        1차 — fdr.StockListing('KRX')                 : 실시간 전종목 스냅샷 (KRX 세션 필요, 현재 차단)
        2차 — 네이버 금융 시세 (BeautifulSoup 병렬)   : 전종목 code+거래량+등락률, 세션 불필요
        3차 — 종목 풀 개별 DataReader 배치 조회       : Yahoo Finance 경유, 최대 100종목 (최후 폴백)

        get_market_ranking()과 get_market_buckets()가 동일 세션에서 연속 호출될 때
        Yahoo Finance 레이트 리밋을 피하기 위해 10분간 결과를 캐싱한다.

        Returns: DataFrame with columns (code, volume, change_pct), or empty DataFrame
        """
        # ── 캐시 확인 (10분 TTL) ─────────────────────────────────────
        now = datetime.now()
        if (
            self._volume_cache is not None
            and self._volume_timestamp is not None
            and (now - self._volume_timestamp).total_seconds() < 600
        ):
            logger.debug("거래량 캐시 히트 — 재조회 생략")
            return self._volume_cache

        # ── 1차: fdr.StockListing('KRX') ─────────────────────────────
        try:
            df = fdr.StockListing('KRX')
            cols = df.columns.tolist()
            mapping: Dict[str, str] = {}
            if 'Code'    in cols: mapping['Code']    = 'code'
            if 'Volume'  in cols: mapping['Volume']  = 'volume'
            if 'Chg'     in cols: mapping['Chg']     = 'change_pct'
            elif 'Changes' in cols: mapping['Changes'] = 'change_pct'
            df = df.rename(columns=mapping)
            if 'code' in df.columns and 'volume' in df.columns:
                keep = ['code', 'volume'] + (['change_pct'] if 'change_pct' in df.columns else [])
                logger.debug("fdr.StockListing('KRX') 성공")
                result = df[keep]
                self._volume_cache = result
                self._volume_timestamp = datetime.now()
                return result
        except Exception as e:
            logger.warning(
                f"fdr.StockListing('KRX') 차단 — 네이버 금융 시세로 전환: {e}"
            )

        # ── 2차: 네이버 금융 시세 (전종목 code+거래량+등락률) ────────
        try:
            result = self._fetch_naver_sise()
            if not result.empty:
                self._volume_cache = result
                self._volume_timestamp = datetime.now()
                return result
            logger.warning("네이버 금융 시세 빈 결과 — 배치 DataReader로 전환")
        except Exception as e:
            logger.warning(f"네이버 금융 시세 실패 — 배치 DataReader로 전환: {e}")

        # ── 3차: 개별 DataReader 배치 조회 (최후 폴백) ──────────────
        candidate_pool = self._get_bulk_candidate_pool(valid_codes)
        logger.info(f"거래량 배치 조회: {len(candidate_pool)}종목 (최후 폴백)")
        result = self._fetch_bulk_volume_change(candidate_pool)
        if not result.empty:
            self._volume_cache = result
            self._volume_timestamp = datetime.now()
        return result

    def _fetch_naver_sise(self, max_workers: int = 20, timeout: int = 30) -> pd.DataFrame:
        """네이버 금융 시세 — 전종목 거래량·등락률 수집 (2차 폴백).

        finance.naver.com/sise/sise_market_sum.naver 를 BeautifulSoup으로 병렬 파싱.
        KOSPI(sosok=0)·KOSDAQ(sosok=1) 전 페이지를 동적으로 탐색하여 수집.
        총 페이지 수는 .pgRR 링크에서 자동 감지하므로 상장 종목 수 변동에 대응.
        ETF·ETN 등이 포함되나 _get_volume_change_df() 에서 valid_codes 필터로 제거됨.

        Returns: DataFrame(code, volume, change_pct) or empty DataFrame
        """
        import requests
        import re
        import concurrent.futures as _cf
        from bs4 import BeautifulSoup

        _headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        def _last_page(sosok: int) -> int:
            r = requests.get(
                'https://finance.naver.com/sise/sise_market_sum.naver',
                params={'sosok': str(sosok), 'page': '1'},
                headers=_headers, timeout=10,
            )
            soup = BeautifulSoup(r.text, 'html.parser')
            pager = soup.select('.pgRR a')
            if pager:
                m = re.search(r'page=(\d+)', pager[-1]['href'])
                if m:
                    return int(m.group(1))
            return 50  # 폴백 상한

        def _col_indices(soup) -> tuple[int, int]:
            """테이블 헤더에서 '등락률'·'거래량' 컬럼 인덱스를 탐지. 실패 시 기본값 반환."""
            tbl = soup.select_one('table.type_2')
            if not tbl:
                return 4, 9
            ths = tbl.select('thead th') or tbl.select('tr:first-child th')
            chg_idx, vol_idx = 4, 9  # 현행 Naver 구조 기본값
            for i, th in enumerate(ths):
                txt = th.get_text(strip=True)
                if '등락률' in txt:
                    chg_idx = i
                elif '거래량' in txt:
                    vol_idx = i
            return chg_idx, vol_idx

        def _fetch_page(sosok: int, page: int) -> list:
            r = requests.get(
                'https://finance.naver.com/sise/sise_market_sum.naver',
                params={'sosok': str(sosok), 'page': str(page)},
                headers=_headers, timeout=10,
            )
            soup = BeautifulSoup(r.text, 'html.parser')
            chg_idx, vol_idx = _col_indices(soup)
            rows = []
            for tr in soup.select('table.type_2 tbody tr'):
                a = tr.select_one('td a[href*="code="]')
                if not a:
                    continue
                code = a['href'].split('code=')[-1].strip()
                tds = tr.select('td')
                try:
                    chg = tds[chg_idx].get_text(strip=True).replace('%', '').replace(',', '').replace('+', '')
                    vol = tds[vol_idx].get_text(strip=True).replace(',', '')
                    rows.append({
                        'code':       code,
                        'change_pct': float(chg) if chg else 0.0,
                        'volume':     float(vol) if vol else 0.0,
                    })
                except (IndexError, ValueError):
                    pass
            return rows

        try:
            kospi_pages  = _last_page(0)
            kosdaq_pages = _last_page(1)
            tasks = (
                [(0, p) for p in range(1, kospi_pages  + 1)] +
                [(1, p) for p in range(1, kosdaq_pages + 1)]
            )
            logger.info(
                f"네이버 금융 시세 수집 시작: KOSPI {kospi_pages}p + KOSDAQ {kosdaq_pages}p = {len(tasks)}p"
            )
            all_rows: list = []
            pool = ThreadPoolExecutor(max_workers=max_workers)
            try:
                futures_map = {pool.submit(_fetch_page, s, p): (s, p) for s, p in tasks}
                done, not_done = _cf.wait(futures_map.keys(), timeout=timeout)
                for future in done:
                    try:
                        all_rows.extend(future.result())
                    except Exception:
                        pass
                if not_done:
                    logger.warning(f"네이버 금융 시세: {len(not_done)}페이지 타임아웃 — {len(all_rows)}행 수집")
                    for f in not_done:
                        f.cancel()
            finally:
                pool.shutdown(wait=False)

            if not all_rows:
                return pd.DataFrame()
            df = pd.DataFrame(all_rows).drop_duplicates(subset='code').reset_index(drop=True)
            logger.info(f"네이버 금융 시세 수집 완료: {len(df)}종목 (ETF 포함, valid_codes 필터 전)")
            return df[['code', 'volume', 'change_pct']]
        except Exception as e:
            logger.warning(f"네이버 금융 시세 수집 실패: {e}")
            return pd.DataFrame()

    def _get_bulk_candidate_pool(self, valid_codes: set, max_size: int = 100) -> List[str]:
        """배치 OHLCV 조회용 종목 풀 구성.

        정적 풀(유동성 상위 97종목)을 기반으로, KIND API 종목 목록에서 보충하여
        최대 max_size개의 대표 풀을 반환한다.
        Yahoo Finance 레이트 리밋 감안: 100종목 × 10 workers ≈ 10~20초 내 완료.
        """
        pool: List[str] = []
        seen: set = set()
        for c in _STATIC_STOCK_POOL:
            if c in valid_codes and c not in seen:
                pool.append(c)
                seen.add(c)
        if len(pool) < max_size:
            try:
                full_list = self.get_stock_list()
                for c in full_list['code'].tolist():
                    if c not in seen and c in valid_codes:
                        pool.append(c)
                        seen.add(c)
                        if len(pool) >= max_size:
                            break
            except Exception:
                pass
        return pool[:max_size]

    def _fetch_bulk_volume_change(
        self,
        codes: List[str],
        max_workers: int = 10,
        timeout: int = 30,
    ) -> pd.DataFrame:
        """종목별 최신 거래량·등락률을 FDR DataReader로 병렬 수집.

        KRX 세션 정책으로 fdr.StockListing('KRX')이 차단될 때 호출된다.
        최근 14일 OHLCV를 조회하여 최신일 거래량 + 전일 대비 등락률을 계산한다.

        구현 주의:
          ThreadPoolExecutor 컨텍스트 매니저는 __exit__ 시 shutdown(wait=True)를 호출하여
          타임아웃 후에도 모든 스레드가 끝날 때까지 블록한다. 이를 방지하기 위해
          concurrent.futures.wait() + shutdown(wait=False)를 직접 사용한다.

        Returns: DataFrame (code, volume, change_pct)
        """
        import concurrent.futures as _cf

        end_str   = datetime.now().strftime('%Y-%m-%d')
        start_str = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')

        def _one(code: str):
            try:
                df = fdr.DataReader(code, start_str, end_str)
                if df is None or df.empty or len(df) < 2:
                    return None
                latest   = df.iloc[-1]
                prev     = df.iloc[-2]
                vol      = float(latest.get('Volume', 0) or 0)
                close    = float(latest.get('Close',  0) or 0)
                prev_cls = float(prev.get('Close',    0) or 0)
                chg      = (close - prev_cls) / prev_cls * 100.0 if prev_cls else 0.0
                return {'code': code, 'volume': vol, 'change_pct': round(chg, 2)}
            except Exception:
                return None

        results = []
        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures_map = {pool.submit(_one, c): c for c in codes}
            done, not_done = _cf.wait(futures_map.keys(), timeout=timeout)
            for future in done:
                try:
                    r = future.result()
                    if r:
                        results.append(r)
                except Exception:
                    pass
            if not_done:
                logger.warning(
                    f"_fetch_bulk_volume_change: {len(not_done)}건 타임아웃 — {len(results)}건 수집"
                )
                for f in not_done:
                    f.cancel()
        finally:
            pool.shutdown(wait=False)  # 잔여 스레드 대기 없이 즉시 반환

        if results:
            logger.info(f"거래량 배치 조회 완료: {len(results)}/{len(codes)}종목")
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_ohlcv(self, code: str, start: str = None, end: str = None, period: str = '1y') -> pd.DataFrame:
        """특정 종목의 OHLCV(시가, 고가, 저가, 종가, 거래량) 데이터를 반환 (5분 캐시 적용)"""
        cache_key = f"{code}_{period}_{start or datetime.now().strftime('%Y-%m-%d')}_{end or ''}"
        now = datetime.now()
        if cache_key in self._ohlcv_cache:
            cached_ts, cached_df = self._ohlcv_cache[cache_key]
            if (now - cached_ts).total_seconds() < config.CACHE_EXPIRE_MARKET:
                return cached_df

        try:
            if not end:
                end = datetime.now().strftime('%Y-%m-%d')

            if not start:
                # period를 바탕으로 start_date 계산
                end_dt = datetime.now()
                if period == '1y':
                    start_dt = end_dt - timedelta(days=365)
                elif period == '2y':
                    start_dt = end_dt - timedelta(days=730)
                elif period == '3m':
                    start_dt = end_dt - timedelta(days=90)
                elif period == '6m':
                    start_dt = end_dt - timedelta(days=180)
                elif period == '1m':
                    start_dt = end_dt - timedelta(days=30)
                else:
                    start_dt = end_dt - timedelta(days=365)
                start = start_dt.strftime('%Y-%m-%d')

            try:
                df = fdr.DataReader(code, start, end)
            except ValueError as _ve:
                # FDR/KIND 내부 월 경계 날짜 계산 버그 (e.g. Feb 31)
                # start를 +1일 조정 후 재시도
                if 'day is out of range' in str(_ve):
                    adjusted = (datetime.strptime(start, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                    logger.debug(f"[{code}] 날짜 경계 버그 재시도: {start} → {adjusted}")
                    df = fdr.DataReader(code, adjusted, end)
                else:
                    raise
            if df.empty:
                return df

            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Change': 'change'
            })
            df.index.name = 'date'
            self._ohlcv_cache[cache_key] = (now, df)
            return df
        except Exception as e:
            # FDR/KIND의 월 경계 날짜 버그는 WARNING으로 처리 (분석은 중립값으로 계속)
            if 'day is out of range' in str(e):
                logger.warning(f"[{code}] OHLCV 날짜 범위 조회 실패 (FDR 월 경계 버그): {e}")
            elif str(e).strip() in ('LOGOUT', 'LOGIN'):
                logger.warning(f"[{code}] OHLCV KRX 세션 만료 (일시적, 자동 갱신됨): {e}")
            else:
                logger.error(f"Error fetching OHLCV for {code}: {e}")
            return pd.DataFrame()

    def get_market_indices(self) -> Dict[str, float]:
        """주요 시장 지수(KOSPI, KOSDAQ, 환율) 정보 반환"""
        now = datetime.now()
        if self._market_cache and self._market_timestamp:
            if (now - self._market_timestamp).total_seconds() < config.CACHE_EXPIRE_MARKET:
                return self._market_cache

        indices = {}
        start_str = (now - timedelta(days=7)).strftime('%Y-%m-%d')
        # '^' 접두사 심볼: FDR이 Yahoo Finance 소스를 사용 (KRX 세션 불필요)
        # 'KS11'/'KQ11'은 FDR KRX 직접 접근으로 LOGOUT 에러 발생 가능
        for symbol, name in [('^KS11', 'KOSPI'), ('^KQ11', 'KOSDAQ'), ('USD/KRW', 'USD_KRW')]:
            try:
                df = fdr.DataReader(symbol, start_str)
                if not df.empty:
                    close_val = df.iloc[-1]['Close']
                    if pd.isna(close_val):
                        continue
                    indices[name] = float(close_val)
                    # Change 컬럼이 없는 소스(Yahoo Finance)도 있으므로 전일비 직접 계산
                    if len(df) >= 2:
                        prev = float(df.iloc[-2]['Close'])
                        curr = float(df.iloc[-1]['Close'])
                        indices[f"{name}_change"] = (curr - prev) / prev if (prev and not np.isnan(prev) and not np.isnan(curr)) else 0.0
                    else:
                        raw_chg = df.iloc[-1].get('Change', 0.0)
                        indices[f"{name}_change"] = 0.0 if pd.isna(raw_chg) else float(raw_chg)
            except Exception as e:
                logger.error(f"Error fetching market indices [{symbol}]: {e}")

        if indices:
            self._market_cache = indices
            self._market_timestamp = now
        return indices

    def get_market_ranking(self, limit: int = 50, market: str = 'ALL') -> List[str]:
        """거래량 및 등락률 상위 종목 코드를 취합하여 반환 (market: 'ALL'|'KOSPI'|'KOSDAQ')"""
        try:
            full_stock_list = self.get_stock_list()
            if market != 'ALL':
                valid_codes = set(full_stock_list[full_stock_list['market'] == market]['code'].tolist())
            else:
                valid_codes = set(full_stock_list['code'].tolist())

            # 거래량+등락률 데이터 수집 (KRX 차단 시 배치 DataReader 폴백)
            df_ranking = self._get_volume_change_df(valid_codes)

            if df_ranking is not None and not df_ranking.empty and 'volume' in df_ranking.columns:
                df_ranking['volume'] = pd.to_numeric(df_ranking['volume'], errors='coerce').fillna(0)
                df_ranking = df_ranking[df_ranking['volume'] > 0]

                top_volume  = df_ranking.sort_values(by='volume', ascending=False).head(limit)
                top_gainers = pd.DataFrame()
                if 'change_pct' in df_ranking.columns:
                    df_ranking['change_pct'] = pd.to_numeric(
                        df_ranking['change_pct'], errors='coerce'
                    ).fillna(0)
                    top_gainers = df_ranking.sort_values(by='change_pct', ascending=False).head(limit)

                vol_codes  = top_volume['code'].tolist()
                gain_codes = top_gainers['code'].tolist() if not top_gainers.empty else []
                seen       = set(vol_codes)
                ordered_codes = vol_codes + [c for c in gain_codes if c not in seen]
                result = [c for c in ordered_codes if c in valid_codes]

                logger.info(f"Market ranking fetched: {len(result)} candidates (volume+gainers, ordered).")
                return result
        except Exception as e:
            logger.error(f"Error fetching market ranking: {e}")

        # ── 최종 폴백: DB 이력 + 정적 풀 ────────────────────────────────
        try:
            from koreanstocks.core.data.database import db_manager
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                if market != 'ALL':
                    cursor.execute(
                        "SELECT DISTINCT code FROM recommendations WHERE json_extract(detail_json, '$.market') = ? ORDER BY session_date DESC LIMIT ?",
                        (market, limit),
                    )
                else:
                    cursor.execute(
                        "SELECT DISTINCT code FROM recommendations ORDER BY session_date DESC LIMIT ?",
                        (limit,),
                    )
                db_codes = [row[0] for row in cursor.fetchall()]
        except Exception as e3:
            logger.error(f"Market ranking DB 폴백 실패: {e3}")
            db_codes = []
        # 최종 폴백: 정적 유동성 종목 풀로 보충 (FDR/KIND 전종목 API 불가 상황)
        if market == 'KOSPI':
            static_pool = _STATIC_KOSPI_POOL
        elif market == 'KOSDAQ':
            static_pool = _STATIC_KOSDAQ_POOL
        else:
            static_pool = _STATIC_STOCK_POOL
        # DB 이력 코드 우선, 정적 풀로 나머지 보충
        db_set = set(db_codes)
        combined = db_codes + [c for c in static_pool if c not in db_set]
        logger.warning(
            f"Market ranking: 정적 종목 풀 폴백 {len(combined[:limit])}개 "
            f"(DB {len(db_codes)}건 + 정적 풀 보충, FDR/KIND 전종목 API 불가)"
        )
        return combined[:limit]

    def get_value_candidates(
        self,
        limit: int = 200,
        market: str = "ALL",
        per_max: float = 30.0,
        roe_min: float = 5.0,
    ) -> List[str]:
        """시가총액 상위 종목에서 PER/ROE 사전 필터를 거쳐 가치주 후보 반환.

        네이버 금융 시가총액 순위(sise_market_sum.naver)에서 PER·ROE 컬럼을 파싱해
        1차 필터링 후 코드 목록을 반환한다. DART 호출 전 사전 선별로 효율 개선.

        Args:
            limit: 반환 종목 수 (필터 통과 기준, 시가총액 순)
            market: ALL | KOSPI | KOSDAQ
            per_max: PER 상한 (0 이하 = 적자 제외)
            roe_min: ROE 하한 (%)
        """
        import re
        import concurrent.futures as _cf
        from bs4 import BeautifulSoup

        _headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        sosok_list = [0, 1] if market == "ALL" else ([0] if market == "KOSPI" else [1])

        def _last_page(sosok: int) -> int:
            try:
                r = requests.get(
                    "https://finance.naver.com/sise/sise_market_sum.naver",
                    params={"sosok": str(sosok), "page": "1"},
                    headers=_headers, timeout=10,
                )
                soup = BeautifulSoup(r.text, "html.parser")
                pager = soup.select(".pgRR a")
                if pager:
                    m = re.search(r"page=(\d+)", pager[-1]["href"])
                    if m:
                        return int(m.group(1))
            except Exception:
                pass
            return 50

        def _fetch_page(sosok: int, page: int):
            try:
                r = requests.get(
                    "https://finance.naver.com/sise/sise_market_sum.naver",
                    params={"sosok": str(sosok), "page": str(page)},
                    headers=_headers, timeout=10,
                )
                soup = BeautifulSoup(r.text, "html.parser")
                table = soup.select_one("table.type_2")
                if not table:
                    return []
                # 헤더에서 PER·ROE 컬럼 인덱스 탐지
                ths = table.select("thead th")
                per_idx, roe_idx = None, None
                for i, th in enumerate(ths):
                    txt = th.get_text(strip=True)
                    if txt == "PER":
                        per_idx = i
                    elif txt == "ROE":
                        roe_idx = i
                rows = []
                for tr in table.select("tbody tr"):
                    a = tr.select_one("td a[href*='code=']")
                    if not a:
                        continue
                    code = a["href"].split("code=")[-1].strip()
                    tds = tr.select("td")
                    per, roe = None, None
                    try:
                        if per_idx is not None and per_idx < len(tds):
                            per = float(tds[per_idx].get_text(strip=True).replace(",", "") or "nan")
                        if roe_idx is not None and roe_idx < len(tds):
                            roe = float(tds[roe_idx].get_text(strip=True).replace(",", "") or "nan")
                    except ValueError:
                        pass
                    rows.append({"code": code, "per": per, "roe": roe})
                return rows
            except Exception:
                return []

        try:
            full_stock_list = self.get_stock_list()
            if market != "ALL":
                valid_codes = set(full_stock_list[full_stock_list["market"] == market]["code"].tolist())
            else:
                valid_codes = set(full_stock_list["code"].tolist())

            # 페이지 수 파악 후 병렬 수집
            tasks = []
            for sosok in sosok_list:
                n = _last_page(sosok)
                tasks += [(sosok, p) for p in range(1, n + 1)]

            all_rows: List[dict] = []
            with _cf.ThreadPoolExecutor(max_workers=20) as ex:
                futures = {ex.submit(_fetch_page, s, p): (s, p) for s, p in tasks}
                for f in _cf.as_completed(futures):
                    all_rows.extend(f.result())

            # 시가총액 순서 유지 (페이지 순서) + PER/ROE 필터
            seen: set = set()
            result: List[str] = []
            for row in all_rows:
                code = row["code"]
                if code in seen or code not in valid_codes:
                    continue
                seen.add(code)
                per = row["per"]
                roe = row["roe"]
                import math
                if per is not None and not math.isnan(per) and (per <= 0 or per > per_max):
                    continue
                if roe is not None and not math.isnan(roe) and roe < roe_min:
                    continue
                result.append(code)
                if len(result) >= limit:
                    break

            logger.info(f"[VALUE 후보] 사전 필터 통과: {len(result)}개 (PER≤{per_max}, ROE≥{roe_min}%)")
            return result

        except Exception as e:
            logger.error(f"get_value_candidates 실패, get_market_ranking 폴백: {e}")
            return self.get_market_ranking(limit=limit, market=market)

    def get_market_buckets(self, market: str = 'ALL') -> Dict[str, List[str]]:
        """거래량·모멘텀·반등 3개 버킷으로 후보군 분류.

        버킷 A (volume)   — 유동성 안정주: 거래량 상위, 급등락 제외
        버킷 B (momentum) — 상승 모멘텀:  등락률 +2%~+15% 구간, 거래량 정렬
        버킷 C (rebound)  — 반등 후보:    거래량 상위 절반 중 -2% 이하 하락주

        공통 사전 필터:
          · 유동성 하한: volume >= 50,000주 (소형·관리종목 제외)
          · 상장 종목 목록 교차 검증
          · 급등락 ±15% 초과 제외 (서킷브레이커·이슈 급등락 제외)

        거래량 데이터 수집:
          1차 — fdr.StockListing('KRX')          : 실시간 전종목 스냅샷
          2차 — 개별 DataReader 배치 조회 (최대 300종목): KRX 차단 시 Yahoo Finance 경유

        Returns:
            {'volume': [...], 'momentum': [...], 'rebound': [...]}
        """
        LIQUIDITY_MIN    = 50_000   # 최소 거래량 (주)
        SURGE_LIMIT_PCT  = 15.0     # 급등락 제외 임계값 (%)
        MOMENTUM_MIN_PCT =  2.0     # 모멘텀 버킷 하한 (%)
        REBOUND_MAX_PCT  = -2.0     # 반등 버킷 상한 (%)
        _empty: Dict[str, List[str]] = {'volume': [], 'momentum': [], 'rebound': []}

        try:
            full_stock_list = self.get_stock_list()
            if market != 'ALL':
                valid_codes = set(full_stock_list[full_stock_list['market'] == market]['code'].tolist())
            else:
                valid_codes = set(full_stock_list['code'].tolist())

            # 거래량+등락률 데이터 수집 (KRX 차단 시 배치 DataReader 폴백)
            df = self._get_volume_change_df(valid_codes)

            if df is None or df.empty or 'code' not in df.columns or 'volume' not in df.columns:
                logger.warning("get_market_buckets: 거래량 데이터 없음 → 순위 기반 3버킷 근사 분류")
                fallback = self.get_market_ranking(limit=200, market=market)
                n = len(fallback)
                cut_v = max(1, n * 2 // 4)          # 상위 ~50% → volume
                cut_m = max(cut_v + 1, n * 3 // 4)  # 다음 ~25% → momentum (위치 기반 근사)
                # 나머지 ~25% → rebound (위치 기반 근사)
                return {
                    'volume':   fallback[:cut_v],
                    'momentum': fallback[cut_v:cut_m],
                    'rebound':  fallback[cut_m:],
                }

            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            df['change_pct'] = (
                pd.to_numeric(df['change_pct'], errors='coerce').fillna(0)
                if 'change_pct' in df.columns else 0.0
            )

            # 공통 사전 필터: 유동성 + 상장 종목 + 급등락 제외
            df = df[
                (df['volume'] >= LIQUIDITY_MIN) &
                (df['code'].isin(valid_codes)) &
                (df['change_pct'].abs() < SURGE_LIMIT_PCT)
            ]

            # 버킷 A — 거래량 상위 (중립 구간: -2%~+2% 포함, 급등락 이미 제외)
            bucket_a = df.sort_values('volume', ascending=False)['code'].tolist()

            # 버킷 B — 상승 모멘텀: +2%~+15%, 거래량 기준 정렬
            df_b = df[df['change_pct'] >= MOMENTUM_MIN_PCT]
            bucket_b = df_b.sort_values('volume', ascending=False)['code'].tolist()

            # 버킷 C — 반등 후보: 거래량 상위 절반 중 -2% 이하 하락주, 하락폭순 정렬
            vol_median = df['volume'].median()
            df_c = df[(df['change_pct'] <= REBOUND_MAX_PCT) & (df['volume'] >= vol_median)]
            bucket_c = df_c.sort_values('change_pct', ascending=True)['code'].tolist()

            logger.info(
                f"Market buckets ({market}): "
                f"volume={len(bucket_a)}, momentum={len(bucket_b)}, rebound={len(bucket_c)}"
            )
            return {'volume': bucket_a, 'momentum': bucket_b, 'rebound': bucket_c}

        except Exception as e:
            logger.error(f"get_market_buckets 실패: {e} → 순위 기반 3버킷 근사 분류")
            fallback = self.get_market_ranking(limit=200, market=market)
            n = len(fallback)
            cut_v = max(1, n * 2 // 4)
            cut_m = max(cut_v + 1, n * 3 // 4)
            return {
                'volume':   fallback[:cut_v],
                'momentum': fallback[cut_v:cut_m],
                'rebound':  fallback[cut_m:],
            }

    def is_trading_day(self, d: Optional[date_type] = None) -> bool:
        """한국 증시 거래일인지 여부 반환.

        d: 확인할 날짜 (None이면 오늘). 미래·과거 날짜도 지원.

        판별 순서:
          1단계 (즉시)  — 토·일 → False
          2단계 (오프라인) — exchange_calendars XKRX: 공휴일·대체공휴일·KRX 전용 휴장
          3단계 (오프라인) — holidays.KR 보완: exchange_calendars가 놓친 선거일 등
          4단계 (온라인)  — FDR 실측 확인 (d=오늘이고 네트워크 가용 시만)
        """
        target = d or datetime.now().date()

        # 1단계: 주말 즉시 판별
        if target.weekday() >= 5:
            return False

        # 2단계: exchange_calendars XKRX (오프라인)
        xkrx_is_trading: Optional[bool] = None
        try:
            cal = _get_xkrx_calendar()
            xkrx_is_trading = cal.is_session(pd.Timestamp(target))
            if not xkrx_is_trading:
                return False
        except Exception as e:
            logger.debug(f"exchange_calendars 체크 실패: {e}")

        # 3단계: holidays.KR 보완 (선거일·임시공휴일 등 exchange_calendars 누락분)
        try:
            kr_h = _get_kr_holidays(target.year)
            if target in kr_h:
                return False
        except Exception as e:
            logger.debug(f"holidays.KR 체크 실패: {e}")

        # 4단계: FDR 온라인 실측 — d=오늘이고 네트워크 가용 시만
        # 삼성전자 당일 OHLCV 조회 후 인덱스 날짜가 오늘인지 검증
        today = datetime.now().date()
        if target == today:
            try:
                today_iso = today.isoformat()
                df = fdr.DataReader('005930', today_iso, today_iso)
                if not df.empty:
                    last_str = df.index[-1].strftime('%Y-%m-%d')
                    if last_str == today_iso:
                        return True  # 오늘 거래 데이터 확인됨
                    else:
                        logger.debug(f"FDR이 이전 거래일({last_str}) 데이터 반환 → 오프라인 판단 사용")
                else:
                    logger.debug("FDR 데이터 없음(장 전 또는 미갱신) → 오프라인 판단 사용")
            except Exception as e:
                logger.warning(f"FDR 거래일 확인 실패: {e} → 오프라인 판단 사용")

        # 오프라인 체크 통과 (FDR 미확인 or 미래/과거 날짜) → 거래일로 판단
        return True

    def get_stocks_by_theme(self, keywords: List[str], market: str = 'ALL') -> pd.DataFrame:
        """업종/산업 분야에서 키워드를 검색하여 관련 종목 리스트 반환"""
        try:
            df = self.get_stock_list() # Sector/Industry 정보가 포함된 리스트
            if df.empty: return df

            # 시장 필터링
            if market != 'ALL':
                df = df[df['market'] == market]

            theme_mask = pd.Series([False] * len(df), index=df.index)

            # 키워드 검색 로직 (더 유연하게)
            search_cols = ['sector', 'industry', 'name'] # 종목명(name)도 검색 대상에 추가

            for keyword in keywords:
                for col in search_cols:
                    if col in df.columns:
                        # 키워드를 포함하는 경우 (대소문자 무시)
                        theme_mask |= df[col].astype(str).str.contains(keyword, na=False, case=False)

            theme_stocks = df[theme_mask]
            logger.info(f"Found {len(theme_stocks)} stocks for keywords: {keywords}")
            return theme_stocks
        except Exception as e:
            logger.error(f"Error filtering theme stocks: {e}")
            return pd.DataFrame()

# Singleton instance
data_provider = StockDataProvider()
