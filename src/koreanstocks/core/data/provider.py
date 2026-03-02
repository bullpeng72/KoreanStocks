import pandas as pd
import numpy as np
import FinanceDataReader as fdr
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

class StockDataProvider:
    """한국 시장 주식 데이터 수집을 담당하는 클래스"""
    
    def __init__(self):
        self._krx_cache = None
        self._krx_timestamp = None
        self._krx_fail_timestamp = None  # KRX API 실패 시각 (단기 재시도 차단)
        self._market_cache = None
        self._market_timestamp = None
        self._ohlcv_cache: Dict[str, tuple] = {}  # key: "code_period" → (timestamp, df)

    @staticmethod
    def _normalize_market_df(df: pd.DataFrame, market_name: str) -> pd.DataFrame:
        """fdr 반환 df에서 기존 market 컬럼을 제거하고 표준 market 레이블을 추가한다."""
        df = df.drop(columns=[c for c in df.columns if c.lower() == 'market'])
        df['market'] = market_name
        return df

    def get_stock_list(self) -> pd.DataFrame:
        """KOSPI, KOSDAQ 상장 종목 리스트를 반환 (캐싱 적용)"""
        now = datetime.now()
        if self._krx_cache is not None and self._krx_timestamp:
            if (now - self._krx_timestamp).total_seconds() < config.CACHE_EXPIRE_STOCKS:
                return self._krx_cache

        # 최근 5분 이내 실패한 경우 KRX API 재시도 차단 (반복 오류 방지)
        if self._krx_fail_timestamp:
            if (now - self._krx_fail_timestamp).total_seconds() < 300:
                return self._krx_cache if self._krx_cache is not None else pd.DataFrame(columns=['code', 'name', 'market', 'sector', 'industry'])

        try:
            # KRX 전체 종목 리스트를 가져오는 것이 더 안정적입니다.
            # 하지만, KRX Listing에는 Sector/Industry 정보가 불충분할 수 있으므로 KOSPI/KOSDAQ를 따로 가져와 병합

            # 일시적 네트워크/API 오류에 대한 재시도 (최대 3회, 지수 백오프)
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

            # 두 데이터프레임을 합치기
            df = pd.concat([kospi_df, kosdaq_df], ignore_index=True)

            # 컬럼명 표준화 (market은 이미 처리했으므로 제외)
            column_mapping = {
                'Code': 'code',
                'Name': 'name',
                'Sector': 'sector',
                'Industry': 'industry',
                'Dept': 'dept',
            }

            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            # 필수 컬럼이 없을 경우 빈 값으로 생성
            for col in ['sector', 'industry', 'market']:
                if col not in df.columns:
                    df[col] = ''
            
            # 'market' 컬럼이 이미 추가되었으므로 중복 처리 방지
            # KOSPI/KOSDAQ 필터링은 이미 데이터 병합 시에 이루어짐
            
            # 필요한 컬럼만 최종 선택
            final_columns = ['code', 'name', 'market', 'sector', 'industry']
            df = df[[col for col in final_columns if col in df.columns]] # 존재하는 컬럼만 선택
            
            # 누락된 컬럼 다시 채워넣기 (예: KRX에는 Sector/Industry가 없는 경우가 있으므로)
            for col in ['sector', 'industry']:
                if col not in df.columns:
                    df[col] = ''
            
            # 중복 종목 제거 (Code 기준)
            df = df.drop_duplicates(subset=['code'])

            self._krx_cache = df
            self._krx_timestamp = now
            return df
        except Exception as e:
            self._krx_fail_timestamp = now  # 5분간 재시도 차단
            logger.error(f"Error fetching stock list: {e}")
            return self._krx_cache if self._krx_cache is not None else pd.DataFrame(columns=['code', 'name', 'market', 'sector', 'industry'])

    def get_ohlcv(self, code: str, start: str = None, end: str = None, period: str = '1y') -> pd.DataFrame:
        """특정 종목의 OHLCV(시가, 고가, 저가, 종가, 거래량) 데이터를 반환 (5분 캐시 적용)"""
        cache_key = f"{code}_{period}_{start}_{end}"
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
                # FDR/pykrx 내부 월 경계 날짜 계산 버그 (e.g. Feb 31)
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
            # FDR/pykrx의 월 경계 날짜 버그는 WARNING으로 처리 (분석은 중립값으로 계속)
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
        start_str = (now - timedelta(days=5)).strftime('%Y-%m-%d')
        for symbol, name in [('KS11', 'KOSPI'), ('KQ11', 'KOSDAQ'), ('USD/KRW', 'USD_KRW')]:
            try:
                df = fdr.DataReader(symbol, start_str)
                if not df.empty:
                    indices[name] = float(df.iloc[-1]['Close'])
                    indices[f"{name}_change"] = float(df.iloc[-1].get('Change', 0.0))
            except Exception as e:
                # LOGOUT: FDR KRX 세션 만료 (일시적, 자동 갱신됨) — WARNING으로 처리
                log = logger.warning if str(e).strip() in ('LOGOUT', 'LOGIN') else logger.error
                log(f"Error fetching market indices [{symbol}]: {e}")

        if indices:
            self._market_cache = indices
            self._market_timestamp = now
        return indices

    def get_market_ranking(self, limit: int = 50, market: str = 'ALL') -> List[str]:
        """거래량 및 등락률 상위 종목 코드를 취합하여 반환 (market: 'ALL'|'KOSPI'|'KOSDAQ')"""
        try:
            full_stock_list = self.get_stock_list()
            
            # 현재 시장 랭킹 정보 (거래량, 등락률)
            df_ranking = fdr.StockListing('KRX') # 최신 거래량/등락률 정보는 KRX 리스팅에서
            
            # 컬럼명 존재 여부 확인 및 표준화
            cols = df_ranking.columns.tolist()
            mapping = {}
            if 'Code' in cols: mapping['Code'] = 'code'
            if 'Volume' in cols: mapping['Volume'] = 'volume'
            if 'Chg' in cols: mapping['Chg'] = 'change_pct'
            elif 'Changes' in cols: mapping['Changes'] = 'change_pct' # 버전 차이 대응
            
            df_ranking = df_ranking.rename(columns=mapping)
            
            # 필수 데이터 확인
            if 'code' not in df_ranking.columns or 'volume' not in df_ranking.columns:
                logger.warning("Required columns (Code, Volume) missing in StockListing('KRX').")
                return df_ranking['Code'].head(limit).tolist() if 'Code' in df_ranking.columns else []

            # 거래량이 있는 종목만 선택
            df_ranking['volume'] = pd.to_numeric(df_ranking['volume'], errors='coerce').fillna(0)
            df_ranking = df_ranking[df_ranking['volume'] > 0]
            
            # 1. 거래량 상위
            top_volume = df_ranking.sort_values(by='volume', ascending=False).head(limit)
            
            # 2. 상승률 상위 (컬럼이 있을 때만)
            top_gainers = pd.DataFrame()
            if 'change_pct' in df_ranking.columns:
                df_ranking['change_pct'] = pd.to_numeric(df_ranking['change_pct'], errors='coerce').fillna(0)
                top_gainers = df_ranking.sort_values(by='change_pct', ascending=False).head(limit)
            
            # 순위 순서 유지하며 합치기: 거래량 상위 → 그 외 등락률 상위
            # (set() 사용 시 순서가 소실되어 candidate_codes[:30]이 코드번호 순으로 잘리는 버그 수정)
            vol_codes  = top_volume['code'].tolist()
            gain_codes = top_gainers['code'].tolist() if not top_gainers.empty else []
            seen       = set(vol_codes)
            ordered_codes = vol_codes + [c for c in gain_codes if c not in seen]

            # 상장 종목 목록에 존재하는 코드만 유지 (비상장·관리종목 제외)
            # market 파라미터가 지정된 경우 해당 시장 종목만 허용
            if market != 'ALL':
                valid_codes = set(full_stock_list[full_stock_list['market'] == market]['code'].tolist())
            else:
                valid_codes = set(full_stock_list['code'].tolist())
            result = [c for c in ordered_codes if c in valid_codes]

            logger.info(f"Market ranking fetched: {len(result)} candidates (volume+gainers, ordered).")
            return result
        except Exception as e:
            logger.error(f"Error fetching market ranking: {e}")
            # KRX FDR 실패 시 PyKrx로 종목 코드 목록 폴백 (거래량 정렬 없음)
            # 비거래일엔 당일 목록이 비어있으므로 최근 거래일까지 최대 10일 역탐색
            try:
                from pykrx import stock as _pykrx
                codes = []
                ref_date = ''
                for i in range(10):
                    d = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
                    if market == 'KOSDAQ':
                        c = _pykrx.get_market_ticker_list(d, market='KOSDAQ')
                    elif market == 'KOSPI':
                        c = _pykrx.get_market_ticker_list(d, market='KOSPI')
                    else:
                        c = (_pykrx.get_market_ticker_list(d, market='KOSPI') +
                             _pykrx.get_market_ticker_list(d, market='KOSDAQ'))
                    if c:
                        codes = c
                        ref_date = d
                        break
                logger.warning(f"Market ranking: PyKrx 폴백 {len(codes)}개 (기준일: {ref_date}, 거래량 정렬 없음)")
                if codes:
                    return codes[:limit]
            except Exception as e2:
                logger.error(f"Market ranking PyKrx 폴백 실패: {e2}")
            # DB 폴백: 최근 추천·분석 이력에서 종목 코드 재사용 (비거래일 강제 실행용)
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
                if db_codes:
                    logger.warning(f"Market ranking: DB 폴백 {len(db_codes)}개 (최근 추천 종목, 비거래일 모드)")
                    return db_codes[:limit]
            except Exception as e3:
                logger.error(f"Market ranking DB 폴백 실패: {e3}")
            return []

    def is_trading_day(self, d: Optional[date_type] = None) -> bool:
        """한국 증시 거래일인지 여부 반환.

        d: 확인할 날짜 (None이면 오늘). 미래·과거 날짜도 지원.

        판별 순서:
          1단계 (즉시)  — 토·일 → False
          2단계 (오프라인) — exchange_calendars XKRX: 공휴일·대체공휴일·KRX 전용 휴장
          3단계 (오프라인) — holidays.KR 보완: exchange_calendars가 놓친 선거일 등
          4단계 (온라인)  — pykrx 실측 확인 (d=오늘이고 네트워크 가용 시만)
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

        # 4단계: pykrx 온라인 실측 — d=오늘인 경우만 (미래 날짜는 KRX 데이터 없음)
        today = datetime.now().date()
        if target == today:
            try:
                from pykrx import stock as _pykrx
                tickers = _pykrx.get_market_ticker_list(today.strftime('%Y%m%d'), market='KOSPI')
                return len(tickers) > 0
            except Exception as e:
                logger.warning(f"pykrx 거래일 확인 실패: {e} → 오프라인 판단 사용")

        # 오프라인 체크 통과 (pykrx 실패 or 미래/과거 날짜) → 거래일로 판단
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
