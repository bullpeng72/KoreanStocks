import json
import sqlite3
import pandas as pd
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
from koreanstocks.core.config import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """SQLite 데이터베이스 관리를 담당하는 클래스 (Singleton)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.db_path = config.DB_PATH
            cls._instance._ensure_db_dir()
            cls._instance.init_db()
        return cls._instance

    def _ensure_db_dir(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def get_connection(self):
        return sqlite3.connect(self.db_path, timeout=30.0)

    def init_db(self):
        """필요한 테이블들을 생성"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. 종목 정보 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stocks (
                    code TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    market TEXT NOT NULL,
                    sector TEXT,
                    industry TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 2. 주가 데이터 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_prices (
                    code TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    change REAL,
                    PRIMARY KEY (code, date),
                    FOREIGN KEY (code) REFERENCES stocks(code)
                )
            ''')
            
            # 3. 추천 기록 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT NOT NULL,
                    type TEXT NOT NULL, -- 'BUY', 'SELL', 'HOLD'
                    score REAL,
                    reason TEXT,
                    target_price REAL,
                    stop_loss REAL,
                    source TEXT, -- 'ML_ENSEMBLE', 'AI_AGENT'
                    detail_json TEXT, -- 전체 분석 결과 JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (code) REFERENCES stocks(code)
                )
            ''')
            # 기존 테이블 마이그레이션
            for migration in [
                "ALTER TABLE recommendations ADD COLUMN detail_json TEXT",
                "ALTER TABLE recommendations ADD COLUMN session_date DATE",
                "ALTER TABLE analysis_history ADD COLUMN detail_json TEXT",
            ]:
                try:
                    cursor.execute(migration)
                except Exception:
                    pass  # 이미 존재하면 무시
            
            # 4. 백테스트 결과 테이블 (Phase 1 강화)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    total_return REAL,
                    win_rate REAL,
                    mdd REAL,
                    sharpe_ratio REAL,
                    start_date DATE,
                    end_date DATE,
                    parameters TEXT, -- JSON string
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 5. 관심 종목 테이블 (Phase 4 신규)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS watchlist (
                    code TEXT PRIMARY KEY,
                    name TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 6. 분석 이력 테이블 (Timeline용)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT NOT NULL,
                    tech_score REAL,
                    ml_score REAL,
                    sentiment_score REAL,
                    action TEXT,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (code) REFERENCES stocks(code)
                )
            ''')

            # 7. 뉴스 감성 캐시 테이블 (GitHub Actions 재실행 비용 절감)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    cache_key  TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 8. 추천 결과 검증 테이블 (피드백 루프용)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendation_outcomes (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    code         TEXT NOT NULL,
                    session_date DATE NOT NULL,
                    action       TEXT NOT NULL,
                    entry_price  REAL NOT NULL,
                    target_price REAL,
                    price_5d     REAL,
                    return_5d    REAL,
                    correct_5d   INTEGER,
                    price_10d    REAL,
                    return_10d   REAL,
                    correct_10d  INTEGER,
                    price_20d    REAL,
                    return_20d   REAL,
                    correct_20d  INTEGER,
                    target_hit   INTEGER,
                    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(code, session_date)
                )
            ''')

            # 9. 펀더멘털 캐시 테이블 (가치주 스크리너용, 당일 유효)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fundamental_cache (
                    code        TEXT NOT NULL,
                    cache_date  TEXT NOT NULL,
                    data_json   TEXT NOT NULL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (code, cache_date)
                )
            ''')

    def get_sentiment_cache(self, cache_key: str) -> Optional[Dict]:
        """당일 감성 분석 캐시 조회. 없으면 None 반환."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT result_json FROM sentiment_cache WHERE cache_key = ?',
                    (cache_key,)
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            logger.warning(f"sentiment_cache 조회 실패: {e}")
        return None

    def save_sentiment_cache(self, cache_key: str, result: Dict) -> None:
        """감성 분석 결과를 SQLite에 저장하고 7일 지난 항목은 정리."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT OR REPLACE INTO sentiment_cache (cache_key, result_json) VALUES (?, ?)',
                    (cache_key, json.dumps(result, ensure_ascii=False))
                )
                # 7일 이상 된 캐시 자동 정리
                cursor.execute(
                    "DELETE FROM sentiment_cache WHERE created_at < datetime('now', '-7 days')"
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"sentiment_cache 저장 실패: {e}")

    def save_analysis_history(self, res: Dict):
        """분석 결과 이력 저장 (요약 + 전체 JSON)"""
        try:
            detail_json = json.dumps(res, ensure_ascii=False, default=str)
        except Exception:
            detail_json = None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analysis_history
                    (code, tech_score, ml_score, sentiment_score, action, summary, detail_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                res['code'],
                res['tech_score'],
                res['ml_score'],
                res['sentiment_score'],
                res['ai_opinion']['action'],
                res['ai_opinion']['summary'],
                detail_json,
            ))
            conn.commit()

    def get_analysis_history(self, code: str, limit: int = 5) -> List[Dict]:
        """특정 종목의 최근 분석 이력 조회 (요약 + 전체 상세 포함)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT tech_score, ml_score, sentiment_score, action, summary,
                       created_at, detail_json
                FROM analysis_history
                WHERE code = ?
                ORDER BY created_at DESC LIMIT ?
            ''', (code, limit))
            rows = cursor.fetchall()
            result = []
            for r in rows:
                item: Dict = {
                    'tech_score':      r[0],
                    'ml_score':        r[1],
                    'sentiment_score': r[2],
                    'action':          r[3],
                    'summary':         r[4],
                    'date':            r[5],
                }
                if r[6]:
                    try:
                        item['detail'] = json.loads(r[6])
                    except Exception:
                        pass
                result.append(item)
            return result

    def get_recommendations_by_date(self, date_str: str) -> List[Dict]:
        """특정 날짜의 추천 종목 목록 반환 (detail_json 우선, 없으면 기본 구조로 폴백)"""
        import json
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT r.code, s.name, r.type, r.score, r.reason,
                       r.target_price, r.created_at, r.detail_json
                FROM recommendations r
                LEFT JOIN stocks s ON r.code = s.code
                WHERE r.session_date = ?
                ORDER BY r.score DESC
            ''', (date_str,))
            rows = cursor.fetchall()

        result = []
        for code, name, action, score, reason, target_price, created_at, detail_json in rows:
            if detail_json:
                try:
                    result.append(json.loads(detail_json))
                    continue
                except Exception:
                    pass
            # 폴백: detail_json 없는 구버전 데이터
            result.append({
                'code': code, 'name': name or code,
                'current_price': 0, 'change_pct': 0,
                'tech_score': score, 'ml_score': score, 'sentiment_score': 0,
                'sentiment_info': {}, 'indicators': {}, 'stats': {},
                'ai_opinion': {
                    'action': action, 'summary': reason or '',
                    'reasoning': '', 'target_price': target_price or 0,
                    'target_rationale': '', 'strength': '', 'weakness': '',
                },
            })
        return result

    def get_recommendation_history(self, days: int = 30) -> List[Dict]:
        """최근 N일간 추천 데이터 전체 반환 (지속성 히트맵용)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT r.code,
                       COALESCE(s.name, json_extract(r.detail_json, '$.name'), r.code) AS name,
                       r.score, r.type, r.session_date
                FROM recommendations r
                LEFT JOIN stocks s ON r.code = s.code
                WHERE r.session_date IS NOT NULL
                  AND r.session_date >= date('now', ?)
                ORDER BY r.session_date ASC
            ''', (f'-{days} days',))
            rows = cursor.fetchall()
        return [
            {'code': r[0], 'name': r[1], 'score': r[2], 'action': r[3], 'date': r[4]}
            for r in rows
        ]

    def get_recommendation_dates(self, limit: int = 30) -> List[str]:
        """추천 데이터가 존재하는 날짜 목록 반환 (최근순)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT session_date
                FROM recommendations
                WHERE session_date IS NOT NULL
                ORDER BY session_date DESC
                LIMIT ?
            ''', (limit,))
            return [r[0] for r in cursor.fetchall()]

    def get_latest_recommendation_date(self) -> Optional[str]:
        """가장 최근 추천 날짜 반환"""
        dates = self.get_recommendation_dates(limit=1)
        return dates[0] if dates else None

    def get_stock_name(self, code: str) -> Optional[str]:
        """로컬 DB에서 종목명 조회 (오프라인 폴백용)"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM stocks WHERE code = ? LIMIT 1", (code,))
            row = cursor.fetchone()
            return row[0] if row and row[0] else None

    def save_stocks(self, df: pd.DataFrame):
        """종목 리스트 저장"""
        if df.empty: return
        # 중복 컬럼 방어: 스키마에 맞는 컬럼만 선택
        target_cols = ['code', 'name', 'market', 'sector', 'industry']
        save_df = df[[c for c in target_cols if c in df.columns]].copy()
        # 누락된 필수 컬럼 채우기
        for col in target_cols:
            if col not in save_df.columns:
                save_df[col] = ''
        with self.get_connection() as conn:
            save_df.to_sql('stocks', conn, if_exists='replace', index=False)

    def save_prices(self, code: str, df: pd.DataFrame):
        """주가 데이터 저장"""
        if df.empty: return
        df = df.copy()
        df['code'] = code
        df = df.reset_index()
        with self.get_connection() as conn:
            df.to_sql('stock_prices', conn, if_exists='append', index=False, 
                      method=None, chunksize=1000)

    def get_prices(self, code: str, start: str = None, end: str = None) -> pd.DataFrame:
        """저장된 주가 데이터 조회"""
        query = "SELECT * FROM stock_prices WHERE code = ?"
        params = [code]
        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)
        query += " ORDER BY date"
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            return df

    def add_to_watchlist(self, code: str, name: str):
        """관심 종목 추가"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT OR REPLACE INTO watchlist (code, name) VALUES (?, ?)', (code, name))
            conn.commit()

    def remove_from_watchlist(self, code: str):
        """관심 종목 삭제"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM watchlist WHERE code = ?', (code,))
            conn.commit()

    def get_watchlist(self) -> List[Dict]:
        """관심 종목 리스트 조회"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT code, name FROM watchlist ORDER BY added_at DESC')
            rows = cursor.fetchall()
            return [{'code': row[0], 'name': row[1]} for row in rows]

# Singleton instance
db_manager = DatabaseManager()
