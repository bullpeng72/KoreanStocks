import logging
import time
from datetime import datetime
from koreanstocks.core.data.provider import data_provider
from koreanstocks.core.data.database import db_manager
from koreanstocks.core.engine.recommendation_agent import recommendation_agent
from koreanstocks.core.utils.notifier import notifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_daily_update(limit: int = 9):
    """매일 수행할 자동화 작업: 데이터 갱신 및 유망 종목 알림"""
    logger.info(f"Starting daily automated update (limit={limit})...")

    try:
        # 0. 지난 추천 성과 기록 (5·10·20 거래일 후 실제 수익률 집계)
        logger.info("Recording past recommendation outcomes...")
        try:
            from koreanstocks.core.utils.outcome_tracker import (
                record_outcomes, get_outcome_stats, get_recent_outcomes
            )
            record_outcomes()
            stats  = get_outcome_stats()
            recent = get_recent_outcomes(days=14)
            notifier.notify_performance_report(stats, recent)
        except Exception as e:
            logger.warning(f"Outcome tracking 실패 (분석은 계속 진행): {e}")

        # 1. 시장 종목 리스트 갱신 및 DB 저장
        logger.info("Updating stock list...")
        try:
            stocks = data_provider.get_stock_list()
        except Exception as e:
            logger.error(f"종목 목록 수집 중 예외 발생: {e}")
            notifier.send_message(f"⚠️ 종목 목록 수집 실패 (기존 DB 유지): {e}")
            stocks = None

        if not stocks is None and not stocks.empty:
            db_manager.save_stocks(stocks)
        else:
            logger.warning("종목 목록이 비어있거나 수집 실패 — DB 업데이트 건너뜀 (기존 데이터 유지)")
        
        # 2. 유망 종목 분석 및 추천 생성
        # recommendation_agent.get_recommendations 내부에서 분석 및 DB 저장이 수행됨
        logger.info("Analyzing market for recommendations...")
        recs = recommendation_agent.get_recommendations(limit=limit)
        
        # 3. 텔레그램 알림 전송
        if recs:
            logger.info(f"Sending notifications for {len(recs)} stocks...")
            notifier.notify_recommendation(recs)
        else:
            logger.info("No high-score recommendations found today.")
            
        logger.info("Daily update completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during daily update: {e}")
        notifier.send_message(f"❌ **자동 갱신 오류 발생:** {str(e)}")

if __name__ == "__main__":
    run_daily_update()
