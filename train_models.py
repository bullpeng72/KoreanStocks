"""
ML 모델 학습 스크립트
======================
학습 로직은 koreanstocks.core.engine.trainer 패키지에 있습니다.
이 스크립트는 직접 실행을 위한 얇은 래퍼입니다.

실행 방법:
    python train_models.py
    python train_models.py --period 2y --future-days 5
    koreanstocks train  ← 동일한 로직을 어디서든 실행
"""

import os
import sys

# macOS에서 PyTorch와 LightGBM/XGBoost 등 OpenMP/MKL 사용 라이브러리 간의
# 충돌로 인해 발생하는 Deadlock(Hang) 현상을 방지하기 위해 환경 변수를 1개 스레드로 강제 설정합니다.
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from koreanstocks.core.engine.trainer import run_training, DEFAULT_TRAINING_STOCKS


def parse_args():
    parser = argparse.ArgumentParser(description='ML 주가 예측 모델 학습')
    parser.add_argument(
        '--period', default='2y',
        choices=['1y', '2y', '3m', '6m'],
        help='학습 데이터 기간 (기본값: 2y)'
    )
    parser.add_argument(
        '--future-days', type=int, default=10,
        help='예측 대상 기간 (거래일 수, 기본값: 10)'
    )
    parser.add_argument(
        '--stocks', nargs='+', default=None,
        help='학습에 사용할 종목 코드 (미지정 시 기본 종목 리스트 사용)'
    )
    parser.add_argument(
        '--test-ratio', type=float, default=0.2,
        help='검증 세트 비율 (기본값: 0.2)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_training(
        period=args.period,
        future_days=args.future_days,
        stocks=args.stocks or DEFAULT_TRAINING_STOCKS,
        test_ratio=args.test_ratio,
    )
