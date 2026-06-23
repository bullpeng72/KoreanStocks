VERSION = "0.5.9"

import os
import sys
import warnings

# macOS에서 PyTorch와 LightGBM/XGBoost 등 OpenMP/MKL 사용 라이브러리 간의
# 충돌로 인해 발생하는 Deadlock(Hang) 현상을 방지하기 위해 환경 변수를 1개 스레드로 강제 설정합니다.
if sys.platform == "darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

