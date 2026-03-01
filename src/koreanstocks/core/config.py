import os
from pathlib import Path
from dotenv import load_dotenv


def _resolve_base_dir() -> str:
    """저장소 루트 결정.

    우선순위:
    1) KOREANSTOCKS_BASE_DIR 환경변수 (임의 경로 지정 시)
    2) __file__ 기준 4단계 상위에 pyproject.toml이 있으면 프로젝트 루트
       (editable install: src/koreanstocks/core/ → src/koreanstocks/ → src/ → 루트/)
    3) ~/.koreanstocks/ — PyPI 전역 설치 시 사용자 홈 디렉토리
    """
    from_env = os.getenv("KOREANSTOCKS_BASE_DIR")
    if from_env:
        return os.path.abspath(from_env)

    candidate = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    if os.path.isfile(os.path.join(candidate, "pyproject.toml")):
        return candidate

    # PyPI 전역 설치: site-packages 구조이므로 사용자 홈 디렉토리로 fallback
    home_base = os.path.join(os.path.expanduser("~"), ".koreanstocks")
    os.makedirs(home_base, exist_ok=True)
    return home_base


# Step 1: CWD 기준 .env 로드 (editable install / 기존 워크플로 호환)
load_dotenv()

# Step 2: BASE_DIR 결정 (위에서 로드한 KOREANSTOCKS_BASE_DIR 반영)
_BASE_DIR = _resolve_base_dir()

# Step 3: BASE_DIR/.env 추가 로드 (PyPI 전역설치, koreanstocks init이 BASE_DIR에 생성한 .env)
#         override=False → 시스템 환경변수 및 CWD .env 값을 덮어쓰지 않음
_env_in_base = Path(_BASE_DIR) / ".env"
if _env_in_base.exists():
    load_dotenv(dotenv_path=_env_in_base, override=False)


class Config:
    # Version
    VERSION = "0.3.1"

    # Project Root — Step 2에서 결정된 _BASE_DIR 재사용 (중복 호출 방지)
    # - editable install (pip install -e .): __file__ 기준 자동 탐지
    # - 전역 설치 또는 경로 오류 시: .env에 KOREANSTOCKS_BASE_DIR=/path/to/project 설정
    BASE_DIR = _BASE_DIR

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
    NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
    DART_API_KEY = os.getenv("DART_API_KEY", "")

    # Database — 상대 경로는 BASE_DIR 기준 절대 경로로 변환 (CWD 의존 방지)
    _db_raw = os.getenv(
        "DB_PATH",
        os.path.join(BASE_DIR, "data", "storage", "stock_analysis.db"),
    )
    DB_PATH = _db_raw if os.path.isabs(_db_raw) else os.path.join(BASE_DIR, _db_raw)
    
    # Model Settings
    DEFAULT_MODEL = "gpt-4o-mini"
    
    # Trading Settings
    TRANSACTION_FEE = 0.00015  # 0.015%
    TAX_RATE = 0.0018         # 0.18%
    
    # GitHub DB 동기화 URL (koreanstocks sync 명령용)
    # 저장소를 포크했거나 private인 경우 KOREANSTOCKS_GITHUB_DB_URL 환경변수로 재정의
    GITHUB_RAW_DB_URL: str = os.getenv(
        "KOREANSTOCKS_GITHUB_DB_URL",
        "https://raw.githubusercontent.com/bullpeng72/KoreanStock/main/data/storage/stock_analysis.db",
    )

    # Cache Settings
    CACHE_EXPIRE_STOCKS = 1800  # 30 mins
    CACHE_EXPIRE_MARKET = 300   # 5 mins

    # Market Constants
    TRADING_DAYS_PER_YEAR = 252

config = Config()
