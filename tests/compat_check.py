"""
Python 3.11~3.13 호환성 빠른 점검 스크립트
실행: python tests/compat_check.py
"""
import sys
import importlib
import traceback

PY = f"Python {sys.version}"
OK  = "✅"
FAIL = "❌"
WARN = "⚠️"

results: list[tuple[str, str, str]] = []  # (항목, 상태, 메모)


def check(label: str, fn):
    try:
        note = fn() or ""
        results.append((label, OK, str(note)))
    except Exception as e:
        results.append((label, FAIL, str(e)[:120]))


# ── 1. 핵심 패키지 임포트 ──────────────────────────────────────────
def _import(name):
    def _():
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", "?")
        return f"v{ver}"
    return _


for pkg in [
    "fastapi", "uvicorn", "typer",
    "openai", "pandas", "sklearn", "xgboost",
    "lightgbm", "catboost", "ta", "plotly", "numpy",
    "scipy", "joblib", "bs4", "schedule",
    "matplotlib", "httpx", "dotenv",
    "requests", "FinanceDataReader",
]:
    check(f"import {pkg}", _import(pkg))

# torch는 선택적 의존성 ([dl] extra) — 미설치 시 경고만 (FAIL 아님)
try:
    import torch as _torch
    results.append(("import torch [dl]", OK, f"v{_torch.__version__} (TCN 활성화)"))
except ImportError:
    results.append(("import torch [dl]", WARN, "미설치 — TCN 비활성화 (pipx inject koreanstocks torch 로 활성화)"))

# ── 2. koreanstocks 패키지 자체 ────────────────────────────────────
check("import koreanstocks", lambda: importlib.import_module("koreanstocks").__version__ if hasattr(importlib.import_module("koreanstocks"), "__version__") else "ok")

def _core_imports():
    from koreanstocks.core.config import config
    from koreanstocks.core.data.database import db_manager
    from koreanstocks.core.engine.indicators import IndicatorCalculator
    from koreanstocks.core.engine.strategy import TechnicalStrategy
    from koreanstocks.core.utils.backtester import Backtester
    return "ok"

check("koreanstocks.core 전체", _core_imports)

def _api_import():
    from koreanstocks.api.app import app
    return f"{len(app.routes)} routes"

check("koreanstocks.api (FastAPI app)", _api_import)

# ── 3. 핵심 기능 동작 테스트 ───────────────────────────────────────
def _backtester_smoke():
    import pandas as pd
    import numpy as np
    from koreanstocks.core.engine.indicators import IndicatorCalculator
    from koreanstocks.core.engine.strategy import TechnicalStrategy
    from koreanstocks.core.utils.backtester import Backtester
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=80, freq="B")
    base = 50000 + rng.integers(-500, 500, 80).cumsum()
    df = pd.DataFrame({
        "open": base, "high": base + 300, "low": base - 300,
        "close": base, "volume": rng.integers(100000, 1000000, 80),
    }, index=dates)
    calc = IndicatorCalculator()
    df_ind = calc.calculate_all(df)
    signals = TechnicalStrategy().generate_signals(df_ind, strategy_type="RSI")
    result = Backtester().run(df_ind, signals)
    assert "total_return_pct" in result, "missing key"
    return f"total_return={result['total_return_pct']:.1f}%"

check("Backtester.run (RSI, 80일)", _backtester_smoke)

def _indicator_smoke():
    import pandas as pd
    import numpy as np
    from koreanstocks.core.engine.indicators import IndicatorCalculator
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=80, freq="B")
    df = pd.DataFrame({
        "open":   50000 + rng.integers(-500, 500, 80).cumsum(),
        "high":   50500 + rng.integers(-500, 500, 80).cumsum(),
        "low":    49500 + rng.integers(-500, 500, 80).cumsum(),
        "close":  50000 + rng.integers(-500, 500, 80).cumsum(),
        "volume": rng.integers(100000, 1000000, 80),
    }, index=dates)
    calc = IndicatorCalculator()
    out = calc.calculate_all(df)
    score = calc.get_composite_score(out)
    assert 0 <= score <= 100
    return f"score={score:.1f}"

check("IndicatorCalculator (RSI/MACD/BB)", _indicator_smoke)

def _numpy_compat():
    import numpy as np
    # numpy 2.x 주요 변경: np.bool, np.int 등 제거
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([True, False, True], dtype=np.bool_)
    assert a.dtype == np.int64
    assert b.dtype == np.bool_
    # random generator (구형 np.random.seed 아닌 새 API)
    rng = np.random.default_rng(42)
    _ = rng.integers(0, 100, 10)
    return f"v{np.__version__}"

check("NumPy 2.x API 호환성", _numpy_compat)

def _pandas_compat():
    import pandas as pd
    # pandas 3.x: copy-on-write 기본 활성화
    df = pd.DataFrame({"a": [1, 2, 3]})
    df2 = df.copy()
    df2["a"] = 10
    assert df["a"].tolist() == [1, 2, 3], "CoW violation"
    return f"v{pd.__version__}"

check("pandas 3.x CoW 호환성", _pandas_compat)

def _sklearn_compat():
    import sklearn
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.random((50, 5))
    y = rng.random(50)
    rf = RandomForestRegressor(n_estimators=5, random_state=0)
    rf.fit(X, y)
    pred = rf.predict(X[:3])
    assert len(pred) == 3
    return f"v{sklearn.__version__}"

check("scikit-learn 1.6+ 기본 동작", _sklearn_compat)

# ── 결과 출력 ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  호환성 검증 결과  —  {PY}")
print(f"{'='*60}")
col_w = max(len(r[0]) for r in results) + 2
for label, status, note in results:
    print(f"  {status}  {label:<{col_w}}  {note}")

fail_count = sum(1 for _, s, _ in results if s == FAIL)
ok_count   = sum(1 for _, s, _ in results if s == OK)
print(f"\n  합계: {OK} {ok_count}개 통과  /  {FAIL} {fail_count}개 실패")
print(f"{'='*60}\n")
sys.exit(1 if fail_count > 0 else 0)
