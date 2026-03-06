"""FastAPI 앱 팩토리"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from koreanstocks import VERSION
from koreanstocks.api.routers import (
    recommendations,
    analysis,
    watchlist,
    backtest,
    market,
    models,
    value,
)

STATIC_DIR = Path(__file__).parent.parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="KoreanStocks API",
        version=VERSION,
        description="KOSPI·KOSDAQ 종목 자동 스크리닝 + 텔레그램 리포트 플랫폼",
    )

    # 라우터 등록
    app.include_router(recommendations.router, prefix="/api")
    app.include_router(analysis.router, prefix="/api")
    app.include_router(watchlist.router, prefix="/api")
    app.include_router(backtest.router, prefix="/api")
    app.include_router(market.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    app.include_router(value.router,  prefix="/api")

    # Static 파일 마운트 (Reveal.js + 대시보드)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/api/version", tags=["meta"])
    async def get_version():
        """앱 버전 반환."""
        return {"version": VERSION}

    @app.get("/", include_in_schema=False)
    async def root():
        """Reveal.js 일일 브리핑"""
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard():
        """인터랙티브 대시보드"""
        return FileResponse(str(STATIC_DIR / "dashboard.html"))

    return app


app = create_app()
