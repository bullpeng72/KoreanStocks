"""Typer CLI — koreanstocks serve / recommend / analyze / train / init / sync / home"""
import typer
from typing import Optional


def _build_env_template(keys: dict) -> str:
    """채워진 키 값으로 .env 템플릿 문자열 생성."""
    def v(key: str) -> str:
        return keys.get(key, "")

    return (
        "# KoreanStocks 환경변수\n"
        "# koreanstocks init 으로 생성됨\n"
        "# 각 항목에 API 키를 입력한 뒤 저장하세요.\n"
        "\n"
        "# ── 필수 ────────────────────────────────────────────────────\n"
        "\n"
        "# OpenAI API 키 — GPT-4o-mini (뉴스 감성 분석, AI 의견 생성)\n"
        "# 발급: https://platform.openai.com/api-keys\n"
        f"OPENAI_API_KEY={v('OPENAI_API_KEY')}\n"
        "\n"
        "# 네이버 검색 API — 종목명 기반 최신 뉴스 수집\n"
        "# 발급: https://developers.naver.com/apps\n"
        f"NAVER_CLIENT_ID={v('NAVER_CLIENT_ID')}\n"
        f"NAVER_CLIENT_SECRET={v('NAVER_CLIENT_SECRET')}\n"
        "\n"
        "# 텔레그램 봇 — 일일 추천 리포트 발송\n"
        "# BOT_TOKEN: @BotFather 에서 /newbot 으로 발급\n"
        "# CHAT_ID:   봇에게 메시지를 보낸 뒤 https://api.telegram.org/bot<TOKEN>/getUpdates 로 확인\n"
        f"TELEGRAM_BOT_TOKEN={v('TELEGRAM_BOT_TOKEN')}\n"
        f"TELEGRAM_CHAT_ID={v('TELEGRAM_CHAT_ID')}\n"
        "\n"
        "# ── 선택 ────────────────────────────────────────────────────\n"
        "\n"
        "# DART Open API 키 — 금융감독원 공시 수집 (설정 시 감성 분석 품질 향상)\n"
        "# 미설정이어도 동작하며, 뉴스만으로 감성 분석을 진행합니다.\n"
        "# 발급: https://opendart.fss.or.kr → 개발자 센터 → API 신청 (무료, 즉시 발급)\n"
        f"DART_API_KEY={v('DART_API_KEY')}\n"
        "\n"
        "# ── 시스템 ───────────────────────────────────────────────────\n"
        "\n"
        "# SQLite DB 경로 (기본값 그대로 사용 권장)\n"
        "DB_PATH=data/storage/stock_analysis.db\n"
        "\n"
        "# 프로젝트 루트 경로 (pip install -e . 로 editable 설치 시 자동 탐지됨)\n"
        "# 전역 설치(pip install koreanstocks) 사용 시 ~/.koreanstocks/ 가 자동 사용됨\n"
        "# 별도 경로를 지정하려면 아래 주석을 해제하고 경로를 입력\n"
        "# KOREANSTOCKS_BASE_DIR=/path/to/data-dir\n"
        "\n"
        "# koreanstocks sync 다운로드 URL (저장소를 포크한 경우에만 변경)\n"
        "# KOREANSTOCKS_GITHUB_DB_URL=https://raw.githubusercontent.com/{owner}/{repo}/main/data/storage/stock_analysis.db\n"
    )

app = typer.Typer(
    name="koreanstocks",
    add_completion=False,
    rich_markup_mode="rich",
    invoke_without_command=True,   # 서브커맨드 없이 실행 가능
    no_args_is_help=False,         # 직접 처리
)


def _version_callback(value: bool):
    if value:
        try:
            import importlib.metadata
            ver = importlib.metadata.version("koreanstocks")
        except Exception:
            from koreanstocks import VERSION
            ver = VERSION
        typer.echo(f"koreanstocks {ver}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None, "--version", "-V",
        help="버전 정보 표시 후 종료",
        callback=_version_callback,
        is_eager=True,
    ),
):
    """
    [bold cyan]KoreanStocks[/bold cyan] — AI/ML 기반 한국 주식 분석 플랫폼

    [dim]KOSPI·KOSDAQ 종목을 기술적 지표, ML 앙상블, 뉴스 감성 분석으로
    자동 스크리닝하고 텔레그램 리포트를 발송합니다.[/dim]

    ────────────────────────────────────────────────

    [bold]분석 파이프라인:[/bold]

      [cyan]1단계[/cyan]  기술적 지표  →  tech_score  [dim](RSI·MACD·BB·ADX·CMF 등, 0~100)[/dim]
      [cyan]2단계[/cyan]  ML 앙상블    →  ml_score    [dim](RF+GB+XGB, 10거래일 후 상위 25% 확률, 0~100)[/dim]
      [cyan]3단계[/cyan]  뉴스 감성    →  sentiment   [dim](Naver 뉴스 + GPT-4o-mini, -100~+100)[/dim]
      [cyan]4단계[/cyan]  AI 종합      →  BUY/HOLD/SELL + 목표가

    [dim]  종합점수 = tech×0.40 + ml×0.35 + sentiment_norm×0.25[/dim]

    ────────────────────────────────────────────────

    [bold]빠른 시작:[/bold]

    [green]  koreanstocks init[/green]            [dim]# .env 설정 파일 대화형 생성[/dim]
    [green]  koreanstocks sync[/green]            [dim]# GitHub에서 최신 분석 DB 다운로드[/dim]
    [green]  koreanstocks serve[/green]           [dim]# 웹 대시보드 실행 (브라우저 자동 열림)[/dim]
    [green]  koreanstocks recommend[/green]       [dim]# 오늘의 추천 종목 분석 + 텔레그램 발송[/dim]
    [green]  koreanstocks analyze 005930[/green]  [dim]# 삼성전자 단일 심층 분석[/dim]
    [green]  koreanstocks train[/green]           [dim]# ML 모델 재학습[/dim]
    [green]  koreanstocks outcomes[/green]        [dim]# 추천 결과 성과 추적 (5·10·20거래일)[/dim]

    ────────────────────────────────────────────────

    [bold]환경변수 (.env 필수 항목):[/bold]

      [yellow]OPENAI_API_KEY[/yellow]       GPT-4o-mini 뉴스 감성·AI 의견 생성
      [yellow]NAVER_CLIENT_ID/SECRET[/yellow]  네이버 뉴스 API
      [yellow]TELEGRAM_BOT_TOKEN[/yellow]   텔레그램 리포트 발송
      [yellow]TELEGRAM_CHAT_ID[/yellow]     수신 채팅방 ID

    커맨드별 상세 도움말: [cyan]koreanstocks [커맨드] --help[/cyan]
    """
    # 서브커맨드 없이 실행됐을 때는 간단한 요약만 출력
    if ctx.invoked_subcommand is None:
        try:
            import importlib.metadata
            ver = importlib.metadata.version("koreanstocks")
        except Exception:
            from koreanstocks import VERSION
            ver = VERSION
        typer.echo(
            f"KoreanStocks v{ver} — AI/ML 기반 한국 주식 분석 플랫폼\n"
            "\n"
            "  koreanstocks init       # 초기 설정 (.env 생성)\n"
            "  koreanstocks sync       # 최신 분석 DB 다운로드\n"
            "  koreanstocks serve      # 웹 대시보드 실행\n"
            "  koreanstocks recommend  # 오늘의 추천 종목 분석\n"
            "  koreanstocks analyze 005930  # 단일 종목 심층 분석\n"
            "\n"
            "자세한 도움말: koreanstocks --help"
        )
        raise typer.Exit()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="바인딩 호스트 주소"),
    port: int = typer.Option(8000, help="포트 번호"),
    reload: bool = typer.Option(False, help="코드 변경 시 자동 재시작 [dim](개발용)[/dim]"),
    no_browser: bool = typer.Option(False, "--no-browser", help="브라우저 자동 실행 비활성화"),
):
    """
    [bold]FastAPI 서버 실행[/bold] — 웹 대시보드 및 Reveal.js 브리핑

    서버 시작 후 브라우저가 자동으로 열립니다.

    [bold]제공 URL:[/bold]
    [green]  /[/green]          일일 브리핑 슬라이드 (Reveal.js)
    [green]  /dashboard[/green] 인터랙티브 대시보드 (6탭)
    [green]  /docs[/green]      API 문서 (Swagger UI)

    [bold]예시:[/bold]
    [dim]  koreanstocks serve[/dim]
    [dim]  koreanstocks serve --port 8080[/dim]
    [dim]  koreanstocks serve --reload --no-browser[/dim]
    """
    import threading
    import time
    import webbrowser
    import uvicorn

    url = f"http://{host}:{port}/dashboard"
    typer.echo(f"서버 시작: {url}")

    if not no_browser:
        def _open_browser():
            time.sleep(1.2)
            webbrowser.open(url)
        threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(
        "koreanstocks.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def recommend(
    market: str = typer.Option("ALL", help="시장 필터: [cyan]ALL[/cyan] | KOSPI | KOSDAQ"),
    limit: int = typer.Option(5, help="추천 종목 수 (1~30)"),
):
    """
    [bold]추천 종목 분석 실행[/bold] — GitHub Actions 자동화 및 수동 실행용

    후보 종목 100개를 병렬 분석하여 상위 N개를 선정합니다.
    결과는 DB에 저장되고 텔레그램으로 알림이 전송됩니다.

    [bold]분석 파이프라인:[/bold]
    [dim]  후보 100종목 → 병렬 분석(최대 30) → 종합점수 산출 → DB저장 → 텔레그램 발송[/dim]

    [bold]예시:[/bold]
    [dim]  koreanstocks recommend[/dim]
    [dim]  koreanstocks recommend --market KOSPI --limit 10[/dim]
    [dim]  koreanstocks recommend --market KOSDAQ[/dim]
    """
    from koreanstocks.core.engine.scheduler import run_daily_update
    typer.echo(f"일일 업데이트 실행 (market={market}, limit={limit})...")
    run_daily_update()
    typer.echo("완료.")


@app.command()
def analyze(
    code: str = typer.Argument(..., help="종목 코드 6자리 (예: 005930)"),
):
    """
    [bold]단일 종목 심층 분석[/bold] — 기술지표·ML·뉴스 감성 통합 분석

    [bold]출력 항목:[/bold]
      기술점수 (Tech)  — RSI, MACD, BB 등 지표 기반 (0~100)
      ML점수           — 10거래일 후 상위 25% 확률 (캘리브레이션 0~100)
      감성점수 (News)  — 뉴스 GPT 감성 분석 (-100~+100)
      AI 의견          — BUY / HOLD / SELL + 한줄 요약 + 목표가

    [bold]예시:[/bold]
    [dim]  koreanstocks analyze 005930[/dim]   [dim]# 삼성전자[/dim]
    [dim]  koreanstocks analyze 000660[/dim]   [dim]# SK하이닉스[/dim]
    [dim]  koreanstocks analyze 035420[/dim]   [dim]# NAVER[/dim]
    """
    from koreanstocks.core.engine.analysis_agent import analysis_agent
    from koreanstocks.core.data.provider import data_provider

    stock_list = data_provider.get_stock_list()
    row = stock_list[stock_list["code"] == code]
    name = row.iloc[0]["name"] if not row.empty else code

    typer.echo(f"[{code}] {name} 분석 중...")
    result = analysis_agent.analyze_stock(code, name)

    if result:
        opinion = result.get("ai_opinion", {})
        action  = opinion.get("action", "-")
        typer.echo(f"  기술점수 : {result.get('tech_score', '-')}")
        typer.echo(f"  ML점수   : {result.get('ml_score', '-')}")
        typer.echo(f"  감성점수 : {result.get('sentiment_score', '-')}")
        typer.echo(f"  의견     : {action} — {opinion.get('summary', '')}")
        if opinion.get("target_price"):
            typer.echo(f"  목표가   : {int(opinion['target_price']):,}원")
    else:
        typer.echo("분석 실패.", err=True)
        raise typer.Exit(1)


@app.command()
def train(
    period: str = typer.Option("2y", help="학습 데이터 기간: 1y | [cyan]2y[/cyan]"),
    future_days: int = typer.Option(10, help="예측 대상 거래일 수 (기본 10 = 2주, 중기 노이즈 최소화)"),
    test_ratio: float = typer.Option(0.2, help="검증 세트 비율 (0~1)"),
):
    """
    [bold]ML 모델 재학습[/bold] — RandomForest·GradientBoosting·XGBoost 앙상블

    [bold]사용 피처 (18개):[/bold]
    [dim]  기술지표 15개 — ATR, ADX, BB, MACD, CMF, VZO, Fisher, vol_ratio 등[/dim]
    [dim]  시장·거시 3개  — 시장 상대강도 3m, VIX 레벨/변화율, S&P500 1m[/dim]

    [bold]타깃:[/bold] 10거래일 후 수익률 상위 25%/하위 25% 이진 분류 (중간 50% neutral zone 제외)

    [bold]예시:[/bold]
    [dim]  koreanstocks train[/dim]
    [dim]  koreanstocks train --period 1y --future-days 10[/dim]
    [dim]  koreanstocks train --test-ratio 0.3[/dim]
    """
    from koreanstocks.core.engine.trainer import run_training, DEFAULT_TRAINING_STOCKS

    run_training(
        period=period,
        future_days=future_days,
        stocks=DEFAULT_TRAINING_STOCKS,
        test_ratio=test_ratio,
    )


@app.command()
def sync(
    force: bool = typer.Option(False, "--force", help="로컬 DB가 있어도 강제 덮어쓰기"),
    token: Optional[str] = typer.Option(
        None, "--token", envvar="GITHUB_TOKEN",
        help="비공개 저장소용 GitHub Personal Access Token",
    ),
):
    """
    [bold]GitHub에서 최신 분석 DB 다운로드[/bold] — 다중 PC 동기화

    GitHub Actions가 평일 16:30 KST에 생성한 추천 데이터를 로컬로 가져옵니다.
    로컬 DB가 이미 존재하면 덮어쓰지 않습니다 ([cyan]--force[/cyan] 옵션으로 강제 가능).

    [bold]저장 위치:[/bold]
    [dim]  pip install -e .    →  (프로젝트루트)/data/storage/stock_analysis.db[/dim]
    [dim]  pip install (PyPI)  →  ~/.koreanstocks/data/storage/stock_analysis.db[/dim]

    [bold]비공개 저장소 사용 시:[/bold]
    [dim]  GitHub → Settings → Developer settings → Personal access tokens[/dim]
    [dim]  repo(read) 권한으로 토큰 발급 후 --token 또는 GITHUB_TOKEN 환경변수로 전달[/dim]

    [bold]예시:[/bold]
    [dim]  koreanstocks sync[/dim]
    [dim]  koreanstocks sync --force[/dim]
    [dim]  koreanstocks sync --token ghp_xxxx[/dim]
    """
    import httpx
    from pathlib import Path
    from koreanstocks.core.config import config

    db_path = Path(config.DB_PATH)

    if db_path.exists() and not force:
        typer.echo(f"로컬 DB가 이미 존재합니다: {db_path}")
        typer.echo("최신 데이터로 덮어쓰려면 --force 옵션을 사용하세요.")
        raise typer.Exit()

    url = config.GITHUB_RAW_DB_URL
    typer.echo(f"다운로드: {url}")

    headers: dict = {}
    if token:
        headers["Authorization"] = f"token {token}"

    tmp_path = db_path.with_suffix(".tmp")
    try:
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code == 401:
                    typer.echo(
                        "인증 실패. --token 옵션으로 GitHub Personal Access Token을 제공하거나\n"
                        "GITHUB_TOKEN 환경변수를 설정하세요.",
                        err=True,
                    )
                    raise typer.Exit(1)
                if response.status_code == 404:
                    typer.echo(
                        "DB 파일을 찾을 수 없습니다 (404).\n"
                        "GitHub Actions가 아직 한 번도 실행되지 않았거나\n"
                        "저장소가 비공개 상태일 수 있습니다.",
                        err=True,
                    )
                    raise typer.Exit(1)
                response.raise_for_status()

                total = int(response.headers.get("content-length", 0))
                db_path.parent.mkdir(parents=True, exist_ok=True)

                downloaded = 0
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            typer.echo(
                                f"\r  {downloaded // 1024:,} KB / {total // 1024:,} KB  ({pct:.0f}%)",
                                nl=False,
                            )

                if total:
                    typer.echo("")  # 진행률 줄바꿈

        # 정상 수신 후 최종 경로로 이동 (원자적 교체)
        tmp_path.replace(db_path)
        typer.echo(f"완료: {db_path}  ({downloaded // 1024:,} KB)")

    except httpx.RequestError as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        typer.echo(f"네트워크 오류: {e}", err=True)
        raise typer.Exit(1)
    except typer.Exit:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


@app.command()
def init(
    non_interactive: bool = typer.Option(
        False, "--non-interactive", "-y",
        help="대화형 입력 건너뜀 — CI·자동화 환경용 (빈 템플릿만 생성)",
    ),
):
    """
    [bold]초기 설정[/bold] — .env 환경변수 파일 대화형 생성

    API 키를 단계적으로 입력받아 [cyan].env[/cyan] 파일을 생성합니다.
    [cyan]--non-interactive[/cyan] 옵션으로 빈 템플릿만 생성할 수 있습니다.

    [bold]생성 위치:[/bold]
    [dim]  pip install -e .    →  (프로젝트루트)/.env[/dim]
    [dim]  pip install (PyPI)  →  ~/.koreanstocks/.env[/dim]

    [bold]예시:[/bold]
    [dim]  koreanstocks init                   # 대화형 입력[/dim]
    [dim]  koreanstocks init --non-interactive  # 빈 템플릿만 생성 (CI용)[/dim]
    """
    from pathlib import Path
    from koreanstocks.core.config import config

    env_file = Path(config.BASE_DIR) / ".env"

    typer.echo(f"생성 위치: {env_file}")

    if env_file.exists():
        if non_interactive:
            typer.echo(".env 파일이 이미 존재합니다.")
            typer.echo(f"  편집: ${{EDITOR:-nano}} {env_file}")
            return
        overwrite = typer.confirm("\n.env 파일이 이미 존재합니다. 덮어쓰겠습니까?", default=False)
        if not overwrite:
            typer.echo(f"취소했습니다. 기존 파일을 편집하세요: {env_file}")
            return

    keys: dict = {}

    if not non_interactive:
        _REQUIRED = [
            ("OPENAI_API_KEY",      "OpenAI API Key",      "https://platform.openai.com/api-keys"),
            ("NAVER_CLIENT_ID",     "Naver Client ID",     "https://developers.naver.com/apps"),
            ("NAVER_CLIENT_SECRET", "Naver Client Secret", None),
            ("TELEGRAM_BOT_TOKEN",  "Telegram Bot Token",  "@BotFather → /newbot"),
            ("TELEGRAM_CHAT_ID",    "Telegram Chat ID",    "getUpdates 로 확인"),
        ]
        _OPTIONAL = [
            ("DART_API_KEY", "DART API Key (선택)", "https://opendart.fss.or.kr"),
        ]

        typer.echo("\n[필수] API 키를 입력하세요 (Enter = 나중에 입력):\n")
        for key, label, hint in _REQUIRED:
            prompt_text = f"  {label}" + (f" [{hint}]" if hint else "")
            keys[key] = typer.prompt(prompt_text, default="", show_default=False).strip()

        typer.echo("\n[선택] 미입력 시 건너뜁니다:\n")
        for key, label, hint in _OPTIONAL:
            prompt_text = f"  {label}" + (f" [{hint}]" if hint else "")
            keys[key] = typer.prompt(prompt_text, default="", show_default=False).strip()

        typer.echo("")

    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(_build_env_template(keys), encoding="utf-8")

    typer.echo(f".env 파일을 생성했습니다.")
    typer.echo(f"  경로: {env_file}")

    if non_interactive:
        typer.echo(f"  편집: ${{EDITOR:-nano}} {env_file}")
    else:
        _required_keys = ["OPENAI_API_KEY", "NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET",
                          "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
        empty = [k for k in _required_keys if not keys.get(k)]
        if empty:
            typer.echo(f"\n  미입력 필수 키: {', '.join(empty)}")
            typer.echo(f"  편집: ${{EDITOR:-nano}} {env_file}")

    typer.echo("\n다음 단계:")
    typer.echo("  koreanstocks sync    # 최신 분석 DB 다운로드")
    typer.echo("  koreanstocks serve   # 웹 대시보드 실행")


@app.command()
def outcomes(
    days: int = typer.Option(90, help="최근 N일간 통계 조회"),
    no_record: bool = typer.Option(False, "--no-record", help="성과 기록 없이 조회만"),
):
    """
    [bold]추천 성과 조회[/bold] — 지난 AI 추천의 실제 수익률 확인

    추천 후 5·10·20 거래일이 경과한 종목의 실제 주가를 수집하여
    정답률·평균 수익률·목표가 달성률을 계산합니다.

    [bold]예시:[/bold]
    [dim]  koreanstocks outcomes[/dim]
    [dim]  koreanstocks outcomes --days 30[/dim]
    [dim]  koreanstocks outcomes --no-record   # 기록 없이 조회만[/dim]
    """
    from koreanstocks.core.utils.outcome_tracker import (
        record_outcomes, get_outcome_stats, get_recent_outcomes
    )

    if not no_record:
        typer.echo("성과 기록 중...")
        updated = record_outcomes()
        typer.echo(f"  새로 업데이트: {updated}건")

    stats    = get_outcome_stats(days=days)
    outcomes_list = get_recent_outcomes(days=days)

    if not stats or stats.get("total", 0) == 0:
        typer.echo("\n성과 데이터가 없습니다. 추천 후 5거래일 이상 지나야 집계됩니다.")
        return

    typer.echo(f"\n=== 최근 {days}일 추천 성과 ({stats['total']}건) ===")
    for n, label in [(5, " 5거래일"), (10, "10거래일"), (20, "20거래일")]:
        ev = stats.get(f"evaluated_{n}d", 0)
        if ev == 0:
            continue
        wr  = stats.get(f"win_rate_{n}d",  0)
        ret = stats.get(f"avg_return_{n}d", 0)
        typer.echo(f"  {label}: 정답률 {wr:.0f}%  평균 {ret:+.2f}%  ({ev}건)")

    thr = stats.get("target_hit_rate")
    if thr is not None:
        typer.echo(f"  목표가 달성률: {thr:.0f}%")

    if outcomes_list:
        typer.echo(f"\n개별 성과 (최근 {min(15, len(outcomes_list))}건):")
        for o in outcomes_list[:15]:
            action = o.get("action", "?")
            r5  = o["outcome_5d"].get("return_pct")
            c5  = o["outcome_5d"].get("correct")
            r20 = o["outcome_20d"].get("return_pct")
            hit = "✅" if c5 == 1 else ("❌" if c5 == 0 else "⏳")
            r5_str  = f"{r5:+.1f}%"  if r5  is not None else "집계중"
            r20_str = f"{r20:+.1f}%" if r20 is not None else "집계중"
            typer.echo(
                f"  {hit} [{o['session_date']}] {o['name']}({o['code']}) "
                f"{action}  5d:{r5_str}  20d:{r20_str}"
            )


@app.command()
def home(
    open_dir: bool = typer.Option(False, "--open", "-o", help="파일 탐색기로 홈 디렉토리 열기"),
    setup: bool = typer.Option(False, "--setup", "-s", help="셸 alias 스니펫 출력"),
):
    """
    [bold]데이터 홈 디렉토리 경로 출력[/bold]

    .env, DB, ML 모델이 저장된 디렉토리의 경로를 출력합니다.
    셸에서 [cyan]cd $(koreanstocks home)[/cyan] 으로 바로 이동할 수 있습니다.

    [bold]저장 파일:[/bold]
    [dim]  .env                              — API 키 설정 파일[/dim]
    [dim]  data/storage/stock_analysis.db   — SQLite 분석 데이터베이스[/dim]
    [dim]  models/saved/                    — 학습된 ML 모델 (.pkl)[/dim]

    [bold]예시:[/bold]
    [dim]  cd $(koreanstocks home)           # 홈 디렉토리로 이동[/dim]
    [dim]  koreanstocks home --open          # 파일 탐색기로 열기[/dim]
    [dim]  koreanstocks home --setup         # 셸 alias 설정 안내 출력[/dim]
    """
    import os
    import subprocess
    import sys
    from pathlib import Path
    from koreanstocks.core.config import config

    base = Path(config.BASE_DIR)

    if open_dir:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(base)], check=True)
            elif sys.platform == "win32":
                subprocess.run(["explorer", str(base)], check=True)
            else:
                subprocess.run(["xdg-open", str(base)], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            typer.echo(f"파일 탐색기를 열 수 없습니다: {e}", err=True)
            typer.echo(f"경로: {base}")
        return

    if setup:
        shell = os.environ.get("SHELL", "")
        rc_file = "~/.zshrc" if "zsh" in shell else "~/.bashrc"
        typer.echo(f"# 아래 내용을 {rc_file} 에 추가하세요:\n")
        typer.echo('alias kshome=\'cd "$(koreanstocks home)"\'')
        typer.echo('alias ksenv=\'${EDITOR:-nano} "$(koreanstocks home)/.env"\'')
        typer.echo(f"\n# 적용: source {rc_file}")
        return

    # 기본: 경로만 출력 (cd $(koreanstocks home) 에서 사용)
    typer.echo(str(base))
