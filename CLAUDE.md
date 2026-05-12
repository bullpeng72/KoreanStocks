# Korean Stocks AI/ML Analysis System `v0.5.6`

KOSPI·KOSDAQ 종목을 기술적 지표, 머신러닝, 뉴스 감성 분석으로 자동 스크리닝하고 텔레그램 리포트를 발송하는 투자 보조 플랫폼.

## 아키텍처 원칙

1. **Decoupling:** 비즈니스 로직(`koreanstocks.core/`)과 API/UI를 엄격히 분리. API 서버 없이도 CLI로 분석 엔진이 독립 동작해야 함.
2. **Validation First:** 모든 전략과 ML 모델은 백테스팅 결과를 동반해야 함.
3. **Cost Control:** LLM(GPT-5.4-nano) 호출 전 전처리로 비용 최적화. `max_tokens` 제한 필수.
4. **Automation:** 데이터 수집·분석·알림은 GitHub Actions 스케줄러가 담당 (평일 16:30 KST). SQLite DB는 저장소에 자동 커밋·푸시되며, GitHub Artifact로도 병행 백업 (90일 보존).

## 기술 스택

- **UI:** FastAPI + Reveal.js (일일 브리핑 슬라이드) + Vanilla JS (인터랙티브 대시보드)
- **CLI:** Typer (`koreanstocks serve / recommend / analyze / train / outcomes / value / quality / init / sync / home`)
- **AI/LLM:** OpenAI GPT-5.4-nano
- **ML:** Scikit-learn (Random Forest, Gradient Boosting), XGBoost Ranker, LightGBM, CatBoost, PyTorch TCN (선택적)
- **기술 지표:** `ta` 라이브러리 (RSI, MACD, BB, SMA, OBV, ADX, VWAP, CMF, MFI, Stochastic, CCI, ATR, Donchian) + `finta` (SQZMI, VZO, Fisher Transform, Williams Fractal)
- **데이터:** FinanceDataReader, Naver News API
- **DB:** SQLite (`data/storage/stock_analysis.db`)
- **자동화:** GitHub Actions, Telegram Bot API
- **시각화:** Plotly, Matplotlib, Chart.js (백테스트 차트)
- **언어:** Python 3.11 ~ 3.13

## 프로젝트 구조

```
pyproject.toml                       # pip 빌드 설정 (koreanstocks CLI 진입점)
requirements.txt                     # 개발/테스트 전용 (pytest 등)
src/
└── koreanstocks/
    ├── __init__.py                  # VERSION = "0.5.6"
    ├── cli.py                       # Typer CLI (serve/recommend/analyze/train/outcomes/value/quality/init/sync/home)
    ├── api/
    │   ├── app.py                   # FastAPI 앱 팩토리, StaticFiles 마운트
    │   ├── dependencies.py          # 공통 의존성 (db_manager, analysis_agent 등)
    │   └── routers/
    │       ├── recommendations.py   # GET/POST /api/recommendations
    │       ├── analysis.py          # GET/POST /api/analysis/{code}
    │       ├── watchlist.py         # CRUD /api/watchlist
    │       ├── backtest.py          # GET /api/backtest
    │       ├── market.py            # GET /api/market, GET /api/macro_context
    │       ├── models.py            # GET /api/model_health, GET/POST/DELETE /api/model_params/{name}
    │       ├── value.py             # GET /api/value_stocks (가치주 스크리닝)
    │       └── quality.py           # GET /api/quality_stocks (우량주 스크리닝)
    ├── static/
    │   ├── index.html               # Reveal.js 일일 브리핑 슬라이드
    │   ├── dashboard.html           # 인터랙티브 대시보드 (8탭)
    │   ├── js/
    │   │   ├── slides.js            # 슬라이드 동적 생성 (API fetch)
    │   │   └── dashboard.js         # 대시보드 인터랙션
    │   └── css/
    │       └── theme.css            # 공통 스타일
    └── core/
        ├── config.py                # 환경변수 및 설정 (dotenv)
        ├── constants.py             # 전역 상수 (버킷 분류 등)
        ├── data/
        │   ├── provider.py              # 주가·뉴스 데이터 수집 (FinanceDataReader + KIND API)
        │   ├── fundamental_provider.py  # DART 기반 펀더멘털 수집 (ROE·부채비율·PER·PBR)
        │   └── database.py              # SQLite CRUD (recommendations, recommendation_outcomes 등)
        ├── engine/
        │   ├── indicators.py            # 기술적 지표 계산 (RSI, MACD, BB 등)
        │   ├── features.py              # 공유 피처 추출 (trainer·prediction_model 양쪽 사용, 28개)
        │   ├── strategy.py              # 전략별 시그널 생성 (TechnicalStrategy)
        │   ├── prediction_model.py      # ML 앙상블 예측 (RF · GB · LGB · CB · XGBRanker · TCN 앙상블)
        │   ├── tcn_model.py             # TCN 딥러닝 모델 (Dilated Causal Conv1D, 선택적)
        │   ├── news_agent.py            # 뉴스 수집 + GPT 감성 분석
        │   ├── macro_news_agent.py      # 거시 뉴스 감성 분석 + 레짐 감지 (risk_on/uncertain/risk_off)
        │   ├── analysis_agent.py        # 종목 심층 분석 오케스트레이터
        │   ├── recommendation_agent.py  # 유망 종목 선정 + 추천 생성
        │   ├── value_screener.py        # 가치주 스크리닝 엔진 (Piotroski F-Score + value_score)
        │   ├── quality_screener.py      # 우량주 스크리닝 엔진 (quality_score, ROE 2개년 평균)
        │   ├── trainer.py               # ML 모델 재학습 워크플로우
        │   └── scheduler.py             # 자동화 워크플로우
        └── utils/
            ├── backtester.py        # 전략 성과 검증 엔진
            ├── notifier.py          # 텔레그램 리포트 발송
            └── outcome_tracker.py   # 추천 결과 검증 (5·10·20거래일 후 성과 기록)
models/saved/                        # 학습된 ML 모델 (.pkl) 및 파라미터 (.json)
data/storage/                        # SQLite DB 파일
train_models.py                      # ML 모델 재학습 스크립트
tests/
├── test_backtester.py               # 백테스터 단위 테스트 (pytest)
└── compat_check.py                  # Python 3.11~3.13 호환성 검증 스크립트
.github/workflows/
└── daily_analysis.yml               # GitHub Actions 자동화 스케줄러
```

## 분석 파이프라인

```
1단계  기술적 지표 → tech_score (0~100)
2단계  ML 앙상블   → ml_score (0~100)  [모델 없으면 tech_score 폴백]
       └─ 6-모델: RF · GB · LGB · CB (이진 분류) + XGBRanker (랭커) + TCN (딥러닝, 선택적)
       └─ 가중치: AUC 기반 Softmax 정규화 / 분류기+TCN 75% + 랜커 25%
3단계  뉴스 감성   → sentiment_score (-100~100)
4단계  GPT 종합    → action (BUY/HOLD/SELL), 요약, 목표가

종합 점수 (ML 활성 + 거시감성) = tech×0.35 + ml×0.35 + 종목감성×0.20 + 거시감성×0.10
종합 점수 (ML 활성, 거시감성 없음) = tech×0.40 + ml×0.35 + 종목감성×0.25
종합 점수 (ML 모델 없을 시) = tech×0.65 + 종목감성×0.35
  ※ sentiment_norm = (sentiment_score + 100) / 2  → 0~100 정규화
```

## 주요 명령어

```bash
# 패키지 설치
pip install -e .              # 개발 / git clone 환경 (editable)
pip install koreanstocks      # PyPI 전역 설치 (DB는 ~/.koreanstocks/ 에 생성)

# 초기 설정 (.env 대화형 생성 — API 키를 프롬프트로 입력)
koreanstocks init                   # 대화형 입력
koreanstocks init --non-interactive  # 빈 템플릿만 생성 (CI용)

# 데이터 홈 디렉토리 (.env, DB, 모델 저장 위치)
koreanstocks home                   # 경로 출력 (cd $(koreanstocks home) 로 이동)
koreanstocks home --open            # 파일 탐색기로 열기
koreanstocks home --setup           # 셸 alias 스니펫 출력 (~/.bashrc / ~/.zshrc)

# GitHub Actions 생성 DB 다운로드 (PyPI 설치 후 추천 데이터 즉시 사용 가능)
koreanstocks sync              # 최초 수신 또는 날짜 갱신
koreanstocks sync --force      # 로컬 DB가 있어도 강제 덮어쓰기

# 웹 대시보드 실행 (브라우저 자동 실행)
koreanstocks serve                     # http://localhost:8000/dashboard
koreanstocks serve --port 8080         # 포트 변경
koreanstocks serve --no-browser        # 브라우저 자동 실행 비활성화

# 오늘의 추천 종목 분석 (GitHub Actions용)
koreanstocks recommend
koreanstocks recommend --market KOSPI --limit 10

# 단일 종목 심층 분석
koreanstocks analyze 005930

# ML 모델 재학습
koreanstocks train
koreanstocks train --auto-tune                         # 품질 미달 모델 자동 탐색·재학습
koreanstocks train --auto-tune --reset-overrides       # override 전체 초기화 후 탐색 (드리프트 복구)
koreanstocks train --auto-tune --max-trials 20         # 랜덤 탐색 횟수 확대
koreanstocks train --auto-tune --no-save-overrides     # 이번만 적용, overrides.json 미저장
python train_models.py                                 # 직접 실행도 가능

# 추천 결과 성과 추적 (5·10·20거래일 후 실적 검증)
koreanstocks outcomes                  # 미검증 추천 결과 업데이트 + 통계 출력
koreanstocks outcomes --days 180       # 최근 180일 성과 조회
koreanstocks outcomes --no-record      # DB 업데이트 없이 통계만 출력

# 가치주 스크리닝 (펀더멘털 기반 중기 관점)
koreanstocks value                     # 기본 필터로 가치주 스크리닝 (상위 20종목)
koreanstocks value --market KOSPI --limit 10
koreanstocks value --per-max 15 --roe-min 12 --f-score-min 5

# 우량주 스크리닝 (펀더멘털 기반 장기 관점)
koreanstocks quality                   # 기본 필터로 우량주 스크리닝 (상위 20종목)
koreanstocks quality --market KOSPI --limit 10
koreanstocks quality --roe-min 15 --op-margin-min 15

# 단위 테스트 실행
pytest tests/

# Python 3.11~3.13 호환성 검증
python tests/compat_check.py
```

## 환경 변수 (`.env`)

```ini
OPENAI_API_KEY=...          # 필수: GPT-5.4-nano (감성 분석, AI 의견 생성)
TELEGRAM_BOT_TOKEN=...      # 필수: 추천 리포트 발송
TELEGRAM_CHAT_ID=...        # 필수: 수신 채팅방 ID
NAVER_CLIENT_ID=...         # 필수: 뉴스 검색 API
NAVER_CLIENT_SECRET=...
DART_API_KEY=...            # 선택: 금융감독원 공시 수집 (미설정 시 뉴스만 사용)
DB_PATH=data/storage/stock_analysis.db

# 경로 재정의 (기본값 그대로 사용 권장)
# KOREANSTOCKS_BASE_DIR=...           # 데이터 루트 강제 지정 (미설정 시 자동 탐지)
#   - editable install (pip install -e .): pyproject.toml 기준 프로젝트 루트
#   - PyPI 전역 설치: ~/.koreanstocks/ 자동 생성·사용
# KOREANSTOCKS_GITHUB_DB_URL=...      # sync 다운로드 URL (저장소 fork 시에만 변경)
```

## 코딩 규칙

- **Error Handling:** 데이터 크롤링 및 API 호출마다 try/except + 로그 필수.
- **Type Hinting:** 함수 시그니처에 타입 힌트 적극 사용.
- **Docstring:** 새 에이전트·유틸리티 함수에 docstring 작성.
- **LLM 비용:** GPT 호출 시 `max_tokens` 제한, 필요 정보만 포함한 프롬프트 유지.
- **ML 모델 경로:** 절대 경로 사용 (`pathlib.Path(__file__).parent` 기준).
- **모델-스케일러 무결성:** 모델 로드 시 반드시 대응하는 스케일러도 함께 로드.

## /techdebt 전용 아키텍처 규칙

글로벌 `/techdebt` skill이 이 섹션을 읽어 KoreanStocks 전용 검사를 추가로 수행합니다.

### 아키텍처 경계
- `src/koreanstocks/core/` 파일에 `import streamlit` 또는 `st.` 호출이 있으면 🔴 High (UI/Core 커플링 위반)
- `src/koreanstocks/core/` 파일이 `src/koreanstocks/api/`를 직접 import하면 🔴 High (역방향 의존성 위반)

### ML 모델 무결성
- 모델 파일(`.pkl`) 로드 시 대응 스케일러를 함께 로드하지 않으면 🔴 High
- 모델 경로가 하드코딩(`"models/saved/..."`)이면 🟡 Medium — `pathlib.Path(__file__).parent` 사용 필수
- `train_models.py` 실행 결과와 `models/saved/model_params/*.json` 불일치 시 🟡 Medium

### LLM 비용 리스크
- `news_agent.py`, `analysis_agent.py`, `recommendation_agent.py`의 GPT 호출에 `max_tokens` 없으면 🔴 High
- 종목 루프 안에서 GPT를 개별 호출하면 🟡 Medium (배치 처리 검토)

### 자동 수정 금지 대상 (Manual Only)
- 종합 점수 가중치 (`tech×0.35 + ml×0.35 + 종목감성×0.20 + 거시감성×0.10`) 변경
- ML 피처 목록 변경 (모델 재학습 필요)
- GitHub Actions 스케줄 변경

## 📝 변경 이력

### v0.5.6-hotfix (2026-05-12) — Auto-Tune 드리프트 방지 강화

- 🐛 `trainer.py`: Phase3 test AUC guard — 튜닝 후 `test_auc < 원본 - 0.005`이면 결과 거부·원본 복원 (`_AT_TEST_AUC_MARGIN`)
- 🐛 `trainer.py`: Phase2 depth 방향 제약 — OVERFIT 시 `max_depth/depth` 증가 시도 차단, UNDERFIT 시 감소 시도 차단 (`_DEPTH_PARAMS`)
- ✨ `trainer.py`: Stale override 경고 — OVERFIT인데 override가 depth를 기본값보다 크게 설정 시 경고 로그 출력
- ✨ `trainer.py` + `cli.py`: `--reset-overrides` 플래그 — 학습 전 `*_overrides.json` 전체 삭제 후 기본 `MODEL_CONFIGS` 복원

### v0.5.6 (2026-05-11) — 추천 성과 개선 Phase 2 + Auto-Tune + 모델 업그레이드

- ✨ `analysis_agent.py`: [N-1] RSI ≥ 65 규칙 기반 BUY→HOLD — [I-1](GPT weakness 텍스트 의존)이 실질 비활성(2/90건)임을 확인하고 수치 기반으로 대체. RSI 65~70 구간 정답률 37.8%, 중앙값 -5.25%가 전 구간 최악임을 n=234로 검증.
- ✨ `recommendation_agent.py`: [N-4] 강긍정 감성(>50) 전시장 추천 제외 — KOSDAQ도 감성 역U자형 패턴 동일 확인 (강긍정 25%, 중앙값 -15.46%)
- ✨ `recommendation_agent.py`: [N-2] 감성≥25 + RSI≥65 복합 추천 제외 — 단일 조건보다 조합 효과가 강함 (n=41, 43.9%, 평균 -2.89%)
- ✨ `recommendation_agent.py`: [N-3] risk_off 레짐 + rebound 버킷 BUY→HOLD — [M-1] 검증 완료 (n=10, 30%, 평균 -7.45%)
- ✨ `trainer.py` + `tcn_model.py` + `cli.py`: Auto-Tune 3단계 추가 — `koreanstocks train --auto-tune` (Phase1 규칙 기반 → Phase2 랜덤 탐색 → Phase3 전체 재학습)
- ✨ `constants.py`: `AUTO_TUNE_THRESHOLDS` 추가 — OVERFIT/UNDERFIT/UNSTABLE/WEAK 진단 임계값 단일 소스화
- 🔧 `config.py`: OpenAI 모델 `gpt-4o-mini` → `gpt-5.4-nano` 업그레이드 (DEFAULT_MODEL)
- 📝 `docs/6_PERFORMANCE_IMPROVEMENT.md`: Phase 2 구현 완료 기록, 사후 시뮬 결과 61.5% → 66.1% (234→171건) 추가

### v0.5.5 (2026-04-03) — 추천 성과 개선 Phase 1 완성 · 버킷 분석 문서화

- ✨ `analysis_agent.py`: GPT 과매수 경고 시 BUY→HOLD 자동 전환 (`action_override` 필드로 이유 기록)
- ✨ `recommendation_agent.py`: 품질 필터 3종 추가 — 거래량 폭증(6x+) 차단 · 급등+강감성 과열 차단 · KOSPI 황금조합 조건 강제 (정답률 35%→75%)
- ✨ `recommendation_agent.py`: 세션 내 복합점수 백분위 순위(`score_percentile`) 산출 — 절대값 집중 구간(64~92점) 변별력 보완
- ✨ `recommendation_agent.py`: 거시 레짐별 composite_score 임계값 필터링 (risk_on: 45 / uncertain: 50 / risk_off: 57)
- 📝 `docs/6_PERFORMANCE_IMPROVEMENT.md`: Phase 1 구현 완료 기록 + [M-4] 버킷 비율 동적 조정 항목 추가 (2026-04-30 이후 버킷별 n≥30 누적 후 재검토 조건 명시)

### v0.5.4 (2026-03-17) — 기술 부채 해소 · 브리핑 UI 개선 · 서버 안정성 강화

- 🔧 `quality_screener.py`: 이중 슬라이싱 버그 제거 (`get_value_candidates`가 이미 `limit` 적용)
- 🔧 `prediction_model.py`: 매직 넘버 상수화 (`_MIN_RMSE_FOR_WEIGHT` · `_DEFAULT_MODEL_WEIGHT` · `_MIN_AUC_WEIGHT`) + `_parse_calibration()` 헬퍼 추출 (모델/TCN 중복 캘리브레이션 검증 제거)
- 🔧 `trainer.py`: `_fetch_stock_base()` 공통 헬퍼 추출 — `_collect_stock_features` / `_collect_stock_tcn` 85% 중복 해소
- 🔧 `constants.py`: `FSCORE_WEIGHT(0.40)` · `VALUE_SCORE_WEIGHT(0.60)` 추출 — `value_screener` 매직 넘버 제거
- 🔧 `provider.py`: `_NAVER_SISE_URL` 상수 추가 — 3곳 URL 하드코딩 제거
- 🐛 `outcome_tracker.py`: `socket.setdefaulttimeout()` → `ThreadPoolExecutor` 격리 타임아웃 교체 + `BaseException` 래퍼 — 서버 크래시 근본 수정
- 🐛 `app.py`: `/favicon.ico` 404 → 204 No Content · `/` 라우트 `Cache-Control: no-store`
- ✨ `slides.js` v4: 마지막 슬라이드 "더 알아보기" → "종합 요약" 테이블 (종목·시그널·점수·상승여력·RSI·MACD) · 표지 대시보드 링크 제거
- ✨ `dashboard.html`: 브리핑·API 링크 `target="_blank"` — 독립 창으로 분리
- ✨ `.github/workflows`: `pre_check` 잡 추가 — 이중 실행 방지 (scheduled + workflow_dispatch 동일일 중복 차단)

### v0.5.3 (2026-03-16) — 모델 파라미터 조정 UI 프론트 구현 완성

- ✨ `dashboard.js`: 모델 파라미터 조정 UI 추가 — 신뢰도 향상 방안 대상 모델(`overfit_gap > 0.10` 또는 `cv_auc_std > 0.05`)에만 ⚙ 파라미터 조정 버튼 표시
- ✨ 슬라이더 UI: 파라미터명 + `기존: N` 표기, 슬라이더, 조정값(파란색) 2행 레이아웃 — 카드 너비 내 깨짐 없음
- ✨ 💾 저장 (`POST /api/model_params/{name}`) / 🔄 초기화 (`DELETE .../override`) 즉시 반영
- 🐛 `dashboard.html`: `?v=30` 캐시 버스팅 갱신 (파라미터 조정 UI 반영)

### v0.5.2 (2026-03-16) — 기술 부채 해소 · 상수 중앙화 · trainer 분해 · 단위 테스트 추가

- 🔧 `constants.py`: `REGIME_SCORE_THRESHOLD`, `ENSEMBLE_CLF_WEIGHT`, `ENSEMBLE_RNK_WEIGHT`, `SOFTMAX_TEMPERATURE`, `MAX_ANALYSIS_WORKERS`, `MAX_SCREEN_WORKERS` 추출 — 매직넘버 중앙화
- 🔧 `provider.py`: `fetch_macro_df()` / `fetch_market_df()` 공유 함수 추출 + `_HEADERS` 모듈 상수화 (3곳 중복 제거)
- 🔧 `trainer.py`: `train_and_save()` 255줄 → `_load_effective_configs()` / `_walk_forward_cv()` / `_train_final_model()` 3개 헬퍼 + ~80줄 오케스트레이터로 분해
- 🔧 `prediction_model.py`: 공유 fetch 함수 위임, `SOFTMAX_TEMPERATURE`·앙상블 가중치 상수 참조
- 🔧 `quality_screener.py`: O(n²) 종목 룩업 → `set_index("code")` O(1) + `MAX_SCREEN_WORKERS` 상수화
- 🔧 `fundamental_provider.py`: `calc_roe_avg()` 공유 함수 추출 (value/quality 스크리너 양쪽 사용)
- 🔧 `value.py` / `quality.py` 라우터: `Depends(get_*_screener)` 패턴 통일 (테스트 인젝션 가능)
- 🔧 `analysis_agent.py`: 지연 import 3개 → 모듈 레벨 이동 (순환 의존성 없음 확인 후 적용)
- 🐛 `backtester.py`: `test_sharpe_zero_when_no_variance` — 전략 수익률 NaN sync-back 제거 (std 왜곡 수정)
- 🐛 `features.py` / `prediction_model.py`: `market_df`·`macro_df` 중복 인덱스 방어 코드 추가
- 🐛 `sync` CLI URL `KoreanStock` → `KoreanStocks` 오타 수정
- ✨ `tests/test_core.py` 신규: `calc_composite_score`, `build_features`, `get_composite_score`, `calc_roe_avg` 단위 테스트 29개 추가
- 📝 `features.py` docstring 20→28개 피처 sync, `ML_ANALYSIS.md` 피처 설명 정확화

### v0.5.1 (2026-03-16) — 모델 파라미터 UI · 신뢰도 권장 파라미터 정확화

- ✨ `models.py` / `dashboard.js`: 모델 파라미터 조회·조정 UI — 학습 파라미터 토글, 오버라이드 저장 (`*_overrides.json`), 재학습 시 자동 병합
- ✨ `GET/POST/DELETE /api/model_params/{name}` 엔드포인트 — 파라미터 오버라이드 CRUD, 서버 측 범위 검증
- 🔧 `신뢰도 향상 방안` 카드 — 모델별 실제 파라미터명으로 조치 텍스트 정확화 (XGBoost: `min_child_weight`, LightGBM: `min_child_samples`, CatBoost: `min_data_in_leaf`)
- 🐛 `dashboard.js`: `innerHTML +=` 반복 시 DOM 재직렬화·`&&` 이중 인코딩 버그 → `createElement` + `addEventListener` 방식으로 교체
- 🐛 `trainer.py`: 학습 루프에 `*_overrides.json` 자동 merge 로직 추가 (원본 `MODEL_CONFIGS` 불변 보장)

### v0.5.0 (2026-03-13) — 거시경제 통합 · MacroNewsAgent · 대시보드 거시 UI · ML 28피처

- ✨ `macro_news_agent.py` 신규: Naver News API + GPT-4o-mini 거시 뉴스 감성 분석 + 레짐 감지 (`risk_on` / `uncertain` / `risk_off`)
- ✨ `features.py`: ML 피처 20개 → 28개 — `vix_change_5d`, `tnx_level`, `tnx_change_1m`, `yield_spread`, `nasdaq_1m`, `gold_1m`, `oil_1m`, `csi300_1m` 추가
- ✨ `prediction_model.py`: `_get_macro_df()` — 8개 심볼(VIX·S&P500·IXIC·TNX·IRX·GC·CL·CSI300) 수집 + 당일 캐싱
- ✨ `analysis_agent.py`: `macro_news_agent.get_macro_context()` 호출 → `macro_regime`, `macro_sentiment`, `macro_summary` 종목 분석 결과에 포함
- ✨ `recommendation_agent.py`: 레짐별 composite_score 임계값 적용 (`risk_on: 45` / `uncertain: 50` / `risk_off: 57`)
- ✨ `market.py`: `GET /api/macro_context` 엔드포인트 신설 — 레짐·감성·요약 반환
- ✨ `dashboard.js` (v27): 거시경제 배너 (Dashboard·AI추천 탭), 레짐 배지 (추천 목록), 거시감성 섹션 (종목 모달)
- ✨ `theme.css` (v3): `.macro-banner`, `.regime-badge` — 레짐별 색상 (risk_on=초록, risk_off=빨강, uncertain=주황)
- 🔧 `constants.py`: 종합 점수 3-케이스 (`with_ml_macro`, `with_ml`, `without_ml`), 거시감성 10% 반영
- 🐛 `trainer.py`: `_fetch_macro_data()` — `^VIX·^GSPC` 2심볼만 수집 → 8심볼 전체 수집으로 수정 (7개 피처 0.00% 중요도 버그 해결)
- 🐛 `prediction_model.py`: tz-aware 인덱스 `tz_localize(None)` TypeError 수정 → tz-aware 감지 후 `tz_convert(None)` 분기
- 🐛 `recommendation_agent.py`: 레짐 필터 후 빈 결과 복원 fallback 로직 수정
- 🐛 `macro_news_agent.py`: URL 인코딩 수정 (f-string → `params=` 사용), HTML 엔티티 디코딩 추가

### v0.4.x (2026-03-06 ~ 2026-03-12) — 가치·우량주 스크리너 · TCN 딥러닝 앙상블 · 히트맵 · 안정성 강화

- ✨ `가치주 추천` 탭 — PER·PBR·ROE·F-Score 필터 + `koreanstocks value` CLI + `GET /api/value_stocks`
- ✨ `우량주 추천` 탭 — ROE·영업이익률·YoY성장 필터 + `koreanstocks quality` CLI + `GET /api/quality_stocks`
- ✨ `tcn_model.py` 신규: Dilated Causal Conv1D TCN — 6-모델 앙상블 완성 (RF · GB · LGB · CB · XGBRanker · TCN)
- ✨ 추천 지속성 히트맵 — 7등급 체계(SS/S/A/B/Cp/C/D), 연속 추천 배지, 5단계 정렬 tiebreaker
- ✨ 추천 성과 Collapse UI, 성과 탭 자동 재시도, `target_hit` 소급 집계
- 🔧 ML 피처 17→20개 (`obv_trend`, `rsi`, `cci_pct`), Walk-Forward CV 강화, TCN 과적합 억제
- 🔧 FDR DataReader read timeout 전역 패치 — 학습 수집 hang 해결
- 🔧 pipx 환경 감지 → `pipx inject koreanstocks torch` 안내 자동 출력
- 🐛 pandas·yfinance FutureWarning 전면 제거, `SettingWithCopyWarning` 수정
- 🐛 `fundamental_provider.py` DART 재작성 — ROE·부채비율 대차대조표 직접 계산

### v0.3.x (2026-02-28 ~ 2026-03-05) — 추천 성과 추적 · 버킷 UI · pykrx 제거 · 5-모델 앙상블 · 기술 부채 해소

- ✨ 추천 성과 추적 기능 (5·10·20거래일 후 실적 검증, `outcomes` CLI 및 Web UI)
- ✨ 추천 버킷(거래량 상위/상승 모멘텀/반등 후보) 배지 UI — 대시보드·슬라이드 동시 반영
- ✨ 기본 추천 종목 수 5 → 9개 상향, 분석 설정 종목 수 자동 선택 로직
- ✨ LightGBM · CatBoost 모델 추가 → 5-모델 앙상블 (RF · GB · LGB · CB · XGBRanker)
- ✨ XGBoost 이진 분류 → XGBRanker (rank:ndcg) 교체 — 크로스섹셔널 순위 직접 최적화
- ✨ `/api/version` 엔드포인트 신설 — 대시보드가 API로 버전 동적 조회
- 🐛 `/api/market` NaN → JSON 직렬화 오류 수정
- 🐛 `prediction_model.py` `mom_accel` 공식 수정 — 학습/추론 불일치 해소
- 🐛 CLI `recommend` 명령 `--limit` 인수 누락 버그 수정
- 🔧 pykrx 완전 제거 → FinanceDataReader + KIND API (KRX 정책 변경 대응)
- 🔧 버전 단일 소스 (`__init__.py`) 구조 확립, 버킷 상수 `core/constants.py` 통합
- 🔧 atr_ratio → rolling 60일 percentile 변환, 피처 교체 (bb_position·mfi·low_52w_ratio)
- 🔧 뉴스 캐시 TTL 1시간, 감성 분석 퀀트 애널리스트 프롬프트 추가
