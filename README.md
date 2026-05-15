# 📈 Korean Stocks AI/ML Analysis System

![version](https://img.shields.io/badge/version-0.5.7-blue)
![python](https://img.shields.io/badge/python-3.11~3.13-green)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

> **KOSPI · KOSDAQ 종목을 AI와 머신러닝으로 분석하는 자동화 투자 보조 플랫폼**

---

## 목차

1. [프로젝트 소개](#-프로젝트-소개)
2. [주요 기능](#-주요-기능)
3. [기술 스택](#-기술-스택)
4. [시스템 아키텍처](#-시스템-아키텍처)
5. [분석 파이프라인](#-분석-파이프라인)
   - [단기 AI 추천](#단기-ai-추천-파이프라인-1-2주)
   - [ML 앙상블 모델](#2단계--ml-앙상블-ml_score-0-100)
   - [가치주 스크리닝](#가치주-스크리닝-파이프라인-중기-3-6개월)
   - [우량주 스크리닝](#우량주-스크리닝-파이프라인-장기-6개월)
6. [점수 체계](#-점수-체계-해석)
7. [대시보드 메뉴](#-대시보드-메뉴-구성-8탭)
8. [실전 투자 활용 가이드](#-실전-투자-활용-가이드)
9. [설치 및 실행](#-설치-및-실행)
10. [API 엔드포인트](#-api-엔드포인트)
11. [자동화 설정](#-자동화-설정-github-actions)
12. [변경 이력](#-변경-이력)
13. [면책 조항](#-면책-조항)

---

## 🚀 프로젝트 소개

`Korean Stocks AI/ML Analysis System`은 기술적 지표 분석, 머신러닝 예측, 뉴스 감성 분석을 통합하여 한국 주식 시장의 유망 종목을 자동으로 발굴하고 리포트를 생성하는 플랫폼입니다.

매일 장 마감 후 자동으로 실행되어 KOSPI·KOSDAQ 전 종목 중 **거래량 상위 · 상승 모멘텀 · 반등 후보** 버킷으로 분류된 종목을 스크리닝하고, 심층 분석 후 텔레그램으로 결과를 전송합니다.

단기 AI 추천 외에, DART 공시 기반 펀더멘털과 **Piotroski F-Score**를 활용한 **가치주 스크리닝**(중기 3~6개월), ROE·영업이익률·재무건전성 기반의 **우량주 스크리닝**(장기 6개월+)도 지원합니다.

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| **AI 종목 추천** | 기술적 지표·ML·뉴스를 종합한 복합 점수로 유망 종목 선정 |
| **버킷 기반 선정** | 거래량 상위·상승 모멘텀·반등 후보 3개 버킷 쿼터 보장 (배지 UI 표시) |
| **날짜별 히스토리** | 과거 30일 분석 결과를 날짜 선택으로 조회 |
| **추천 지속성 히트맵** | SS~D 7등급 체계·연속/반복 배지(🔥🔄📌)·최신 점수 인라인·teal Cp 보더로 신호 신뢰도 시각화 |
| **DB 우선 조회 & 캐시** | 당일 저장된 DB 결과 우선 표시, 메뉴 이탈 후 재진입 시 세션 캐시 유지 |
| **DB 자동 동기화** | GitHub Actions 완료 후 DB를 저장소에 자동 커밋·푸시 → `koreanstocks sync` 한 번으로 최신 결과 반영 |
| **텔레그램 알림** | 종합점수 바·당일 등락률·RSI·뉴스 헤드라인·AI 강점 포함 구조화 리포트 발송 |
| **전략 백테스팅** | RSI · MACD · COMPOSITE 전략 시뮬레이션 (단순보유 비교, 초보자 해석 가이드 포함) |
| **관심 종목 관리** | Watchlist 등록 및 분석 이력 타임라인 제공 |
| **추천 성과 추적** | 5·10·20거래일 후 실제 수익률 자동 검증, 승률·목표가 달성률 통계 제공 (조회 기간: 2/3/6개월) |
| **가치주 스크리닝** | PER·PBR·ROE·부채비율·Piotroski F-Score 필터 + value_score 정렬, 당일 인메모리 캐시 |
| **우량주 스크리닝** | ROE·영업이익률·YoY성장·부채비율·PBR 필터 + quality_score 정렬, ROE 2개년 평균으로 지속성 확인 |
| **모델 신뢰도 대시보드** | ML 모델 AUC·과적합 갭·드리프트 등급·피처 중요도·재학습 권장 여부 확인 |

---

## 🛠 기술 스택

```
UI          FastAPI + Reveal.js (일일 브리핑) + Vanilla JS (인터랙티브 대시보드 8탭)
CLI         Typer (10개 명령어: serve / recommend / analyze / train / outcomes / value / quality / init / sync / home)
AI/LLM      OpenAI gpt-5.4-nano (뉴스 감성 분석, AI 종합 의견)
ML          Scikit-learn (Random Forest, Gradient Boosting) + XGBoost Ranker + LightGBM + CatBoost
            + PyTorch TCN (선택적, pip install koreanstocks[dl])
            → 6-모델 앙상블 (분류기+TCN 75% + 랜커 25%, AUC 기반 Softmax 가중치)
기술 지표    ta (RSI, MACD, BB, SMA, OBV, ADX, VWAP, CMF, MFI, Stochastic, CCI, ATR, Donchian)
             + finta (SQZMI, VZO, Fisher Transform, Williams Fractal)
ML 피처     28개 (변동성·추세강도·시장 상대강도·모멘텀·finta·거래량·거시경제 10개·극값감지)
데이터       FinanceDataReader, KIND API (KRX 전종목), Naver News API, DART Open API (선택)
             Yahoo Finance (VIX·S&P500·NASDAQ·10Y금리·장단기스프레드·금·유가·CSI300)
DB          SQLite (data/storage/stock_analysis.db)
자동화       GitHub Actions (평일 16:30 KST), Telegram Bot API
시각화       Plotly, Matplotlib, Chart.js (백테스트 차트)
언어         Python 3.11 ~ 3.13
```

---

## 🏗 시스템 아키텍처

세 가지 독립적인 분석 파이프라인: **단기 AI 추천** · **중기 가치주 스크리닝** · **장기 우량주 스크리닝**

```mermaid
graph TD
    CLI["🖥 CLI<br/>serve · recommend · analyze · train<br/>outcomes · value · quality · init · sync · home"]
    USER["👤 사용자<br/>브라우저"]

    subgraph API["⚡ FastAPI  (koreanstocks.api)"]
        RA["AI 추천 · 분석 · Watchlist<br/>백테스트 · 시장 · 모델"]
        RV["가치주 · 우량주<br/>GET /api/value_stocks<br/>GET /api/quality_stocks"]
    end

    subgraph FRONTEND["🌐 대시보드 (dashboard.html — 8탭)"]
        F1["① Dashboard<br/>시장지수 · AI추천 · 히트맵"]
        F2["② Watchlist<br/>관심종목 관리"]
        F3["③ AI 추천<br/>ML 추천 · 성과 추적"]
        F4["④ 가치주 추천<br/>F-Score · value_score"]
        F5["⑤ 우량주 추천<br/>quality_score · ROE 지속성"]
        F6["⑥ 백테스트<br/>전략 시뮬레이션"]
        F7["⑦ 모델 신뢰도<br/>AUC · 피처 중요도"]
        F8["⑧ 설정<br/>수동 실행 · 데이터 소스"]
    end

    subgraph PIPELINE_AI["🤖 단기 AI 추천 (1~2주)"]
        E1["indicators.py<br/>기술적 지표 → tech_score"]
        E2["prediction_model.py<br/>6-모델 앙상블 → ml_score<br/>RF · GB · LGB · CB · XGBRanker · TCN"]
        E3["news_agent.py<br/>뉴스 감성 → sentiment_score<br/>gpt-5.4-nano"]
        E4["analysis_agent.py<br/>4단계 심층 분석"]
        E5["recommendation_agent.py<br/>버킷 기반 종목 선정<br/>composite score 산출"]
    end

    subgraph PIPELINE_FUND["💰 펀더멘털 스크리닝"]
        V1["value_screener.py<br/>중기 가치주 (3~6개월)<br/>Piotroski F-Score + value_score"]
        Q1["quality_screener.py<br/>장기 우량주 (6개월+)<br/>quality_score · ROE 2개년 평균"]
    end

    subgraph DATA["📊 Data Layer"]
        D1["provider.py<br/>FinanceDataReader · KIND API<br/>OHLCV · 시장지수 · 종목목록"]
        D2["fundamental_provider.py<br/>DART Open API<br/>PER · PBR · ROE · 부채비율"]
        D3["database.py<br/>SQLite CRUD<br/>추천결과 · 분석이력 · 뉴스캐시"]
    end

    subgraph EXT["🌐 외부 데이터"]
        X1["FinanceDataReader / KIND API<br/>OHLCV · 종목목록"]
        X2["Naver News API<br/>종목 뉴스"]
        X3["DART Open API<br/>재무제표 공시"]
        X4["OpenAI gpt-5.4-nano<br/>감성 분석 · AI 의견"]
        X5["Yahoo Finance<br/>VIX · S&P500"]
    end

    subgraph STORAGE["💾 저장소"]
        S1[("stock_analysis.db")]
        S2["models/saved/*.pkl<br/>학습된 ML 모델"]
    end

    CLI -->|서버 기동| API
    CLI -->|직접 실행| PIPELINE_AI
    CLI -->|value / quality| PIPELINE_FUND

    USER --> FRONTEND
    FRONTEND -->|REST| API
    API --> RA --> PIPELINE_AI
    API --> RV --> PIPELINE_FUND

    PIPELINE_AI --> E1 & E2 & E3
    E1 & E2 & E3 --> E4 --> E5 --> D3

    PIPELINE_FUND --> V1 & Q1
    V1 & Q1 --> D1 & D2

    D1 --> X1
    D2 --> X3
    E3 --> X2 & X4
    E4 --> X4
    E1 & E2 --> X5

    D3 --> S1
    E2 --> S2
```

---

## 🔬 분석 파이프라인

### 단기 AI 추천 파이프라인 (1~2주)

#### 버킷 기반 후보군 선정

```mermaid
flowchart TD
    A["FinanceDataReader + KIND API<br/>KOSPI · KOSDAQ 전체 종목"] --> B["시장 필터<br/>KOSPI / KOSDAQ / ALL"]
    B --> V["🟦 거래량 상위<br/>40% 쿼터"]
    B --> M["🟩 상승 모멘텀<br/>+2%~+15%<br/>35% 쿼터"]
    B --> R["🟥 반등 후보<br/>거래량 상위 중 하락<br/>25% 쿼터"]

    V & M & R --> POOL["분석 풀 구성<br/>min(limit × 8, 80)개<br/>limit=9 → 최대 72종목"]
```

#### 종목별 심층 분석 (4단계 병렬)

```mermaid
flowchart TD
    POOL["분석 풀 (최대 80종목)"] --> PAR["병렬 분석<br/>max_workers=5 · timeout=60s"]

    PAR --> T["1단계 기술적 지표<br/>tech_score 0~100<br/>추세(40) + 모멘텀(30) + BB/CMF/거래량(30)"]
    PAR --> ML["2단계 ML 앙상블<br/>ml_score 0~100<br/>RF · GB · LGB · CB + XGBRanker<br/>28개 피처 · 101분위 캘리브레이션"]
    PAR --> N["3단계 뉴스 감성<br/>sentiment_score -100~100<br/>gpt-5.4-nano · 지수감쇠 시간가중"]
    PAR --> MN["3단계-B 거시감성<br/>macro_sentiment -100~100<br/>MacroNewsAgent · 레짐 감지"]

    T & ML & N & MN --> GPT["4단계 gpt-5.4-nano<br/>AI 종합 의견<br/>BUY/HOLD/SELL · 목표가 · 강점/약점"]
    GPT --> C["종합 점수 산출<br/>tech×0.35 + ml×0.35<br/>+ 종목감성×0.20 + 거시감성×0.10"]
    C --> Q["버킷 쿼터 기반 최종 N종목<br/>섹터 다양성 고려"]
    Q --> DB[("SQLite DB<br/>날짜별 저장")]
    Q --> TG["📱 텔레그램<br/>구조화 리포트"]
```

#### 종합 점수 공식

```mermaid
flowchart LR
    T["Tech Score<br/>0~100"] -->|"× 0.35"| SUM["종합 점수<br/>0~100"]
    ML["ML Score<br/>0~100"] -->|"× 0.35"| SUM
    N["종목 감성<br/>0~100 정규화"] -->|"× 0.20"| SUM
    MN["거시 감성<br/>0~100 정규화"] -->|"× 0.10"| SUM
    SUM --> FB1["※ 거시감성 없을 때<br/>tech×0.40 + ml×0.35 + 종목감성×0.25"]
    SUM --> FB2["※ ML 모델 없을 때<br/>tech×0.65 + sentiment_norm×0.35"]
    style FB1 fill:#ffe,stroke:#999,stroke-dasharray:5 5
    style FB2 fill:#f9f,stroke:#999,stroke-dasharray:5 5
```

> `sentiment_norm = (sentiment_score + 100) / 2`  → 0 ~ 100 정규화

> `macro_norm = (macro_sentiment_score + 100) / 2`  → 0 ~ 100 정규화

---

### 2단계 — ML 앙상블 (ml_score, 0~100)

#### 6-모델 앙상블 구조

```mermaid
flowchart LR
    subgraph FEAT["입력 피처 (28개)"]
        F1["변동성·추세강도<br/>atr_ratio · adx<br/>bb_width · bb_position"]
        F2["시장 상대강도<br/>rs_vs_mkt_3m"]
        F3["모멘텀·추세<br/>high_52w_ratio · mom_accel<br/>macd_diff · macd_slope_5d<br/>price_sma_5_ratio"]
        F4["finta 지표<br/>fisher · bullish_fractal_5d"]
        F5["거래량·강도<br/>mfi · vzo · obv_trend<br/>low_52w_ratio"]
        F6["극값감지·반전<br/>rsi · cci_pct"]
        F7["거시경제 (10개)<br/>vix_level · vix_change_5d<br/>sp500_1m · nasdaq_1m<br/>tnx_level · tnx_change_1m<br/>yield_spread<br/>gold_1m · oil_1m · csi300_1m"]
    end

    subgraph MODELS["6-모델 앙상블"]
        RF["Random Forest<br/>(분류기)"]
        GB["Gradient Boosting<br/>(분류기)"]
        LGB["LightGBM<br/>(분류기)"]
        CB["CatBoost<br/>(분류기)"]
        TCN["TCN<br/>(딥러닝, 선택적)"]
        XGB["XGBoost Ranker<br/>(rank:ndcg)"]
    end

    subgraph AGG["집계"]
        CLS["분류기+TCN 평균<br/>AUC 기반 Softmax 가중치<br/>75%"]
        RNK["랜커 점수<br/>25%"]
        CAL["101분위수 캘리브레이션<br/>→ 0~100 균등 스케일"]
    end

    FEAT --> MODELS
    RF & GB & LGB & CB & TCN --> CLS
    XGB --> RNK
    CLS & RNK --> CAL
```

**ML 학습 설정:**
- **타깃:** 10거래일 후 수익률 상위 25% = 1 / 하위 25% = 0 (중간 50% neutral zone 제외)
- **Walk-Forward CV:** VAL_STEP=10 거래일, 약 48 fold, Purging 20 거래일
- **품질 게이트:** test_AUC ≥ 0.52 통과 시에만 저장 (미통과 시 tech_score 폴백)

---

### 가치주 스크리닝 파이프라인 (중기 3~6개월)

```mermaid
flowchart TD
    A["Naver 시가총액 순위 페이지<br/>병렬 스크래핑<br/>탐색 범위: 100 / 200 / 300종목"] --> B["사전 필터<br/>PER > 0 · ROE > 0<br/>시가총액 500억 이상"]
    B --> C["DART Open API<br/>펀더멘털 수집<br/>PER · PBR · ROE · 부채비율 · 영업이익YoY<br/>※ 연도 폴백: 전년사업보고서 → 반기 → 전전년"]
    C --> D["6단계 필터<br/>PER ≤ 25 · PBR ≤ 3<br/>ROE ≥ 8% · 부채비율 ≤ 150%<br/>영업이익YoY ≥ -15% · F-Score ≥ 4"]
    D --> E["Piotroski F-Score<br/>9점 만점<br/>수익성(P1~P3) + 안전성(L1~L3) + 성장성(E1~E3)"]
    D --> F["value_score<br/>0~100점<br/>PER(25) + PBR(15) + ROE(20)<br/>부채비율(15) + 영업이익YoY(30) + 배당(10)"]
    E & F --> G["복합 정렬<br/>value_score × 0.7 + F-Score_normalized × 0.3"]
    G --> H["결과 반환<br/>당일 인메모리 캐시<br/>동일 조건 재실행 → 즉시 반환"]
```

#### Piotroski F-Score 구성 (9점)

| 구분 | 항목 | 기준 |
|------|------|------|
| **수익성 (P1~P3)** | P1 ROA | 당기순이익 / 총자산 > 0 |
| | P2 영업현금흐름 | 영업이익 > 0 |
| | P3 ROA 개선 | 전년 대비 ROA 증가 |
| **안전성 (L1~L3)** | L1 부채비율 감소 | 전년 대비 부채비율 하락 |
| | L2 유동비율 개선 | 부채비율 하락 (대리지표) |
| | L3 무상증자 없음 | PBR 정상 범위 |
| **성장성 (E1~E3)** | E1 영업이익률 개선 | 전년 대비 영업이익률 상승 |
| | E2 자산회전율 개선 | 전년 대비 매출/총자산 증가 |
| | E3 OCF > 순이익 | 영업이익YoY > 5% (대리지표) |

---

### 우량주 스크리닝 파이프라인 (장기 6개월+)

```mermaid
flowchart TD
    A["Naver 시가총액 순위 페이지<br/>탐색 범위: 100 / 200 / 300종목"] --> B["사전 필터<br/>ROE > 0 · 시가총액 500억 이상"]
    B --> C["DART Open API<br/>ROE 2개년 평균 · 영업이익률<br/>영업이익YoY · 부채비율 · 배당수익률<br/>※ 연도 폴백: 전년사업보고서 → 반기"]
    C --> D["5단계 필터<br/>ROE ≥ roe_min · 영업이익률 ≥ op_margin_min<br/>영업이익YoY ≥ yoy_min · 부채비율 ≤ debt_max<br/>PBR ≤ pbr_max"]
    D --> E["quality_score<br/>0~100점<br/>ROE(30) + 영업이익률(25)<br/>영업이익YoY(20) + 부채비율(15) + 배당(10)"]
    E --> F["ROE 2개년 평균<br/>일시적 고ROE 필터링<br/>지속 성장 기업 확인"]
    F --> G["quality_score 내림차순 정렬<br/>당일 인메모리 캐시"]
```

---

## 📊 점수 체계 해석

### Tech Score (기술적 지표 종합, 0~100)

| 점수 | 해석 |
|------|------|
| 80–100 | 매우 강세 |
| 60–79 | 강세 |
| 40–59 | 중립 |
| 0–39 | 약세 |

**세부 구성 (합계 100점)**

| 구성 | 최대 | 주요 지표 |
|------|------|-----------|
| ① 추세 | 40점 | SMA5/20/60, MACD 골든크로스, ADX DI+/DI− |
| ② 모멘텀 | 30점 | RSI × MACD 방향 맥락 보정, BB 폭 보정 |
| ③ 위치·자금흐름 | 30점 | BB 위치(20), CMF(5), 거래량 확인(5) |

> MACD 방향에 따라 RSI 최적 구간이 반전됩니다 (상승추세: 55~75 최적 / 하락추세: 35~50 최적).

---

### ML Score (머신러닝 예측, 0~100)

10거래일 후 수익률 **상위 25% 진입 확률**의 캘리브레이션 점수.

| 점수 | 해석 |
|------|------|
| 70–100 | 강한 상승 기대 (상위 25% 고확률) |
| 50–69 | 중간 이상 — 양호 |
| 30–49 | 중립~약세 |
| 0–29 | 하위권 예상 |

---

### News Sentiment Score (뉴스 감성, -100~100)

| 점수 | 해석 |
|------|------|
| 51–100 | Very Bullish (매우 긍정) |
| 1–50 | Bullish (긍정) |
| 0 | Neutral |
| -49~-1 | Bearish (부정) |
| -100~-50 | Very Bearish (매우 부정) |

---

### Value Score (가치주, 0~100)

| 항목 | 배점 | 최고점 기준 |
|------|------|------------|
| PER | 25pt | 업종 중앙값 기준 상대 평가 |
| PBR | 15pt | 낮을수록 최고 / 3.0 이상 0pt |
| ROE | 20pt | ≥ 30% 최고 (2개년 평균) |
| 부채비율 | 15pt | 낮을수록 최고 / 150% 이상 0pt |
| 영업이익YoY | 30pt | ≥ +30% 최고 / -30% 이하 0pt |
| 배당수익률 | 10pt | ≥ 3% 최고 (데이터 없으면 제외) |

**복합 정렬:** `value_score × 0.7 + (F-Score / 9 × 100) × 0.3`

| value_score | 해석 |
|-------------|------|
| 70–100 | 우수한 저평가 종목 — 중기 매수 검토 대상 |
| 50–69 | 양호 — 추가 검증 후 판단 |
| 30–49 | 보통 — 일부 지표 취약 |
| 0–29 | 미달 |

---

### Quality Score (우량주, 0~100)

| 항목 | 배점 | 최고점 기준 |
|------|------|------------|
| ROE | 30pt | ≥ 20% 최고 (2개년 평균) |
| 영업이익률 | 25pt | ≥ 20% 최고 |
| 영업이익YoY | 20pt | ≥ 30% 최고 |
| 부채비율 | 15pt | 낮을수록 최고 / 100% 이상 0pt |
| 배당수익률 | 10pt | ≥ 3% 최고 |

| quality_score | 해석 |
|---------------|------|
| 70–100 | 최우량 — 장기 핵심 보유 후보 |
| 50–69 | 양호 — 장기 투자 검토 대상 |
| 30–49 | 보통 — 추가 검증 필요 |
| 0–29 | 미달 |

---

## 🖥 대시보드 메뉴 구성 (8탭)

> **권장 브라우저: Chrome / Firefox (최신 버전)**

```mermaid
flowchart LR
    DASH["http://localhost:8000/dashboard"]

    DASH --> T1["① Dashboard<br/>시장지수 · 포트폴리오 요약<br/>날짜별 AI 추천 리포트<br/>추천 지속성 히트맵 SS~D 7등급 🔥🔄📌"]
    DASH --> T2["② Watchlist<br/>관심 종목 등록·삭제<br/>실시간 심층 분석<br/>분석 이력 타임라인"]
    DASH --> T3["③ AI 추천<br/>테마·시장별 추천 생성<br/>날짜 선택 히스토리<br/>📊 성과 추적 (2/3/6개월 기간 선택)"]
    DASH --> T4["④ 가치주 추천<br/>PER·PBR·ROE·부채비율·F-Score 필터<br/>value_score 복합 정렬<br/>탐색 범위 100/200/300종목"]
    DASH --> T5["⑤ 우량주 추천<br/>ROE·영업이익률·YoY·부채비율 필터<br/>quality_score 정렬<br/>ROE 2개년 지속성 확인"]
    DASH --> T6["⑥ 백테스트<br/>RSI / MACD / COMPOSITE<br/>단순보유 비교 차트<br/>초보자 해석 가이드"]
    DASH --> T7["⑦ 모델 신뢰도<br/>6모델 AUC · 과적합 갭<br/>드리프트 등급 · 피처 중요도<br/>재학습 권장 여부<br/>🔧 파라미터 슬라이더 조정 · 오버라이드 저장"]
    DASH --> T8["⑧ 설정<br/>수동 일일 업데이트 실행<br/>텔레그램·데이터소스 상태 확인"]
```

| 탭 | 주요 기능 | 투자 관점 |
|----|----------|-----------|
| **Dashboard** | 시장지수, AI 추천, 히트맵 | 당일 현황 파악 |
| **Watchlist** | 관심종목 관리, 분석 이력 | 지속 모니터링 |
| **AI 추천** | ML 추천 생성, 성과 추적 | 단기 1~2주 |
| **가치주 추천** | F-Score + value_score 스크리닝 | 중기 3~6개월 |
| **우량주 추천** | quality_score 스크리닝 | 장기 6개월+ |
| **백테스트** | 전략별 과거 성과 시뮬레이션 | 전략 검증 |
| **모델 신뢰도** | ML 모델 헬스체크 · 파라미터 조정 | 신호 신뢰성 판단 · 과적합 완화 |
| **설정** | 수동 실행, 환경 설정 확인 | 운영 관리 |

### 🔧 모델 파라미터 조정 (모델 신뢰도 탭)

과적합 갭이 크거나 CV 성능이 불안정한 모델에 대해 **재학습 없이 파라미터를 미리 조정**하고 저장할 수 있다.

```
① 모델 카드 하단 [파라미터 조정 ▼] 클릭
   → 현재 학습에 사용된 파라미터값 표시 (models/saved/model_params/*.json 기준)
   → 오버라이드 적용 중이면 🔧 오버라이드 적용 중 배지 표시

② 슬라이더 / 입력란으로 값 조정
   조정 가능 파라미터 (모델별):
   ┌─────────────────┬────────────────────────────────────────────────────┐
   │ CatBoost        │ depth (2~6), l2_leaf_reg (1~20), min_data_in_leaf (20~100) │
   │ LightGBM        │ max_depth (1~4), min_child_samples (50~200), reg_lambda (1~15) │
   │ XGBoost Ranker  │ max_depth (2~5), min_child_weight (15~60), reg_lambda (1~10) │
   │ Random Forest   │ max_depth (3~8), min_samples_leaf (15~60)                │
   │ Gradient Boost  │ max_depth (1~4), min_samples_leaf (15~60)                │
   └─────────────────┴────────────────────────────────────────────────────┘

③ [오버라이드 저장] 클릭
   → models/saved/model_params/{name}_overrides.json 에 저장
   → 다음 `koreanstocks train` 실행 시 학습 파라미터에 자동 병합

④ [오버라이드 초기화] 클릭
   → 오버라이드 파일 삭제, 원래 파라미터 복원
```

> **과적합 갭 완화 예시**: CatBoost val-train AUC 갭이 0.10 이상이면 `depth 3→2`, `min_data_in_leaf 40→60`으로 조정 후 재학습.

> **⚠️ Override 드리프트 주의**: override가 누적되어 depth가 기본값보다 커지면 오히려 과적합이 심해집니다. 모델 신뢰도 지표가 갑자기 악화되면 `koreanstocks train --reset-overrides` 로 전체 초기화 후 재학습하세요.

---

## 💡 실전 투자 활용 가이드

> ⚠️ 본 시스템은 **투자 보조 도구**입니다. 최종 투자 결정은 반드시 본인이 직접 판단하세요.

### 투자 시계 (Investment Horizon) 선택

```mermaid
flowchart TD
    START["투자 목적 선택"] --> S1["단기 트레이딩<br/>1~2주"]
    START --> S2["중기 투자<br/>3~6개월"]
    START --> S3["장기 투자<br/>6개월+"]

    S1 --> A1["AI 추천 탭 사용<br/>tech + ml + 종목감성 + 거시감성<br/>종합 점수 45~57+ (레짐별 임계) · BUY 신호 우선"]
    S2 --> A2["가치주 추천 탭 사용<br/>value_score 60+ · F-Score 6+<br/>PER ≤ 15 · ROE ≥ 10%"]
    S3 --> A3["우량주 추천 탭 사용<br/>quality_score 70+<br/>ROE 2개년 ≥ 15% · 영업이익률 ≥ 15%"]

    A1 & A2 & A3 --> CHECK["두 신호 이상 일치 시<br/>최우선 검토 대상"]
```

### 단기 AI 추천 활용 단계별 가이드

```
Step 1 — 스크리닝 (매일 자동)
  → 텔레그램 알림으로 오늘의 추천 9종목 확인
  → 종합 점수 상위 2~3종목을 후보로 선정

Step 2 — 지속성 확인 (신뢰도 검증)
  → 추천 지속성 히트맵에서 연속 추천 일수 확인
  → 🔥 (연속 2일+) 배지 종목은 신호 신뢰도 높음

Step 3 — 성과 데이터 확인
  → AI 추천 탭 → "추천 성과 추적" 섹션
  → 과거 추천의 5·10·20거래일 승률 · 목표가 달성률 확인

Step 4 — 심층 검증 (수동)
  → Dashboard 또는 AI 추천 탭 상세 리포트 확인
  → 강점/약점, 뉴스 원문 링크, 목표가 근거 직접 검토
  → 백테스트 탭에서 해당 전략의 과거 성과 확인

Step 5 — 최종 판단 기준
  아래 조건 중 2개 이상 충족 시 매수 검토 ✅
  ✓ 최근 5일 거래량 ≥ 20일 평균의 150%
  ✓ 52주 저점 대비 -20% 이내
  ✓ 뉴스 감성 Bullish 이상 (score > 20)
  ✓ 추천 지속성 히트맵 🔥 배지 (연속 2일+)
  ✓ ML Score ≥ 60 + Tech Score ≥ 65
```

### 강력 매수 후보 판단 기준

```
강력 매수 후보 (모든 조건 충족 시)
  ✅ Tech Score ≥ 65
  ✅ ML Score ≥ 60
  ✅ News Score > 20
  ✅ AI action = BUY
  ✅ RSI: 35~50 구간 (과매도 탈출 또는 중립 하단)
  ✅ MACD: 골든크로스 발생 또는 유지

관망 권고
  ✗ Tech < 50 이고 MACD 데드크로스 상태
  ✗ News Score < -30 (강한 악재 뉴스)
  ✗ RSI > 75 (과열 구간)

매도 검토
  ✗ AI action = SELL + Tech Score < 40
  ✗ RSI > 75 + MACD 데드크로스 동시 발생
```

### 리스크 관리 원칙

| 원칙 | 설명 |
|------|------|
| **분산 투자** | 동일 섹터에 몰리지 않도록 1~2종목만 선택 |
| **손절 기준** | 매수가 대비 7~8% 하락 시 손절 고려 |
| **비중 관리** | 단일 종목에 총 자산의 10% 이상 집중 지양 |
| **재검증** | 매수 후 3~5일 내 재분석으로 의견 변화 모니터링 |

---

## ⚙️ 설치 및 실행

### 방법 A — PyPI 설치 (권장: 분석 결과 조회 전용)

분석 실행 없이 GitHub Actions가 생성한 추천 결과를 대시보드로 조회할 때 사용합니다.

```bash
# 시스템 라이브러리 (XGBoost / LightGBM 구동에 필요)
sudo apt-get install -y libomp-dev   # Ubuntu / Debian
# brew install libomp               # macOS

pip install koreanstocks             # 기본 설치 (TCN 비활성화)
pip install "koreanstocks[dl]"       # TCN 딥러닝 앙상블 포함 (~700MB)
```

```bash
koreanstocks init    # API 키 대화형 설정
koreanstocks sync    # GitHub Actions 생성 DB 다운로드
koreanstocks serve   # http://localhost:8000/dashboard 자동 열림
```

> `.env`·DB·ML 모델은 `~/.koreanstocks/`에 저장됩니다.

#### pipx로 설치 (CLI 격리 권장)

```bash
pip install pipx && pipx ensurepath
pipx install koreanstocks          # 기본 설치 (TCN 비활성화)
```

TCN 딥러닝 앙상블을 활성화하려면 (선택적, ~700MB):

```bash
# 방법 A: 처음부터 dl extra 포함 설치 (권장)
pipx install "koreanstocks[dl]"

# 방법 B: 이미 설치한 경우 inject
pipx inject koreanstocks torch

# 방법 C: 이미 설치되어 있고 dl extra를 추가하려면 --force 재설치
pipx install "koreanstocks[dl]" --force
```

GPU(CUDA) 환경에서 torch CUDA 버전으로 교체하려면:

```bash
pipx install koreanstocks
pipx inject koreanstocks torch --index-url https://download.pytorch.org/whl/cu121
```

> **주의**: pipx는 격리 venv를 사용하므로 `pip install koreanstocks[dl]`로는 TCN을 활성화할 수 없습니다. 반드시 위 pipx 방식을 사용하세요.

---

### 방법 B — 저장소 클론 (개발 / 자체 분석 실행)

```bash
git clone https://github.com/bullpeng72/KoreanStock.git
cd KoreanStock

conda create -n stocks_env python=3.11
conda activate stocks_env

sudo apt-get install -y libomp-dev   # Ubuntu/Debian
pip install -e .                     # editable 설치 (TCN 비활성화)
pip install -e ".[dl]"               # TCN 딥러닝 앙상블 포함 (~700MB)
```

---

### API 키 설정 — `koreanstocks init`

```bash
koreanstocks init                   # 대화형 입력 (권장)
koreanstocks init --non-interactive  # 빈 템플릿 생성 (CI용)
```

#### 환경 변수 목록

```ini
# ── 필수 ──────────────────────────────────────────────────────
OPENAI_API_KEY=sk-proj-...         # gpt-5.4-nano 뉴스 감성·AI 의견
NAVER_CLIENT_ID=abc123             # Naver News API
NAVER_CLIENT_SECRET=xyz789
TELEGRAM_BOT_TOKEN=123456:ABC-...  # 추천 리포트 발송
TELEGRAM_CHAT_ID=-1001234567890

# ── 선택 ──────────────────────────────────────────────────────
DART_API_KEY=                      # 미설정 시 뉴스만으로 감성 분석

# ── 시스템 (기본값 사용 권장) ──────────────────────────────────
DB_PATH=data/storage/stock_analysis.db
# KOREANSTOCKS_BASE_DIR=           # 데이터 루트 경로 강제 지정
# KOREANSTOCKS_GITHUB_DB_URL=      # fork 시 sync URL 재정의
```

| 변수 | 발급처 | 필수 |
|------|--------|:----:|
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | ✅ |
| `NAVER_CLIENT_ID/SECRET` | [developers.naver.com](https://developers.naver.com) — 검색 API | ✅ |
| `TELEGRAM_BOT_TOKEN` | 텔레그램 [@BotFather](https://t.me/BotFather) → `/newbot` | ✅ |
| `TELEGRAM_CHAT_ID` | `api.telegram.org/bot<TOKEN>/getUpdates` | ✅ |
| `DART_API_KEY` | [opendart.fss.or.kr](https://opendart.fss.or.kr) (무료) | ☑️ |

---

### 주요 CLI 명령어

```bash
# 웹 대시보드
koreanstocks serve                     # http://localhost:8000/dashboard
koreanstocks serve --port 8080         # 포트 변경
koreanstocks serve --no-browser        # 브라우저 자동 실행 비활성화

# 일일 추천 분석 (GitHub Actions용)
koreanstocks recommend
koreanstocks recommend --market KOSPI --limit 10

# 단일 종목 심층 분석
koreanstocks analyze 005930

# ML 모델 재학습
koreanstocks train
koreanstocks train --period 2y --future-days 10
koreanstocks train --auto-tune                         # 품질 미달 모델 자동 탐색·재학습
koreanstocks train --auto-tune --reset-overrides       # override 초기화 후 기본값으로 탐색
koreanstocks train --auto-tune --max-trials 20         # 랜덤 탐색 횟수 확대
koreanstocks train --auto-tune --no-save-overrides     # 이번만 적용, overrides.json 미저장
koreanstocks train --reset-overrides                   # override만 초기화 후 기본 재학습 (드리프트 복구)

# DB 동기화 (PyPI 설치 환경)
koreanstocks sync              # 최초 수신 또는 날짜 갱신
koreanstocks sync --force      # 강제 덮어쓰기

# 추천 성과 추적
koreanstocks outcomes                  # 미검증 결과 업데이트 + 통계 출력
koreanstocks outcomes --days 180       # 최근 180일 조회
koreanstocks outcomes --no-record      # DB 업데이트 없이 통계만

# 가치주 스크리닝 (중기 3~6개월)
koreanstocks value                     # 기본 필터 (상위 20종목)
koreanstocks value --per-max 15 --roe-min 10
koreanstocks value --f-score-min 6 --candidate-limit 300

# 우량주 스크리닝 (장기 6개월+)
koreanstocks quality                   # 기본 필터 (상위 20종목)
koreanstocks quality --roe-min 15 --margin-min 15
koreanstocks quality --market KOSPI --candidate-limit 200

# 데이터 홈 디렉토리
koreanstocks home                      # 경로 출력
koreanstocks home --open               # 파일 탐색기로 열기
koreanstocks home --setup              # 셸 alias 스니펫 출력

# 테스트
pytest tests/
python tests/compat_check.py          # Python 3.11~3.13 호환성 검증
```

---

## 📡 API 엔드포인트

서버 실행 후 `/docs`에서 Swagger UI로 전체 API 문서를 확인할 수 있습니다.

| 라우터 | 엔드포인트 | 메서드 | 설명 |
|--------|-----------|--------|------|
| **market** | `/api/market` | GET | 시장 지수 (KS11/KQ11) |
| | `/api/market/trading-day` | GET | 거래일 여부 확인 |
| | `/api/market/ranking` | GET | 시장 등락 순위 |
| **recommendations** | `/api/recommendations` | GET | 날짜별 추천 목록 |
| | `/api/recommendations/run` | POST | 추천 분석 실행 |
| | `/api/recommendations/history` | GET | 30일 히스토리 |
| | `/api/recommendations/outcomes` | GET | 성과 추적 통계 |
| **analysis** | `/api/analysis/{code}` | GET/POST | 종목 심층 분석 |
| | `/api/analysis/{code}/history` | GET | 분석 이력 타임라인 |
| **watchlist** | `/api/watchlist` | GET/POST | 관심 종목 조회/등록 |
| | `/api/watchlist/{code}` | DELETE | 관심 종목 삭제 |
| **backtest** | `/api/backtest` | GET | 전략 백테스팅 |
| **value** | `/api/value_stocks` | GET | 가치주 스크리닝 결과 |
| | `/api/value_stocks/filters` | GET | 필터 기본값 |
| **quality** | `/api/quality_stocks` | GET | 우량주 스크리닝 결과 |
| | `/api/quality_stocks/filters` | GET | 필터 기본값 |
| **models** | `/api/model_health` | GET | ML 모델 헬스체크 |
| | `/api/model_params/{name}` | GET | 학습 파라미터 + 오버라이드 조회 |
| | `/api/model_params/{name}` | POST | 파라미터 오버라이드 저장 |
| | `/api/model_params/{name}/override` | DELETE | 오버라이드 초기화 |
| | `/api/macro_context` | GET | 거시경제 레짐·감성·요약 |
| **version** | `/api/version` | GET | API 버전 정보 |

---

## 🤖 자동화 설정 (GitHub Actions)

**실행 시점:** 평일 오후 16:30 KST (UTC 07:30) — 장 마감 후 자동 실행

```mermaid
flowchart TD
    TRIGGER["⏰ 16:30 KST 평일 자동 실행<br/>(또는 수동 workflow_dispatch)"]
    TRIGGER --> TRADING{"한국 증시 거래일?"}

    TRADING -- 휴장 --> SKIP["📅 분석 건너뜀 (알림 없음)"]

    TRADING -- 거래일 --> OUTCOME["지난 추천 성과 기록<br/>5·10·20거래일 후 수익률 집계<br/>→ 텔레그램 성과 리포트"]
    OUTCOME --> STOCKLIST["KOSPI + KOSDAQ 전체 종목 갱신"]
    STOCKLIST --> BUCKETS["버킷 기반 후보군 선정<br/>거래량 상위 / 상승 모멘텀 / 반등 후보"]
    BUCKETS --> ANALYSIS["심층 분석 (병렬)<br/>기술 + ML + 뉴스 + GPT"]
    ANALYSIS --> SELECT["종합 점수 상위 9종목 선정<br/>버킷 쿼터 + 섹터 다양성"]
    SELECT --> SAVE[("SQLite DB 저장")]
    SAVE --> ARTIFACT["GitHub Artifact 백업 (90일)"]
    SAVE --> COMMIT["저장소에 DB 커밋·푸시"]
    COMMIT --> NOTIFY["📱 텔레그램 추천 리포트"]

    COMMIT --> SYNC1["git clone 환경 → git pull"]
    COMMIT --> SYNC2["PyPI 설치 → koreanstocks sync"]
```

**GitHub Secrets 등록 (Settings > Secrets and variables > Actions):**

```
OPENAI_API_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
NAVER_CLIENT_ID
NAVER_CLIENT_SECRET
DART_API_KEY          (선택)
```

---

## 📁 프로젝트 구조

```
KoreanStocks/
├── pyproject.toml                       # pip 빌드 설정 (koreanstocks CLI 진입점)
├── requirements.txt                     # 개발/테스트 전용 (pytest 등)
├── train_models.py                      # ML 모델 재학습 스크립트
├── src/
│   └── koreanstocks/
│       ├── __init__.py                  # VERSION = "0.5.7"
│       ├── cli.py                       # Typer CLI (10개 명령어)
│       ├── api/
│       │   ├── app.py                   # FastAPI 앱 팩토리
│       │   ├── dependencies.py          # 공통 의존성
│       │   └── routers/
│       │       ├── recommendations.py   # AI 추천 · 성과 추적
│       │       ├── analysis.py          # 종목 심층 분석
│       │       ├── watchlist.py         # 관심 종목 CRUD
│       │       ├── backtest.py          # 전략 백테스팅
│       │       ├── market.py            # 시장 지수 · 거래일
│       │       ├── models.py            # ML 모델 헬스체크
│       │       ├── value.py             # 가치주 스크리닝
│       │       └── quality.py           # 우량주 스크리닝
│       ├── static/
│       │   ├── index.html               # Reveal.js 일일 브리핑 슬라이드
│       │   ├── dashboard.html           # 인터랙티브 대시보드 (8탭)
│       │   ├── js/
│       │   │   ├── slides.js
│       │   │   └── dashboard.js
│       │   └── css/theme.css
│       └── core/
│           ├── config.py                # 환경변수 및 설정 (dotenv)
│           ├── constants.py             # 버킷 상수 등 공유 상수
│           ├── data/
│           │   ├── provider.py              # 주가 · 종목목록 수집
│           │   ├── fundamental_provider.py  # DART 펀더멘털 수집
│           │   └── database.py              # SQLite CRUD
│           ├── engine/
│           │   ├── indicators.py            # 기술적 지표 계산
│           │   ├── features.py              # ML 피처 추출 (28개, 공유)
│           │   ├── strategy.py              # 전략별 시그널 생성
│           │   ├── prediction_model.py      # 6-모델 앙상블 추론 (트리 5 + TCN)
│           │   ├── tcn_model.py             # TCN 딥러닝 모델 (Dilated Causal Conv1D, 선택적)
│           │   ├── news_agent.py            # 뉴스 수집 + gpt-5.4-nano 감성
│           │   ├── macro_news_agent.py      # 거시 뉴스 감성 + 레짐 감지 (risk_on/uncertain/risk_off)
│           │   ├── analysis_agent.py        # 종목 심층 분석 오케스트레이터
│           │   ├── recommendation_agent.py  # 버킷 기반 추천 생성
│           │   ├── value_screener.py        # 가치주 스크리닝 (F-Score + value_score)
│           │   ├── quality_screener.py      # 우량주 스크리닝 (quality_score)
│           │   ├── trainer.py               # ML 모델 재학습 워크플로우
│           │   └── scheduler.py             # 자동화 워크플로우
│           └── utils/
│               ├── backtester.py        # 전략 성과 검증
│               ├── notifier.py          # 텔레그램 리포트
│               └── outcome_tracker.py   # 추천 결과 성과 추적
├── models/saved/                        # 학습된 ML 모델 (.pkl) · 파라미터 (.json)
├── data/storage/                        # SQLite DB 파일
├── docs/
│   ├── 1_TECHNICAL_ANALYSIS.md          # 기술적 분석 시스템 기술 문서
│   ├── 2_ML_ANALYSIS.md                 # ML 앙상블 시스템 기술 문서
│   ├── 3_NEWS_ANALYSIS.md               # 뉴스 감성 분석 시스템 기술 문서
│   ├── 4_VALUE_SCREENING.md             # 가치주 스크리닝 기술 문서
│   ├── 5_QUALITY_SCREENING.md           # 우량주 스크리닝 기술 문서
│   └── 6_PERFORMANCE_IMPROVEMENT.md     # 추천 성과 분석 및 개선 계획
├── tests/
│   ├── test_backtester.py               # 백테스터 단위 테스트
│   └── compat_check.py                  # Python 3.11~3.13 호환성 검증
└── .github/workflows/
    └── daily_analysis.yml               # GitHub Actions 스케줄러
```

---

## 📝 변경 이력

### v0.5.7 (2026-05-15) — 세그폴트 수정 · GPT API 호환

- 🐛 `prediction_model.py`: torch+LightGBM OpenMP 충돌 세그폴트 수정 — `tcn_model` 지연 임포트 전환 (Python 3.13 + PyTorch 2.12 + LightGBM 4.6 환경에서 `koreanstocks serve` 즉시 크래시)
- 🐛 `analysis_agent.py`: GPT 응답 JSON 잘림 수정
- 🔧 `analysis_agent.py`: `max_tokens` → `max_completion_tokens` 전환 (gpt-5.4-nano 호환)

### v0.5.6-hotfix (2026-05-12) — Auto-Tune 드리프트 방지 강화

- 🐛 `trainer.py`: Phase3 test AUC guard — 튜닝 후 `test_auc < 원본 - 0.005`이면 결과 거부·원본 모델 복원 (`_AT_TEST_AUC_MARGIN = 0.005`)
- 🐛 `trainer.py`: Phase2 depth 방향 제약 — OVERFIT 진단 시 `max_depth/depth` 증가 시도 자동 차단 (이번 사고의 직접 원인)
- ✨ `trainer.py`: Stale override 경고 — OVERFIT인데 override가 depth를 기본값보다 크게 설정 시 `train --reset-overrides` 권장 로그 출력
- ✨ `trainer.py` + `cli.py`: `--reset-overrides` 플래그 추가 — 학습 전 모든 `*_overrides.json` 삭제 후 기본 `MODEL_CONFIGS`로 복원
- ✨ `trainer.py`: `_DEPTH_PARAMS` 상수 추가 — 모델별 depth 파라미터명 단일 소스화

### v0.5.6 (2026-05-11) — 추천 성과 개선 Phase 2 + Auto-Tune + 모델 업그레이드

- ✨ `analysis_agent.py`: [N-1] RSI ≥ 65 규칙 기반 BUY→HOLD (n=234 검증, RSI 65~70 구간 정답률 37.8%)
- ✨ `recommendation_agent.py`: [N-2] 감성≥25 + RSI≥65 복합 제외 · [N-3] risk_off+rebound BUY→HOLD · [N-4] 강긍정 감성(>50) 전시장 제외
- ✨ `trainer.py` + `tcn_model.py` + `cli.py`: Auto-Tune 3단계 — `koreanstocks train --auto-tune` (Phase1 규칙·Phase2 랜덤·Phase3 재학습)
- 🔧 `config.py`: OpenAI 모델 `gpt-4o-mini` → `gpt-5.4-nano` 업그레이드
- 📝 `docs/6_PERFORMANCE_IMPROVEMENT.md`: 사후 시뮬 결과 61.5% → 66.1% (234→171건) 추가

### v0.5.5 (2026-04-03) — 추천 성과 개선 Phase 1 완성 · 버킷 분석 문서화

- ✨ `analysis_agent.py`: GPT 과매수 경고 시 BUY→HOLD 자동 전환 (action_override 필드 기록)
- ✨ `recommendation_agent.py`: 품질 필터 3종 — 거래량 6x+ 차단 · 급등+강감성 과열 차단 · KOSPI 황금조합 강제 적용
- ✨ `recommendation_agent.py`: 세션 내 복합점수 백분위 순위(score_percentile) 산출
- ✨ `recommendation_agent.py`: 거시 레짐별 composite_score 임계값 필터링
- 📝 `docs/6_PERFORMANCE_IMPROVEMENT.md`: Phase 1 완료 기록 + [M-4] 버킷 비율 동적 조정 분석 추가

### v0.5.4 (2026-03-17) — 기술 부채 해소 · 브리핑 UI 개선 · 서버 안정성 강화

- 🔧 기술 부채 해소: `quality_screener` 이중 슬라이싱 버그 · `prediction_model` 매직 넘버 상수화 + `_parse_calibration()` 헬퍼 · `trainer` `_fetch_stock_base()` 공통 헬퍼 · `constants` 가중치 상수화 · `provider` URL 상수화
- 🐛 `outcome_tracker`: `socket.setdefaulttimeout()` → `ThreadPoolExecutor` 격리 타임아웃 + `BaseException` 래퍼 — `/api/macro_context` 이후 서버 크래시 근본 수정
- 🐛 `app.py`: `/favicon.ico` 404 → 204 · `Cache-Control: no-store`
- ✨ `slides.js` v4: 마지막 슬라이드 "종합 요약" 테이블 (종목·시그널·점수·상승여력·RSI·MACD) · 표지 대시보드 링크 제거
- ✨ `dashboard.html`: 브리핑·API 링크 새 창 분리 (`target="_blank"`)
- ✨ GitHub Actions `pre_check` 잡: 동일일 이중 실행 방지

### v0.5.3 (2026-03-16) — 모델 파라미터 조정 UI 프론트 구현 완성

- ✨ 신뢰도 향상 방안 대상 모델에만 ⚙ 파라미터 조정 버튼 표시 (`overfit_gap > 0.10` 또는 `cv_auc_std > 0.05`)
- ✨ 슬라이더 2행 레이아웃 — 파라미터명 + `기존: N` + 조정값 표기, 카드 내 오버플로 수정
- ✨ 💾 저장 / 🔄 초기화 즉시 반영 (`POST/DELETE /api/model_params/{name}`)

### v0.5.2 (2026-03-16) — 기술 부채 해소 · 상수 중앙화 · 단위 테스트 추가

- 🔧 매직넘버 `constants.py` 중앙화, `trainer.py` 분해, `quality_screener.py` O(n²)→O(1) 최적화
- 🐛 Sharpe 계산 왜곡·중복 인덱스 방어·`sync` URL 오타 수정
- ✨ `tests/test_core.py` 단위 테스트 29개 추가

### v0.5.1 (2026-03-16) — 모델 파라미터 API · 신뢰도 향상 방안 카드

- ✨ `GET/POST/DELETE /api/model_params/{name}` — 파라미터 오버라이드 CRUD, 서버 측 범위 검증
- ✨ 신뢰도 향상 방안 카드 — 모델별 구체적 조치 텍스트 (과적합 갭·레짐 갭·CV 불안정)
- 🐛 `dashboard.js` DOM 재직렬화 버그 수정, `trainer.py` overrides 자동 merge 추가

### v0.5.0 (2026-03-13) — 거시경제 통합 · ML 28피처

- ✨ `macro_news_agent.py`: 거시 뉴스 감성 + 레짐 감지 (`risk_on` / `uncertain` / `risk_off`)
- ✨ ML 피처 20 → 28개 (VIX·금리·나스닥·금·원유·CSI300 추가), 종합 점수 거시감성 10% 반영
- ✨ `GET /api/macro_context`, 대시보드 레짐 배너·배지 UI 추가
- 🐛 `trainer._fetch_macro_data()` 2심볼 → 8심볼 (피처 중요도 0% 버그 해결)

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

### v0.3.x (2026-02-28 ~ 2026-03-05) — 추천 성과 추적 · 5-모델 앙상블 · pykrx 제거

- ✨ 추천 성과 추적 (5·10·20거래일 후 실적 검증, `outcomes` CLI 및 Web UI)
- ✨ 버킷 배지 UI (거래량 상위/상승 모멘텀/반등 후보) — 대시보드·슬라이드 동시 반영
- ✨ LightGBM · CatBoost 추가 → 5-모델 앙상블
- ✨ XGBoost 이진 분류 → XGBRanker (rank:ndcg) 교체
- ✨ `/api/version` 엔드포인트 신설
- 🔧 pykrx 완전 제거 → FinanceDataReader + KIND API

---

## ⚠️ 면책 조항

본 소프트웨어는 **교육 및 정보 제공 목적**으로만 제작되었습니다.

- 본 시스템의 분석 결과는 투자 권유 또는 금융 조언이 아닙니다.
- AI 및 ML 모델의 예측은 미래 수익을 보장하지 않습니다.
- 주식 투자에는 원금 손실의 위험이 있습니다.
- 최종 투자 결정과 그에 따른 손익은 전적으로 투자자 본인에게 있습니다.

---

## 📄 라이선스

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.

---

*(C) 2026. All rights reserved.*
