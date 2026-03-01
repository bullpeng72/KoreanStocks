# 기술적 분석 시스템 기술 문서

> Korean Stocks AI/ML Analysis System `v0.3.1`
> 최종 업데이트: 2026-02-28

---

## 목차

1. [개요](#1-개요)
2. [지표 계산](#2-지표-계산)
3. [종합 점수 산출](#3-종합-점수-산출)
4. [전략별 시그널](#4-전략별-시그널)
5. [백테스팅 엔진](#5-백테스팅-엔진)
6. [설계 원칙 및 한계](#6-설계-원칙-및-한계)

---

## 1. 개요

기술적 분석은 OHLCV(시가·고가·저가·종가·거래량) 데이터를 기반으로 `tech_score`를 산출하고, 전략별 매수/매도 시그널을 생성한다.

주요 담당 클래스:
- `src/koreanstocks/core/engine/indicators.py` → `IndicatorCalculator` (지표 계산 + 종합 점수)
- `src/koreanstocks/core/engine/strategy.py` → `TechnicalStrategy` (시그널 생성)
- `src/koreanstocks/core/utils/backtester.py` → `Backtester` (전략 성과 검증)

```
OHLCV 데이터
  → IndicatorCalculator.calculate_all()   # 8종 지표 계산
  → IndicatorCalculator.get_composite_score()  → tech_score (0~100)
  → TechnicalStrategy.generate_signals()  → 매수/매도 시그널
  → Backtester.run()                      → 수익률·MDD·샤프 지수
```

`tech_score`는 종합 점수(composite)의 40%(ML 모델 활성) 또는 65%(ML 없음)를 차지한다.

---

## 2. 지표 계산

담당: `IndicatorCalculator.calculate_all(df)` — `ta` 라이브러리 사용

**최소 데이터 요건:** 30행 이상 (필수 지표 기준)

### 2-1. 이동평균 (Trend)

| 지표 | 파라미터 | 활성 조건 |
|------|---------|---------|
| SMA 5 | window=5 | 항상 |
| SMA 20 | window=20 | 항상 |
| SMA 60 | window=60 | 데이터 60행 이상 |
| SMA 120 | window=120 | 데이터 120행 이상 |

### 2-2. MACD (Trend)

| 지표 | 파라미터 |
|------|---------|
| macd | 단기 12 / 장기 26 EMA 차이 |
| macd_signal | MACD의 9일 EMA |
| macd_diff | MACD − Signal (히스토그램) |

### 2-3. RSI — 모멘텀 (Momentum)

| 지표 | 파라미터 |
|------|---------|
| rsi | window=14 |

### 2-4. 볼린저 밴드 (Volatility)

| 지표 | 파라미터 |
|------|---------|
| bb_high | window=20, σ=2 상단 밴드 |
| bb_mid | 20일 이동평균 (중심선) |
| bb_low | window=20, σ=2 하단 밴드 |

### 2-5. 거래량 지표 (Volume)

| 지표 | 파라미터 | 설명 |
|------|---------|------|
| vol_sma_20 | window=20 | 거래량 20일 이동평균 |
| obv | — | On-Balance Volume |

### 2-6. 스토캐스틱 (Momentum)

| 지표 | 파라미터 |
|------|---------|
| stoch_k | window=14, smooth=3 |
| stoch_d | %K의 3일 이동평균 (시그널) |

### 2-7. CCI — Commodity Channel Index

| 지표 | 파라미터 |
|------|---------|
| cci | window=20 (고·저·종가 사용) |

### 2-8. ATR — Average True Range

| 지표 | 파라미터 |
|------|---------|
| atr | window=14 |

### 2-9. dropna 정책

NaN 제거는 **필수 지표 4개**(`rsi`, `macd`, `macd_signal`, `bb_mid`)를 기준으로만 수행한다.
SMA 60/120·OBV·Stochastic 등의 NaN으로 인해 초기 데이터 행이 과도하게 손실되는 것을 방지한다.

---

## 3. 종합 점수 산출

담당: `IndicatorCalculator.get_composite_score(df)` → `float` (0 ~ 100)

데이터 미충족(`df.empty` 또는 `rsi` 컬럼 없음) 시 기본값 **50.0** 반환.

### 3-1. 추세 점수 (최대 40pt)

| 조건 | 점수 |
|------|-----|
| `close > SMA20` | +10pt |
| `SMA5 > SMA20` | +10pt |
| `MACD > Signal` _(SMA60 있음)_ | +15pt |
| `close > SMA60` _(SMA60 있음)_ | +5pt |
| `MACD > Signal` _(SMA60 없음)_ | +20pt (MACD에 집중) |

> SMA60 가용 여부에 따라 MACD 배점을 조정하여 합계 40pt를 유지한다.

### 3-2. 모멘텀 점수 (최대 30pt)

RSI 구간에 따라 차등 배점한다. **45~65 구간(상승 추세 내 적정)이 최고점**이며, 과매수·과매도로 갈수록 감점된다.

| RSI 구간 | 점수 | 해석 |
|---------|-----|------|
| 45 ≤ RSI ≤ 65 | 30pt | 상승 추세 내 적정 구간 (최적) |
| 35 ≤ RSI < 45 | 22pt | 하락 완화, 반등 준비 구간 |
| 65 < RSI ≤ 75 | 18pt | 강한 상승 (과매수 경계) |
| 30 ≤ RSI < 35 | 16pt | 과매도 근접, 반등 기대 |
| RSI > 75 | 8pt | 강한 과매수 (오버히팅) |
| RSI < 30 | 12pt | 깊은 과매도, 반등 가능성 인정 |

### 3-3. 가격 위치 + 거래량 점수 (최대 30pt)

#### BB 위치 (최대 25pt)

볼린저 밴드 내 상대 위치 `bb_pos = (close − BB하단) / (BB상단 − BB하단)` 을 산출하고,
**MACD 방향(상승/하락추세)**에 따라 최적 구간을 달리 적용한다.

**상승 추세 (MACD > Signal):** 추세 추종 — 중상단 구간 선호

| bb_pos | 점수 |
|--------|-----|
| 0.40 ~ 0.75 | 25pt |
| 0.75 ~ 0.90 | 18pt |
| 0.20 ~ 0.40 | 14pt |
| > 0.90 | 8pt |
| 나머지 (하단 이탈) | 3pt |

**하락/중립 추세 (MACD ≤ Signal):** 반등 매수 — 중하단 구간 선호

| bb_pos | 점수 |
|--------|-----|
| 0.20 ~ 0.50 | 25pt |
| 0.50 ~ 0.70 | 18pt |
| 0.10 ~ 0.20 | 12pt |
| 0.70 ~ 0.90 | 8pt |
| 나머지 (밴드 이탈) | 3pt |

#### 거래량 확인 (최대 5pt)

`volume / vol_sma_20 ≥ 1.5` 이면 +5pt (추세 신뢰도 가점).

---

## 4. 전략별 시그널

담당: `TechnicalStrategy.generate_signals(df, strategy_type)`

시그널 값: `1` = 보유(매수), `0` = 미보유(매도/관망)
포지션 보유 로직 포함 — 새로운 시그널이 발생할 때까지 기존 포지션 유지.

### 4-1. RSI 전략

```python
RSI < 40  →  매수 (과매도 완화 기준)
RSI > 60  →  매도 (과매수 완화 기준)
```

고전적 30/70 기준보다 완화하여 시그널 빈도를 높이고 추세 초기 구간에서 진입한다.

### 4-2. MACD 전략

```python
MACD > Signal  →  골든크로스 → 매수
MACD < Signal  →  데드크로스 → 매도
```

### 4-3. COMPOSITE 전략

```python
RSI < 50  AND  MACD > Signal  →  매수   (두 조건 모두 충족)
RSI > 60  OR   MACD < Signal  →  매도   (하나라도 해당)
```

RSI 모멘텀과 MACD 추세를 동시에 만족할 때만 진입하여 오신호를 줄인다.

---

## 5. 백테스팅 엔진

담당: `Backtester.run(df, signals, initial_capital)`

전략 성과를 과거 데이터로 검증하고 주요 성과 지표를 반환한다.

### 5-1. 기본 설정

| 항목 | 값 |
|------|-----|
| 기본 초기 자본 | 10,000,000원 |
| 거래 수수료 | 0.015% (편도) |
| 거래세 | 0.18% |
| 연간 거래일 | 252일 (샤프 지수 연환산 기준) |

### 5-2. 수익률 계산 방식

```python
strategy_returns[t] = signal[t-1] × pct_change[t]  # 1일 지연 (lookahead bias 방지)
```

매매 발생 시 (`signal.diff() ≠ 0`) 해당 일의 전략 수익률에서 `수수료 + 거래세` 차감.

### 5-3. 반환 지표

| 지표 | 설명 |
|------|------|
| `total_return_pct` | 전체 기간 누적 수익률 (%) |
| `mdd_pct` | 최대 낙폭 (Maximum Drawdown, %) |
| `win_rate` | 일별 양수 수익률 발생 비율 (%) |
| `sharpe_ratio` | 샤프 지수 (연환산, 무위험 이자율 0% 가정) |
| `final_capital` | 최종 자본금 (원) |
| `daily_results` | 날짜별 close·signal·cum_returns·cum_capital |

샤프 지수: `(일평균 수익률 / 일수익률 표준편차) × √252`

---

## 6. 설계 원칙 및 한계

### 고정값 (임의 수정 금지)

| 항목 | 값 | 이유 |
|------|----|------|
| tech_score 배점 구조 | 추세 40pt / 모멘텀 30pt / 위치·거래량 30pt | 백테스트 검증 기반 |
| RSI 최고점 구간 | 45~65 (30pt) | 상승 추세 내 적정 구간, 과매수 오신호 방지 |
| BB 위치 점수 분기 | MACD 방향 기준 | 추세 맥락 반영, 단순 중심값 기준 대비 정확도 향상 |
| 거래량 가점 임계 | 1.5배 (5pt) | 추세 신뢰도 확인 기준 |
| RSI 전략 기준 | 40/60 (30/70 대신) | 시그널 빈도 확보, 추세 초기 진입 |

### 알려진 한계

| 항목 | 내용 | 영향도 |
|------|------|-------|
| 단순 지표 조합 | 지표 간 상관관계 미고려. RSI·MACD 동시 과열 시 점수 과대 계상 가능 | 중간 |
| SMA120 미활용 | `calculate_all()`에서 계산하나 `get_composite_score()`에서 미사용 | 낮음 |
| Stochastic·CCI·ATR 미활용 | 지표 계산 후 점수 반영 없음 — 향후 확장 여지 | 낮음 |
| 백테스팅 무위험 이자율 0% | 샤프 지수 산출 시 무위험 이자율 미반영 → 절대값 과대 평가 | 낮음 |
| 미래 데이터 없음 | 백테스터는 1일 지연(shift)으로 lookahead bias 방지. SMA·RSI 계산 자체는 look-forward 없음 | — |

### 향후 개선 방향

| 우선순위 | 항목 | 기대 효과 |
|---------|------|---------|
| 1 | Stochastic / CCI / ATR 점수 반영 — 현재 계산만 하고 미사용 | tech_score 다양화 |
| 2 | SMA120 장기 추세 반영 — 추세 점수 조건 추가 | 장기 하락 종목 필터링 강화 |
| 3 | 백테스팅 무위험 이자율 반영 (연 3~4% 국채 기준) | 샤프 지수 현실화 |
