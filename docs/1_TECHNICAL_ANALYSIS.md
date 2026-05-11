# 기술적 분석 시스템 기술 문서

> Korean Stocks AI/ML Analysis System `v0.5.6`
> 최종 업데이트: 2026-05-11

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

> **투자 시계:** 핵심 판단 기준(SMA5·20, MACD 12/26, RSI·BB·ADX 14~20일)이 모두 단기~중단기 지표다.
> SMA120(6개월)은 계산되지만 `get_composite_score()`에서 사용되지 않아 **단기(5~60일) 관점**에 집중된다.

주요 담당 클래스:
- `src/koreanstocks/core/engine/indicators.py` → `IndicatorCalculator` (지표 계산 + 종합 점수)
- `src/koreanstocks/core/engine/strategy.py` → `TechnicalStrategy` (시그널 생성)
- `src/koreanstocks/core/utils/backtester.py` → `Backtester` (전략 성과 검증)

```
OHLCV 데이터
  → IndicatorCalculator.calculate_all()   # 14종 지표 계산 (ta + finta)
  → IndicatorCalculator.get_composite_score()  → tech_score (0~100)
  → TechnicalStrategy.generate_signals()  → 매수/매도 시그널
  → Backtester.run()                      → 수익률·MDD·샤프 지수
```

`tech_score`는 종합 점수(composite)의 **35%(ML+거시감성 활성)**, **40%(ML만 활성)**, **65%(ML 없음)** 를 차지한다.

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

### 2-9. ADX — Average Directional Index

| 지표 | 파라미터 | 설명 |
|------|---------|------|
| adx | window=14 | 추세 강도 (0~100) |
| adx_pos | window=14 | DI+ (상승 방향 지수) |
| adx_neg | window=14 | DI- (하락 방향 지수) |

### 2-10. VWAP — Volume Weighted Average Price

| 지표 | 파라미터 |
|------|---------|
| vwap | window=14 (롤링) |

### 2-11. Donchian Channel

| 지표 | 파라미터 | 설명 |
|------|---------|------|
| dc_high | window=20 | 20일 최고가 채널 |
| dc_low | window=20 | 20일 최저가 채널 |

### 2-12. CMF — Chaikin Money Flow

| 지표 | 파라미터 | 범위 |
|------|---------|------|
| cmf | window=20 | −1 ~ +1 (매도/매수 압력) |

### 2-13. MFI — Money Flow Index

| 지표 | 파라미터 | 설명 |
|------|---------|------|
| mfi | window=14 | 거래량 가중 RSI (0~100) |

### 2-14. finta 지표 (선택적)

`finta` 패키지 설치 시 활성. 미설치 시 기본값 0으로 폴백.

| 지표 | 설명 |
|------|------|
| sqzmi | Squeeze Momentum Indicator |
| vzo | Volume Zone Oscillator |
| fisher | Fisher Transform (±5 클리핑) |
| bullish_fractal | Williams Fractal (강세 프랙탈) |

### 2-15. dropna 정책

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
| `ADX DI+ > DI−` _(추세 방향 확인)_ | +3pt (최대 40pt 캡) |

> SMA60 가용 여부에 따라 MACD 배점을 조정하여 기본 합계 40pt를 유지한다.
> ADX 보너스는 `min(40, trend_score + 3)`으로 적용 — 40pt 초과 불가.

### 3-2. 모멘텀 점수 (최대 30pt)

RSI 점수는 **MACD 방향(상승/하락 추세)**에 따라 최적 구간이 달라진다.

**상승 추세 (MACD > Signal):** 강한 RSI가 추세 강도를 확인하므로 과매수 패널티 최소화.

| RSI 구간 | 점수 | 해석 |
|---------|-----|------|
| 55 ≤ RSI ≤ 75 | 30pt | 핵심 상승 구간 (최적) |
| RSI > 75 | 24pt | 강한 과매수 — 모멘텀 강함 |
| 45 ≤ RSI < 55 | 20pt | 추세 초입 |
| 35 ≤ RSI < 45 | 12pt | 추세 약화 경고 |
| RSI < 35 | 6pt | 상승 추세인데 하락 — 신뢰 저하 |

**하락/중립 추세 (MACD ≤ Signal):** 과매도 반등 구간이 최적.

| RSI 구간 | 점수 | 해석 |
|---------|-----|------|
| 35 ≤ RSI ≤ 50 | 30pt | 과매도 탈출, 반등 준비 (최적) |
| 30 ≤ RSI < 35 | 24pt | 깊은 과매도, 반등 기대 |
| RSI < 30 | 18pt | 심한 과매도, 단기 반등 가능 |
| 50 < RSI ≤ 65 | 14pt | 중립~완만한 상승 |
| 65 < RSI ≤ 75 | 8pt | 하락 추세인데 RSI 높음 |
| RSI > 75 | 4pt | 과열 경고 |

**BB 폭 신뢰도 보정 (±3pt):**

| 조건 | 조정 |
|------|------|
| bb_width / bb_mid < 3% (극단적 스퀴즈) | −3pt (방향 불확실) |
| bb_width / bb_mid > 12% (밴드 확장) | +2pt (추세 명확), max(30pt) |

### 3-3. 가격 위치 + CMF + 거래량 점수 (최대 30pt)

#### BB 위치 (최대 20pt)

볼린저 밴드 내 상대 위치 `bb_pos = (close − BB하단) / (BB상단 − BB하단)` 을 산출하고,
**MACD 방향(상승/하락추세)**에 따라 최적 구간을 달리 적용한다.

**상승 추세 (MACD > Signal):** 추세 추종 — 중상단 구간 선호

| bb_pos | 점수 |
|--------|-----|
| 0.40 ~ 0.75 | 20pt |
| 0.75 ~ 0.90 | 14pt |
| 0.20 ~ 0.40 | 11pt |
| > 0.90 | 6pt |
| 나머지 (하단 이탈) | 2pt |

**하락/중립 추세 (MACD ≤ Signal):** 반등 매수 — 중하단 구간 선호

| bb_pos | 점수 |
|--------|-----|
| 0.20 ~ 0.50 | 20pt |
| 0.50 ~ 0.70 | 14pt |
| 0.10 ~ 0.20 | 10pt |
| 0.70 ~ 0.90 | 6pt |
| 나머지 (밴드 이탈) | 2pt |

#### CMF 자금흐름 (최대 5pt)

`Chaikin Money Flow` 값으로 매수/매도 압력을 확인한다.

| CMF 값 | 점수 |
|--------|-----|
| > 0.05 | +5pt (자금 유입 확인) |
| > 0 | +3pt (약한 자금 유입) |
| ≤ 0 | 0pt |

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
| tech_score 배점 구조 | 추세 40pt / 모멘텀 30pt / 위치·CMF·거래량 30pt | 백테스트 검증 기반 |
| RSI 점수 방식 | MACD 방향 맥락별 차등 (상승/하락 추세 최적 구간 상이) | 추세 맥락 반영, 오신호 방지 |
| ADX DI 보너스 | DI+ > DI- 시 +3pt (최대 40pt 캡) | 추세 방향 확인 |
| CMF 자금흐름 | 0.05 초과 +5pt / 0 초과 +3pt | 매수 압력 확인 |
| BB 위치 점수 분기 | MACD 방향 기준 (상승 20pt 구간 상이) | 추세 맥락 반영 |
| 거래량 가점 임계 | 1.5배 (5pt) | 추세 신뢰도 확인 기준 |
| RSI 전략 기준 | 40/60 (30/70 대신) | 시그널 빈도 확보, 추세 초기 진입 |

### 알려진 한계

| 항목 | 내용 | 영향도 |
|------|------|-------|
| 단순 지표 조합 | 지표 간 상관관계 미고려. RSI·MACD 동시 과열 시 점수 과대 계상 가능 | 중간 |
| SMA120 미활용 | `calculate_all()`에서 계산하나 `get_composite_score()`에서 미사용 — 장기 추세 필터링 없음 | 낮음 |
| **단기 특화** | 지표 파라미터 최대 60일(SMA60). 반년~1년 이상 보유 목적에는 부적합 | 높음 |
| Stochastic·CCI·ATR 미활용 | 계산 후 점수 반영 없음 (CMF·ADX는 반영됨) — 향후 확장 여지 | 낮음 |
| VWAP·Donchian 미활용 | 계산하나 tech_score·ML 피처 모두 미사용 — 향후 확장 여지 | 낮음 |
| 백테스팅 무위험 이자율 0% | 샤프 지수 산출 시 무위험 이자율 미반영 → 절대값 과대 평가 | 낮음 |
| 미래 데이터 없음 | 백테스터는 1일 지연(shift)으로 lookahead bias 방지. SMA·RSI 계산 자체는 look-forward 없음 | — |

### 향후 개선 방향

| 우선순위 | 항목 | 기대 효과 |
|---------|------|---------|
| 1 | Stochastic / CCI / ATR 점수 반영 — 현재 계산만 하고 미사용 | tech_score 다양화 |
| 2 | SMA120 장기 추세 반영 — 추세 점수 조건 추가 | 장기 하락 종목 필터링 강화 |
| 3 | 백테스팅 무위험 이자율 반영 (연 3~4% 국채 기준) | 샤프 지수 현실화 |
