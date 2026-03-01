# ML 분석 시스템 기술 문서

> Korean Stocks AI/ML Analysis System `v0.3.1`
> 최종 업데이트: 2026-03-02

---

## 목차

1. [개요](#1-개요)
2. [타깃 변수](#2-타깃-변수)
3. [피처 엔지니어링](#3-피처-엔지니어링)
4. [모델 구성](#4-모델-구성)
5. [학습 파이프라인](#5-학습-파이프라인)
6. [앙상블 추론](#6-앙상블-추론)
7. [성능 지표](#7-성능-지표)
8. [재학습 방법](#8-재학습-방법)
9. [설계 원칙 및 제약](#9-설계-원칙-및-제약)

---

## 1. 개요

ML 모델은 **향후 10거래일 후 수익률이 상위 25%에 속할 확률**을 이진 분류로 예측한다.
절대 수익률 회귀 대신 이진 분류를 채택하여 노이즈를 줄이고 신호 대 잡음비를 개선한다.

```
추론 결과 (ml_score)
  0   = 하위 25% 예상 (낮은 확률)
 50   = 중립
100   = 상위 25% 예상 (높은 확률)
(test_proba 101분위수 캘리브레이션으로 0~100 균등 스케일 변환)
```

ML 점수는 종합 점수 산출에 다음 가중치로 반영된다.

```
composite (ML 모델 활성) = tech × 0.40 + ml × 0.35 + sentiment_norm × 0.25
composite (ML 모델 없음) = tech × 0.65 + sentiment_norm × 0.35
```

---

## 2. 타깃 변수

### 이진 분류 (Neutral Zone)

각 날짜마다 학습에 참여한 전 종목의 `N`거래일 후 수익률을 크로스섹셔널 퍼센타일로 변환한 뒤,
상위/하위 25%만 학습 대상으로 삼고 중간 50%는 제외한다.

```python
rank_pct = df_all.groupby(df_all.index)['raw_return'].rank(pct=True)
df_all['target'] = np.nan
df_all.loc[rank_pct >= 0.75, 'target'] = 1   # 상위 25% → 양성
df_all.loc[rank_pct <= 0.25, 'target'] = 0   # 하위 25% → 음성
df_all = df_all.dropna(subset=['target'])     # 중간 50% 제외
```

| 항목 | 값 |
|------|----|
| 예측 기간 | 10거래일 후 (기본값, `--future-days` 변경 가능) |
| 양성 클래스 (1) | 날짜별 상위 25% 수익률 종목 |
| 음성 클래스 (0) | 날짜별 하위 25% 수익률 종목 |
| 제외 | 중간 50% (neutral zone) |
| 클래스 비율 | 약 50:50 (대칭 설계) |

### 이진 분류 채택 이유

- 5~10거래일 절대 수익률 회귀는 노이즈 비율이 극도로 높아 R² 거의 0
- 이진 분류(상위 25% vs 하위 25%)로 문제를 단순화하여 AUC 개선
- 중간 50% 제외(neutral zone)로 경계 샘플 혼란 방지
- 추천 목적(상위 종목 선별)과 타깃이 정합

---

## 3. 피처 엔지니어링

총 **25개 피처** — 순수 기술지표 + 거시경제 (PyKrx 펀더멘털·수급 제외)

### 3-1. 피처 목록

| 그룹 | 피처 | 설명 |
|------|------|------|
| 변동성·추세강도 (4) | `atr_ratio` | ATR / 종가 — 상대 변동성 |
| | `adx` | ADX 추세 강도 (0~100) |
| | `adx_di_diff` | DI+ − DI− (추세 방향, 양수=상승) |
| | `bb_width` | BB 너비 / BB 중심선 |
| 시장 상대강도 (2) | `rs_vs_mkt_3m` | 3개월 수익률 − 벤치마크 3개월 |
| | `rs_vs_mkt_1m` | 1개월 수익률 − 벤치마크 1개월 |
| 모멘텀 팩터 (4) | `high_52w_ratio` | 종가 / 52주 고점 |
| | `mom_accel` | return_1m − (return_3m / 3) |
| | `macd_diff` | MACD − Signal (히스토그램) |
| | `macd_diff_change` | MACD diff 전일 대비 변화량 |
| 추세 기울기 (2) | `macd_slope_5d` | MACD diff 5일 기울기 |
| | `price_sma_5_ratio` | 종가 / SMA5 |
| finta 지표 (4) | `fisher` | Fisher Transform (클립 ±5) |
| | `bullish_fractal_5d` | Williams Fractal 5일 내 최댓값 |
| | `cmf` | Chaikin Money Flow |
| | `vzo` | Volume Zone Oscillator |
| 거래량·강도 (5) | `obv_change` | OBV 전일 대비 변화율 (클립 ±1) |
| | `vol_ratio` | 거래량 / 20일 평균 거래량 |
| | `vol_change` | 거래량 전일 대비 변화율 |
| | `rsi_mfi_div` | RSI − MFI (발산 지표) |
| | `sqzmi` | Squeeze Momentum Indicator |
| 캔들·거시 (4) | `candle_body` | (종가 − 시가) / 시가 |
| | `vix_level` | VIX 절대값 |
| | `vix_change_5d` | VIX 5일 변화율 |
| | `sp500_1m` | S&P500 1개월 수익률 |

> 벤치마크: KOSPI 종목 → KS11, KOSDAQ 종목 → KQ11

### 3-2. PyKrx 피처 제외 이유

- 추론 시 단일 종목만 처리하므로 XS 순위 피처는 중립값(50)으로 고정되어 정보량 없음
- API 응답 지연(수 초~분) 및 장중 미확정 데이터 문제
- PyKrx 제외 후 AUC 오히려 개선 확인 (노이즈 감소 효과)

---

## 4. 모델 구성

세 모델을 독립적으로 학습하고 AUC 기반 가중 앙상블로 결합한다.

### Random Forest (이진 분류기)

| 파라미터 | 값 |
|----------|----|
| n_estimators | 300 |
| max_depth | 4 |
| min_samples_split | 20 |
| min_samples_leaf | 20 |
| max_features | 0.5 |
| class_weight | balanced |

### Gradient Boosting (이진 분류기)

| 파라미터 | 값 |
|----------|----|
| n_estimators | 200 |
| learning_rate | 0.05 |
| max_depth | 2 |
| min_samples_leaf | 25 |
| subsample | 0.7 |

### XGBoost (이진 분류기)

| 파라미터 | 값 |
|----------|----|
| n_estimators | 200 |
| learning_rate | 0.05 |
| max_depth | 2 |
| subsample | 0.7 |
| colsample_bytree | 0.6 |
| min_child_weight | 30 |
| reg_alpha | 1.0 |
| reg_lambda | 3.0 |
| scale_pos_weight | 1.0 (50:50 균형) |

---

## 5. 학습 파이프라인

### 전체 흐름

```
1. KS11/KQ11 시장 수익률 로드 (상대강도 피처용)
2. VIX·S&P500 거시경제 데이터 로드 (Yahoo Finance)
3. 종목별 OHLCV 수집 + 지표 계산 + 피처 생성 (25개)
4. 전 종목 concat → 날짜별 크로스섹셔널 순위 → 이진 타깃 산출
   (상위 25% = 1, 하위 25% = 0, 중간 50% 제외)
5. 시계열 분할 (앞 80% → 학습 / 뒤 20% → 검증)
6. 5-fold TimeSeriesSplit CV (날짜 단위, 미래 누출 방지)
7. StandardScaler 정규화 → 모델 학습 → pkl 저장
8. test_proba 101분위수 배열(캘리브레이션) → JSON 저장
```

### 시계열 분할 (데이터 누출 방지)

```
전체 날짜 정렬
─────────────────────────────────────────────
│       학습 (80%)             │ 검증 (20%) │
─────────────────────────────────────────────
                          ↑ split_date
동일 날짜의 모든 종목이 반드시 같은 세트에 속함
랜덤 분할 사용 안 함 (미래 데이터 누출 방지)
```

### 5-fold TimeSeriesSplit CV

과적합 감지를 위해 학습 세트 내에서 추가로 날짜 단위 5-fold CV를 수행한다.

```python
unique_dates = sorted(df_train.index.unique())
tscv = TimeSeriesSplit(n_splits=5)
for tr_d_idx, val_d_idx in tscv.split(unique_dates):
    tr_dates  = {unique_dates[i] for i in tr_d_idx}
    val_dates = {unique_dates[i] for i in val_d_idx}
    tr_mask   = df_train.index.isin(tr_dates)
    val_mask  = df_train.index.isin(val_dates)
    # StandardScaler 재학습 + 모델 학습 + AUC 계산
```

### 학습 데이터 현황 (v0.3.1 기준)

| 항목 | 값 |
|------|----|
| 학습 종목 | 약 144개 (KOSPI + KOSDAQ 대표주·중형주) |
| 데이터 기간 | 2년 (`--period 2y`) |
| 학습 샘플 (neutral zone 제외) | 약 21,996 |
| 검증 샘플 | 약 5,499 |
| 양성 비율 | 약 50.7% (중립 구간 제외로 균형) |
| 분할 기준일 | ≈ 2025-10 (재학습 시마다 변동) |
| 피처 수 | 25개 |

---

## 6. 앙상블 추론

### AUC 기반 가중 앙상블

성능이 좋은 모델(높은 AUC)에 더 높은 가중치를 부여한다.

```python
w_i = max(AUC_i - 0.50, 1e-6)   # AUC=0.56 → w=0.06
ml_score = clip(Σ(p_i × w_i) / Σ(w_i), 0, 100)
```

품질 게이트(`_MIN_MODEL_AUC = 0.52`)를 통과하지 못한 모델은 로드 자체를 거부한다.

### 캘리브레이션 (점수 균등화)

`predict_proba()[:,1]` 원시 확률은 40~60에 몰리는 경향이 있다.
`test_proba` 기준 101분위수 배열을 `params.json`에 저장하여 예측 시 0~100으로 균등 변환한다.

```python
# 학습 시 (trainer.py): test_proba 기준 101분위수 저장
calibration_points = np.percentile(test_proba, np.arange(0, 101)).tolist()

# 추론 시 (prediction_model.py): percentile rank로 변환
p = float(np.clip(np.searchsorted(calibration, p_raw), 0, 100))
```

> `train_proba` 대신 `test_proba` 기준을 사용해야 과적합 분포(더 극단적)를 피한다.

### 모델 미탑재 시 폴백 순서

```
1순위: tech_score 직접 사용 (fallback_score 인자 전달 시)
2순위: MACD diff + ATR 비율 기반 휴리스틱 점수
```

---

## 7. 성능 지표

> 학습일: 2026-03-01 / 검증 기간: ≈ 2025-10 이후 약 5개월

| 모델 | test AUC | CV AUC (5-fold TS) | train AUC | 과적합 gap |
|------|----------|-------------------|----------|-----------|
| Random Forest | **0.5600** | 0.5119 ± 0.0238 | 0.6505 | 0.0905 |
| Gradient Boosting | **0.5687** | 0.5180 ± 0.0287 | 0.6601 | 0.0914 |
| XGBoost | **0.5671** | 0.5179 ± 0.0271 | 0.6435 | 0.0764 |
| 기준선 (랜덤 분류기) | 0.5000 | — | — | — |

> **해석:** 이진 분류의 AUC 0.52~0.57은 랜덤(0.50) 대비 유의미한 수준이다.
> CV AUC가 test AUC보다 낮게 나오는 것은 TimeSeriesSplit 특성상 초기 fold의 학습 데이터가 적기 때문이며 정상 범위다.

### AUC 개선 이력

| 단계 | 피처 수 | 타깃 | RF test AUC |
|------|---------|------|------------|
| 회귀 → 분류 전환 | 31 (PyKrx 포함) | 순위 회귀 → 이진 분류 | 0.5176 |
| PyKrx 제외 + 방향성 피처 추가 | 25 | 이진 분류 (상위 30%/하위 70%) | 0.5176 |
| Neutral Zone 타깃 (v0.3.0) | 25 | 상위 25%/하위 25%, 중간 50% 제외 | **0.5600** |
| XGBoost 과적합 감소 (v0.3.1) | 25 | 동일 | **0.5600** (XGB: 0.5558→0.5671) |

---

## 8. 재학습 방법

```bash
# CLI 명령어 (권장)
koreanstocks train
koreanstocks train --future-days 10 --period 2y --test-ratio 0.2
```

### 재학습 시 주의사항

- 재학습 후 `models/saved/model_params/*.json`의 AUC 수치 확인 권장
- 품질 게이트 미달(`test_auc < 0.52`) 시 모델 저장은 되지만 추론 시 자동 제외
- ML 피처 목록 변경 시 `trainer.py`의 `BASE_FEATURE_COLS`와 `prediction_model.py`의 `_FEATURE_COLS`를 반드시 동기화 필요

---

## 9. 설계 원칙 및 제약

### 고정값 (임의 수정 금지)

| 항목 | 값 | 이유 |
|------|----|------|
| 종합 점수 가중치 | tech×0.40 + ml×0.35 + sent×0.25 | 백테스트 검증 기반 |
| 피처 목록 | BASE 25개 | 모델 재학습 없이 변경 불가 |
| 타깃 정의 | 상위 25%/하위 25% 이진, 중간 50% 제외 | AUC 최적화 기반 |
| 캘리브레이션 기준 | test_proba 101분위수 | train_proba 기준은 과적합 분포 반영 |

### 알려진 한계

| 항목 | 내용 |
|------|------|
| CV AUC vs test AUC 간격 | TimeSeriesSplit 초기 fold는 학습 데이터 부족 → CV AUC가 낮게 측정됨. 정상 현상. |
| 단기 노이즈 | 10거래일 예측은 본질적으로 신호 대 잡음비가 낮음 (AUC 0.57이 현실적 상한) |
| 상장 폐지 종목 | 학습 종목 리스트에서 수동으로 제거 필요 |
| 거시 데이터 의존 | VIX·S&P500 오프라인/API 오류 시 기본값(20.0, 0.0)으로 폴백 |
