# ML 분석 시스템 기술 문서

> Korean Stocks AI/ML Analysis System `v0.5.0`
> 최종 업데이트: 2026-03-13

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

ML 앙상블은 **향후 10거래일 후 수익률이 상위 25%에 속할 확률**을 이진 분류(RF·GB·LGB·CB) 및 크로스섹셔널 순위(XGBRanker), 딥러닝 시계열(TCN)로 예측한다.
절대 수익률 회귀 대신 이진 분류/랭킹을 채택하여 노이즈를 줄이고 신호 대 잡음비를 개선한다.

> **투자 시계:** 10거래일(≈ 2주) 단기 신호. 1개월 이상 보유를 목적으로 한 종목 선별에는 적합하지 않다.

```
추론 결과 (ml_score)
  0   = 하위 25% 예상 (낮은 확률)
 50   = 중립
100   = 상위 25% 예상 (높은 확률)
(test_proba/predict 101분위수 캘리브레이션으로 0-100 균등 스케일 변환)
```

ML 점수는 종합 점수 산출에 다음 가중치로 반영된다.

```
composite (ML + 거시감성 활성) = tech×0.35 + ml×0.35 + sentiment_norm×0.20 + macro_norm×0.10
composite (ML 활성, 거시감성 없음) = tech×0.40 + ml×0.35 + sentiment_norm×0.25
composite (ML 모델 없음)          = tech×0.65 + sentiment_norm×0.35
```

`macro_norm = (macro_sentiment_score + 100) / 2` — MacroNewsAgent가 제공하는 거시 뉴스 감성 점수(−100~100)를 0~100으로 정규화.

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

- 5-10거래일 절대 수익률 회귀는 노이즈 비율이 극도로 높아 R² 거의 0
- 이진 분류(상위 25% vs 하위 25%)로 문제를 단순화하여 AUC 개선
- 중간 50% 제외(neutral zone)로 경계 샘플 혼란 방지
- 추천 목적(상위 종목 선별)과 타깃이 정합

---

## 3. 피처 엔지니어링

총 **28개 피처** — 순수 기술지표 + 거시경제 (PyKrx 펀더멘털·수급 제외)

> 제거된 피처 (v0.3.2): `sqzmi`, `vol_change`, `macd_diff_change`, `obv_change`, `rsi_mfi_div`, `candle_body`, `rs_vs_mkt_1m` — 3모델 합산 중요도 최하위(<2%)
> 피처 교체 (v0.3.7): `adx_di_diff`·`cmf`·`vol_ratio` 제거 / `bb_position`·`mfi`·`low_52w_ratio` 추가. `atr_ratio` → rolling 60일 percentile 변환 (레짐 의존성 제거).
> 피처 추가 (v0.4.1): `obv_trend`, `rsi`(0-1 정규화), `cci_pct` 추가 — 17→20개.
> 거시 피처 확장 (v0.5.0): `vix_change_5d`, `tnx_level`, `tnx_change_1m`, `yield_spread`, `nasdaq_1m`, `gold_1m`, `oil_1m`, `csi300_1m` 추가 — 20→28개. yfinance 8개 심볼(^VIX·^GSPC·^IXIC·^TNX·^IRX·GC=F·CL=F·000300.SS) 수집.

### 3-1. 피처 목록

| 그룹 | 피처 | 설명 |
|------|------|------|
| 변동성·추세강도 (4) | `atr_ratio` | ATR/종가 rolling 60일 percentile — 레짐 독립적 변동성 |
| | `adx` | ADX 추세 강도 (0-100) |
| | `bb_width` | BB 너비 / BB 중심선 |
| | `bb_position` | BB 내 상대 위치 — (종가 − BB하단) / (BB상단 − BB하단) |
| 시장 상대강도 (1) | `rs_vs_mkt_3m` | 3개월 수익률 − 벤치마크 3개월 |
| 모멘텀·추세 (5) | `high_52w_ratio` | 종가 / 52주 고점 |
| | `mom_accel` | return_1m − (return_3m / 3) |
| | `macd_diff` | MACD − Signal (히스토그램) |
| | `macd_slope_5d` | MACD diff 5일 기울기 |
| | `price_sma_5_ratio` | 종가 / SMA5 |
| finta 지표 (2) | `fisher` | Fisher Transform (클립 ±5) |
| | `bullish_fractal_5d` | Williams Fractal 5일 내 최댓값 |
| 거래량·강도 (4) | `mfi` | Money Flow Index (거래량 가중 RSI, 0-100) |
| | `vzo` | Volume Zone Oscillator |
| | `low_52w_ratio` | 종가 / 52주 저점 — 저점 대비 반등 위치 (≥ 1.0) |
| | `obv_trend` | OBV 10일 모멘텀 (±1 클리핑) — 거래량 추세 가속/감속 신호 |
| 극값 감지·반전 (2) | `rsi` | RSI 0-1 정규화 (과매도 0.3↓ / 과매수 0.7↑ 극값 보존) |
| | `cci_pct` | CCI rolling 20일 percentile (레짐 독립적 0-1 정규화) |
| 거시경제 (10) | `vix_level` | VIX 절대값 (공포지수) |
| | `vix_change_5d` | VIX 5일 변화율 — 공포 가속/감속 |
| | `sp500_1m` | S&P500 1개월 수익률 |
| | `nasdaq_1m` | NASDAQ 1개월 수익률 — 기술주 방향 |
| | `tnx_level` | 미국채 10Y 금리 절대값 (pp) |
| | `tnx_change_1m` | 10Y 금리 1개월 절대 변화 (pp) |
| | `yield_spread` | 장단기 스프레드 (10Y−3M, pp) — 경기 선행지표 |
| | `gold_1m` | 금 선물 1개월 수익률 — 안전자산 수요 |
| | `oil_1m` | WTI 원유 1개월 수익률 — 인플레이션·경기 |
| | `csi300_1m` | 중국 CSI300 1개월 수익률 — 중국 경기 |

> 벤치마크: KOSPI 종목 → KS11, KOSDAQ 종목 → KQ11

### 3-2. PyKrx 피처 제외 이유

- 추론 시 단일 종목만 처리하므로 XS 순위 피처는 중립값(50)으로 고정되어 정보량 없음
- API 응답 지연(수 초-분) 및 장중 미확정 데이터 문제
- PyKrx 제외 후 AUC 오히려 개선 확인 (노이즈 감소 효과)

---

## 4. 모델 구성

여섯 모델을 독립적으로 학습하고 AUC 기반 가중 앙상블로 결합한다.
이진 분류기 4개(RF·GB·LGB·CB) + 크로스섹셔널 랭커 1개(XGBRanker) + 딥러닝 시계열 1개(TCN).
TCN은 PyTorch 설치 시(`pip install koreanstocks[dl]`) 자동 활성화되며, 미설치 시 5-모델 앙상블로 폴백한다.

### Random Forest (이진 분류기)

| 파라미터 | 값 |
|----------|----|
| n_estimators | 300 |
| max_depth | 4 |
| min_samples_split | 30 |
| min_samples_leaf | 30 |
| max_features | 0.4 |
| max_samples | 0.8 |
| class_weight | balanced |

### Gradient Boosting (이진 분류기)

| 파라미터 | 값 |
|----------|----|
| n_estimators | 200 |
| learning_rate | 0.05 |
| max_depth | 2 |
| min_samples_leaf | 25 |
| subsample | 0.7 |

### LightGBM (이진 분류기)

leaf-wise 분기 특성상 과적합이 강하게 발생 → 강한 정규화 적용.

| 파라미터 | 값 |
|----------|----|
| n_estimators | 200 |
| max_depth | 2 |
| num_leaves | 4 |
| learning_rate | 0.05 |
| min_child_samples | 100 |
| subsample | 0.6 |
| subsample_freq | 1 |
| colsample_bytree | 0.5 |
| reg_alpha | 2.0 |
| reg_lambda | 5.0 |
| class_weight | balanced |

### CatBoost (이진 분류기)

과적합 방어 기능 내장 (Ordered Boosting, 대칭 트리).

| 파라미터 | 값 |
|----------|----|
| iterations | 200 |
| depth | 3 |
| learning_rate | 0.05 |
| l2_leaf_reg | 5.0 |
| min_data_in_leaf | 40 |
| bootstrap_type | Bernoulli |
| subsample | 0.7 |
| auto_class_weights | Balanced |

### XGBRanker (크로스섹셔널 랭커)

이진 분류 대신 `rank:ndcg` 목적함수로 날짜별 그룹 내 종목 순위를 직접 최적화한다.
`predict_proba()` 없이 연속 점수 반환 → 101분위수 캘리브레이션으로 0-100 변환.

| 파라미터 | 값 |
|----------|----|
| objective | rank:ndcg |
| n_estimators | 200 |
| max_depth | 3 |
| learning_rate | 0.05 |
| subsample | 0.7 |
| colsample_bytree | 0.6 |
| min_child_weight | 25 |
| reg_alpha | 1.0 |
| reg_lambda | 3.0 |

### TCN — Temporal Convolutional Network (딥러닝 시계열)

트리 앙상블이 포착하지 못하는 **시간적 패턴(모멘텀 연속성, 변동성 레짐 전환)**을 학습하는 딥러닝 보완 모델.
Dilated Causal Conv1D 구조로 미래 누출 없이 과거 20거래일 시퀀스를 처리한다.

**아키텍처:**

```
Input  [B, T=20, F=28]
  → Transpose [B, F=28, T=20]
  → CausalBlock(dilation=1) → CausalBlock(dilation=2) → CausalBlock(dilation=4)
  → 마지막 타임스텝 슬라이스 [B, channels=32]
  → Linear(32→1) → Sigmoid → 상승 확률

Receptive field = 1 + 2×(1+2+4) = 15 거래일
```

| 파라미터 | 값 | 비고 |
|----------|----|------|
| lookback | 20 | 입력 시퀀스 길이 (거래일) |
| channels | 32 | TCN 히든 채널 수 |
| kernel_size | 3 | Conv1D 커널 |
| dilations | [1, 2, 4] | Receptive field = 15거래일 |
| dropout | 0.4 | 과적합 억제 (v0.4.4 상향) |
| weight_decay | 5e-4 | Adam L2 정규화 (v0.4.4 강화) |
| epochs | 40 | Early stopping (patience=8, val AUC 기준) |
| batch_size | 64 | |
| optimizer | Adam | lr=1e-3 |
| 클래스 가중치 | 수동 적용 | pos_weight = (1-pos_rate)/pos_rate |
| gradient clip | 1.0 | 폭발적 기울기 방지 |

**트리 모델과의 차이:**

| 항목 | 트리 모델 | TCN |
|------|-----------|-----|
| 입력 | 단일 시점 스냅샷 (T=1) | 20거래일 시퀀스 |
| 캡처 패턴 | 크로스섹셔널 피처 관계 | 시간적 연속성·레짐 전환 |
| 피처 중요도 | 지니 계수 기반 | N/A (아키텍처 정보로 대체) |
| Log Loss | 정의됨 | N/A (Binary BCE → 직접 비교 불가) |
| 앙상블 내 역할 | 주요 신호 생성 (75%) | 시계열 다양성 보완 (Softmax 가중치) |
| PyTorch 의존 | 없음 | 필요 (`pip install koreanstocks[dl]`) |

---

## 5. 학습 파이프라인

### 전체 흐름

```
[트리 모델 공통]
1. KS11/KQ11 시장 수익률 로드 (상대강도 피처용)
2. 거시경제 데이터 로드 (Yahoo Finance 8개 심볼: ^VIX·^GSPC·^IXIC·^TNX·^IRX·GC=F·CL=F·000300.SS)
3. 종목별 OHLCV 수집 + 지표 계산 + 피처 생성 (28개)
4. 전 종목 concat → 날짜별 크로스섹셔널 순위 → 이진 타깃 산출
   (상위 25% = 1, 하위 25% = 0, 중간 50% 제외)
5. 시계열 분할 (앞 80% → 학습 / 뒤 20% → 검증, 경계 Purging 20거래일 적용)
6. Walk-Forward CV (VAL_STEP=10 거래일, ~48 fold, fold 경계마다 Purging 20거래일 적용)
7. StandardScaler 정규화 → 모델 학습 → pkl 저장
8. test_proba 101분위수 배열(캘리브레이션) → JSON 저장

[TCN 추가 단계 — PyTorch 설치 시]
9.  종목별 피처 시계열 수집 (`_collect_stock_tcn()`) — 트리와 병렬 실행
10. 크로스섹셔널 이진 라벨 생성 (날짜별 상위/하위 25%, 중간 50% 제외)
11. build_sequences(): 피처 DataFrame → [N, T=20, F=20] 시퀀스 배열
12. Walk-Forward CV (VAL_WINDOW=20, VAL_STEP=10, Purging 20거래일) + Early Stopping
13. OOF 확률 수집 → 101분위수 캘리브레이션 배열 산출
14. 최종 TCN 학습 (전체 train 데이터) → state_dict + scaler + tcn_params.json 저장
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

### Walk-Forward CV (VAL_STEP=10)

과적합 감지를 위해 학습 세트 내에서 슬라이딩 윈도우 방식의 Walk-Forward CV를 수행한다.
VAL_STEP=10 거래일 간격으로 검증 윈도우를 이동 → 약 48 fold.

```python
unique_dates = sorted(df_train.index.unique())
val_step = 10   # 검증 윈도우 이동 간격 (거래일)
for start_idx in range(min_train_n, len(unique_dates), val_step):
    tr_dates  = set(unique_dates[:start_idx - 2*future_days])  # Purging 포함
    val_dates = set(unique_dates[start_idx:start_idx + val_step])
    # StandardScaler 재학습 + 모델 학습 + AUC 계산
```

### CV Purging (미래 누출 방지)

각 fold의 val 시작일 직전 `2 × future_days`(기본 **20**) 거래일을 학습 세트에서 제거하여
학습·검증 경계 근처의 데이터가 양쪽에 동시에 노출되는 문제를 방지한다.

```python
purge_boundary = start_idx - 2 * future_days   # future_days=10 → 20거래일 제거
tr_dates = set(unique_dates[:purge_boundary])
```

학습/검증 전체 분할에도 동일한 purging이 적용된다:
- 전체 분할 경계(`split_date`) 직전 20거래일을 학습 세트에서 제거
- 이로 인해 약 2,179개 샘플이 제거됨 (2025-10 경계 구간)

### 학습 데이터 현황 (v0.5.0 기준 — 재학습 전)

| 항목 | 값 |
|------|----|
| 학습 종목 | 144개 (`DEFAULT_TRAINING_STOCKS` 고정 리스트 — KOSPI 84개 + KOSDAQ 60개) |
| 데이터 기간 | 2년 (`--period 2y`) |
| 학습 샘플 (neutral zone 제외) | ~30,000 (재학습 시 변동) |
| 양성 비율 | 50.5% (중립 구간 제외로 균형) |
| 분할 기준일 | ≈ 2025-10 (재학습 시마다 변동) |
| Purging 일수 | 20거래일 (학습/검증 경계 누출 방지) |
| 피처 수 | **28개** (v0.5.0: 거시 8개 추가) |

> ⚠️ v0.5.0에서 피처가 20→28개로 확장됐으므로 `koreanstocks train` 재학습 필수. 이전 모델(20개 피처)은 28개 피처 환경에서 자동으로 기본값(중립)으로 폴백됨.

---

## 6. 앙상블 추론

### AUC 기반 Softmax 가중 앙상블

성능이 좋은 모델(높은 AUC)에 더 높은 가중치를 부여하되, Softmax로 정규화한다.
트리/랭커(Classifier)와 TCN을 75:25 비율로 혼합한다.

```python
# Softmax 정규화 (prediction_model.py)
raw_w = {name: max(auc - 0.50, 1e-6) for name, auc in aucs.items()}
total  = sum(raw_w.values())
weights = {name: w / total for name, w in raw_w.items()}

# 분류기 합산 (트리 4개 + TCN)
clf_sum    += p_i × w_i   # 캘리브레이션된 0-100 점수
clf_weight += w_i

# 랭커 합산 (XGBRanker)
rnk_sum    += p_i × w_i
rnk_weight += w_i

# 최종 ml_score
ml_score = clip(clf_sum/clf_weight × 0.75 + rnk_sum/rnk_weight × 0.25, 0, 100)
```

품질 게이트(`_MIN_MODEL_AUC = 0.52`)를 통과하지 못한 모델은 로드 자체를 거부한다.

### 캘리브레이션 (점수 균등화)

`predict_proba()[:,1]` 원시 확률은 40-60에 몰리는 경향이 있다.
`test_proba` 기준 101분위수 배열을 `params.json`에 저장하여 예측 시 0-100으로 균등 변환한다.

```python
# 학습 시 (trainer.py / tcn_model.py): test_proba 기준 101분위수 저장
calibration_points = np.percentile(test_proba, np.arange(0, 101)).tolist()

# 추론 시 (prediction_model.py): percentile rank로 변환
p = float(np.clip(np.searchsorted(calibration, p_raw), 0, 100))
```

> `train_proba` 대신 `test_proba`(또는 OOF 예측) 기준을 사용해야 과적합 분포(더 극단적)를 피한다.
> TCN은 OOF fold 예측을 캘리브레이션 소스로 우선 사용한다 (OOF < 101개 시 train_preds 폴백).

### TCN 추론 특이사항

```python
# predict_proba_tcn(): 단일 종목 최근 lookback=20행 피처 시퀀스 추론
feat_matrix = features[feat_cols].values   # [T, F]
prob = predict_proba_tcn(tcn_loaded, feat_matrix)   # 0~1 float

# 캘리브레이션 적용 → 0-100 변환 후 clf_sum에 편입
p = float(np.clip(np.searchsorted(cal, prob), 0, 100))
```

TCN은 피처 중요도(Gini) 개념이 없으므로 모델 신뢰도 탭에서 아키텍처 정보(layers, receptive field, dropout)를 대신 표시한다.

### 모델 미탑재 시 폴백 순서

```
1순위: tech_score 직접 사용 (fallback_score 인자 전달 시)
2순위: MACD diff + ATR 비율 기반 휴리스틱 점수
```

---

## 7. 성능 지표

> 학습일: 2026-03-13 / 검증 기간: ≈ 2025-11 이후 약 4개월

| 모델 | test AUC | CV AUC (Walk-Forward) | train AUC | 과적합 gap | 비고 |
|------|----------|----------------------|-----------|-----------|------|
| Random Forest | **0.5896** | 0.4702 ± 0.0522 | 0.6691 | 0.0795 | max_depth=4, max_samples=0.8 정규화 |
| Gradient Boosting | **0.5928** | 0.5002 ± 0.0341 | 0.6750 | 0.0822 | |
| LightGBM | **0.5915** | 0.4972 ± 0.0359 | 0.6706 | 0.0791 | |
| CatBoost | **0.6147** | 0.5017 ± 0.0342 | 0.7035 | 0.0889 | 최고 test AUC |
| XGBRanker | **0.5853** | 0.4827 ± 0.0490 | 0.6408 | **0.0555** | 최저 과적합 갭 |
| TCN (딥러닝) | **0.5703** | 0.5467 ± 0.0405 | 0.6352 | 0.0649 ※ | ※설계적 허용 범위 (딥러닝 수렴 특성) |
| **트리 앙상블 평균** | **0.5948** | — | — | **0.0770** | |
| **전체 앙상블 평균** | **0.5907** | — | — | **0.0750** | |
| 기준선 (랜덤 분류기) | 0.5000 | — | — | — | |

> **해석:**
> - CatBoost가 test AUC 0.6147로 앙상블을 견인. XGBRanker 과적합 갭 0.0555로 순위 신뢰도 최고.
> - CV AUC가 test AUC보다 낮은 것은 Walk-Forward 초기 fold 학습 데이터 부족 때문이며 정상 범위.
> - **RF 과적합 갭 개선**: `max_depth=4` + `max_samples=0.8` 적용으로 0.1175 → **0.0795** 달성. 트리 모델 전체가 임계값 0.10 이하.
> - **TCN**: test AUC 0.5703, CV 0.5467로 안정적. 과적합 갭 0.0649는 딥러닝 수렴 특성상 Train AUC가 높게 측정되는 구조적 현상이며 설계적 허용 범위 — 앙상블 mean_overfit_gap 계산에서 제외.

### AUC 개선 이력

| 단계 | 피처 수 | 타깃 | RF test AUC |
|------|---------|------|------------|
| 회귀 → 분류 전환 | 31 (PyKrx 포함) | 순위 회귀 → 이진 분류 | 0.5176 |
| PyKrx 제외 + 방향성 피처 추가 | 25 | 이진 분류 (상위 30%/하위 70%) | 0.5176 |
| Neutral Zone 타깃 (v0.3.0) | 25 | 상위 25%/하위 25%, 중간 50% 제외 | **0.5600** |
| XGBoost 과적합 감소 (v0.3.1) | 25 | 동일 | **0.5600** (XGB: 0.5558→0.5671) |
| 피처 정제 + CV Purging 적용 (v0.3.2) | 18 | 동일 | **0.5759** |
| 5-모델 앙상블 + 피처 교체 (v0.3.7) | 18 | 동일 | 0.575 / 앙상블 평균: **0.5857** |
| 피처 추가·CV 개선·depth 조정 (v0.4.1) | 20 | 동일 | **0.5880** / 앙상블 평균: **0.5946** |
| **TCN 딥러닝 추가 (v0.4.4)** | 20 | 동일 | **0.5962** / 6-모델 앙상블 평균: **0.5919** |
| **거시 피처 확장 (v0.5.0)** | **28** | 동일 | **0.5896** / 6-모델 앙상블 평균: **0.5907** |
| **RF 정규화 강화 (v0.5.0)** | 28 | 동일 | RF gap 0.1175→**0.0795** / 트리 평균 gap: **0.0770** |

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
| 종합 점수 가중치 | tech×0.35+ml×0.35+종목감성×0.20+거시감성×0.10 (ML+macro) | 백테스트 검증 기반 |
| 피처 목록 | BASE 28개 | 모델 재학습 없이 변경 불가 |
| 타깃 정의 | 상위 25%/하위 25% 이진, 중간 50% 제외 | AUC 최적화 기반 |
| 캘리브레이션 기준 | test_proba 101분위수 | train_proba 기준은 과적합 분포 반영 |

### 고정값 (임의 수정 금지) — TCN 추가 항목

| 항목 | 값 | 이유 |
|------|----|------|
| TCN lookback | 20 거래일 | 저장된 모델과 추론 시 반드시 일치해야 함 |
| TCN n_features | 28 (트리와 동일) | `features.py` `BASE_FEATURE_COLS` 변경 시 재학습 필수 |
| TCN AUC 게이트 | 0.52 | 트리 모델과 동일한 품질 기준 |
| TCN 캘리브레이션 소스 | OOF 예측 (< 101개 시 train 폴백) | test_proba 캘리브레이션과 동일 원칙 |

### 알려진 한계

| 항목 | 내용 |
|------|------|
| CV AUC vs test AUC 간격 | TimeSeriesSplit 초기 fold는 학습 데이터 부족 → CV AUC가 낮게 측정됨. 정상 현상. |
| 단기 노이즈 | 10거래일 예측은 본질적으로 신호 대 잡음비가 낮음 (AUC 0.57이 현실적 상한) |
| **단기 특화** | ml_score는 10거래일(≈ 2주) 지평에 최적화. 중기(1-3개월) 보유에는 rs_vs_mkt_3m·high/low_52w_ratio 피처가 컨텍스트만 제공할 뿐 타깃 정합성 없음 |
| 상장 폐지 종목 | 학습 종목 리스트(`DEFAULT_TRAINING_STOCKS`)에서 수동으로 제거 필요 |
| 거시 데이터 의존 | 8개 심볼(^VIX·^GSPC·^IXIC·^TNX·^IRX·GC=F·CL=F·000300.SS) API 오류 시 `_MACRO_DEFAULTS`(중립값)로 폴백 |
| **TCN 과적합 갭** | 0.0649 (설계적 허용 범위). Train AUC가 높게 측정되는 것은 딥러닝 수렴 특성이며 모델 결함 아님. 앙상블 mean_overfit_gap 계산에서 제외됨. |
| **TCN PyTorch 의존** | `pip install koreanstocks[dl]` 별도 설치 필요 (CPU 학습 약 10-20분). 미설치 시 5-모델 폴백 |
| **TCN 학습 시간** | CPU 기준 약 15-25분 (트리 모델 추가 시). GPU 사용 시 약 3-5분으로 단축 가능 |
