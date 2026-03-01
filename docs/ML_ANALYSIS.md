# ML 분석 시스템 기술 문서

> Korean Stocks AI/ML Analysis System `v0.3.1`
> 최종 업데이트: 2026-02-28

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

ML 모델은 **향후 5거래일 후 크로스섹셔널 퍼센타일 순위**를 예측한다.
절대 수익률 예측 대신 상대 순위를 타깃으로 삼아 시장 방향(상승/하락장)에 관계없이
강세 종목을 선별하는 데 초점을 맞춘다.

```
추론 결과 (ml_score)
  0  = 당일 전 종목 중 최하위 상대강도 예상
 50  = 평균 수준
100  = 최상위 상대강도 예상
```

ML 점수는 종합 점수 산출에 다음 가중치로 반영된다.

```
composite (ML 모델 활성) = tech × 0.40 + ml × 0.35 + sentiment_norm × 0.25
composite (ML 모델 없음) = tech × 0.65 + sentiment_norm × 0.35
```

---

## 2. 타깃 변수

### 크로스섹셔널 퍼센타일 순위

각 날짜마다 학습에 참여한 전 종목의 `N`거래일 후 수익률을 퍼센타일 순위(0~100)로 변환한다.

```python
df_all['target'] = (
    df_all.groupby(df_all.index)['raw_return']
    .rank(pct=True) * 100.0
)
```

| 항목 | 값 |
|------|----|
| 예측 기간 | 5거래일 후 (기본값, `--future-days` 변경 가능) |
| 학습 종목 수 | 144종목 (146개 지정, 2개 데이터 없음) |
| 날짜별 평균 종목 수 | 143.8 |
| 최소 종목 수 기준 | 5종목 미만 날짜는 순위 신뢰도 부족으로 제외 |

### 절대 수익률 대비 장점

- 상승장 / 하락장 무관하게 상대적 강세 종목을 학습
- 시장 전체 방향성 노이즈 제거
- 추천 목적(순위가 높은 종목 선별)과 타깃이 정합

---

## 3. 피처 엔지니어링

총 **31개 피처** = BASE 22개 + PyKrx 9개

### 3-1. BASE 피처 (22개) — 기술적 지표

| 그룹 | 피처 | 설명 |
|------|------|------|
| 기본 지표 (4) | `rsi` | RSI(14) |
| | `macd_diff` | MACD - Signal |
| | `price_sma_20_ratio` | 종가 / SMA20 |
| | `vol_change` | 거래량 전일 대비 변화율 |
| 추세 변화 (3) | `price_sma_5_ratio` | 종가 / SMA5 |
| | `rsi_change` | RSI 전일 대비 변화량 |
| | `macd_diff_change` | MACD diff 전일 대비 변화량 |
| 볼린저 밴드 (2) | `bb_position` | BB 내 종가 위치 (0=하단, 1=상단) |
| | `bb_width` | BB 너비 / BB 중심선 |
| 거래량 (1) | `vol_ratio` | 거래량 / 20일 평균 거래량 |
| 모멘텀 오실레이터 (3) | `stoch_k` | Stochastic %K |
| | `stoch_d` | Stochastic %D |
| | `cci` | CCI |
| 변동성·캔들 (3) | `atr_ratio` | ATR / 종가 |
| | `candle_body` | (종가 - 시가) / 시가 |
| | `obv_change` | OBV 전일 대비 변화율 (clip ±1) |
| 모멘텀 팩터 (4) | `return_1m` | 20일 수익률 |
| | `return_3m` | 60일 수익률 |
| | `high_52w_ratio` | 종가 / 52주 고점 |
| | `mom_accel` | return_1m − (return_3m / 3) |
| 시장 상대강도 (2) | `rs_vs_mkt_1m` | return_1m − 벤치마크 return_1m |
| | `rs_vs_mkt_3m` | return_3m − 벤치마크 return_3m |

> 벤치마크: KOSPI 종목 → KS11, KOSDAQ 종목 → KQ11

### 3-2. PyKrx 피처 (9개) — 펀더멘털 · 수급

| 그룹 | 피처 | 설명 | 소스 |
|------|------|------|------|
| 펀더멘털 raw (3) | `pbr` | PBR | `get_market_fundamental_by_date` |
| | `per` | PER | |
| | `div` | 배당수익률 | |
| 펀더멘털 XS 순위 (2) | `pbr_xs` | PBR 날짜별 퍼센타일 순위 (0~100) | 학습 시 groupby 계산 |
| | `per_xs` | PER 날짜별 퍼센타일 순위 | |
| 수급 raw (2) | `foreign_5d_ratio` | 외국인 5일 누적 순매수 / 5일 거래대금 | `get_market_trading_value_by_date` |
| | `inst_5d_ratio` | 기관 5일 누적 순매수 / 5일 거래대금 | |
| 수급 XS 순위 (2) | `foreign_xs` | foreign_5d_ratio 날짜별 퍼센타일 순위 | 학습 시 groupby 계산 |
| | `inst_xs` | inst_5d_ratio 날짜별 퍼센타일 순위 | |

**수급 비율 계산식:**

```python
turnover_5d = (close × volume).rolling(5).sum()
foreign_5d_ratio = 외국인합계.rolling(5).sum() / turnover_5d  # clip(−0.5, 0.5)
```

**추론 시 XS 순위 처리:**
단일 종목 추론 시에는 날짜 전체 분포를 알 수 없으므로 XS 피처는 중립값 **50.0** 고정.
raw 피처(pbr, per, foreign_5d_ratio 등)는 실시간 PyKrx 조회값을 사용한다.

---

## 4. 모델 구성

세 모델을 독립적으로 학습하고 RMSE 역수 가중 앙상블로 결합한다.

### Random Forest

| 파라미터 | 값 |
|----------|----|
| n_estimators | 300 |
| max_depth | 4 |
| min_samples_split | 20 |
| min_samples_leaf | 15 |
| max_features | 0.5 |

### Gradient Boosting

| 파라미터 | 값 |
|----------|----|
| n_estimators | 200 |
| learning_rate | 0.05 |
| max_depth | 3 |
| min_samples_leaf | 20 |
| subsample | 0.7 |

### XGBoost

| 파라미터 | 값 |
|----------|----|
| n_estimators | 300 |
| learning_rate | 0.05 |
| max_depth | 3 |
| subsample | 0.7 |
| colsample_bytree | 0.7 |
| min_child_weight | 15 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |

---

## 5. 학습 파이프라인

### 전체 흐름

```
1. KS11 시장 수익률 로드 (시장 상대강도 피처용)
2. PyKrx 전 종목 일괄 수집 (파일 캐시 활용)
   → data/storage/pykrx_cache_{hash}.pkl
3. 종목별 OHLCV 수집 + 지표 계산 + 피처 생성
   → PyKrx 피처 합산 (BASE dropna 후 PyKrx NaN은 중위값 채우기)
4. 전 종목 concat → 날짜별 크로스섹셔널 순위 타깃 산출
5. XS 순위 피처 계산 (pbr_xs, per_xs, foreign_xs, inst_xs)
6. 시계열 전역 분할 (앞 80% → 학습 / 뒤 20% → 검증)
7. StandardScaler 정규화 → 모델 학습 → pkl 저장
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

### 학습 데이터 현황 (v0.2.3 기준)

| 항목 | 값 |
|------|----|
| 학습 종목 | 144개 (KOSPI 72 + KOSDAQ 72) |
| 데이터 기간 | 2년 (`--period 2y`) |
| 학습 샘플 | 44,444 |
| 검증 샘플 | 11,205 |
| 분할 기준일 | ≈ 2025-10 (재학습 시마다 변동) |
| 피처 수 | 31개 |

### PyKrx 캐시

재학습 시 전 종목의 PyKrx API 호출(약 20분)을 반복하지 않도록 파일 캐시를 사용한다.

```
캐시 키: MD5(sorted(codes) + period)[:8]
캐시 경로: data/storage/pykrx_cache_{hash}.pkl
캐시 유효: 종목 리스트 또는 기간이 달라지면 자동 재수집
```

---

## 6. 앙상블 추론

### RMSE 역수 가중 앙상블

성능이 좋은 모델(낮은 RMSE)에 더 높은 가중치를 부여한다.

```python
w_i = 1 / RMSE_i

ml_score = clip(Σ(p_i × w_i) / Σ(w_i), 0, 100)
```

### 구버전 모델 호환 (backward compatibility)

`scaler.n_features_in_` 값으로 피처 수를 자동 감지해 22피처(구버전)와 31피처(신규) 모델을 함께 사용할 수 있다.

```python
x = latest_x_full  if scaler.n_features_in_ > 22  else latest_x
```

### 모델 미탑재 시 폴백 순서

```
1순위: tech_score 직접 사용 (fallback_score 인자 전달 시)
2순위: RSI + MACD + 가격/SMA 기반 휴리스틱 점수
```

---

## 7. 성능 지표

> 학습일: 2026-02-27 / 검증 기간: ≈ 2025-10 이후 약 4개월
> train R² 및 과적합 gap은 현재 모델 파라미터 파일에 저장되지 않음.

| 모델 | test RMSE | test R² | train R² | 과적합 gap | 기준선 대비 개선 |
|------|-----------|---------|---------|-----------|----------------|
| Random Forest | 28.7945 | **+0.0050** | – | – | +0.25% |
| Gradient Boosting | 28.7977 | **+0.0048** | – | – | +0.24% |
| XGBoost | 28.8319 | **+0.0024** | – | – | +0.12% |
| 기준선 (평균 예측) | ≈ 28.867 | 0.0000 | — | — | — |

> **해석:** 주가 5일 예측은 본질적으로 노이즈가 크기 때문에 R² 절댓값 자체는 낮다.
> 세 모델 모두 양수 R²를 유지하며 기준선을 꾸준히 상회하는 것이 핵심 목표다.

### 피처 추가에 따른 R² 개선 이력

| 단계 | 피처 수 | 학습 종목 | RF R² |
|------|---------|----------|-------|
| 초기 (랜덤 분할, 데이터 누출) | 4 | 37 | 과대평가 |
| 시계열 분할 + 22피처 | 22 | 37 | −0.003 |
| 학습 종목 144개로 확장 | 22 | 144 | +0.0024 |
| PyKrx 9피처 추가 (v0.2.2) | **31** | 144 | +0.0042 |
| 재학습 (v0.2.3, 2026-02-27) | **31** | 144 | **+0.0050** |

---

## 8. 재학습 방법

```bash
# CLI 명령어 (권장)
koreanstocks train
koreanstocks train --future-days 5 --period 2y --test-ratio 0.2

# 스크립트 직접 실행
python train_models.py
python train_models.py --future-days 5 --period 2y --test-ratio 0.2
```

### 재학습 시 주의사항

- PyKrx 캐시(`pykrx_cache_*.pkl`)가 있으면 API 재호출 없이 재사용
- 종목 리스트나 기간이 달라지면 캐시 키가 바뀌어 자동 재수집 (약 20분 소요)
- 재학습 후 `models/saved/model_params/*.json`의 성능 수치 확인 권장
- ML 피처 목록 변경 시 `train_models.py`와 `prediction_model.py`를 함께 수정해야 함

---

## 9. 설계 원칙 및 제약

### 고정값 (임의 수정 금지)

| 항목 | 값 | 이유 |
|------|----|------|
| 종합 점수 가중치 | tech×0.40 + ml×0.35 + sent×0.25 | 백테스트 검증 기반 |
| 피처 목록 | BASE 22 + PyKrx 9 = 31 | 모델 재학습 없이 변경 불가 |
| 타깃 정규화 | 크로스섹셔널 퍼센타일 순위 | 절대 수익률로 변경 시 R² 악화 |

### 알려진 한계

| 항목 | 내용 |
|------|------|
| XS 피처 추론 정확도 | 추론 시 단일 종목만 처리하므로 XS 순위 피처는 중립값(50) 고정 |
| 수급 데이터 지연 | PyKrx는 당일 장 마감 이후 데이터 확정 — 장중 추론 시 전일 기준 |
| 단기 노이즈 | 5거래일 예측은 본질적으로 신호 대 잡음비가 낮음 |
| 상장 폐지 종목 | 학습 종목 리스트에서 수동으로 제거 필요 (048260, 196180 현재 제외) |
