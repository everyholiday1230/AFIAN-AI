# 📊 데이터 설정 가이드

## ⚠️ 중요

**데이터 파일은 GitHub에 포함되어 있지 않습니다** (파일 크기: ~1.5GB)

로컬에서 학습하기 전에 데이터를 다운로드해야 합니다.

---

## 🚀 빠른 데이터 다운로드

### **방법 1: 자동 스크립트 실행 (추천)**

```bash
# 프로젝트 디렉토리에서 실행
cd webapp

# 데이터 다운로드 (2019-2024, 6년치)
python scripts/download_year_by_year.py \
  --symbols BTCUSDT ETHUSDT \
  --start-date 2019-01-01 \
  --end-date 2024-12-31 \
  --output-dir data/historical
```

**예상 시간**: 30-60분  
**다운로드 크기**: ~401 MB (raw data)

---

### **방법 2: 전처리 및 기능 생성 포함**

```bash
# 1. 데이터 다운로드
python scripts/download_year_by_year.py \
  --symbols BTCUSDT ETHUSDT \
  --start-date 2019-01-01 \
  --end-date 2024-12-31 \
  --output-dir data/historical

# 2. 데이터 전처리 (아웃라이어 제거, 정규화)
python scripts/preprocess_historical.py \
  --input-dir data/historical \
  --output-dir data/historical_processed

# 3. 5분봉 리샘플링
python scripts/resample_to_5min.py \
  --input-dir data/historical_processed \
  --output-dir data/historical_5min

# 4. 기술적 지표 생성 (44개 features)
bash scripts/generate_features_5min.sh
```

**총 예상 시간**: 1-2시간  
**최종 데이터 크기**: ~454 MB (features)

---

## 📦 다운로드 완료 후 구조

```
data/
├── historical/                      # 원시 데이터 (401 MB)
│   ├── BTCUSDT_2019_1m.parquet
│   ├── BTCUSDT_2020_1m.parquet
│   ├── BTCUSDT_2021_1m.parquet
│   ├── BTCUSDT_2022_1m.parquet
│   ├── BTCUSDT_2023_1m.parquet
│   ├── BTCUSDT_2024_1m.parquet
│   └── [ETHUSDT 동일]
│
├── historical_processed/            # 전처리 데이터 (663 MB)
│   └── [동일 구조]
│
├── historical_5min/                 # 5분봉 데이터 (139 MB)
│   └── [동일 구조]
│
└── historical_5min_features/        # 학습용 데이터 (454 MB) ⭐
    ├── BTCUSDT_2019_1m.parquet     (104,693 rows × 44 features)
    ├── BTCUSDT_2020_1m.parquet     (105,089 rows × 44 features)
    ├── BTCUSDT_2021_1m.parquet     (104,845 rows × 44 features)
    ├── BTCUSDT_2022_1m.parquet     (105,059 rows × 44 features)
    ├── BTCUSDT_2023_1m.parquet     (105,029 rows × 44 features)
    ├── BTCUSDT_2024_1m.parquet     (105,347 rows × 44 features)
    └── [ETHUSDT 동일]
```

---

## ✅ 데이터 검증

다운로드 완료 후 확인:

```bash
# 파일 개수 확인
ls -lh data/historical_5min_features/

# 데이터 로드 테스트
python -c "
import pandas as pd
from pathlib import Path

data_dir = Path('data/historical_5min_features')
files = sorted(data_dir.glob('*.parquet'))

print(f'Found {len(files)} files')
for f in files:
    df = pd.read_parquet(f)
    print(f'{f.name}: {len(df):,} rows × {len(df.columns)} columns')
"
```

**예상 출력**:
```
Found 12 files
BTCUSDT_2019_1m.parquet: 104,693 rows × 44 columns
BTCUSDT_2020_1m.parquet: 105,089 rows × 44 columns
...
```

---

## 🎯 학습 시작

데이터 다운로드 완료 후:

```bash
# AI 모델 학습 시작
python scripts/train_production_models.py --all
```

---

## 📝 44개 기술적 지표 (Features)

다운로드된 데이터에 포함된 Features:

### **Trend Indicators (8개)**
- SMA_10, SMA_20, SMA_50
- EMA_12, EMA_26
- MACD, MACD_signal, MACD_hist

### **Momentum Indicators (4개)**
- RSI_14
- Stochastic_K, Stochastic_D
- CCI

### **Volatility Indicators (8개)**
- BB_upper, BB_middle, BB_lower, BB_width
- ATR_14, ATR_period_high, ATR_period_low
- Keltner_Channel

### **Volume Indicators (4개)**
- OBV (On-Balance Volume)
- volume_ma, volume_ma_ratio
- VWAP

### **Price Features (5개)**
- close, open, high, low, volume

### **Returns (4개)**
- returns_1, returns_3, returns_12, returns_60

### **Volatility (3개)**
- volatility_12, volatility_48, volatility_240

### **Time Features (3개)**
- hour (0-23)
- day_of_week (0-6)
- is_trading_hour

### **기타 (5개)**
- price_ma_ratio
- volume_ratio
- trend_strength
- momentum_score
- volatility_regime

---

## 🔧 문제 해결

### **다운로드 실패 시**
```bash
# 특정 연도만 재다운로드
python scripts/download_year_by_year.py \
  --symbols BTCUSDT \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --output-dir data/historical
```

### **메모리 부족 시**
- 연도별로 하나씩 다운로드
- 처리 스크립트의 `chunk_size` 줄이기

### **네트워크 오류 시**
- VPN 사용
- Binance Vision 직접 접속: https://data.binance.vision/

---

## 📞 지원

데이터 다운로드 문제 발생 시:
1. GitHub Issues에 문의
2. 로그 파일 확인: `logs/download.log`
3. Binance API 상태 확인: https://www.binance.com/en/support/announcement

---

## 💡 팁

- **디스크 공간**: 최소 10GB 확보 권장
- **인터넷 속도**: 빠를수록 좋음 (30-60분 소요)
- **백그라운드 실행**: `nohup python ... &` 사용 가능
- **재개 기능**: 스크립트는 이미 다운로드된 파일 건너뜀

---

**데이터 준비 완료 후 `docs/QUICK_START_TRAINING.md`를 참고하여 학습을 시작하세요!** 🚀
