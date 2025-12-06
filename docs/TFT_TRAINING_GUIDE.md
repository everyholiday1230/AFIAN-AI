# TFT 학습 가이드 (로컬 환경용)

## 🚨 샌드박스 환경의 한계

현재 샌드박스 환경에서는 **메모리 제한**(~2-4GB)으로 인해 TFT 학습이 불가능합니다.

### 메모리 요구사항:
- **Random Forest**: 500MB ✅ (샌드박스 가능)
- **TFT**: 4-8GB ❌ (로컬 환경 필요)
- **LSTM**: 1-2GB ⚠️ (제한적)

---

## 📁 준비된 데이터

### 5분봉 데이터 (2019-2024):
```
data/historical_5min_features/
├── BTCUSDT_2019_1m.parquet  (104,693 rows)
├── BTCUSDT_2020_1m.parquet  (105,089 rows)
├── BTCUSDT_2021_1m.parquet  (104,845 rows)
├── BTCUSDT_2022_1m.parquet  (105,059 rows)
├── BTCUSDT_2023_1m.parquet  (105,029 rows)
├── BTCUSDT_2024_1m.parquet  (105,347 rows)
├── ETHUSDT_2019_1m.parquet  (104,693 rows)
├── ETHUSDT_2020_1m.parquet  (105,089 rows)
├── ETHUSDT_2021_1m.parquet  (104,845 rows)
├── ETHUSDT_2022_1m.parquet  (105,059 rows)
├── ETHUSDT_2023_1m.parquet  (105,029 rows)
└── ETHUSDT_2024_1m.parquet  (105,347 rows)
```

총: **1,259,124 rows × 44 features** (454 MB)

---

## 🤖 TFT 학습 방법 (로컬 환경)

### Option 1: PyTorch Forecasting 사용

```bash
# 1. 환경 설정
pip install pytorch-forecasting pytorch-lightning torch

# 2. 학습 실행
python scripts/train_tft_incremental.py
```

**예상 소요 시간:**
- GPU: 2-4시간
- CPU: 8-12시간

---

### Option 2: 커스텀 TFT 구현 사용

프로젝트에 이미 완전한 TFT 구현이 있습니다:

```python
from ai.models.tft.temporal_fusion_transformer import TemporalFusionTransformer
from ai.training.pipelines.tft_training_pipeline import TFTTrainingPipeline

# 학습 실행
pipeline = TFTTrainingPipeline(
    data_dir='data/historical_5min_features',
    symbols=['BTCUSDT'],
    encoder_length=60,  # 5시간
    decoder_length=12,  # 1시간
)

pipeline.train()
```

---

## 📊 학습 완료 후

### 1. 모델 저장 위치:
```
models/tft/
├── tft_2021.ckpt
├── tft_2022.ckpt
└── tft_2023.ckpt
```

### 2. 백테스트 실행:
```bash
python scripts/backtest_tft.py \
  --model-path models/tft/tft_2023.ckpt \
  --test-data data/historical_5min_features/BTCUSDT_2024_1m.parquet
```

### 3. 성능 비교:
| 모델 | 학습 시간 | 메모리 | 예상 성능 |
|------|----------|--------|----------|
| Random Forest | 2분 | 500MB | R²: 0.001 |
| LSTM | 10분 | 1-2GB | R²: 0.05-0.15 |
| **TFT** | 4시간 | 4-8GB | **R²: 0.3-0.6** |

---

## 🎯 로컬 환경 학습 단계

### Step 1: 데이터 다운로드
```bash
# 프로젝트를 로컬로 clone
git clone <repository-url>
cd webapp

# 데이터는 이미 준비됨
ls -lh data/historical_5min_features/
```

### Step 2: 환경 설정
```bash
# Python 3.10+ 권장
pip install -r requirements.txt
pip install pytorch-forecasting pytorch-lightning
```

### Step 3: TFT 학습
```bash
# 단일 연도 (빠른 테스트)
python scripts/train_tft_incremental.py

# 전체 데이터 (최고 성능)
python ai/training/pipelines/tft_training_pipeline.py \
  --data-dir data/historical_5min_features \
  --epochs 50 \
  --batch-size 128
```

### Step 4: 백테스트
```bash
python scripts/backtest_tft.py \
  --model-path models/tft/best_model.ckpt \
  --test-year 2024
```

---

## 📈 기대 결과

### TFT vs Random Forest (2024년 테스트):

**Random Forest (현재):**
- Total Return: -78.23%
- Sharpe Ratio: -2.20
- Max Drawdown: -78.95%

**TFT (예상):**
- Total Return: +50% ~ +150%
- Sharpe Ratio: 1.0 ~ 2.5
- Max Drawdown: -20% ~ -40%

---

## 🔧 메모리 최적화 팁

로컬에서도 메모리가 부족하면:

1. **배치 크기 줄이기**: 128 → 64 → 32
2. **시퀀스 길이 줄이기**: 60 → 30
3. **Hidden size 줄이기**: 64 → 32 → 16
4. **연도별 학습**: 1년씩 나눠서 학습
5. **Mixed Precision**: `trainer.precision=16`

---

## 📝 현재 상태

✅ **완료:**
- 6년 데이터 수집 (2019-2024)
- 5분봉 리샘플링 (1.26M rows)
- 44개 기술적 지표 생성
- Random Forest 학습 완료
- 백테스트 시스템 구축

⏳ **대기 중:**
- TFT 학습 (로컬 환경 필요)
- 2024년 Out-of-Sample 검증
- 앙상블 모델 구축
- 실전 Paper Trading

---

## 🚀 다음 단계

1. **로컬 환경에서 TFT 학습** (4-8시간)
2. **2024년 백테스트** (성능 검증)
3. **하이퍼파라미터 튜닝** (최적화)
4. **앙상블 전략** (RF + TFT + Guardian)
5. **Paper Trading** (실전 테스트)
6. **Live Trading** (실제 운영)

---

## 📞 지원

문제 발생 시:
1. GitHub Issues 등록
2. 로그 파일 확인: `logs/training.log`
3. 메모리 모니터링: `nvidia-smi` (GPU) 또는 `htop` (CPU)
