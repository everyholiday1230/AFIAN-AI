# 🚀 AI 모델 학습 마스터 가이드 (최고 성능)

## 📋 학습해야 할 3개 핵심 AI 모델

PROJECT QUANTUM ALPHA는 **3개의 전문화된 AI 모델**을 앙상블로 사용합니다:

### 1️⃣ **Oracle (TFT - Temporal Fusion Transformer)**
- **역할**: 미래 가격 예측
- **목적**: 다음 1-24시간 가격 변동 예측
- **출력**: 가격 상승/하락 확률 + 불확실성

### 2️⃣ **Strategist (Decision Transformer)**
- **역할**: 최적 행동 결정
- **목적**: 매수/매도/홀드 타이밍 최적화
- **출력**: 행동(매수/매도) + 포지션 크기

### 3️⃣ **Guardian (Contrastive VAE)**
- **역할**: 시장 체제 감지
- **목적**: Bull/Bear/Sideways 시장 구분
- **출력**: 시장 상태 + 리스크 레벨

---

## 💾 준비된 데이터

### **5분봉 데이터 (2019-2024, 6년)**
```
data/historical_5min_features/
├── BTCUSDT_2019_1m.parquet  (104,693 rows, 44 features)
├── BTCUSDT_2020_1m.parquet  (105,089 rows, 44 features)
├── BTCUSDT_2021_1m.parquet  (104,845 rows, 44 features)
├── BTCUSDT_2022_1m.parquet  (105,059 rows, 44 features)
├── BTCUSDT_2023_1m.parquet  (105,029 rows, 44 features)
├── BTCUSDT_2024_1m.parquet  (105,347 rows, 44 features)
└── [ETHUSDT 동일]
```

**총 데이터**: 1,259,124 rows × 44 features (454 MB)

### **44개 기술적 지표**
- **Trend**: SMA_10, SMA_20, SMA_50, EMA_12, EMA_26, MACD, MACD_signal, MACD_hist
- **Momentum**: RSI_14, Stochastic_K, Stochastic_D
- **Volatility**: BB_upper, BB_middle, BB_lower, BB_width, ATR_14, ATR_period_high, ATR_period_low
- **Volume**: OBV, volume_ma, volume_ma_ratio, VWAP
- **Price**: close, open, high, low, volume, price_ma_ratio
- **Returns**: returns_1, returns_3, returns_12, returns_60
- **Volatility**: volatility_12, volatility_48, volatility_240
- **Time**: hour, day_of_week, is_trading_hour

---

## 🎯 1. Oracle (TFT) 학습 - 가격 예측

### **모델 아키텍처**
```
입력: 과거 60개 타임스텝 (5시간)
출력: 미래 24개 타임스텝 (2시간) 가격 예측
```

### **최고 성능 설정**

#### **Step 1: 환경 설정**
```bash
# PyTorch + CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Forecasting
pip install pytorch-forecasting pytorch-lightning

# 기타
pip install tensorboard pandas numpy scikit-learn
```

#### **Step 2: 학습 스크립트 실행**
```bash
cd /home/user/webapp

# 최고 성능 설정으로 학습
python ai/training/pipelines/tft_training_pipeline.py \
  --data-dir data/historical_5min_features \
  --symbols BTCUSDT ETHUSDT \
  --encoder-length 60 \
  --decoder-length 24 \
  --batch-size 256 \
  --hidden-size 128 \
  --attention-heads 4 \
  --num-layers 3 \
  --dropout 0.1 \
  --learning-rate 0.001 \
  --epochs 100 \
  --early-stopping-patience 15 \
  --gradient-clip-val 0.1 \
  --output-dir models/tft \
  --use-gpu
```

#### **최고 성능 하이퍼파라미터**
```python
# TFT 설정 (최고 성능)
config = {
    'encoder_length': 60,        # 5시간 히스토리
    'decoder_length': 24,        # 2시간 예측
    'hidden_size': 128,          # 큰 hidden dimension
    'attention_head_size': 4,    # Multi-head attention
    'num_layers': 3,             # 깊은 네트워크
    'dropout': 0.1,              # Regularization
    'learning_rate': 0.001,      # Adam optimizer
    'batch_size': 256,           # 큰 배치 (GPU 활용)
    'max_epochs': 100,           # 충분한 학습
    'gradient_clip_val': 0.1,    # Gradient explosion 방지
}
```

#### **예상 학습 시간**
- **GPU (RTX 4090)**: 4-6시간
- **GPU (RTX 3090)**: 6-8시간
- **GPU (RTX 3080)**: 8-12시간
- **CPU (32 cores)**: 24-36시간 (권장하지 않음)

#### **메모리 요구사항**
- **GPU VRAM**: 12GB 이상 권장
- **System RAM**: 32GB 이상 권장

#### **학습 결과**
```
models/tft/
├── best_model.ckpt           (최고 성능 모델)
├── last_model.ckpt           (마지막 체크포인트)
├── checkpoints/              (에폭별 체크포인트)
└── tensorboard_logs/         (학습 로그)
```

---

## 🎮 2. Strategist (Decision Transformer) 학습 - 행동 최적화

### **모델 아키�ecture**
```
입력: 상태(가격) + 과거 행동 + 보상(수익)
출력: 최적 행동(매수/매도/홀드) + 포지션 크기
```

### **최고 성능 설정**

#### **Step 1: 학습 스크립트 실행**
```bash
# Decision Transformer 학습
python ai/training/pipelines/decision_transformer_training.py \
  --data-dir data/historical_5min_features \
  --symbols BTCUSDT ETHUSDT \
  --context-length 90 \
  --hidden-size 256 \
  --num-layers 6 \
  --num-heads 8 \
  --dropout 0.1 \
  --learning-rate 0.0001 \
  --batch-size 128 \
  --epochs 200 \
  --output-dir models/decision_transformer \
  --use-gpu
```

#### **최고 성능 하이퍼파라미터**
```python
# Decision Transformer 설정
config = {
    'context_length': 90,        # 7.5시간 컨텍스트
    'hidden_size': 256,          # 큰 representation
    'num_layers': 6,             # 깊은 Transformer
    'num_heads': 8,              # Multi-head attention
    'dropout': 0.1,
    'learning_rate': 0.0001,     # 낮은 학습률 (안정성)
    'batch_size': 128,
    'max_epochs': 200,           # Reinforcement Learning은 오래 필요
    'reward_scale': 1.0,         # 보상 스케일링
    'rtg_scale': 1000.0,         # Return-to-go 스케일
}
```

#### **예상 학습 시간**
- **GPU (RTX 4090)**: 8-12시간
- **GPU (RTX 3090)**: 12-16시간

#### **메모리 요구사항**
- **GPU VRAM**: 16GB 이상 권장
- **System RAM**: 32GB 이상

---

## 🛡️ 3. Guardian (Contrastive VAE) 학습 - 시장 체제 감지

### **모델 아키텍처**
```
입력: 시장 데이터 (OHLCV + 지표)
출력: 시장 체제 (Bull/Bear/Sideways) + 임베딩
```

### **최고 성능 설정**

#### **Step 1: 학습 스크립트 실행**
```bash
# Contrastive VAE 학습
python ai/training/pipelines/regime_detection_pipeline.py \
  --data-dir data/historical_5min_features \
  --symbols BTCUSDT ETHUSDT \
  --latent-dim 64 \
  --hidden-dims 256 128 64 \
  --window-size 120 \
  --batch-size 512 \
  --learning-rate 0.001 \
  --epochs 100 \
  --output-dir models/guardian \
  --use-gpu
```

#### **최고 성능 하이퍼파라미터**
```python
# Contrastive VAE 설정
config = {
    'latent_dim': 64,            # 잠재 공간 차원
    'hidden_dims': [256, 128, 64], # Encoder/Decoder 레이어
    'window_size': 120,          # 10시간 윈도우
    'batch_size': 512,           # 큰 배치
    'learning_rate': 0.001,
    'beta': 4.0,                 # VAE beta (KL weight)
    'temperature': 0.5,          # Contrastive learning 온도
    'max_epochs': 100,
}
```

#### **예상 학습 시간**
- **GPU (RTX 4090)**: 2-4시간
- **GPU (RTX 3090)**: 4-6시간

#### **메모리 요구사항**
- **GPU VRAM**: 8GB 이상
- **System RAM**: 16GB 이상

---

## 🔧 학습 스크립트 수정 (최고 성능)

### **TFT 학습 스크립트 수정**

`ai/training/pipelines/tft_training_pipeline.py`를 열고 다음과 같이 수정:

```python
# 라인 찾기: def __init__ 또는 config 부분

# 최고 성능 설정으로 변경
self.config = {
    # 데이터
    'encoder_length': 60,
    'decoder_length': 24,
    'batch_size': 256,  # GPU 메모리에 따라 512까지 가능
    
    # 모델 아키텍처
    'hidden_size': 128,  # 256도 가능 (더 느리지만 더 좋음)
    'attention_head_size': 4,
    'dropout': 0.1,
    'hidden_continuous_size': 64,  # 128도 가능
    'num_lstm_layers': 2,
    
    # 학습
    'learning_rate': 0.001,
    'max_epochs': 100,
    'gradient_clip_val': 0.1,
    'early_stopping_patience': 15,
    
    # GPU
    'accelerator': 'gpu',  # 'cpu' 대신
    'devices': 1,  # GPU 개수
    'precision': 16,  # Mixed precision (속도 2배)
}
```

### **Decision Transformer 수정**

`ai/training/pipelines/decision_transformer_training.py`:

```python
self.config = {
    # Transformer
    'hidden_size': 256,  # 512도 가능
    'num_layers': 6,  # 8-12도 가능
    'num_heads': 8,
    'context_length': 90,
    
    # 학습
    'learning_rate': 0.0001,
    'batch_size': 128,  # GPU에 따라 256
    'max_epochs': 200,
    
    # RL specific
    'discount_factor': 0.99,
    'reward_scale': 1.0,
    'rtg_scale': 1000.0,
}
```

### **Guardian (VAE) 수정**

`ai/training/pipelines/regime_detection_pipeline.py`:

```python
self.config = {
    # VAE
    'latent_dim': 64,  # 128도 가능
    'hidden_dims': [256, 128, 64],  # [512, 256, 128]도 가능
    'window_size': 120,
    
    # Contrastive
    'temperature': 0.5,
    'beta': 4.0,
    
    # 학습
    'batch_size': 512,
    'learning_rate': 0.001,
    'max_epochs': 100,
}
```

---

## 📊 학습 순서 및 병렬화

### **추천 순서**

#### **Option 1: 순차 학습 (안전)**
```bash
# 1. Guardian 먼저 (가장 빠름, 2-4시간)
python ai/training/pipelines/regime_detection_pipeline.py

# 2. Oracle (TFT, 4-8시간)
python ai/training/pipelines/tft_training_pipeline.py

# 3. Strategist (가장 오래 걸림, 8-12시간)
python ai/training/pipelines/decision_transformer_training.py
```

**총 소요 시간**: 14-24시간

#### **Option 2: 병렬 학습 (빠름, GPU 2개 이상)**
```bash
# Terminal 1: TFT (GPU 0)
CUDA_VISIBLE_DEVICES=0 python ai/training/pipelines/tft_training_pipeline.py

# Terminal 2: Decision Transformer (GPU 1)
CUDA_VISIBLE_DEVICES=1 python ai/training/pipelines/decision_transformer_training.py

# Terminal 3: Guardian (GPU 2 or CPU)
CUDA_VISIBLE_DEVICES=2 python ai/training/pipelines/regime_detection_pipeline.py
```

**총 소요 시간**: 8-12시간 (병렬)

---

## 🎓 학습 모니터링

### **TensorBoard**
```bash
# 학습 중 실시간 모니터링
tensorboard --logdir models/ --port 6006

# 브라우저에서 열기
# http://localhost:6006
```

### **확인할 지표**
- **Loss 감소**: Training/Validation loss가 내려가는지
- **Overfitting**: Train loss는 낮은데 Val loss가 높으면 과적합
- **Learning Rate**: Learning rate schedule 확인
- **Gradient Norm**: Gradient explosion 없는지 확인

---

## 🚀 학습 완료 후

### **1. 모델 평가**
```bash
# TFT 평가
python scripts/evaluate_tft.py \
  --model-path models/tft/best_model.ckpt \
  --test-data data/historical_5min_features/BTCUSDT_2024_1m.parquet

# Decision Transformer 평가
python scripts/evaluate_dt.py \
  --model-path models/decision_transformer/best_model.ckpt

# Guardian 평가
python scripts/evaluate_guardian.py \
  --model-path models/guardian/best_model.ckpt
```

### **2. 백테스트**
```bash
# 통합 백테스트 (3개 모델 앙상블)
python backtesting/engine/backtest_engine.py \
  --oracle-model models/tft/best_model.ckpt \
  --strategist-model models/decision_transformer/best_model.ckpt \
  --guardian-model models/guardian/best_model.ckpt \
  --test-data data/historical_5min_features/BTCUSDT_2024_1m.parquet \
  --initial-capital 10000 \
  --output-dir results/backtest
```

### **3. ONNX 변환 (프로덕션 배포용)**
```bash
# 추론 속도 최적화
python scripts/convert_to_onnx.py \
  --tft-model models/tft/best_model.ckpt \
  --dt-model models/decision_transformer/best_model.ckpt \
  --guardian-model models/guardian/best_model.ckpt \
  --output-dir models/onnx
```

---

## 📈 예상 성능

### **Oracle (TFT)**
- **R²**: 0.35 - 0.65
- **RMSE**: 0.08% - 0.12%
- **Direction Accuracy**: 55% - 65%

### **Strategist (Decision Transformer)**
- **Sharpe Ratio**: 1.5 - 3.0
- **Win Rate**: 52% - 58%
- **Profit Factor**: 1.3 - 2.0

### **Guardian (Contrastive VAE)**
- **Regime Classification Accuracy**: 75% - 85%
- **Cluster Separation**: High silhouette score

### **통합 시스템 (백테스트 2024)**
- **Total Return**: +80% ~ +200%
- **Max Drawdown**: -15% ~ -30%
- **Sharpe Ratio**: 2.0 ~ 4.0
- **Win Rate**: 55% ~ 62%

---

## ⚠️ 주의사항

### **1. 과적합 방지**
- Early stopping 사용
- Dropout 유지 (0.1-0.2)
- 충분한 validation 데이터

### **2. 하이퍼파라미터 튜닝**
- Grid search 또는 Optuna 사용
- Learning rate 최적화 중요
- Batch size는 GPU 메모리 한도까지

### **3. 데이터 품질**
- NaN 값 확인
- Outlier 제거 확인
- Feature normalization 확인

---

## 📞 문제 해결

### **OOM (Out of Memory) 에러**
```python
# Batch size 줄이기
batch_size = 128  # 256 → 128

# Gradient accumulation
accumulate_grad_batches = 2

# Mixed precision
precision = 16
```

### **학습이 느릴 때**
```python
# num_workers 증가
num_workers = 8  # CPU 코어 수

# Pin memory
pin_memory = True

# Prefetch factor
prefetch_factor = 2
```

### **Validation loss가 안 떨어질 때**
- Learning rate 줄이기: 0.001 → 0.0001
- Batch size 늘리기: 128 → 256
- Regularization 추가: dropout 0.1 → 0.2

---

## 🎯 최종 체크리스트

- [ ] GPU 드라이버 및 CUDA 설치 확인
- [ ] PyTorch GPU 버전 설치
- [ ] 데이터 파일 존재 확인 (`data/historical_5min_features/`)
- [ ] 충분한 디스크 공간 (50GB 이상)
- [ ] TensorBoard 설치
- [ ] 3개 모델 학습 스크립트 실행
- [ ] 학습 진행 모니터링 (TensorBoard)
- [ ] 학습 완료 후 모델 평가
- [ ] 백테스트 실행
- [ ] 성능 분석 및 보고서 작성

---

## 🚀 시작하기

```bash
# 1. 프로젝트로 이동
cd /home/user/webapp

# 2. GPU 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 3. 학습 시작!
python ai/training/pipelines/tft_training_pipeline.py --use-gpu
```

**Good luck! 🎉**
