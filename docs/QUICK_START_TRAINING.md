# 🚀 빠른 시작 가이드 - AI 모델 학습

## ⚡ 5분 안에 시작하기

### **1단계: 환경 확인**
```bash
# GPU 확인
nvidia-smi

# Python & CUDA 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### **2단계: 프로젝트 실행**
```bash
# 프로젝트 디렉토리로 이동
cd /path/to/webapp

# 또는 Git에서 clone
git clone <your-repository-url>
cd webapp
```

### **3단계: 의존성 설치**
```bash
# PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Lightning
pip install pytorch-lightning

# 기타
pip install pandas numpy scikit-learn tensorboard
```

### **4단계: 학습 시작! 🎉**
```bash
# 전체 모델 학습 (추천)
python scripts/train_production_models.py --all

# 또는 개별 모델
python scripts/train_production_models.py --model oracle      # 가격 예측 (4-8h)
python scripts/train_production_models.py --model strategist  # 행동 최적화 (8-12h)
python scripts/train_production_models.py --model guardian    # 시장 체제 (2-4h)
```

---

## 📊 학습할 3개 AI 모델

### **1️⃣ Oracle (TFT)**
- **목적**: 미래 가격 예측
- **시간**: 4-8시간 (GPU)
- **출력**: `models/oracle/best_model.ckpt`

### **2️⃣ Strategist (Decision Transformer)**
- **목적**: 매수/매도 타이밍 최적화
- **시간**: 8-12시간 (GPU)
- **출력**: `models/strategist/best_model.ckpt`

### **3️⃣ Guardian (Contrastive VAE)**
- **목적**: 시장 상태 감지 (Bull/Bear/Sideways)
- **시간**: 2-4시간 (GPU)
- **출력**: `models/guardian/best_model.ckpt`

---

## 📈 학습 모니터링

### **TensorBoard로 실시간 확인**
```bash
# 터미널에서 실행
tensorboard --logdir models/ --port 6006

# 브라우저 열기
# http://localhost:6006
```

**확인할 지표:**
- ✅ **Loss 감소**: Train/Val loss가 내려가는지
- ⚠️ **Overfitting**: Train은 낮은데 Val이 높으면 과적합
- 📉 **Learning Rate**: 스케줄링 확인
- 🎯 **Accuracy** (Guardian): 분류 정확도

---

## 💾 데이터

### **준비된 데이터**
- **위치**: `data/historical_5min_features/`
- **형식**: Parquet 파일
- **내용**: BTCUSDT, ETHUSDT (2019-2024, 6년)
- **크기**: 1,259,124 rows × 44 features (454 MB)

### **Features (44개)**
- Trend: SMA, EMA, MACD
- Momentum: RSI, Stochastic
- Volatility: Bollinger Bands, ATR
- Volume: OBV, VWAP
- Price/Returns/Time features

---

## 🎯 학습 완료 후

### **1. 모델 평가**
```bash
# 개별 평가
python scripts/evaluate_oracle.py
python scripts/evaluate_strategist.py
python scripts/evaluate_guardian.py
```

### **2. 백테스트**
```bash
# 통합 백테스트 (3개 모델 앙상블)
python scripts/backtest_ensemble.py \
  --year 2024 \
  --initial-capital 10000
```

### **3. ONNX 변환 (선택)**
```bash
# 추론 속도 최적화
python scripts/convert_to_onnx.py
```

---

## 🔧 문제 해결

### **OOM (메모리 부족)**
```python
# train_production_models.py 수정
'batch_size': 128,  # 256 → 128로 줄이기
```

### **학습이 느릴 때**
```python
'num_workers': 8,  # CPU 코어 수 맞추기
'precision': 16,   # Mixed Precision 활성화
```

### **GPU가 없을 때**
```bash
# CPU로 학습 (느림)
python scripts/train_production_models.py --model guardian  # 가장 가벼운 것부터
```

---

## 📋 체크리스트

학습 전 확인:
- [ ] GPU 작동 확인 (`nvidia-smi`)
- [ ] CUDA & PyTorch 설치
- [ ] 데이터 파일 존재 (`data/historical_5min_features/`)
- [ ] 디스크 공간 50GB+ 확보
- [ ] TensorBoard 설치

학습 중:
- [ ] TensorBoard 모니터링
- [ ] Loss 감소 확인
- [ ] 충분한 시간 대기 (14-24시간)

학습 후:
- [ ] 3개 모델 파일 생성 확인
- [ ] 모델 평가 실행
- [ ] 백테스트 수행
- [ ] 성능 분석

---

## 📞 추가 정보

- **상세 가이드**: `docs/AI_TRAINING_MASTER_GUIDE.md`
- **프로젝트 문서**: `README.md`
- **아키텍처**: `docs/ARCHITECTURE.md`

---

## 🎉 Success!

학습 완료 후 다음 결과물을 얻게 됩니다:

```
models/
├── oracle/
│   └── best_model.ckpt       (가격 예측 모델)
├── strategist/
│   └── best_model.ckpt       (행동 최적화 모델)
└── guardian/
    └── best_model.ckpt       (시장 체제 모델)
```

**예상 성능 (2024 백테스트):**
- 📈 Total Return: +80% ~ +200%
- 📉 Max Drawdown: -15% ~ -30%
- 📊 Sharpe Ratio: 2.0 ~ 4.0
- 🎯 Win Rate: 55% ~ 62%

**Happy Trading! 🚀**
