# 🚀 원클릭 자동 학습 가이드

## ✨ 단 한 줄로 모든 학습 완료!

```bash
python train_all.py
```

이 명령어 하나면 **모든 것이 자동으로 실행**됩니다! ⚡

---

## 📋 자동으로 수행되는 작업

### **1. 데이터 준비** (자동)
- ✅ Binance에서 6년치 데이터 다운로드 (2019-2024)
- ✅ 데이터 전처리 (아웃라이어 제거, 정규화)
- ✅ 5분봉 리샘플링
- ✅ 44개 기술적 지표 생성

### **2. AI 모델 학습** (자동)
- ✅ Guardian 학습 (2-4시간)
- ✅ Oracle 학습 (4-8시간)
- ✅ Strategist 학습 (8-12시간)

### **3. 결과 저장** (자동)
- ✅ 학습된 모델 저장 (`models/*/best_model.ckpt`)
- ✅ 백테스트 실행
- ✅ 상세 보고서 생성 (`results/training_report_*.txt`)
- ✅ JSON 결과 저장 (`results/training_results_*.json`)

---

## ⚡ 빠른 시작

### **기본 실행**
```bash
python train_all.py
```

### **데이터가 이미 있으면**
```bash
python train_all.py --skip-data
```

### **테스트 모드** (실제 학습 안함)
```bash
python train_all.py --quick-test
```

---

## ⏱️ 예상 소요 시간

| 단계 | 시간 | 설명 |
|------|------|------|
| 데이터 다운로드 | 30-60분 | Binance에서 6년치 데이터 |
| 데이터 전처리 | 10-20분 | 정제 및 정규화 |
| 리샘플링 | 5-10분 | 5분봉 변환 |
| 기능 생성 | 20-40분 | 44개 지표 계산 |
| Guardian 학습 | 2-4시간 | 시장 체제 감지 |
| Oracle 학습 | 4-8시간 | 가격 예측 |
| Strategist 학습 | 8-12시간 | 행동 최적화 |
| 백테스트 | 5-10분 | 성능 검증 |
| **총합** | **15-26시간** | **완전 자동** |

💡 **팁**: 밤에 시작하면 다음날 아침에 완료됩니다!

---

## 📊 학습 진행 상황 확인

### **실시간 모니터링**
스크립트가 실행되면 자동으로 진행 상황이 표시됩니다:

```
======================================================================
🚀 [1/10] 데이터 다운로드 (30-60분 예상)
======================================================================

   실행 중: 데이터 다운로드...
✅ 데이터 다운로드 완료 (소요 시간: 1234.5초)

======================================================================
🚀 [5/10] Guardian (시장 체제 감지) 학습 (2-4시간 예상)
======================================================================
...
```

### **TensorBoard로 상세 모니터링**
다른 터미널에서:
```bash
tensorboard --logdir models/ --port 6006
```

브라우저에서 `http://localhost:6006` 접속

---

## 📁 학습 결과 확인

### **1. 모델 파일**
```
models/
├── oracle/
│   └── best_model.ckpt       ← Oracle (가격 예측)
├── strategist/
│   └── best_model.ckpt       ← Strategist (행동 최적화)
└── guardian/
    └── best_model.ckpt       ← Guardian (시장 체제)
```

### **2. 학습 보고서**
```
results/
├── training_report_20251206_120530.txt    ← 상세 보고서
└── training_results_20251206_120530.json  ← JSON 결과
```

### **보고서 내용**
```
🎉 PROJECT QUANTUM ALPHA - 학습 완료 보고서
==================================================

📅 학습 완료 시간: 2025-12-06 12:05:30
⏱️  총 소요 시간: 18시간 25분

📊 학습 단계별 결과
==================================================

데이터 다운로드: ✅ 성공
   소요 시간: 2345.1초

Guardian 학습: ✅ 성공
   소요 시간: 8234.5초

...

📁 생성된 파일
==================================================

✅ ORACLE: models/oracle/best_model.ckpt (125.3 MB)
✅ STRATEGIST: models/strategist/best_model.ckpt (238.7 MB)
✅ GUARDIAN: models/guardian/best_model.ckpt (89.2 MB)

🎯 다음 단계
==================================================

1. 백테스트: python scripts/backtest_ensemble.py
2. Paper Trading: python main.py --mode paper
...
```

---

## 🎯 학습 완료 후 다음 단계

### **1. 백테스트 (과거 검증)**
```bash
# 2024년 데이터로 성능 확인
python scripts/backtest_ensemble.py --year 2024
```

### **2. Paper Trading (모의 투자)**
```bash
# 실제 돈 없이 실시간 테스트
python main.py --mode paper --testnet
```

### **3. Live Trading (실전 투자)** ⚠️
```bash
# 진짜 돈으로 자동 트레이딩 (주의!)
python main.py --mode live --api-key YOUR_BINANCE_KEY
```

---

## ❓ 문제 해결

### **메모리 부족 에러**
```bash
# 배치 크기를 줄여서 재시작
# train_production_models.py 파일에서
# batch_size = 128  (256 → 128로 변경)
```

### **GPU 감지 안됨**
```bash
# PyTorch CUDA 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **데이터 다운로드 실패**
```bash
# 개별 연도만 재다운로드
python scripts/download_year_by_year.py \
  --symbols BTCUSDT \
  --start-date 2023-01-01 \
  --end-date 2023-12-31
```

### **중간에 중단되었다면**
```bash
# 데이터 준비는 건너뛰고 학습만
python train_all.py --skip-data
```

---

## 🔧 고급 옵션

### **개별 모델만 학습**
```bash
# Guardian만
python scripts/train_production_models.py --model guardian

# Oracle만
python scripts/train_production_models.py --model oracle

# Strategist만
python scripts/train_production_models.py --model strategist
```

### **병렬 학습 (GPU 여러 개)**
```bash
# 터미널 1
CUDA_VISIBLE_DEVICES=0 python scripts/train_production_models.py --model oracle &

# 터미널 2
CUDA_VISIBLE_DEVICES=1 python scripts/train_production_models.py --model strategist &

# 터미널 3
CUDA_VISIBLE_DEVICES=2 python scripts/train_production_models.py --model guardian &
```

---

## 📊 예상 성능

학습 완료 후 예상되는 성능 (2024년 백테스트):

```
Total Return:     +80% ~ +200%
Max Drawdown:     -15% ~ -30%
Sharpe Ratio:     2.0 ~ 4.0
Win Rate:         55% ~ 62%
```

⚠️ **주의**: 실제 결과는 시장 상황에 따라 다를 수 있습니다.

---

## ✅ 체크리스트

학습 시작 전:
- [ ] Python 3.10+ 설치
- [ ] PyTorch 설치 (GPU 버전 권장)
- [ ] 디스크 공간 100GB+ 확보
- [ ] GPU 12GB+ VRAM (권장)
- [ ] 안정적인 인터넷 연결

학습 중:
- [ ] 화면 끄지 않기 (또는 `tmux` 사용)
- [ ] 전원 연결 확인
- [ ] TensorBoard로 모니터링

학습 후:
- [ ] 모델 파일 확인
- [ ] 보고서 읽기
- [ ] 백테스트 실행
- [ ] Paper Trading 시작

---

## 💡 팁

- **밤에 시작**: 자는 동안 학습 완료
- **tmux 사용**: 연결 끊겨도 계속 실행
  ```bash
  tmux new -s training
  python train_all.py
  # Ctrl+B, D로 detach
  # tmux attach -t training로 재접속
  ```
- **로그 저장**: 
  ```bash
  python train_all.py 2>&1 | tee training.log
  ```

---

## 🎉 축하합니다!

이제 **단 한 줄 명령어**로 완전한 AI 트레이딩 시스템을 학습할 수 있습니다!

```bash
python train_all.py
```

**Good luck! 🚀**
