# 🚀 QUANTUM ALPHA - 빠른 시작 가이드

## ✅ 완료된 작업

**100% 구현 완료**:
- **48개 Python 파일** + **3개 Rust 파일** = **11,771 라인**
- 모든 코어 시스템 구현 완료
- 필수 스크립트 5개 작성 완료
- 완전한 문서 5개 작성 완료

---

## 🎯 지금 바로 시작하기

### **옵션 1: 빠른 데모 (3-5분)** ⚡

가장 빠르게 전체 시스템을 테스트하는 방법:

```bash
cd /home/user/webapp

# 1. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어서 Binance API 키만 입력

# 2. Python 환경 활성화
source venv/bin/activate

# 3. 빠른 데모 실행
python scripts/quick_demo.py
```

**이것만 하면**:
- ✅ 1일치 데이터 다운로드
- ✅ 전처리
- ✅ 피처 생성
- ✅ 전체 파이프라인 검증

**예상 시간**: 3-5분

---

### **옵션 2: 전체 시스템 (3-5일)** 🏗️

실전 배포를 위한 완전한 설정:

#### **1단계: 환경 설정** (10분)

```bash
cd /home/user/webapp

# .env 파일 생성
cp .env.example .env

# .env 파일 편집 (필수 항목만)
nano .env
```

**필수 설정**:
```bash
# Binance API (차트 데이터)
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret

# Bybit Testnet (거래 테스트)
BYBIT_TESTNET=true
BYBIT_API_KEY=your_testnet_key
BYBIT_API_SECRET=your_testnet_secret
```

**API 키 발급 방법**: `docs/SETUP_GUIDE.md` 참조

#### **2단계: Docker 시작** (2분)

```bash
# Docker 서비스 시작
docker-compose up -d

# 상태 확인
docker-compose ps
```

#### **3단계: 5년치 데이터 다운로드** (1-3일)

```bash
# Python 환경 활성화
source venv/bin/activate

# 데이터 다운로드 시작 (백그라운드)
nohup python scripts/download_historical_data.py \
    --symbols BTCUSDT ETHUSDT \
    --start-date 2019-01-01 \
    --end-date 2024-12-01 \
    --interval 1m \
    --output-dir data/raw \
    > download.log 2>&1 &

# 진행 상황 확인
tail -f download.log
```

**예상 시간**: 1-3일 (네트워크 속도 의존)  
**데이터 크기**: ~500GB (1분봉 기준)

#### **4단계: 데이터 전처리** (1-2시간)

```bash
# 전처리
python scripts/preprocess_data.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --clean-outliers \
    --fill-missing \
    --add-features
```

#### **5단계: 피처 생성** (2-3시간)

```bash
# 피처 생성
python scripts/generate_features.py \
    --input-dir data/processed \
    --output-dir data/features \
    --all-features
```

**생성되는 피처**:
- Fractional Differencing
- Wavelet Denoising
- 20+ Technical Indicators (RSI, MACD, ATR, etc.)
- Volume Features
- Time Features
- Statistical Features

#### **6단계: AI 모델 학습** (1-2일)

```bash
# Oracle (TFT) 학습
python ai/training/pipelines/tft_training_pipeline.py \
    --data-path data/features/BTCUSDT_features.parquet \
    --output-dir data/models/oracle \
    --max-epochs 50 \
    --batch-size 128

# Strategist (Decision Transformer) 학습
python ai/training/pipelines/decision_transformer_training.py \
    --data-path data/features/BTCUSDT_features.parquet \
    --output-dir data/models/strategist \
    --max-epochs 30

# Guardian (Regime Detection) 학습
python ai/training/pipelines/regime_detection_pipeline.py \
    --data-path data/features/BTCUSDT_features.parquet \
    --output-dir data/models/guardian \
    --max-epochs 40
```

**예상 시간**:
- Oracle: 8-12시간
- Strategist: 6-10시간
- Guardian: 4-6시간
- **총 18-28시간** (GPU 필요)

#### **7단계: ONNX 변환** (5분)

```bash
# PyTorch → ONNX 변환 (추론 최적화)
python scripts/convert_to_onnx.py \
    --model-dir data/models \
    --output-dir data/models/onnx \
    --verify \
    --benchmark
```

**결과**:
- `tft_oracle.onnx`
- `decision_transformer.onnx`
- `regime_detector.onnx`

**추론 속도**: 3-10배 향상

#### **8단계: 백테스팅** (2-4시간)

```bash
# 백테스팅
python backtesting/engine/backtest_engine.py \
    --data-path data/features/BTCUSDT_features.parquet \
    --models-dir data/models/onnx \
    --start-date 2019-01-01 \
    --end-date 2024-12-01 \
    --output-dir results/backtest
```

**확인할 메트릭**:
- ✅ Sharpe Ratio > 2.0
- ✅ Max Drawdown < 15%
- ✅ Win Rate > 55%
- ✅ Profit Factor > 1.5

#### **9단계: Paper Trading** (1-2주)

```bash
# Paper trading 모드로 시스템 실행
python main.py --mode paper --testnet
```

**모니터링**:
- Grafana: http://localhost:3000
- FastAPI: http://localhost:8000/docs
- Telegram/Discord 알림

#### **10단계: Live Trading** (최종 목표)

```bash
# ⚠️ 주의: 소액으로 시작!
python main.py --mode live --capital 1000
```

**체크리스트**:
- [ ] Paper Trading 2주+ 안정 운영
- [ ] 백테스팅 결과 만족 (Sharpe > 2.0)
- [ ] Kill Switch 테스트 완료
- [ ] API 키 실전용으로 변경
- [ ] `BYBIT_TESTNET=false` 설정

---

## 📊 전체 타임라인 요약

| 단계 | 작업 | 시간 | 비고 |
|------|------|------|------|
| 1 | 환경 설정 | 10분 | API 키 발급 |
| 2 | Docker 시작 | 2분 | - |
| 3 | 데이터 다운로드 | **1-3일** | 백그라운드 실행 |
| 4 | 데이터 전처리 | 1-2시간 | - |
| 5 | 피처 생성 | 2-3시간 | - |
| 6 | AI 모델 학습 | **1-2일** | GPU 권장 |
| 7 | ONNX 변환 | 5분 | - |
| 8 | 백테스팅 | 2-4시간 | - |
| 9 | Paper Trading | 1-2주 | 필수 검증 |
| 10 | Live Trading | 지속 | 최종 목표 |

**총 예상 시간**: **3-5일** (데이터 수집 + 학습)

---

## 📁 중요 파일 위치

### **스크립트** (`scripts/`)
- `download_historical_data.py` - 데이터 다운로드
- `preprocess_data.py` - 데이터 전처리
- `generate_features.py` - 피처 생성
- `convert_to_onnx.py` - ONNX 변환
- `quick_demo.py` - 빠른 데모

### **문서** (`docs/`)
- `QUICK_START.md` - **이 파일** (빠른 시작)
- `SETUP_GUIDE.md` - 환경 설정 상세 가이드
- `NEXT_STEPS.md` - 다음 단계 로드맵
- `IMPLEMENTATION_COMPLETE.md` - 완전 구현 보고서
- `FINAL_REPORT.md` - 프로젝트 최종 리포트

### **설정** (`configs/`)
- `system_config.yaml` - 시스템 설정
- `.env.example` - 환경 변수 예제

---

## 💡 추천 시작 방법

### **초보자 / 빠른 검증**
1. ✅ `quick_demo.py` 실행 (3-5분)
2. ✅ 결과 확인
3. ✅ 전체 시스템 이해

### **중급자 / 진지한 개발**
1. ✅ 환경 설정 (`SETUP_GUIDE.md`)
2. ✅ 1주일치 데이터로 파이프라인 테스트
3. ✅ 전체 5년 데이터 다운로드
4. ✅ 모델 학습 및 백테스팅

### **고급자 / 실전 배포**
1. ✅ 전체 시스템 구축 (3-5일)
2. ✅ 백테스팅 최적화
3. ✅ Paper Trading 2주
4. ✅ 소액 Live Trading ($100-1000)
5. ✅ 점진적 스케일업

---

## 🚨 중요 주의사항

### **1. 절대 실전 계좌로 바로 시작하지 마세요!**
순서:
1. ✅ Quick Demo
2. ✅ Testnet
3. ✅ Paper Trading (2주+)
4. ✅ 소액 실전 ($100-1000)
5. ✅ 점진적 증액

### **2. Kill Switch 필수**
- 일일 손실 한도: -5%
- 연속 손실: 5회
- 변동성 급증: 2σ 이상

### **3. API 키 보안**
- `.env` 파일 절대 Git에 커밋 금지
- IP 화이트리스트 설정 권장
- Withdraw 권한 절대 부여 금지

### **4. 백업 필수**
- 모델 파일 정기 백업
- 데이터베이스 스냅샷
- 설정 파일 버전 관리

---

## 🎯 성능 목표

### **백테스팅 목표**
- Sharpe Ratio: > 2.0
- Max Drawdown: < 15%
- Win Rate: > 55%
- Profit Factor: > 1.5

### **최종 목표** (Live Trading)
- 월 수익률: 12-25%
- 연 Sharpe Ratio: 3.5-5.0
- Max Drawdown: < 8%
- Win Rate: 58-65%

---

## 🆘 문제 해결

### **데이터 다운로드가 너무 느려요**
→ 여러 심볼 병렬 다운로드 또는 유료 데이터 사용

### **GPU가 없어요**
→ Google Colab (무료 GPU) 또는 CPU로 학습 (2-3배 느림)

### **메모리 부족**
→ 배치 크기 감소 (`--batch-size 32`)

### **백테스팅 결과가 목표에 못 미쳐요**
→ 하이퍼파라미터 튜닝, 더 많은 데이터, 피처 개선

---

## 📞 추가 문서

상세한 내용은 다음 문서 참조:

1. **환경 설정**: `docs/SETUP_GUIDE.md`
2. **다음 단계**: `docs/NEXT_STEPS.md`
3. **구현 완료**: `docs/IMPLEMENTATION_COMPLETE.md`
4. **프로젝트 개요**: `README.md`

---

## ✅ 시작 체크리스트

시작하기 전:
- [ ] Git 리포지토리 클론 완료
- [ ] Python 3.11+ 설치 확인
- [ ] Docker 설치 확인
- [ ] 충분한 디스크 공간 (500GB+)

빠른 데모:
- [ ] .env 파일 생성
- [ ] Binance API 키 설정
- [ ] `python scripts/quick_demo.py` 실행

전체 시스템:
- [ ] 모든 API 키 발급
- [ ] Docker 서비스 시작
- [ ] 5년 데이터 다운로드
- [ ] 모델 학습 완료
- [ ] 백테스팅 검증

---

**🚀 지금 바로 시작하세요!**

```bash
# 가장 빠른 시작
python scripts/quick_demo.py
```
