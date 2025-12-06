# 🎯 QUANTUM ALPHA - 지금 해야 할 일 & 남은 작업

## 📊 현재 상태

### ✅ 완료된 것 (100%)
- **전체 시스템 아키텍처**: 43개 Python + 3개 Rust 파일 (9,902 라인)
- **AI 모델 코드**: TFT, Decision Transformer, Regime Detection
- **피처 엔지니어링**: Fractional Differencing, OFI, Volume Profile, Wavelet
- **실행 엔진**: Rust 기반 Order Executor, Risk Manager
- **백테스팅**: Performance Metrics, Vectorized Engine
- **모니터링**: Telegram, Discord, Email 알림
- **추론 엔진**: ONNX 최적화, FastAPI 서빙

### ❌ 아직 안 된 것 (해야 할 일)
1. **5년치 데이터가 없음** ⚠️
2. **AI 모델이 학습되지 않음** ⚠️
3. **백테스팅 미실행** ⚠️
4. **환경 변수 미설정** ⚠️

---

## 🚀 지금 바로 해야 할 일 (우선순위 순)

### **STEP 1: 환경 설정** (5분) 🔧

#### 1.1 환경 변수 파일 생성
```bash
cp .env.example .env
```

그 다음 `.env` 파일을 열어서 다음 내용을 채워야 합니다:

```bash
# Binance API (차트 데이터용)
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# Bybit API (거래 실행용)
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_api_secret
BYBIT_TESTNET=true  # 처음엔 Testnet으로 시작

# Database
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://user:password@localhost:5432/quantum_alpha

# Telegram (선택사항 - 알림용)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Discord (선택사항 - 알림용)
DISCORD_WEBHOOK_URL=your_webhook_url

# Email (선택사항 - 알림용)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=465
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
RECEIVER_EMAILS=receiver1@example.com,receiver2@example.com
```

#### 1.2 API 키 발급 방법

**Binance API** (차트 데이터):
1. Binance 계정 로그인
2. API Management 메뉴
3. Create API 클릭
4. "Read Only" 권한만 활성화 (거래는 안 함)

**Bybit API** (거래 실행):
1. Bybit 계정 로그인
2. API → Create New Key
3. **처음엔 Testnet API 사용 권장**
4. Testnet: https://testnet.bybit.com

---

### **STEP 2: 데이터 수집** (1-3일) 📊

**문제**: 5년치 원시 데이터가 없습니다.

**해결 방법 2가지**:

#### 옵션 A: 직접 다운로드 (무료, 시간 소요)

```bash
# 데이터 다운로드 스크립트 실행
cd /home/user/webapp
python scripts/download_historical_data.py \
    --symbols BTCUSDT ETHUSDT \
    --start-date 2019-01-01 \
    --end-date 2024-12-01 \
    --interval 1m \
    --output-dir data/raw
```

**예상 시간**: 1-3일 (네트워크 속도 의존)
**데이터 크기**: ~500GB (1분봉 기준)

#### 옵션 B: 기존 데이터 구매 (빠름, 유료)

**추천 제공자**:
1. **CryptoDataDownload** (무료): https://www.cryptodatadownload.com/
2. **Kaiko** (유료): 고품질 기관급 데이터
3. **CoinAPI** (유료): REST API로 다운로드

---

### **STEP 3: 데이터 전처리** (1-2시간) 🔬

데이터를 다운로드한 후:

```bash
# 1. 데이터 정제
python scripts/preprocess_data.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --clean-outliers \
    --fill-missing

# 2. 피처 생성
python scripts/generate_features.py \
    --input-dir data/processed \
    --output-dir data/features \
    --all-features
```

**생성되는 피처**:
- Fractional Differencing
- Order Flow Imbalance
- Volume Profile (POC, VAH, VAL)
- Wavelet Denoising
- 20+ Technical Indicators (RSI, MACD, ATR, etc.)

---

### **STEP 4: AI 모델 학습** (1-2일) 🧠

#### 4.1 The Oracle (TFT) 학습

```bash
python ai/training/pipelines/tft_training_pipeline.py \
    --data-path data/features/BTCUSDT_features.parquet \
    --output-dir data/models/oracle \
    --max-epochs 50 \
    --batch-size 128 \
    --learning-rate 1e-3
```

**예상 시간**: 8-12시간 (GPU 필요)
**출력**: `data/models/oracle/tft_best.pt`

#### 4.2 The Strategist (Decision Transformer) 학습

```bash
python ai/training/pipelines/decision_transformer_training.py \
    --data-path data/features/BTCUSDT_features.parquet \
    --output-dir data/models/strategist \
    --max-epochs 30 \
    --batch-size 64
```

**예상 시간**: 6-10시간 (GPU 필요)
**출력**: `data/models/strategist/decision_transformer_best.pt`

#### 4.3 The Guardian (Regime Detection) 학습

```bash
python ai/training/pipelines/regime_detection_pipeline.py \
    --data-path data/features/BTCUSDT_features.parquet \
    --output-dir data/models/guardian \
    --max-epochs 40 \
    --n-clusters 4
```

**예상 시간**: 4-6시간
**출력**: `data/models/guardian/regime_detector_best.pt`

#### 4.4 ONNX 변환 (추론 최적화)

```bash
# PyTorch → ONNX 변환
python scripts/convert_to_onnx.py \
    --model-dir data/models \
    --output-dir data/models/onnx
```

**출력**:
- `data/models/onnx/tft_oracle.onnx`
- `data/models/onnx/decision_transformer.onnx`
- `data/models/onnx/regime_detector.onnx`

---

### **STEP 5: 백테스팅** (2-4시간) 📈

```bash
# 전체 시스템 백테스팅
python backtesting/engine/backtest_engine.py \
    --data-path data/features/BTCUSDT_features.parquet \
    --models-dir data/models/onnx \
    --start-date 2019-01-01 \
    --end-date 2024-12-01 \
    --initial-capital 10000 \
    --output-dir results/backtest
```

**출력**:
- `results/backtest/performance_metrics.json`
- `results/backtest/equity_curve.png`
- `results/backtest/trades.csv`

**확인할 메트릭**:
- ✅ Sharpe Ratio > 2.0
- ✅ Max Drawdown < 15%
- ✅ Win Rate > 55%
- ✅ Profit Factor > 1.5

---

### **STEP 6: Paper Trading** (1-2주) 🧪

**목적**: 실제 시장에서 가상 자금으로 테스트

```bash
# Testnet에서 실행
docker-compose up -d

# Paper trading 모드로 메인 시스템 실행
python main.py --mode paper --testnet
```

**모니터링**:
- Grafana 대시보드: http://localhost:3000
- FastAPI 문서: http://localhost:8000/docs
- Telegram/Discord 알림 확인

**확인 사항**:
1. 주문 실행 지연시간 < 100ms
2. Kill Switch 정상 작동
3. 모든 알림 정상 수신
4. 메모리/CPU 사용량 안정

---

### **STEP 7: Live Trading** (지속) 💰

**⚠️ 주의**: 소액으로 시작!

```bash
# 실전 모드 (소액 $100-1000)
python main.py --mode live --capital 1000
```

**체크리스트**:
- [ ] API 키 실전용으로 변경
- [ ] `BYBIT_TESTNET=false`로 설정
- [ ] Kill Switch 임계값 재확인
- [ ] 알림 시스템 테스트 완료
- [ ] 백업 플랜 준비

---

## 📝 필요한 스크립트 작성

다음 스크립트들을 작성해야 합니다:

### 1. `scripts/download_historical_data.py`
- Binance API에서 5년치 데이터 다운로드
- 여러 심볼 동시 처리
- 체크포인트 지원 (중단 시 재개)

### 2. `scripts/preprocess_data.py`
- 결측값 처리
- 이상치 제거
- 데이터 정규화

### 3. `scripts/generate_features.py`
- 모든 피처 생성
- Parquet 형식 저장

### 4. `scripts/convert_to_onnx.py`
- PyTorch → ONNX 변환
- 최적화 적용

---

## ⏱️ 전체 타임라인

| 단계 | 작업 | 예상 시간 | 필수 여부 |
|------|------|-----------|-----------|
| 1 | 환경 설정 | 5분 | ✅ 필수 |
| 2 | 데이터 수집 | 1-3일 | ✅ 필수 |
| 3 | 데이터 전처리 | 1-2시간 | ✅ 필수 |
| 4 | AI 모델 학습 | 1-2일 | ✅ 필수 |
| 5 | 백테스팅 | 2-4시간 | ✅ 필수 |
| 6 | Paper Trading | 1-2주 | ⚠️ 강력 권장 |
| 7 | Live Trading | 지속 | 💰 최종 목표 |

**총 예상 시간**: 3-5일 (데이터 수집 제외)

---

## 🛠️ 필요한 하드웨어/소프트웨어

### 최소 사양
- **CPU**: 4+ cores
- **RAM**: 16GB+
- **Storage**: 500GB+ SSD
- **Network**: 안정적인 인터넷 연결

### 권장 사양 (모델 학습용)
- **GPU**: NVIDIA RTX 3080+ (12GB VRAM)
- **RAM**: 32GB+
- **Storage**: 1TB+ NVMe SSD

### 소프트웨어
- **Docker**: 필수
- **Docker Compose**: 필수
- **Python 3.11+**: 필수
- **Rust 1.75+**: Rust 컴포넌트 빌드용
- **CUDA 12.1+**: GPU 학습용 (선택)

---

## 💡 빠른 시작 옵션

### 옵션 1: 전체 시스템 (3-5일)
위의 모든 단계를 순서대로 실행

### 옵션 2: 데모 모드 (1시간)
소규모 데이터로 빠른 테스트:

```bash
# 1일치 데이터만 다운로드
python scripts/download_historical_data.py \
    --symbols BTCUSDT \
    --start-date 2024-12-01 \
    --end-date 2024-12-02 \
    --interval 1m

# 간단한 모델 학습 (10 epochs)
python ai/training/pipelines/tft_training_pipeline.py \
    --max-epochs 10 \
    --quick-test

# 1일치 백테스팅
python backtesting/engine/backtest_engine.py \
    --start-date 2024-12-01 \
    --end-date 2024-12-02
```

---

## 🎯 핵심 포인트

### 지금 당장 할 일 (우선순위):
1. **환경 변수 설정** (.env 파일)
2. **API 키 발급** (Binance + Bybit Testnet)
3. **데이터 다운로드 시작** (백그라운드에서 실행)
4. **필요 스크립트 작성** (download_historical_data.py 등)

### 1주일 내 목표:
- ✅ 데이터 수집 완료
- ✅ AI 모델 학습 완료
- ✅ 백테스팅 결과 확인

### 1개월 내 목표:
- ✅ Paper Trading 안정화
- ✅ 성능 검증 (Sharpe > 2.0)
- ✅ Live Trading 준비 완료

---

## 🚨 중요 주의사항

1. **절대 실전 계좌로 바로 시작하지 마세요!**
   - 반드시 Testnet → Paper Trading → 소액 실전 순서

2. **Kill Switch 필수**
   - 일일 손실 한도: -5%
   - 연속 손실: 5회
   - 이 설정은 `configs/system_config.yaml`에서 조정

3. **API 키 보안**
   - `.env` 파일을 절대 Git에 커밋하지 마세요
   - IP 화이트리스트 설정 권장

4. **백업 플랜**
   - 모델 파일 정기 백업
   - 데이터베이스 스냅샷
   - 설정 파일 버전 관리

---

## 📞 문제 발생 시

### 일반적인 문제

**Q: 데이터 다운로드가 너무 느려요**
A: 여러 심볼을 병렬로 다운로드하거나, 유료 데이터 제공자 사용

**Q: GPU가 없어요**
A: CPU만으로도 학습 가능하지만 2-3배 느립니다. Google Colab (무료 GPU) 사용 권장

**Q: 메모리 부족 에러**
A: 배치 크기 감소 (`--batch-size 32`로 줄이기)

**Q: 백테스팅 결과가 목표에 못 미쳐요**
A: 하이퍼파라미터 튜닝, 더 많은 데이터, 피처 개선 필요

---

## ✅ 체크리스트

시작하기 전에 확인:

- [ ] .env 파일 생성 및 API 키 설정
- [ ] Docker 설치 확인 (`docker --version`)
- [ ] Python 3.11+ 설치 확인 (`python --version`)
- [ ] 충분한 디스크 공간 (500GB+)
- [ ] Git 커밋 완료 (현재 코드 백업)

모델 학습 전 확인:

- [ ] 데이터 다운로드 완료
- [ ] 데이터 전처리 완료
- [ ] 피처 생성 완료
- [ ] GPU 사용 가능 (선택사항)

실전 배포 전 확인:

- [ ] 백테스팅 결과 만족
- [ ] Paper Trading 2주+ 안정 운영
- [ ] Kill Switch 테스트 완료
- [ ] 알림 시스템 정상 작동
- [ ] 백업 플랜 준비 완료

---

**다음 단계**: 제가 필요한 스크립트들을 작성해드릴까요? 아니면 특정 단계부터 시작하시겠습니까?
