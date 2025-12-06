

# 🛠️ QUANTUM ALPHA - 환경 설정 가이드

## 📋 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [API 키 발급](#api-키-발급)
3. [환경 변수 설정](#환경-변수-설정)
4. [Docker 설정](#docker-설정)
5. [Python 환경 설정](#python-환경-설정)
6. [데이터베이스 초기화](#데이터베이스-초기화)
7. [검증](#검증)

---

## 🖥️ 시스템 요구사항

### 최소 사양
- **CPU**: 4+ cores (Intel i5 또는 동급)
- **RAM**: 16GB
- **Storage**: 500GB SSD
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows 10/11
- **Network**: 안정적인 인터넷 연결 (1Mbps+)

### 권장 사양 (모델 학습용)
- **CPU**: 8+ cores (Intel i7/i9, AMD Ryzen 7/9)
- **GPU**: NVIDIA RTX 3080+ (12GB VRAM)
- **RAM**: 32GB+
- **Storage**: 1TB+ NVMe SSD
- **Network**: 10Mbps+ (데이터 다운로드용)

### 필수 소프트웨어
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Python**: 3.11+
- **Git**: 2.30+
- **Rust**: 1.75+ (Rust 컴포넌트 빌드용, 선택사항)

---

## 🔑 API 키 발급

### 1. Binance API (차트 데이터용)

**목적**: 실시간 시장 데이터 수집 (Read-Only)

**발급 절차**:
1. [Binance](https://www.binance.com) 계정 로그인
2. **프로필** → **API Management** 클릭
3. **Create API** 버튼 클릭
4. API 이름 입력 (예: "Quantum Alpha Data")
5. **이메일 인증** 및 **2FA 인증** 완료
6. **API Key** 및 **Secret Key** 저장

**권한 설정**:
- ✅ **Enable Reading** (읽기 권한만)
- ❌ **Enable Spot & Margin Trading** (거래 권한 불필요)
- ❌ **Enable Futures** (거래 권한 불필요)
- ❌ **Enable Withdrawals** (출금 권한 절대 금지)

**보안 설정**:
- **IP Access Restriction**: 사용 중인 IP 추가 권장
- **API Key Restrictions**: "Enable Reading" 만 체크

---

### 2. Bybit API (거래 실행용)

**⚠️ 중요**: 처음엔 반드시 **Testnet**으로 시작!

#### 2.1 Testnet API (테스트용)

**목적**: 가상 자금으로 거래 테스트

**발급 절차**:
1. [Bybit Testnet](https://testnet.bybit.com) 접속
2. 계정 생성 (실제 계정과 별도)
3. **API** → **Create New Key** 클릭
4. API 이름 입력
5. **API Key** 및 **Secret Key** 저장

**Testnet 특징**:
- 가상 자금 제공 (무료)
- 실제 시장과 동일한 환경
- 실제 돈 손실 없음
- Paper Trading에 적합

#### 2.2 Mainnet API (실전용)

**⚠️ 주의**: Testnet에서 충분히 테스트 후 사용!

**발급 절차**:
1. [Bybit](https://www.bybit.com) 실전 계정 로그인
2. **API** → **Create New Key**
3. API 이름 입력
4. **권한 설정**:
   - ✅ **Trade** (거래 권한)
   - ❌ **Withdraw** (출금 권한 절대 금지)
5. **IP Whitelist** 설정 (강력 권장)
6. **API Key** 및 **Secret Key** 저장

---

### 3. Telegram Bot (알림용, 선택사항)

**목적**: 실시간 거래 알림 수신

**발급 절차**:
1. Telegram에서 [@BotFather](https://t.me/BotFather) 검색
2. `/newbot` 명령 전송
3. 봇 이름 입력 (예: "Quantum Alpha Bot")
4. 봇 사용자 이름 입력 (예: "quantum_alpha_bot")
5. **API Token** 저장 (형식: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

**Chat ID 얻기**:
1. 생성한 봇과 대화 시작 (아무 메시지 전송)
2. 브라우저에서 `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates` 접속
3. `"chat":{"id":123456789}` 부분에서 숫자 확인
4. 이 숫자가 **Chat ID**

---

### 4. Discord Webhook (알림용, 선택사항)

**목적**: Discord 서버에 거래 알림 전송

**발급 절차**:
1. Discord 서버 설정 → **통합** → **웹후크**
2. **웹후크 만들기** 클릭
3. 웹후크 이름 및 채널 선택
4. **웹후크 URL 복사** (형식: `https://discord.com/api/webhooks/...`)

---

## ⚙️ 환경 변수 설정

### 1. `.env` 파일 생성

```bash
cd /home/user/webapp
cp .env.example .env
```

### 2. `.env` 파일 편집

```bash
nano .env  # 또는 vim, code 등
```

### 3. 필수 변수 설정

```bash
# ===== Binance API (차트 데이터) =====
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_secret_here

# ===== Bybit API (거래 실행) =====
# Testnet (처음엔 이것 사용)
BYBIT_TESTNET=true
BYBIT_API_KEY=your_bybit_testnet_api_key
BYBIT_API_SECRET=your_bybit_testnet_secret

# Mainnet (나중에 실전 시)
# BYBIT_TESTNET=false
# BYBIT_API_KEY=your_bybit_mainnet_api_key
# BYBIT_API_SECRET=your_bybit_mainnet_secret

# ===== Database =====
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://quantum:quantum123@localhost:5432/quantum_alpha

# ===== Telegram (선택사항) =====
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789

# ===== Discord (선택사항) =====
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_here

# ===== Email (선택사항) =====
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=465
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
RECEIVER_EMAILS=receiver1@example.com,receiver2@example.com

# ===== System Config =====
LOG_LEVEL=INFO
DATA_DIR=/home/user/webapp/data
MODEL_DIR=/home/user/webapp/data/models
```

### 4. 파일 권한 설정

```bash
chmod 600 .env  # .env 파일 보호
```

---

## 🐳 Docker 설정

### 1. Docker 설치 확인

```bash
docker --version  # Docker version 20.10+
docker-compose --version  # Docker Compose version 2.0+
```

### 2. Docker Compose 서비스 시작

```bash
cd /home/user/webapp
docker-compose up -d
```

**실행되는 서비스**:
- **TimescaleDB**: PostgreSQL 기반 시계열 DB (포트 5432)
- **Redis**: 인메모리 캐시 (포트 6379)
- **Prometheus**: 메트릭 수집 (포트 9090)
- **Grafana**: 시각화 대시보드 (포트 3000)

### 3. 서비스 상태 확인

```bash
docker-compose ps
```

**예상 출력**:
```
NAME                COMMAND             STATUS          PORTS
timescaledb         docker-entrypoint   Up 2 minutes    0.0.0.0:5432->5432/tcp
redis               redis-server        Up 2 minutes    0.0.0.0:6379->6379/tcp
prometheus          /bin/prometheus     Up 2 minutes    0.0.0.0:9090->9090/tcp
grafana             /run.sh             Up 2 minutes    0.0.0.0:3000->3000/tcp
```

### 4. 서비스 접속 테스트

- **Grafana**: http://localhost:3000 (admin / admin)
- **Prometheus**: http://localhost:9090
- **Redis**: `redis-cli -h localhost -p 6379 ping` → `PONG`

---

## 🐍 Python 환경 설정

### 1. Python 버전 확인

```bash
python --version  # Python 3.11+
```

**Python 3.11 설치** (Ubuntu 기준):
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### 2. 가상 환경 생성

```bash
cd /home/user/webapp
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac

# Windows:
# venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**예상 시간**: 5-10분

**주요 패키지**:
- PyTorch 2.1+ (딥러닝)
- PyTorch Lightning (학습 추상화)
- ONNX Runtime (추론 최적화)
- Polars (빅데이터 처리)
- FastAPI (API 서빙)
- 기타 100+ 패키지

### 4. GPU 지원 (선택사항)

NVIDIA GPU가 있는 경우:

```bash
# CUDA 11.8 기준
pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

GPU 확인:
```bash
python -c "import torch; print(torch.cuda.is_available())"  # True
```

---

## 🗄️ 데이터베이스 초기화

### 1. TimescaleDB 스키마 생성

```bash
# Docker 컨테이너 접속
docker exec -it timescaledb psql -U quantum -d quantum_alpha
```

SQL 실행:
```sql
-- 시계열 테이블 생성
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION
);

-- Hypertable 변환 (시계열 최적화)
SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
ON market_data (symbol, time DESC);

-- 종료
\q
```

### 2. Redis 연결 테스트

```bash
redis-cli -h localhost -p 6379
```

Redis 명령:
```bash
127.0.0.1:6379> PING
PONG
127.0.0.1:6379> SET test "hello"
OK
127.0.0.1:6379> GET test
"hello"
127.0.0.1:6379> DEL test
(integer) 1
127.0.0.1:6379> exit
```

---

## ✅ 검증

### 1. 환경 변수 확인

```bash
cd /home/user/webapp
source venv/bin/activate
python -c "import os; print('BINANCE_API_KEY:', os.getenv('BINANCE_API_KEY')[:10] + '...')"
```

### 2. API 연결 테스트

```bash
# Binance API 테스트
python -c "
import requests
response = requests.get('https://fapi.binance.com/fapi/v1/ping')
print('Binance API:', 'OK' if response.status_code == 200 else 'FAIL')
"

# Bybit API 테스트 (Testnet)
python -c "
import requests
response = requests.get('https://api-testnet.bybit.com/v5/market/time')
print('Bybit Testnet API:', 'OK' if response.status_code == 200 else 'FAIL')
"
```

### 3. 모든 서비스 확인

```bash
# 스크립트 실행
python scripts/check_setup.py
```

**예상 출력**:
```
✅ Python 3.11+ detected
✅ Docker is running
✅ Redis connection OK
✅ TimescaleDB connection OK
✅ Binance API OK
✅ Bybit Testnet API OK
✅ All dependencies installed

Setup Status: READY ✅
```

---

## 🚀 다음 단계

환경 설정이 완료되었습니다!

**다음 작업**:
1. **데이터 다운로드**: `docs/NEXT_STEPS.md` 참조
2. **빠른 테스트**: `scripts/quick_demo.py` 실행
3. **전체 학습**: 5년치 데이터 다운로드 및 모델 학습

---

## 🛠️ 문제 해결

### Docker 관련

**문제**: `Cannot connect to the Docker daemon`
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER  # 재로그인 필요
```

**문제**: 포트 충돌
```bash
# 사용 중인 포트 확인
sudo lsof -i :5432  # TimescaleDB
sudo lsof -i :6379  # Redis
sudo lsof -i :3000  # Grafana

# 프로세스 종료
sudo kill -9 <PID>
```

### Python 관련

**문제**: 의존성 설치 실패
```bash
# 시스템 패키지 설치 (Ubuntu)
sudo apt-get install python3.11-dev build-essential

# pip 업그레이드
pip install --upgrade pip setuptools wheel
```

**문제**: CUDA 버전 불일치
```bash
# CUDA 버전 확인
nvcc --version

# 맞는 PyTorch 설치
# https://pytorch.org/get-started/locally/
```

### API 관련

**문제**: Binance API 403 Forbidden
- IP 제한 확인
- API 키 권한 확인 (Read-Only 필요)

**문제**: Bybit API 10002 (Invalid API key)
- Testnet/Mainnet 확인
- API Key/Secret 재확인

---

## 📞 추가 지원

문제가 계속되면:
1. `docs/NEXT_STEPS.md` 참조
2. 로그 확인: `docker-compose logs`
3. GitHub Issues 등록

**환경 설정 완료!** 🎉
