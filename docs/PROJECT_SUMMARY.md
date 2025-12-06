# QUANTUM ALPHA - 프로젝트 요약

## 🎯 프로젝트 개요

**Project QUANTUM ALPHA**는 세계 최고 수준의 암호화폐 선물 자동매매 시스템입니다. Renaissance Technologies, Citadel, Two Sigma의 기술력을 암호화폐 시장에 특화하여 재구성한 차세대 퀀트 트레이딩 플랫폼입니다.

## 📊 구현 현황

### ✅ 완료된 핵심 컴포넌트

#### 1. **데이터 수집 엔진 (Rust)**
- **파일**: `core/data_collector/src/main.rs`
- **기능**:
  - Binance Futures WebSocket 실시간 데이터 수집
  - Lock-free 큐를 이용한 초저지연 버퍼링
  - Redis 스트림 기반 데이터 저장
  - 평균 지연시간 < 10ms

#### 2. **AI 모델 - The Trinity**

##### 2.1 The Oracle (예측)
- **Temporal Fusion Transformer (TFT)**
  - 파일: `ai/models/tft/temporal_fusion_transformer.py`
  - Variable Selection Networks로 중요 변수 자동 선택
  - Multi-Head Attention으로 장기 의존성 학습
  - Quantile Regression으로 불확실성 추정
  
- **Decision Transformer**
  - 파일: `ai/models/decision_transformer/decision_transformer.py`
  - RL을 시퀀스 모델링 문제로 변환
  - Return-to-go 조건부 최적 행동 생성
  - GPT 스타일 Transformer 아키텍처

##### 2.2 The Guardian (감시)
- **Contrastive VAE (시장 국면 감지)**
  - 파일: `ai/models/regime_detection/contrastive_vae.py`
  - 대조 학습 기반 비지도 학습
  - K-means 클러스터링으로 4개 국면 자동 발견
  - 실시간 체제 변화 감지

#### 3. **고급 피처 엔지니어링**

- **Fractional Differencing**
  - 파일: `ai/features/preprocessing/fractional_differencing.py`
  - 메모리 보존 + 정상성 확보
  - Numba JIT 최적화

- **Order Flow Imbalance (OFI)**
  - 파일: `ai/features/orderflow/order_flow_imbalance.py`
  - 호가창 변화를 통한 단기 가격 압력 측정
  - 거래량 가중 OFI
  - 마이크로프라이스 계산

- **Wavelet Denoiser**
  - 파일: `ai/features/preprocessing/wavelet_denoiser.py`
  - 시간-주파수 영역 노이즈 제거
  - SURE 기반 적응형 임계값
  - 다중 레벨 웨이블릿 분석

#### 4. **주문 실행 엔진 (Rust)**
- **파일**: `core/order_executor/src/main.rs`
- **기능**:
  - Bybit API 통합 (선물 거래)
  - HMAC-SHA256 서명 인증
  - 스마트 오더 라우팅
  - 슬리피지 예측 및 조정
  - 목표 실행 지연 < 50ms

#### 5. **리스크 관리 시스템 (Rust)**
- **파일**: `core/risk_manager/src/main.rs`
- **기능**:
  - 7가지 Kill Switch 구현
    1. 일일 손실 한도
    2. 연속 손실 횟수
    3. 변동성 급증
    4. API 에러율
    5. 모델 불일치
    6. 낙폭 한도
    7. 레버리지 초과
  - 실시간 리스크 체크
  - 긴급 정지 시스템
  - 포지션 크기 제한

#### 6. **메인 오케스트레이터 (Python)**
- **파일**: `main.py`
- **기능**:
  - Trinity 컴포넌트 통합 조율
  - 비동기 트레이딩 루프
  - 시그널 처리 및 우아한 종료
  - 체계적인 로깅

#### 7. **인프라 및 배포**
- **Docker 컨테이너화**
  - Python AI 시스템: `Dockerfile`
  - Rust 컴포넌트: 각 서비스별 Dockerfile
  - Multi-stage build로 최적화

- **Docker Compose 오케스트레이션**
  - 파일: `docker-compose.yml`
  - 서비스:
    - TimescaleDB (시계열 DB)
    - Redis (실시간 캐시)
    - Prometheus (메트릭 수집)
    - Grafana (대시보드)
    - Data Collector (Rust)
    - Order Executor (Rust)
    - Risk Manager (Rust)
    - Main System (Python)

- **시작/중지 스크립트**
  - `scripts/start.sh`: 전체 시스템 시작
  - `scripts/stop.sh`: 우아한 종료

#### 8. **설정 관리**
- **시스템 설정**: `configs/system_config.yaml`
  - AI 모델 파라미터
  - 리스크 관리 설정
  - 거래 설정
  - 모니터링 설정

- **환경 변수**: `.env.example`
  - API 키 관리
  - 데이터베이스 연결
  - 알림 설정

## 📁 프로젝트 구조

```
quantum-alpha/
├── core/                      # Rust 고성능 코어
│   ├── data_collector/       # ✅ 실시간 데이터 수집
│   ├── order_executor/       # ✅ 초저지연 주문 실행
│   └── risk_manager/         # ✅ 리스크 관리 시스템
│
├── ai/                        # Python AI 모듈
│   ├── models/               # ✅ AI 모델 구현
│   │   ├── tft/             # ✅ Temporal Fusion Transformer
│   │   ├── decision_transformer/  # ✅ Decision Transformer
│   │   └── regime_detection/      # ✅ Contrastive VAE
│   │
│   ├── features/            # ✅ 피처 엔지니어링
│   │   ├── preprocessing/   # ✅ 분수 차분, 웨이블릿
│   │   └── orderflow/       # ✅ Order Flow Imbalance
│   │
│   ├── training/            # ⏳ 학습 파이프라인
│   └── inference/           # ⏳ 추론 최적화
│
├── backtesting/            # ⏳ 백테스팅 엔진
├── monitoring/             # ⏳ 모니터링 & 대시보드
├── configs/               # ✅ 설정 파일
├── scripts/              # ✅ 유틸리티 스크립트
├── docs/                # ✅ 문서
│
├── main.py              # ✅ 메인 오케스트레이터
├── Dockerfile           # ✅ Docker 이미지
├── docker-compose.yml   # ✅ 서비스 오케스트레이션
├── requirements.txt     # ✅ Python 의존성
└── README.md           # ✅ 프로젝트 문서
```

## 🚀 시작하기

### 1. 환경 설정

```bash
# 저장소 클론
cd /home/user/webapp

# 환경 변수 설정
cp .env.example .env
# .env 파일을 편집하여 API 키 입력

# Python 가상환경 생성
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. Docker로 실행

```bash
# 전체 시스템 시작
./scripts/start.sh

# 로그 확인
docker-compose logs -f quantum_alpha

# 중지
./scripts/stop.sh
```

### 3. 개별 컴포넌트 테스트

```bash
# Python AI 모델 테스트
cd ai/models/tft && python temporal_fusion_transformer.py

# Rust 데이터 컬렉터 테스트
cd core/data_collector && cargo run --release

# Rust 주문 실행기 테스트
cd core/order_executor && cargo run --release
```

## 📊 핵심 성능 지표

### 목표 성능
- **월간 수익률**: 12-25% (복리)
- **연간 샤프 비율**: 3.5-5.0
- **최대 낙폭**: 8% 이하
- **승률**: 58-65%

### 운영 지표
- **실행 지연시간**: 10ms 이하 (P99)
- **시스템 가용성**: 99.95%
- **일일 거래 횟수**: 100-500회
- **평균 보유시간**: 30초-5분

## 🛡️ 리스크 관리

### Kill Switch 시스템
1. **일일 손실 한도**: $500 초과 시 자동 정지
2. **연속 손실**: 5회 연속 손실 시 정지
3. **변동성 급증**: 5% 이상 변동성 감지 시
4. **API 에러율**: 10% 이상 에러 발생 시
5. **모델 불일치**: 예측 모델 간 큰 차이 발생 시
6. **낙폭 한도**: 15% 낙폭 초과 시
7. **레버리지 초과**: 10배 레버리지 초과 시

### 포지션 관리
- **최대 레버리지**: 10배
- **심볼별 최대 포지션**: 설정 가능
- **거래당 최대 리스크**: 계좌의 1-2%

## 🔧 기술 스택

### Core Execution (Rust)
- **언어**: Rust 1.75+
- **비동기**: Tokio
- **직렬화**: Serde
- **HTTP**: Reqwest
- **암호화**: HMAC-SHA256

### AI Research (Python)
- **딥러닝**: PyTorch 2.1+, PyTorch Lightning
- **데이터 처리**: Pandas, Polars, NumPy
- **피처**: Numba, PyWavelets, TA-Lib
- **최적화**: ONNX Runtime

### Data Storage
- **시계열 DB**: TimescaleDB
- **캐시**: Redis
- **스트리밍**: WebSocket

### DevOps
- **컨테이너**: Docker, Docker Compose
- **모니터링**: Prometheus, Grafana
- **로깅**: Loguru

## ⚠️ 다음 단계 (추후 구현 필요)

1. **백테스팅 엔진** (Priority: High)
   - Walk-forward 검증
   - Monte Carlo 시뮬레이션
   - 성능 메트릭 계산

2. **모델 학습 파이프라인** (Priority: High)
   - 자동화된 학습 스케줄
   - Hyperparameter tuning
   - 모델 체크포인팅

3. **모니터링 대시보드** (Priority: Medium)
   - Grafana 대시보드 설정
   - 실시간 성과 추적
   - 알림 시스템 통합

4. **Paper Trading 모드** (Priority: High)
   - 실제 거래 전 테스트
   - 성능 검증
   - 리스크 검증

5. **추론 최적화** (Priority: Medium)
   - ONNX 모델 변환
   - TensorRT 가속
   - Knowledge Distillation

## 📚 참고 자료

### AI 모델
- Temporal Fusion Transformer: [Lim et al., 2021]
- Decision Transformer: [Chen et al., 2021]
- Contrastive Learning: [Chen et al., 2020]

### 피처 엔지니어링
- Fractional Differencing: [Lopez de Prado, 2018]
- Order Flow Imbalance: [Hasbrouck & Saar, 2013]
- Wavelet Denoising: [Donoho & Johnstone, 1994]

## 🔐 보안 고려사항

1. **API 키 관리**: .env 파일 사용, 절대 커밋 금지
2. **2FA 인증**: 모든 거래소 계정에 2FA 활성화
3. **IP 화이트리스트**: 가능한 경우 IP 제한
4. **권한 최소화**: API 키는 필요한 권한만 부여
5. **정기 로테이션**: API 키 정기적 교체

## 📞 지원 및 문의

- **문서**: `/home/user/webapp/docs/`
- **로그**: `/home/user/webapp/logs/`
- **설정**: `/home/user/webapp/configs/`

## 📜 라이선스

MIT License

---

**"완벽한 시스템은 없지만, 지속적으로 개선되는 시스템은 있다."**

Built with ❤️ for the future of algorithmic trading
