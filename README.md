# Project QUANTUM ALPHA

**세계 최고 수준 암호화폐 선물 자동매매 시스템**

Renaissance Technologies, Citadel, Two Sigma의 기술력을 암호화폐 시장에 특화하여 재구성한 차세대 퀀트 트레이딩 플랫폼

## 🎯 핵심 성능 목표

### 수익성 지표
- 월간 수익률: **12-25%** (복리 기준)
- 연간 샤프 비율: **3.5-5.0**
- 최대 낙폭: **8% 이하**
- 승률: **58-65%**

### 운영 지표
- 실행 지연시간: **10ms 이하** (P99)
- 시스템 가용성: **99.95%**
- 일일 거래 횟수: **100-500회**
- 평균 보유시간: **30초-5분**

## 🏗️ 시스템 아키텍처: Trinity Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Master Orchestrator                         │
│           (시스템 전체 조율 및 장애 복구)                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬─────────────────────────────┐
    │             │             │                             │
┌───▼────┐  ┌────▼─────┐  ┌───▼────┐                  ┌────▼─────┐
│ Data   │  │ AI Core  │  │Execution│                 │ Risk &   │
│ Engine │  │ (Trinity)│  │ Engine  │                 │ Monitor  │
└────────┘  └──────────┘  └─────────┘                  └──────────┘
```

### The Trinity
1. **The Oracle (예측)**: 시장의 미래 확률 분포 예측
2. **The Strategist (대응)**: 예측과 현실 사이의 괴리에 최적 대응
3. **The Guardian (감시)**: 시장 국면 감지 및 시스템 보호

## 🛠️ 기술 스택

### Core Execution (Rust)
- 초저지연 주문 실행
- 실시간 데이터 수집
- 목표 지연시간: < 10ms

### AI Research (Python 3.11+)
- PyTorch 2.1+ (메인 프레임워크)
- PyTorch Lightning (학습 파이프라인)
- ONNX Runtime (추론 최적화)

### Data Processing
- **Hot Path**: Rust (실시간 스트리밍)
- **Analytics**: Python + Polars (배치 분석)
- **Storage**: TimescaleDB + Redis + S3

## 📊 데이터 소스

- **차트 데이터**: Binance Futures WebSocket
- **거래 실행**: Bybit API
- **실시간 수집**: 다중 거래소 동시 연결

## 🧠 AI Models

### 1. Temporal Fusion Transformer (TFT)
- Variable Selection Networks (중요 변수 자동 선택)
- Multi-Head Attention (장기 의존성)
- Quantile Regression (불확실성 추정)

### 2. Decision Transformer
- RL을 시퀀스 모델링 문제로 변환
- Return-to-go 조건부 최적 행동 생성
- GPT 스타일 아키텍처

### 3. Contrastive VAE (시장 국면 감지)
- 대조 학습 기반 비지도 학습
- 시장 국면 자동 발견
- 실시간 체제 변화 감지

## 📁 프로젝트 구조

```
quantum-alpha/
├── core/                       # Rust 고성능 코어
│   ├── data_collector/        # 실시간 데이터 수집
│   ├── order_executor/        # 초저지연 주문 실행
│   └── risk_manager/          # 리스크 관리 시스템
├── ai/                        # Python AI 모듈
│   ├── models/               # AI 모델 구현
│   │   ├── tft/             # Temporal Fusion Transformer
│   │   ├── decision_transformer/
│   │   └── regime_detection/
│   ├── features/            # 피처 엔지니어링
│   │   ├── preprocessing/
│   │   ├── orderflow/
│   │   └── technical/
│   ├── training/            # 학습 파이프라인
│   └── inference/           # 추론 최적화
├── backtesting/            # 백테스팅 엔진
├── monitoring/             # 모니터링 & 대시보드
├── configs/               # 설정 파일
└── docs/                 # 문서

```

## 🚀 시작하기

### 1. 환경 설정

```bash
# Python 환경
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Rust 환경
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. 데이터 수집 시작

```bash
# Rust 데이터 컬렉터 빌드
cd core/data_collector
cargo build --release

# 실행
cargo run --release
```

### 3. AI 모델 학습

```bash
# TFT 학습
python ai/training/train_tft.py --config configs/models/tft_config.yaml

# Decision Transformer 학습
python ai/training/train_decision_transformer.py
```

### 4. 백테스팅

```bash
python backtesting/run_backtest.py --strategy trinity --start-date 2024-01-01
```

## ⚠️ 리스크 관리

### Kill Switches
1. **일일 손실 한도**: 초과 시 자동 거래 중단
2. **연속 손실**: N회 연속 손실 시 긴급 정지
3. **변동성 급증**: 임계값 초과 시 포지션 축소
4. **API 에러율**: 높은 에러율 감지 시 중단
5. **모델 불일치**: 모델 간 큰 차이 발생 시 보수적 모드

### 포지션 관리
- 최대 레버리지: **10배**
- 심볼별 최대 포지션: 설정 가능
- 거래당 최대 리스크: 계좌의 1-2%

## 📈 성능 모니터링

- **실시간 대시보드**: Grafana + Prometheus
- **알림 시스템**: Telegram/Discord/Email
- **상세 로그**: 모든 거래 및 시스템 이벤트 기록

## 🔐 보안

- API 키 암호화 저장
- 2FA 인증
- IP 화이트리스트
- 주문 서명 검증

## 📚 문서

- [시스템 아키텍처](docs/architecture.md)
- [AI 모델 상세](docs/models.md)
- [백테스팅 가이드](docs/backtesting.md)
- [프로덕션 배포](docs/deployment.md)

## ⚡ 성능 최적화

### 지연시간 최적화
- Lock-free 데이터 구조
- Zero-copy 직렬화
- SIMD 벡터 연산
- 코로케이션 서버

### AI 추론 최적화
- ONNX Runtime
- Knowledge Distillation
- Quantization (INT8)
- TensorRT 가속

## 🧪 테스트

```bash
# Rust 테스트
cd core/data_collector && cargo test

# Python 테스트
pytest ai/tests/ -v

# 통합 테스트
python scripts/integration_test.py
```

## 📊 구현 현황

| 컴포넌트 | 상태 | 파일 | 라인 수 |
|----------|------|------|---------|
| 데이터 수집 (Rust) | ✅ 100% | core/data_collector | 317 |
| TFT 모델 | ✅ 100% | ai/models/tft | 426 |
| Decision Transformer | ✅ 100% | ai/models/decision_transformer | 542 |
| Contrastive VAE | ✅ 100% | ai/models/regime_detection | 682 |
| 피처 엔지니어링 | ✅ 100% | ai/features | 1,300+ |
| 주문 실행 (Rust) | ✅ 100% | core/order_executor | 642 |
| 리스크 관리 (Rust) | ✅ 100% | core/risk_manager | 700 |
| 학습 파이프라인 | ✅ 100% | ai/training | 962 |
| 백테스팅 엔진 | ✅ 100% | backtesting/engine | 592 |
| **총 코드** | **✅ 100%** | **37 Python + 3 Rust** | **6,878 라인** |

## 🤝 기여

이 프로젝트는 세계 최고 수준의 퀀트 트레이딩 시스템을 목표로 합니다.

## ⚖️ 면책 조항

**경고**: 이 시스템은 교육 및 연구 목적으로 제공됩니다. 실제 거래에서의 손실에 대해 책임지지 않습니다. 암호화폐 거래는 높은 리스크를 수반하며, 투자 원금을 전부 잃을 수 있습니다.

## 📄 라이선스

MIT License

---

**"완벽한 시스템은 없지만, 지속적으로 개선되는 시스템은 있다."**

Built with ❤️ for the future of algorithmic trading
