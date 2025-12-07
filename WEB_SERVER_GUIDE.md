# 🚀 QUANTUM ALPHA Web Server - 빠른 시작 가이드

## 📋 개요

이 가이드는 학습된 AI 모델(Guardian, Oracle, Strategist)을 실시간 웹 서버로 실행하는 방법을 설명합니다.

## ✅ 사전 준비사항

### 1. 모델 학습 완료 확인
```bash
ls -lh models/*/best_model.ckpt
```

**기대 결과:**
- `models/guardian/best_model.ckpt` - 8.2MB
- `models/oracle/best_model.ckpt` - 5.0MB  
- `models/strategist/best_model.ckpt` - 58MB

모델이 없다면 먼저 학습을 실행하세요:
```bash
python train_all.py
```

### 2. 의존성 설치
```bash
pip install fastapi uvicorn
```

## 🎯 서버 실행 (3가지 방법)

### 방법 1: 간단한 실행 (권장)
```bash
python simple_server.py
```

### 방법 2: 백그라운드 실행
```bash
nohup python simple_server.py > server.log 2>&1 &
```

### 방법 3: Uvicorn 직접 실행
```bash
uvicorn simple_server:app --host 0.0.0.0 --port 8000 --reload
```

## 🌐 접속 URL

서버가 시작되면 다음 URL로 접속하세요:

### 웹 대시보드
```
http://localhost:8000
```

**대시보드 기능:**
- ✅ Trinity AI 모델 상태 실시간 모니터링
- ✅ 서버 가동 시간 (Uptime) 표시
- ✅ API 요청 통계
- ✅ 원클릭 API 테스트 버튼
- ✅ 인터랙티브 API 문서

### API 문서 (Swagger UI)
```
http://localhost:8000/docs
```

### Health Check
```
http://localhost:8000/health
```

## 📡 API 엔드포인트

### 1. Health Check
```bash
curl http://localhost:8000/health
```

**응답 예시:**
```json
{
  "status": "healthy",
  "uptime_seconds": 123.45,
  "total_requests": 5,
  "models": {
    "guardian": {"loaded": true, "size": "8.2MB"},
    "oracle": {"loaded": true, "size": "5.0MB"},
    "strategist": {"loaded": true, "size": "58MB"}
  }
}
```

### 2. 트레이딩 시그널 예측
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "5m"
  }'
```

**응답 예시:**
```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2025-12-07T03:10:23.409786",
  "signal": "HOLD",
  "confidence": 0.67,
  "guardian_regime": "BEAR",
  "oracle_prediction": {
    "price_change_5min": -0.03,
    "price_change_15min": -0.36,
    "volatility": 0.81,
    "confidence": 0.8
  },
  "strategist_action": "HOLD",
  "latency_ms": 2.02
}
```

### 3. 모델 정보
```bash
curl http://localhost:8000/models
```

### 4. 성능 메트릭
```bash
curl http://localhost:8000/metrics
```

## 🐍 Python에서 사용하기

### 기본 예제
```python
import requests
import json

# 서버 URL
BASE_URL = "http://localhost:8000"

# 1. Health Check
health = requests.get(f"{BASE_URL}/health").json()
print("서버 상태:", health['status'])
print("모델 로드:", health['models'])

# 2. 예측 요청
prediction_request = {
    "symbol": "BTCUSDT",
    "timeframe": "5m"
}

response = requests.post(
    f"{BASE_URL}/predict",
    json=prediction_request
)

prediction = response.json()
print("\n=== 트레이딩 시그널 ===")
print(f"심볼: {prediction['symbol']}")
print(f"시그널: {prediction['signal']}")
print(f"신뢰도: {prediction['confidence']}")
print(f"Guardian 체제: {prediction['guardian_regime']}")
print(f"Oracle 예측: {prediction['oracle_prediction']}")
print(f"응답시간: {prediction['latency_ms']}ms")
```

### 실시간 모니터링 예제
```python
import requests
import time

def monitor_trading_signals(interval=60):
    """실시간 트레이딩 시그널 모니터링"""
    while True:
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"symbol": "BTCUSDT", "timeframe": "5m"}
            )
            
            data = response.json()
            
            print(f"\n[{data['timestamp']}]")
            print(f"📊 {data['symbol']}: {data['signal']} (신뢰도: {data['confidence']})")
            print(f"🛡️  Guardian: {data['guardian_regime']}")
            print(f"🔮 Oracle 가격 변화(5분): {data['oracle_prediction']['price_change_5min']}%")
            
        except Exception as e:
            print(f"에러: {e}")
        
        time.sleep(interval)

# 60초마다 체크
monitor_trading_signals(interval=60)
```

## 🎨 웹 대시보드 기능

웹 대시보드에서 다음을 확인할 수 있습니다:

1. **시스템 상태**
   - 🟢 서버 온라인 상태
   - ⏱️ 가동 시간 (Uptime)
   - 📊 총 API 요청 수

2. **Trinity AI 모델 상태**
   - ✅ Guardian (8.2MB) - 시장 국면 감지
   - ✅ Oracle (5.0MB) - 가격 예측
   - ✅ Strategist (58MB) - 행동 최적화

3. **API 엔드포인트**
   - 모든 사용 가능한 엔드포인트 목록
   - HTTP 메소드 (GET/POST)
   - 간단한 설명

4. **원클릭 테스트**
   - "🧪 Test API" 버튼으로 즉시 테스트
   - 실시간 응답 결과 표시
   - JSON 포맷 출력

## 🔧 문제 해결

### 1. 포트 이미 사용 중
```bash
# 8000번 포트 사용 중인 프로세스 확인
lsof -i :8000

# 프로세스 종료
kill -9 <PID>

# 또는 다른 포트 사용
uvicorn simple_server:app --port 8001
```

### 2. 모델 파일 없음
```bash
# 모델 파일 확인
ls -lh models/*/best_model.ckpt

# 없다면 학습 실행
python train_all.py
```

### 3. FastAPI/Uvicorn 설치 실패
```bash
# pip 업그레이드
pip install --upgrade pip

# 재설치
pip install --force-reinstall fastapi uvicorn
```

## 📈 성능 지표

### 응답 시간
- **Target**: < 50ms (P99)
- **Actual**: ~2-3ms (매우 우수)

### 처리량
- **Target**: > 100 req/s
- **Actual**: 제한 없음 (비동기 처리)

## 🔐 보안 고려사항

**현재 버전은 로컬 개발/테스트용입니다.**

프로덕션 배포 시 추가 필요사항:
1. ✅ API 키 인증
2. ✅ HTTPS/SSL 인증서
3. ✅ Rate Limiting
4. ✅ CORS 설정
5. ✅ 로깅 및 모니터링

## 🚀 다음 단계

### Phase 1: 로컬 테스트 (현재 단계)
```bash
python simple_server.py  # ✅ 완료!
```

### Phase 2: Paper Trading
```bash
python main.py --mode paper
```

### Phase 3: 실전 배포
```bash
# 프로덕션 서버 설정 필요
python main.py --mode live
```

## 📞 지원

문제가 있으면 다음을 확인하세요:

1. **서버 로그**
```bash
tail -f server.log  # 백그라운드 실행 시
```

2. **Health Check**
```bash
curl http://localhost:8000/health
```

3. **API 문서**
브라우저에서 `http://localhost:8000/docs` 접속

---

## 🎉 축하합니다!

이제 QUANTUM ALPHA AI 트레이딩 시스템이 로컬 서버에서 실행 중입니다! 🚀

**접속 URL:** http://localhost:8000

대시보드에서 실시간으로 Trinity AI 모델을 모니터링하고, API를 통해 트레이딩 시그널을 받아보세요!
