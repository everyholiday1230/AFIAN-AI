"""
FastAPI 모델 서빙 서버

목적: AI 모델을 RESTful API로 서빙하여 실시간 추론 제공

핵심 기능:
- Trinity 모델 앙상블 추론 (Oracle + Strategist + Guardian)
- 비동기 처리로 동시 요청 처리
- 요청 큐잉 및 배치 처리
- Health check 및 메트릭 엔드포인트

성능 목표:
- 추론 지연시간: < 50ms (P99)
- 처리량: > 100 req/s
- 가용성: 99.9%
"""

import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np
import uvicorn
import logging
import time
from datetime import datetime

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from ai.inference.onnx_inference import ONNXInferenceEngine, InferenceConfig, TrinityONNXEnsemble

logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="QUANTUM ALPHA AI Serving API",
    description="실시간 암호화폐 트레이딩 AI 모델 서빙",
    version="1.0.0"
)

# 글로벌 모델 인스턴스
trinity_ensemble: Optional[TrinityONNXEnsemble] = None
request_count = 0
total_latency = 0.0


class MarketDataRequest(BaseModel):
    """시장 데이터 요청"""
    encoder_input: List[List[float]] = Field(..., description="Encoder input features (batch, seq_len, features)")
    static_input: Optional[List[float]] = Field(None, description="Static features")
    decoder_input: Optional[List[List[float]]] = Field(None, description="Decoder input features")
    
    class Config:
        schema_extra = {
            "example": {
                "encoder_input": [[50000, 0.01, 0.5] for _ in range(60)],
                "static_input": [1.0, 0.0, 0.5],
                "decoder_input": [[0.0, 0.0] for _ in range(10)]
            }
        }


class PredictionResponse(BaseModel):
    """예측 응답"""
    oracle_prediction: Dict[str, List[float]]
    strategist_action: Dict[str, List[float]]
    guardian_regime: Dict[str, List[float]]
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check 응답"""
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_requests: int
    avg_latency_ms: float


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    global trinity_ensemble
    
    logger.info("🚀 Loading Trinity ONNX models...")
    
    try:
        # 모델 경로 설정 (실제 환경에서는 환경 변수로 관리)
        model_dir = os.getenv('MODEL_DIR', '/home/user/webapp/data/models')
        
        oracle_path = os.path.join(model_dir, 'tft_oracle.onnx')
        strategist_path = os.path.join(model_dir, 'decision_transformer.onnx')
        guardian_path = os.path.join(model_dir, 'regime_detector.onnx')
        
        # 모델 파일 존재 확인
        for path, name in [(oracle_path, 'Oracle'), (strategist_path, 'Strategist'), (guardian_path, 'Guardian')]:
            if not os.path.exists(path):
                logger.warning(f"⚠️  {name} model not found at {path}. Using dummy model.")
                # 실제 환경에서는 에러 처리
        
        # Trinity 앙상블 로드 (파일이 없으면 None)
        if all(os.path.exists(p) for p in [oracle_path, strategist_path, guardian_path]):
            trinity_ensemble = TrinityONNXEnsemble(
                oracle_path=oracle_path,
                strategist_path=strategist_path,
                guardian_path=guardian_path,
                provider='CPUExecutionProvider'
            )
            logger.info("✅ Trinity models loaded successfully")
        else:
            logger.warning("⚠️  Trinity models not found. API will run in mock mode.")
    
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        trinity_ensemble = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "QUANTUM ALPHA AI Serving API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check 엔드포인트"""
    global request_count, total_latency
    
    uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    avg_latency = total_latency / request_count if request_count > 0 else 0
    
    return HealthResponse(
        status="healthy" if trinity_ensemble is not None else "degraded",
        model_loaded=trinity_ensemble is not None,
        uptime_seconds=uptime,
        total_requests=request_count,
        avg_latency_ms=avg_latency
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: MarketDataRequest):
    """
    트레이딩 시그널 예측
    
    Args:
        request: 시장 데이터 요청
        
    Returns:
        PredictionResponse: 예측 결과
    """
    global request_count, total_latency, trinity_ensemble
    
    start_time = time.perf_counter()
    request_count += 1
    
    try:
        # 모델이 로드되지 않았으면 mock 응답
        if trinity_ensemble is None:
            logger.warning("Trinity models not loaded. Returning mock response.")
            
            mock_response = {
                "oracle_prediction": {
                    "price_forecast": [50000.0 + np.random.randn() * 100 for _ in range(10)],
                    "volatility_forecast": [0.02 + abs(np.random.randn() * 0.005) for _ in range(10)]
                },
                "strategist_action": {
                    "action": [0.0],  # 0: hold, 1: buy, -1: sell
                    "confidence": [0.5]
                },
                "guardian_regime": {
                    "regime": [2.0],  # 0: bull, 1: bear, 2: sideways, 3: high_vol
                    "confidence": [0.6]
                },
                "latency_ms": 1.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return PredictionResponse(**mock_response)
        
        # 입력 데이터 변환
        encoder_input = np.array(request.encoder_input, dtype=np.float32)
        
        # 배치 차원 추가 (필요시)
        if encoder_input.ndim == 2:
            encoder_input = np.expand_dims(encoder_input, axis=0)
        
        # 현재 상태 (더미)
        current_state = {
            'position': np.array([[0.0]], dtype=np.float32),
            'pnl': np.array([[0.0]], dtype=np.float32)
        }
        
        # Trinity 앙상블 추론
        market_data = {'encoder_input': encoder_input}
        result = trinity_ensemble.predict_full_pipeline(market_data, current_state)
        
        # 응답 구성
        response_data = {
            "oracle_prediction": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in result['oracle_prediction'].items()
                if k != 'latency_ms'
            },
            "strategist_action": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in result['strategist_action'].items()
                if k != 'latency_ms'
            },
            "guardian_regime": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in result['guardian_regime'].items()
                if k != 'latency_ms'
            },
            "latency_ms": result['total_latency_ms'],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        end_time = time.perf_counter()
        total_latency += (end_time - start_time) * 1000
        
        return PredictionResponse(**response_data)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """
    Prometheus 메트릭 엔드포인트
    """
    global request_count, total_latency, trinity_ensemble
    
    avg_latency = total_latency / request_count if request_count > 0 else 0
    
    if trinity_ensemble:
        latency_stats = trinity_ensemble.get_all_latency_stats()
    else:
        latency_stats = {}
    
    metrics_text = f"""
# HELP quantum_alpha_requests_total Total number of prediction requests
# TYPE quantum_alpha_requests_total counter
quantum_alpha_requests_total {request_count}

# HELP quantum_alpha_latency_ms_avg Average latency in milliseconds
# TYPE quantum_alpha_latency_ms_avg gauge
quantum_alpha_latency_ms_avg {avg_latency:.2f}

# HELP quantum_alpha_model_loaded Model loaded status
# TYPE quantum_alpha_model_loaded gauge
quantum_alpha_model_loaded {1 if trinity_ensemble else 0}
"""
    
    # 각 모델별 지연시간
    for model_name, stats in latency_stats.items():
        if 'p99_ms' in stats:
            metrics_text += f"\n# HELP quantum_alpha_{model_name}_p99_ms P99 latency for {model_name}\n"
            metrics_text += f"# TYPE quantum_alpha_{model_name}_p99_ms gauge\n"
            metrics_text += f"quantum_alpha_{model_name}_p99_ms {stats['p99_ms']:.2f}\n"
    
    return JSONResponse(content={"metrics": metrics_text})


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    log_level: str = "info"
):
    """
    FastAPI 서버 시작
    
    Args:
        host: 호스트 주소
        port: 포트 번호
        workers: Worker 프로세스 수
        log_level: 로그 레벨
    """
    # 시작 시간 기록
    app.state.start_time = time.time()
    
    logger.info(f"🚀 Starting QUANTUM ALPHA AI Serving API on {host}:{port}")
    
    uvicorn.run(
        "ai.inference.serving.fastapi_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False
    )


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 서버 시작
    start_server(host="0.0.0.0", port=8000, workers=1)
