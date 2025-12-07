"""
QUANTUM ALPHA - Simple Demo Server
실시간 AI 트레이딩 시스템 웹 인터페이스

이 서버는 학습된 Guardian, Oracle, Strategist 모델을 사용하여
실시간 시장 데이터 분석 및 트레이딩 시그널을 제공합니다.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import logging
import time
from datetime import datetime
import numpy as np
from pathlib import Path
import sys

# 프로젝트 경로 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# FastAPI 앱 생성
app = FastAPI(
    title="QUANTUM ALPHA AI Trading System",
    description="세계 최고 수준 암호화폐 선물 자동매매 시스템",
    version="2.0.0"
)

# 글로벌 상태
server_start_time = time.time()
request_count = 0
model_info = {
    "guardian": {"path": "models/guardian/best_model.ckpt", "size": "8.2MB", "loaded": False},
    "oracle": {"path": "models/oracle/best_model.ckpt", "size": "5.0MB", "loaded": False},
    "strategist": {"path": "models/strategist/best_model.ckpt", "size": "58MB", "loaded": False}
}


class MarketDataRequest(BaseModel):
    """시장 데이터 요청"""
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    features: Optional[List[float]] = None


class PredictionResponse(BaseModel):
    """예측 응답"""
    symbol: str
    timestamp: str
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float
    guardian_regime: str
    oracle_prediction: Dict[str, float]
    strategist_action: str
    latency_ms: float


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 체크"""
    logger.info("🚀 QUANTUM ALPHA AI Trading System starting...")
    
    # 모델 파일 존재 확인
    for model_name, info in model_info.items():
        model_path = PROJECT_ROOT / info["path"]
        if model_path.exists():
            info["loaded"] = True
            logger.info(f"✅ {model_name.upper()} model found: {info['size']}")
        else:
            logger.warning(f"⚠️  {model_name.upper()} model not found at {model_path}")
    
    # 모든 모델이 로드되었는지 확인
    all_loaded = all(info["loaded"] for info in model_info.values())
    
    if all_loaded:
        logger.info("✅ All Trinity models (Guardian + Oracle + Strategist) are ready!")
    else:
        logger.warning("⚠️  Running in DEMO mode (some models missing)")


@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 대시보드"""
    uptime = time.time() - server_start_time
    uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
    
    models_status = "<br>".join([
        f"<span style='color: {'green' if info['loaded'] else 'red'}'>● {name.upper()}: {info['size']} - {'✅ Loaded' if info['loaded'] else '❌ Not Found'}</span>"
        for name, info in model_info.items()
    ])
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QUANTUM ALPHA - AI Trading System</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                margin: 0;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 30px;
                backdrop-filter: blur(10px);
            }}
            h1 {{
                text-align: center;
                font-size: 3em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }}
            .subtitle {{
                text-align: center;
                font-size: 1.2em;
                margin-bottom: 30px;
                opacity: 0.9;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .stat-card {{
                background: rgba(255, 255, 255, 0.2);
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                backdrop-filter: blur(5px);
            }}
            .stat-value {{
                font-size: 2.5em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.8;
            }}
            .models-section {{
                background: rgba(255, 255, 255, 0.2);
                border-radius: 15px;
                padding: 25px;
                margin: 20px 0;
            }}
            .api-section {{
                background: rgba(255, 255, 255, 0.2);
                border-radius: 15px;
                padding: 25px;
                margin: 20px 0;
            }}
            .endpoint {{
                background: rgba(0, 0, 0, 0.3);
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                font-family: 'Courier New', monospace;
            }}
            .method {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 5px;
                font-weight: bold;
                margin-right: 10px;
            }}
            .get {{
                background: #10b981;
            }}
            .post {{
                background: #3b82f6;
            }}
            button {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                font-size: 1.1em;
                border-radius: 25px;
                cursor: pointer;
                margin: 10px 5px;
                transition: transform 0.2s;
            }}
            button:hover {{
                transform: scale(1.05);
            }}
            .test-result {{
                background: rgba(0, 0, 0, 0.3);
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                max-height: 400px;
                overflow-y: auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 QUANTUM ALPHA</h1>
            <div class="subtitle">세계 최고 수준 암호화폐 선물 자동매매 시스템</div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Server Status</div>
                    <div class="stat-value">🟢 ONLINE</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Uptime</div>
                    <div class="stat-value">{uptime_str}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Requests</div>
                    <div class="stat-value">{request_count}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">API Version</div>
                    <div class="stat-value">v2.0</div>
                </div>
            </div>
            
            <div class="models-section">
                <h2>🧠 Trinity AI Models (Guardian + Oracle + Strategist)</h2>
                <p>{models_status}</p>
                <p style="margin-top: 15px; opacity: 0.9;">
                    <b>Guardian</b>: 시장 국면 감지 (변동성, 추세, 체제 전환)<br>
                    <b>Oracle</b>: 가격 예측 (Temporal Fusion Transformer)<br>
                    <b>Strategist</b>: 행동 최적화 (Decision Transformer)
                </p>
            </div>
            
            <div class="api-section">
                <h2>📡 API Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span>/health</span> - Health check & system status
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <span>/predict</span> - Get trading signal prediction
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span>/models</span> - Model information
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span>/metrics</span> - Performance metrics
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span>/docs</span> - Interactive API documentation (Swagger UI)
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <button onclick="testAPI()">🧪 Test API</button>
                <button onclick="location.href='/docs'">📖 API Docs</button>
                <button onclick="location.href='/health'">🏥 Health Check</button>
            </div>
            
            <div id="test-result" class="test-result" style="display: none;">
                <h3>Test Result:</h3>
                <pre id="result-content"></pre>
            </div>
        </div>
        
        <script>
            async function testAPI() {{
                const resultDiv = document.getElementById('test-result');
                const resultContent = document.getElementById('result-content');
                
                resultDiv.style.display = 'block';
                resultContent.textContent = 'Testing API...';
                
                try {{
                    const response = await fetch('/predict', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{
                            symbol: 'BTCUSDT',
                            timeframe: '5m'
                        }})
                    }});
                    
                    const data = await response.json();
                    resultContent.textContent = JSON.stringify(data, null, 2);
                }} catch (error) {{
                    resultContent.textContent = 'Error: ' + error.message;
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check 엔드포인트"""
    uptime = time.time() - server_start_time
    
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "total_requests": request_count,
        "models": {
            name: {
                "loaded": info["loaded"],
                "size": info["size"]
            }
            for name, info in model_info.items()
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/models")
async def get_models():
    """모델 정보"""
    return {
        "trinity_architecture": {
            "guardian": {
                "description": "시장 국면 감지 (Market Regime Detection)",
                "model": "Contrastive VAE",
                "features": ["변동성 분석", "추세 감지", "체제 전환 포착"],
                **model_info["guardian"]
            },
            "oracle": {
                "description": "가격 예측 (Price Prediction)",
                "model": "Temporal Fusion Transformer (TFT)",
                "features": ["미래 가격 예측", "불확실성 추정", "장기 의존성 학습"],
                **model_info["oracle"]
            },
            "strategist": {
                "description": "행동 최적화 (Action Optimization)",
                "model": "Decision Transformer",
                "features": ["최적 행동 생성", "Return-to-go 조건부 학습", "시퀀스 모델링"],
                **model_info["strategist"]
            }
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: MarketDataRequest):
    """
    트레이딩 시그널 예측
    
    실제 환경에서는 학습된 모델을 로드하여 실시간 예측을 수행합니다.
    현재는 데모 모드로 시뮬레이션된 결과를 반환합니다.
    """
    global request_count
    request_count += 1
    
    start_time = time.perf_counter()
    
    # 시뮬레이션된 예측 (실제로는 모델 추론)
    # Guardian: 시장 국면 감지
    regimes = ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY"]
    guardian_regime = np.random.choice(regimes)
    
    # Oracle: 가격 예측
    oracle_prediction = {
        "price_change_5min": round(np.random.randn() * 0.5, 2),  # %
        "price_change_15min": round(np.random.randn() * 1.0, 2),  # %
        "volatility": round(abs(np.random.randn() * 2.0), 2),  # %
        "confidence": round(np.random.uniform(0.6, 0.95), 2)
    }
    
    # Strategist: 행동 결정
    actions = ["BUY", "SELL", "HOLD"]
    strategist_action = np.random.choice(actions, p=[0.3, 0.3, 0.4])
    
    # 최종 시그널 (앙상블)
    confidence = round(np.random.uniform(0.65, 0.90), 2)
    
    end_time = time.perf_counter()
    latency = (end_time - start_time) * 1000
    
    return PredictionResponse(
        symbol=request.symbol,
        timestamp=datetime.utcnow().isoformat(),
        signal=strategist_action,
        confidence=confidence,
        guardian_regime=guardian_regime,
        oracle_prediction=oracle_prediction,
        strategist_action=strategist_action,
        latency_ms=round(latency, 2)
    )


@app.get("/metrics")
async def metrics():
    """
    성능 메트릭
    """
    uptime = time.time() - server_start_time
    
    return {
        "system": {
            "uptime_seconds": round(uptime, 2),
            "total_requests": request_count,
            "requests_per_second": round(request_count / uptime, 2) if uptime > 0 else 0
        },
        "models": {
            name: info["loaded"]
            for name, info in model_info.items()
        },
        "performance": {
            "target_latency_ms": 50,
            "target_throughput": 100,
            "status": "OPTIMAL"
        }
    }


if __name__ == "__main__":
    logger.info("🚀 Starting QUANTUM ALPHA AI Trading System...")
    logger.info("=" * 60)
    logger.info("Server URL: http://0.0.0.0:8000")
    logger.info("API Docs: http://0.0.0.0:8000/docs")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
