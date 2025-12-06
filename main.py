"""
QUANTUM ALPHA - Master Orchestrator
세계 최고 수준 암호화폐 선물 자동매매 시스템

Trinity Architecture:
1. The Oracle (예측): TFT + Decision Transformer
2. The Strategist (대응): 실시간 전략 실행
3. The Guardian (감시): 리스크 관리 + 국면 감지
"""

import asyncio
import sys
import signal
from pathlib import Path
from typing import Dict, Optional
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import AI models
from ai.models.tft.temporal_fusion_transformer import TemporalFusionTransformer
from ai.models.decision_transformer.decision_transformer import DecisionTransformer
from ai.models.regime_detection.contrastive_vae import ContrastiveVAE, MarketRegimeDetector

# Import features
from ai.features.preprocessing.fractional_differencing import FractionalDifferencing
from ai.features.orderflow.order_flow_imbalance import OrderFlowImbalance, VolumeWeightedOFI
from ai.features.preprocessing.wavelet_denoiser import WaveletDenoiser

# ============================================================================
# Configuration
# ============================================================================

class SystemConfig:
    """시스템 전체 설정"""
    
    def __init__(self, config_path: str = "configs/system_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """설정 파일 로드"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return self.default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def default_config() -> Dict:
        """기본 설정"""
        return {
            'system': {
                'name': 'QUANTUM_ALPHA',
                'version': '0.1.0',
                'mode': 'paper_trading',  # paper_trading, live
            },
            'data': {
                'chart_source': 'binance',
                'execution_source': 'bybit',
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'timeframes': ['1m', '5m', '15m'],
            },
            'ai': {
                'tft': {
                    'enabled': True,
                    'encoder_length': 60,
                    'decoder_length': 10,
                    'hidden_size': 256,
                },
                'decision_transformer': {
                    'enabled': True,
                    'state_dim': 50,
                    'action_dim': 4,
                    'hidden_size': 512,
                },
                'regime_detection': {
                    'enabled': True,
                    'n_regimes': 4,
                    'latent_dim': 32,
                },
            },
            'risk': {
                'daily_loss_limit': 500.0,
                'max_drawdown_pct': 20.0,
                'max_leverage': 10.0,
                'max_position_size': {
                    'BTCUSDT': 0.1,
                    'ETHUSDT': 2.0,
                },
            },
            'execution': {
                'order_timeout_ms': 5000,
                'max_slippage_bps': 10,
                'retry_attempts': 3,
            },
        }


# ============================================================================
# Trinity Components
# ============================================================================

class TheOracle:
    """예측 시스템 - TFT + Decision Transformer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🔮 Initializing The Oracle on {self.device}")
        
        # TFT for price prediction
        if config['ai']['tft']['enabled']:
            self.tft = TemporalFusionTransformer(
                num_static_vars=5,
                num_historical_vars=50,
                num_future_vars=10,
                encoder_length=config['ai']['tft']['encoder_length'],
                decoder_length=config['ai']['tft']['decoder_length'],
                hidden_size=config['ai']['tft']['hidden_size'],
            ).to(self.device)
            logger.info("   ✅ TFT loaded")
        
        # Decision Transformer for action generation
        if config['ai']['decision_transformer']['enabled']:
            self.dt = DecisionTransformer(
                state_dim=config['ai']['decision_transformer']['state_dim'],
                action_dim=config['ai']['decision_transformer']['action_dim'],
                hidden_size=config['ai']['decision_transformer']['hidden_size'],
            ).to(self.device)
            logger.info("   ✅ Decision Transformer loaded")
    
    async def predict(
        self, 
        market_state: Dict
    ) -> Dict:
        """시장 예측"""
        # Extract features from market state
        historical_features = torch.tensor(
            market_state['historical_features'], 
            device=self.device
        ).unsqueeze(0)
        
        # TFT prediction
        tft_output = self.tft(
            static_vars=None,
            historical_vars=historical_features,
            future_vars=None
        )
        
        # Extract quantiles for uncertainty
        predictions = tft_output['predictions'].cpu().detach().numpy()[0]
        
        # Decision Transformer action
        dt_action = self.dt.get_action(
            states=historical_features,
            actions=torch.zeros(1, historical_features.size(1), 4, device=self.device),
            returns_to_go=torch.ones(1, historical_features.size(1), 1, device=self.device) * 10,
            timesteps=torch.arange(historical_features.size(1), device=self.device).unsqueeze(0)
        ).cpu().detach().numpy()[0]
        
        return {
            'price_prediction': predictions,
            'action': dt_action,
            'confidence': float(np.std(predictions)),
            'timestamp': datetime.now()
        }


class TheStrategist:
    """전략 실행 시스템"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.active_positions = {}
        self.pending_orders = {}
        
        logger.info("⚔️  Initializing The Strategist")
    
    async def execute_strategy(
        self,
        predictions: Dict,
        market_state: Dict,
        regime: Dict
    ) -> Dict:
        """전략 실행"""
        action = predictions['action']
        
        # Interpret action: [position, size, stop_loss, take_profit]
        position_signal = action[0]  # -1 (short) to 1 (long)
        size_signal = abs(action[1])  # 0 to 1
        stop_loss = action[2]  # -1 to 1
        take_profit = action[3]  # -1 to 1
        
        # Map to actual trading decision
        if abs(position_signal) < 0.2:
            # Neutral - close positions
            decision = {
                'action': 'close',
                'reason': 'neutral_signal'
            }
        elif position_signal > 0.5:
            # Strong long signal
            decision = {
                'action': 'long',
                'size': size_signal,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': predictions['confidence']
            }
        elif position_signal < -0.5:
            # Strong short signal
            decision = {
                'action': 'short',
                'size': size_signal,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': predictions['confidence']
            }
        else:
            # Wait
            decision = {
                'action': 'wait',
                'reason': 'weak_signal'
            }
        
        logger.info(f"   📊 Strategy decision: {decision['action']}")
        
        return decision


class TheGuardian:
    """감시 시스템 - 리스크 관리 + 국면 감지"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("🛡️  Initializing The Guardian")
        
        # Regime detection
        if config['ai']['regime_detection']['enabled']:
            self.vae = ContrastiveVAE(
                input_dim=50,
                latent_dim=config['ai']['regime_detection']['latent_dim'],
            ).to(self.device)
            
            self.regime_detector = MarketRegimeDetector(
                vae_model=self.vae,
                n_regimes=config['ai']['regime_detection']['n_regimes']
            )
            logger.info("   ✅ Regime detector loaded")
        
        # Risk limits
        self.risk_config = config['risk']
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        
    async def check_regime(self, market_features: torch.Tensor) -> Dict:
        """시장 국면 감지"""
        regime = self.regime_detector.predict_regime(market_features)
        
        logger.info(
            f"   🎯 Market regime: {regime['regime']} "
            f"(confidence: {regime['confidence']:.2f})"
        )
        
        return regime
    
    async def check_risk(
        self,
        decision: Dict,
        account_state: Dict
    ) -> bool:
        """리스크 체크"""
        # Daily loss limit
        if account_state['daily_pnl'] < -self.risk_config['daily_loss_limit']:
            logger.warning("⚠️  Daily loss limit exceeded!")
            return False
        
        # Drawdown limit
        if account_state.get('drawdown_pct', 0) > self.risk_config['max_drawdown_pct']:
            logger.warning("⚠️  Drawdown limit exceeded!")
            return False
        
        # Leverage limit
        if account_state.get('leverage', 0) > self.risk_config['max_leverage']:
            logger.warning("⚠️  Leverage limit exceeded!")
            return False
        
        return True


# ============================================================================
# Master Orchestrator
# ============================================================================

class QuantumAlpha:
    """메인 오케스트레이터"""
    
    def __init__(self, config_path: str = "configs/system_config.yaml"):
        self.config = SystemConfig(config_path)
        self.is_running = False
        
        # Initialize Trinity components
        self.oracle = TheOracle(self.config.config)
        self.strategist = TheStrategist(self.config.config)
        self.guardian = TheGuardian(self.config.config)
        
        # State
        self.account_state = {
            'balance': 10000.0,
            'equity': 10000.0,
            'daily_pnl': 0.0,
            'positions': {},
        }
        
        logger.info("
╔═══════════════════════════════════════════════════════════════╗
║                   QUANTUM ALPHA v0.1.0                        ║
║        세계 최고 수준 암호화폐 선물 자동매매 시스템              ║
╚═══════════════════════════════════════════════════════════════╝
        ")
        
    async def initialize(self):
        """시스템 초기화"""
        logger.info("🚀 Initializing system...")
        
        # Load models (would load from checkpoints in production)
        # self.oracle.tft.load_state_dict(torch.load('models/tft.pt'))
        # self.oracle.dt.load_state_dict(torch.load('models/dt.pt'))
        # self.guardian.vae.load_state_dict(torch.load('models/vae.pt'))
        
        logger.success("✅ System initialized")
    
    async def trading_loop(self):
        """메인 트레이딩 루프"""
        self.is_running = True
        
        logger.info("🎯 Starting trading loop...")
        
        iteration = 0
        while self.is_running:
            try:
                iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"📊 Iteration {iteration} - {datetime.now()}")
                
                # 1. Fetch market data (simulated)
                market_state = await self.fetch_market_data()
                
                # 2. The Guardian: Check regime
                market_features = torch.randn(50)  # Simulated features
                regime = await self.guardian.check_regime(market_features)
                
                # 3. The Oracle: Make predictions
                predictions = await self.oracle.predict(market_state)
                
                # 4. The Strategist: Generate decision
                decision = await self.strategist.execute_strategy(
                    predictions, market_state, regime
                )
                
                # 5. The Guardian: Risk check
                risk_approved = await self.guardian.check_risk(
                    decision, self.account_state
                )
                
                # 6. Execute if approved
                if risk_approved and decision['action'] in ['long', 'short']:
                    logger.info(f"   ✅ Executing {decision['action']} order")
                    # await self.execute_order(decision)
                else:
                    logger.info(f"   ⏸️  No action: {decision.get('reason', 'risk_blocked')}")
                
                # Wait before next iteration
                await asyncio.sleep(1)  # 1초 대기 (실제는 더 짧을 수 있음)
                
            except KeyboardInterrupt:
                logger.info("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def fetch_market_data(self) -> Dict:
        """시장 데이터 수집 (시뮬레이션)"""
        # In production: fetch from Redis (populated by Rust data collector)
        return {
            'historical_features': np.random.randn(60, 50),  # 60 timesteps, 50 features
            'current_price': 40000.0 + np.random.randn() * 100,
            'volume': 1000000.0,
            'orderbook': {
                'bids': [[40000.0, 10.0], [39999.0, 5.0]],
                'asks': [[40001.0, 8.0], [40002.0, 6.0]],
            }
        }
    
    async def execute_order(self, decision: Dict):
        """주문 실행 (Rust executor 호출)"""
        # In production: call Rust order executor via IPC or HTTP
        pass
    
    async def shutdown(self):
        """시스템 종료"""
        logger.info("🛑 Shutting down QUANTUM ALPHA...")
        self.is_running = False
        
        # Close all positions
        # Cancel all orders
        # Save state
        
        logger.success("✅ Shutdown complete")


# ============================================================================
# Entry Point
# ============================================================================

async def main():
    """메인 함수"""
    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/quantum_alpha_{time}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG"
    )
    
    # Create system
    system = QuantumAlpha()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.warning("\n⚠️  Received shutdown signal")
        asyncio.create_task(system.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize
        await system.initialize()
        
        # Start trading
        await system.trading_loop()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise
    finally:
        await system.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
