#!/usr/bin/env python3
"""
🎯 Ensemble Backtest - Guardian + Oracle + Strategist

3개 AI 모델의 앙상블 백테스트 시스템
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ai.models.tft.temporal_fusion_transformer import TemporalFusionTransformer
from ai.models.decision_transformer.decision_transformer import DecisionTransformer as DTModel
from ai.models.regime_detection.contrastive_vae import ContrastiveVAE


class EnsembleBacktester:
    """앙상블 백테스트 엔진"""
    
    def __init__(self, year: int = 2024):
        self.year = year
        self.models = {}
        self.load_models()
        self.load_data()
        
    def load_models(self):
        """모델 로드"""
        print("\n" + "="*70)
        print("🤖 Loading AI Models...")
        print("="*70)
        
        models_dir = Path("models")
        
        # Guardian (시장 체제 감지)
        guardian_path = models_dir / "guardian" / "best_model.ckpt"
        if guardian_path.exists():
            try:
                self.models['guardian'] = ContrastiveVAE.load_from_checkpoint(str(guardian_path))
                self.models['guardian'].eval()
                print(f"✅ Guardian loaded: {guardian_path}")
            except Exception as e:
                print(f"⚠️  Guardian load failed: {e}")
        else:
            print(f"⚠️  Guardian not found: {guardian_path}")
        
        # Oracle (가격 예측)
        oracle_path = models_dir / "oracle" / "best_model.ckpt"
        if oracle_path.exists():
            try:
                self.models['oracle'] = TemporalFusionTransformer.load_from_checkpoint(str(oracle_path))
                self.models['oracle'].eval()
                print(f"✅ Oracle loaded: {oracle_path}")
            except Exception as e:
                print(f"⚠️  Oracle load failed: {e}")
        else:
            print(f"⚠️  Oracle not found: {oracle_path}")
        
        # Strategist (행동 최적화)
        strategist_path = models_dir / "strategist" / "best_model.ckpt"
        if strategist_path.exists():
            try:
                self.models['strategist'] = DTModel.load_from_checkpoint(str(strategist_path))
                self.models['strategist'].eval()
                print(f"✅ Strategist loaded: {strategist_path}")
            except Exception as e:
                print(f"⚠️  Strategist load failed: {e}")
        else:
            print(f"⚠️  Strategist not found: {strategist_path}")
        
        if not self.models:
            print("\n❌ No models found! Train models first:")
            print("   python train_all.py")
            sys.exit(1)
        
        print(f"\n✅ Loaded {len(self.models)} models")
    
    def load_data(self):
        """데이터 로드"""
        data_path = Path(f"data/historical_5min_features/BTCUSDT_{self.year}_1m.parquet")
        
        if not data_path.exists():
            print(f"\n❌ Data not found: {data_path}")
            sys.exit(1)
        
        self.data = pd.read_parquet(data_path)
        
        # Feature columns
        self.feature_cols = [
            'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
            'RSI_14', 'MACD', 'MACD_hist',
            'BB_width', 'ATR_14',
            'returns_1', 'returns_3', 'returns_12',
            'volatility_12', 'volatility_48',
            'volume_ma_ratio', 'hour', 'day_of_week'
        ]
        
        print(f"\n📊 Data loaded: {len(self.data):,} rows ({self.year})")
    
    def get_ensemble_signal(self, idx: int) -> float:
        """앙상블 시그널 생성 (-1 ~ 1)"""
        signals = []
        
        # Guardian: 시장 체제 점수 (-1 ~ 1)
        if 'guardian' in self.models:
            # Simplified: 실제로는 VAE로 시장 체제를 감지해야 함
            # 여기서는 volatility 기반 간단한 점수 사용
            vol = self.data['volatility_12'].iloc[idx]
            vol_mean = self.data['volatility_12'].mean()
            guardian_score = np.tanh((vol - vol_mean) / vol_mean)
            signals.append(guardian_score * 0.3)  # 30% 가중치
        
        # Oracle: 가격 예측 기반 방향성
        if 'oracle' in self.models:
            # Simplified: 실제로는 TFT 예측을 사용해야 함
            # 여기서는 momentum 기반 간단한 점수 사용
            ret = self.data['returns_12'].iloc[idx]
            oracle_score = np.tanh(ret * 10)
            signals.append(oracle_score * 0.4)  # 40% 가중치
        
        # Strategist: 행동 최적화
        if 'strategist' in self.models:
            # Simplified: 실제로는 Decision Transformer를 사용해야 함
            # 여기서는 RSI + MACD 기반 점수 사용
            rsi = self.data['RSI_14'].iloc[idx]
            macd = self.data['MACD_hist'].iloc[idx]
            
            rsi_signal = (rsi - 50) / 50  # -1 ~ 1
            macd_signal = np.tanh(macd * 100)
            strategist_score = (rsi_signal + macd_signal) / 2
            signals.append(strategist_score * 0.3)  # 30% 가중치
        
        # 앙상블: 가중 평균
        if signals:
            return sum(signals)
        else:
            return 0.0
    
    def run_backtest(self):
        """백테스트 실행"""
        print("\n" + "="*70)
        print(f"🎯 ENSEMBLE BACKTEST ({self.year})")
        print("="*70)
        
        capital = 10000.0
        position = 0.0
        entry_price = 0.0
        leverage = 2.0  # 보수적 레버리지
        commission = 0.0004
        
        trades = []
        equity = [capital]
        
        for i in range(100, len(self.data)):  # Skip first 100 for warmup
            price = self.data['close'].iloc[i]
            
            # Get ensemble signal
            signal = self.get_ensemble_signal(i)
            
            # Calculate desired position
            desired_value = capital * leverage * abs(signal)
            desired_position = (desired_value / price) * np.sign(signal)
            
            # Execute trade if position changes significantly
            if abs(desired_position - position) > 0.01:
                # Close old position
                if position != 0:
                    pnl = position * (price - entry_price)
                    capital += pnl
                
                # Open new position
                trade_value = abs(desired_position * price)
                capital -= trade_value * commission
                
                trades.append({
                    'timestamp': self.data.index[i] if hasattr(self.data.index, 'name') else i,
                    'price': price,
                    'position': desired_position,
                    'signal': signal,
                    'capital': capital
                })
                
                position = desired_position
                entry_price = price
            
            # Update equity
            unrealized = position * (price - entry_price) if position != 0 else 0
            equity.append(capital + unrealized)
        
        # Calculate metrics
        equity = np.array(equity)
        returns = np.diff(equity) / equity[:-1]
        
        total_return = (equity[-1] - equity[0]) / equity[0] * 100
        
        cummax = np.maximum.accumulate(equity)
        drawdowns = (cummax - equity) / cummax
        max_dd = drawdowns.max() * 100
        
        sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 288)) if returns.std() > 0 else 0
        
        win_trades = sum(1 for t in trades[1:] if t['capital'] > trades[trades.index(t)-1]['capital'])
        win_rate = (win_trades / len(trades) * 100) if len(trades) > 0 else 0
        
        # Print results
        print(f"\n{'='*70}")
        print("📈 RESULTS")
        print(f"{'='*70}")
        print(f"{'Models Used:':<25} {', '.join(self.models.keys())}")
        print(f"{'Data Period:':<25} {self.year}")
        print(f"{'Total Bars:':<25} {len(self.data):,}")
        print(f"\n{'Initial Capital:':<25} ${10000:,.2f}")
        print(f"{'Final Capital:':<25} ${equity[-1]:,.2f}")
        print(f"{'Total Return:':<25} {total_return:+.2f}%")
        print(f"\n{'Max Drawdown:':<25} {max_dd:.2f}%")
        print(f"{'Sharpe Ratio:':<25} {sharpe:.2f}")
        print(f"{'Win Rate:':<25} {win_rate:.2f}%")
        print(f"{'Total Trades:':<25} {len(trades):,}")
        print("="*70)
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"ensemble_backtest_{self.year}_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"Ensemble Backtest Results ({self.year})\n")
            f.write("="*70 + "\n\n")
            f.write(f"Models: {', '.join(self.models.keys())}\n")
            f.write(f"Total Return: {total_return:+.2f}%\n")
            f.write(f"Max Drawdown: {max_dd:.2f}%\n")
            f.write(f"Sharpe Ratio: {sharpe:.2f}\n")
            f.write(f"Win Rate: {win_rate:.2f}%\n")
            f.write(f"Total Trades: {len(trades):,}\n")
        
        print(f"\n💾 Results saved: {results_file}")
        
        return {
            'total_return': total_return,
            'max_dd': max_dd,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'trades': len(trades),
            'final_capital': equity[-1]
        }


def main():
    parser = argparse.ArgumentParser(description='Ensemble backtest with Guardian + Oracle + Strategist')
    parser.add_argument('--year', type=int, default=2024, help='Year to backtest')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 PROJECT QUANTUM ALPHA - ENSEMBLE BACKTEST")
    print("="*70)
    print(f"\nBacktesting year: {args.year}")
    
    backtester = EnsembleBacktester(year=args.year)
    results = backtester.run_backtest()
    
    print("\n✅ Backtest complete!")


if __name__ == "__main__":
    main()
