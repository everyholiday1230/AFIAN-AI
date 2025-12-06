"""
간단한 백테스팅 스크립트 (테스트용)
Rule-based 전략으로 빠르게 성능 검증
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SimpleBacktester:
    """간단한 백테스팅 엔진"""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.0004,  # 0.04% (Binance maker fee)
        slippage: float = 0.0001,    # 0.01% slippage
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Portfolio state
        self.capital = initial_capital
        self.position = 0.0  # 현재 포지션 크기
        self.entry_price = 0.0
        
        # Track history
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[pd.Timestamp] = []
        
    def calculate_position_size(self, price: float, signal: float) -> float:
        """
        Calculate position size based on signal strength
        signal: -1.0 to 1.0 (negative = short, positive = long)
        """
        # Use fixed leverage for simplicity
        leverage = 3.0
        max_position_value = self.capital * leverage * abs(signal)
        position_size = max_position_value / price
        
        return position_size * np.sign(signal)
    
    def execute_trade(
        self,
        timestamp: pd.Timestamp,
        price: float,
        signal: float,
    ) -> None:
        """Execute trade based on signal"""
        
        # Calculate desired position
        desired_position = self.calculate_position_size(price, signal)
        
        # Calculate position change
        position_change = desired_position - self.position
        
        if abs(position_change) < 0.001:  # Skip tiny changes
            return
        
        # Apply slippage and commission
        execution_price = price * (1 + self.slippage * np.sign(position_change))
        trade_value = abs(position_change * execution_price)
        commission_cost = trade_value * self.commission
        
        # Close existing position PnL
        if self.position != 0:
            pnl = self.position * (price - self.entry_price)
            self.capital += pnl
        
        # Execute trade
        self.capital -= commission_cost
        self.position = desired_position
        self.entry_price = execution_price
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'price': execution_price,
            'position': self.position,
            'position_change': position_change,
            'signal': signal,
            'commission': commission_cost,
            'capital': self.capital,
        })
    
    def update_equity(self, timestamp: pd.Timestamp, price: float) -> None:
        """Update equity curve"""
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        if self.position != 0:
            unrealized_pnl = self.position * (price - self.entry_price)
        
        total_equity = self.capital + unrealized_pnl
        self.equity_curve.append(total_equity)
        self.timestamps.append(timestamp)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(self.equity_curve) < 2:
            return {}
        
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # Basic metrics
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # Calculate annualized metrics (assuming 1-minute data)
        minutes_per_year = 365 * 24 * 60
        periods = len(returns)
        years = periods / minutes_per_year
        
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(minutes_per_year)
        
        # Sharpe Ratio (assume 3% risk-free rate)
        risk_free_rate = 0.03
        sharpe = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(minutes_per_year) if len(downside_returns) > 0 else 0
        sortino = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Max Drawdown
        cummax = np.maximum.accumulate(equity)
        drawdowns = (equity - cummax) / cummax
        max_drawdown = np.min(drawdowns)
        
        # Calmar Ratio
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        trade_returns = []
        for i in range(1, len(self.trades)):
            prev_trade = self.trades[i-1]
            curr_trade = self.trades[i]
            if prev_trade['position'] != 0:
                ret = (curr_trade['price'] - prev_trade['price']) / prev_trade['price'] * np.sign(prev_trade['position'])
                trade_returns.append(ret)
        
        winning_trades = [r for r in trade_returns if r > 0]
        win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        losing_trades = [abs(r) for r in trade_returns if r < 0]
        gross_loss = sum(losing_trades) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_return': total_return * 100,
            'annualized_return': annualized_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown * 100,
            'calmar_ratio': calmar,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'final_capital': equity[-1],
        }


def simple_trend_strategy(df: pd.DataFrame) -> pd.Series:
    """
    Simple trend-following strategy
    Uses EMA crossover + RSI for signal generation
    
    Returns:
        signals: pd.Series with values from -1.0 to 1.0
    """
    signals = pd.Series(0.0, index=df.index)
    
    # Calculate additional indicators if needed
    ema_fast = df['EMA_12']
    ema_slow = df['EMA_26']
    rsi = df['RSI_14']
    
    # Trend signal from EMA crossover
    trend_signal = np.where(ema_fast > ema_slow, 1.0, -1.0)
    
    # RSI filter (avoid overbought/oversold extremes)
    # When RSI > 70, reduce long exposure
    # When RSI < 30, reduce short exposure
    rsi_filter = np.ones_like(rsi)
    rsi_filter[rsi > 70] = 0.5  # Reduce long signal
    rsi_filter[rsi < 30] = 0.5 if False else 0.5  # Reduce short signal
    
    # Combine signals
    signals = trend_signal * rsi_filter
    
    # Use MACD histogram for signal strength
    if 'MACD_hist' in df.columns:
        macd_strength = np.tanh(df['MACD_hist'] / df['MACD_hist'].std())
        signals = signals * (0.7 + 0.3 * abs(macd_strength))
    
    return signals


def ml_model_strategy(df: pd.DataFrame, model_path: str) -> pd.Series:
    """
    ML Model-based strategy
    Uses trained Random Forest to predict price changes
    
    Returns:
        signals: pd.Series with values from -1.0 to 1.0
    """
    import joblib
    
    # Load model
    model = joblib.load(model_path)
    
    # Prepare features (same as training)
    feature_cols = [
        'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
        'RSI_14', 'MACD', 'MACD_hist',
        'BB_width', 'ATR_14',
        'returns_1', 'returns_3', 'returns_12',
        'volatility_12', 'volatility_48',
        'volume_ma_ratio', 'hour', 'day_of_week'
    ]
    
    X = df[feature_cols].fillna(0)
    
    # Predict
    predictions = model.predict(X)
    
    # Convert predictions to signals (-1 to 1)
    # Normalize by standard deviation
    pred_std = predictions.std()
    signals = np.tanh(predictions / (pred_std * 3))  # Scale to [-1, 1]
    
    return pd.Series(signals, index=df.index)


def simple_mean_reversion_strategy(df: pd.DataFrame) -> pd.Series:
    """
    Simple mean reversion strategy
    Uses Bollinger Bands + RSI
    
    Returns:
        signals: pd.Series with values from -1.0 to 1.0
    """
    signals = pd.Series(0.0, index=df.index)
    
    close = df['close']
    bb_upper = df['BB_upper']
    bb_lower = df['BB_lower']
    bb_middle = df['BB_middle']
    rsi = df['RSI_14']
    
    # Calculate distance from bands (normalized)
    bb_width = bb_upper - bb_lower
    price_position = (close - bb_middle) / (bb_width / 2)
    
    # Mean reversion signal (opposite of trend)
    # When price is high, go short; when price is low, go long
    reversion_signal = -np.tanh(price_position)
    
    # RSI confirmation
    rsi_signal = np.where(rsi < 30, 1.0, np.where(rsi > 70, -1.0, 0.0))
    
    # Combine signals (average)
    signals = (reversion_signal * 0.6 + rsi_signal * 0.4)
    
    return signals


def main():
    parser = argparse.ArgumentParser(description='Run simple backtest')
    parser.add_argument('--data-dir', type=str, default='data/test_features',
                       help='Directory containing feature data')
    parser.add_argument('--strategy', type=str, default='trend',
                       choices=['trend', 'mean_reversion', 'ml_model'],
                       help='Strategy to test')
    parser.add_argument('--model-path', type=str, default='',
                       help='Path to ML model (required for ml_model strategy)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Symbol to backtest')
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                       help='Initial capital')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    print(f"\n{'='*70}")
    print("🎯 SIMPLE BACKTEST")
    print(f"{'='*70}\n")
    
    print(f"📊 Strategy: {args.strategy}")
    print(f"💰 Initial Capital: ${args.initial_capital:,.2f}")
    print(f"📈 Symbol: {args.symbol}\n")
    
    # Load data
    data_file = data_dir / f"{args.symbol}_1m_features.parquet"
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return
    
    print(f"📥 Loading data from {data_file.name}...")
    df = pd.read_parquet(data_file)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
    
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}\n")
    
    # Generate signals
    print(f"🧠 Generating {args.strategy} signals...")
    if args.strategy == 'trend':
        signals = simple_trend_strategy(df)
    elif args.strategy == 'mean_reversion':
        signals = simple_mean_reversion_strategy(df)
    elif args.strategy == 'ml_model':
        if not args.model_path:
            print("❌ --model-path required for ml_model strategy")
            return
        print(f"🤖 Loading ML model: {args.model_path}")
        signals = ml_model_strategy(df, args.model_path)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    print(f"   Signal range: [{signals.min():.2f}, {signals.max():.2f}]")
    print(f"   Signal mean: {signals.mean():.2f}")
    print(f"   Signal std: {signals.std():.2f}\n")
    
    # Run backtest
    print(f"⚙️  Running backtest...")
    backtester = SimpleBacktester(initial_capital=args.initial_capital)
    
    for i, (timestamp, row) in enumerate(df.iterrows()):
        signal = signals.iloc[i]
        price = row['close']
        
        # Execute trades
        if i % 60 == 0:  # Trade every hour (60 minutes)
            backtester.execute_trade(timestamp, price, signal)
        
        # Update equity
        backtester.update_equity(timestamp, price)
    
    # Close final position
    if backtester.position != 0:
        final_price = df['close'].iloc[-1]
        backtester.execute_trade(df.index[-1], final_price, 0.0)
    
    print(f"   Completed {len(backtester.trades)} trades\n")
    
    # Calculate metrics
    print(f"{'='*70}")
    print("📊 PERFORMANCE METRICS")
    print(f"{'='*70}\n")
    
    metrics = backtester.get_performance_metrics()
    
    print(f"💰 Returns:")
    print(f"   Total Return:       {metrics['total_return']:>10.2f}%")
    print(f"   Annualized Return:  {metrics['annualized_return']:>10.2f}%")
    print(f"   Final Capital:      ${metrics['final_capital']:>10,.2f}")
    
    print(f"\n📈 Risk Metrics:")
    print(f"   Volatility:         {metrics['volatility']:>10.2f}%")
    print(f"   Max Drawdown:       {metrics['max_drawdown']:>10.2f}%")
    print(f"   Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}")
    print(f"   Sortino Ratio:      {metrics['sortino_ratio']:>10.2f}")
    print(f"   Calmar Ratio:       {metrics['calmar_ratio']:>10.2f}")
    
    print(f"\n🎯 Trading Stats:")
    print(f"   Total Trades:       {metrics['total_trades']:>10}")
    print(f"   Win Rate:           {metrics['win_rate']:>10.2f}%")
    print(f"   Profit Factor:      {metrics['profit_factor']:>10.2f}")
    
    print(f"\n{'='*70}\n")
    
    # Performance assessment
    print("🎖️  PERFORMANCE ASSESSMENT:\n")
    
    if metrics['sharpe_ratio'] > 2.0:
        print("   ✅ EXCELLENT Sharpe Ratio (>2.0)")
    elif metrics['sharpe_ratio'] > 1.0:
        print("   ✅ GOOD Sharpe Ratio (>1.0)")
    else:
        print("   ⚠️  LOW Sharpe Ratio (<1.0)")
    
    if abs(metrics['max_drawdown']) < 15:
        print("   ✅ GOOD Max Drawdown (<15%)")
    elif abs(metrics['max_drawdown']) < 25:
        print("   ⚠️  MODERATE Max Drawdown (<25%)")
    else:
        print("   ❌ HIGH Max Drawdown (>25%)")
    
    if metrics['win_rate'] > 55:
        print("   ✅ GOOD Win Rate (>55%)")
    else:
        print("   ⚠️  LOW Win Rate (<55%)")
    
    print(f"\n🎉 Backtest completed successfully!")
    print(f"   Next steps:")
    print(f"   1. Try different strategies (trend vs mean_reversion)")
    print(f"   2. Optimize parameters")
    print(f"   3. Test on longer time periods")
    print(f"   4. Train AI models for better signals\n")


if __name__ == "__main__":
    main()
