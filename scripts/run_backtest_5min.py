"""
5분봉 백테스팅 (6년 데이터)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_backtest_simple import SimpleBacktester, simple_trend_strategy, simple_mean_reversion_strategy


def load_all_data(data_dir: Path, symbol: str):
    """모든 년도 데이터 로드"""
    all_data = []
    
    for year in range(2019, 2025):
        file = data_dir / f"{symbol}_{year}_1m.parquet"
        if file.exists():
            df = pd.read_parquet(file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No data found for {symbol}")
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/historical_5min_features')
    parser.add_argument('--strategy', type=str, default='trend', choices=['trend', 'mean_reversion'])
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--initial-capital', type=float, default=10000.0)
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    print(f"\n{'='*70}")
    print("🎯 5-MINUTE BACKTEST (6 YEARS)")
    print(f"{'='*70}\n")
    
    print(f"📊 Strategy: {args.strategy}")
    print(f"💰 Initial Capital: ${args.initial_capital:,.2f}")
    print(f"📈 Symbol: {args.symbol}\n")
    
    # Load data
    print(f"📥 Loading 6-year data...")
    df = load_all_data(data_dir, args.symbol)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
    
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Period: {(df.index[-1] - df.index[0]).days} days\n")
    
    # Generate signals
    print(f"🧠 Generating {args.strategy} signals...")
    if args.strategy == 'trend':
        signals = simple_trend_strategy(df)
    else:
        signals = simple_mean_reversion_strategy(df)
    
    print(f"   Signal range: [{signals.min():.2f}, {signals.max():.2f}]")
    print(f"   Signal mean: {signals.mean():.2f}\n")
    
    # Run backtest
    print(f"⚙️  Running backtest...")
    backtester = SimpleBacktester(initial_capital=args.initial_capital)
    
    # Trade every 12 bars (1 hour for 5min bars)
    for i, (timestamp, row) in enumerate(df.iterrows()):
        signal = signals.iloc[i]
        price = row['close']
        
        if i % 12 == 0:  # Every hour
            backtester.execute_trade(timestamp, price, signal)
        
        backtester.update_equity(timestamp, price)
    
    # Close final position
    if backtester.position != 0:
        backtester.execute_trade(df.index[-1], df['close'].iloc[-1], 0.0)
    
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
    
    # Save results
    output_dir = Path('backtest_results')
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / f"backtest_{args.symbol}_{args.strategy}_5min.txt"
    with open(results_file, 'w') as f:
        f.write(f"Backtest Results\n")
        f.write(f"================\n\n")
        f.write(f"Symbol: {args.symbol}\n")
        f.write(f"Strategy: {args.strategy}\n")
        f.write(f"Period: {df.index[0]} to {df.index[-1]}\n")
        f.write(f"Initial Capital: ${args.initial_capital:,.2f}\n\n")
        f.write(f"Returns:\n")
        f.write(f"  Total Return: {metrics['total_return']:.2f}%\n")
        f.write(f"  Annualized: {metrics['annualized_return']:.2f}%\n")
        f.write(f"  Final Capital: ${metrics['final_capital']:,.2f}\n\n")
        f.write(f"Risk:\n")
        f.write(f"  Volatility: {metrics['volatility']:.2f}%\n")
        f.write(f"  Max DD: {metrics['max_drawdown']:.2f}%\n")
        f.write(f"  Sharpe: {metrics['sharpe_ratio']:.2f}\n")
        f.write(f"  Sortino: {metrics['sortino_ratio']:.2f}\n")
        f.write(f"  Calmar: {metrics['calmar_ratio']:.2f}\n\n")
        f.write(f"Trading:\n")
        f.write(f"  Total Trades: {metrics['total_trades']}\n")
        f.write(f"  Win Rate: {metrics['win_rate']:.2f}%\n")
        f.write(f"  Profit Factor: {metrics['profit_factor']:.2f}\n")
    
    print(f"💾 Results saved: {results_file}\n")


if __name__ == "__main__":
    main()
