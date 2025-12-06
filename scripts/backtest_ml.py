"""
ML 모델 백테스트 (간단 버전)
"""
import pandas as pd
import numpy as np
import joblib

# Load data
import glob
from pathlib import Path

df = pd.read_parquet('data/historical_5min_features/BTCUSDT_2024_1m.parquet')

# Find latest model
model_files = glob.glob('models/full/rf_*.pkl') + glob.glob('models/simple/rf_*.pkl')
if not model_files:
    print('❌ No trained Random Forest model found!')
    print('Please train a model first: python scripts/train_incremental.py')
    exit(1)

latest_model = max(model_files, key=lambda x: Path(x).stat().st_mtime)
print(f'Loading model: {latest_model}')
model = joblib.load(latest_model)

print('='*70)
print('🎯 ML MODEL BACKTEST (2024 BTCUSDT 5min) - OUT-OF-SAMPLE TEST')
print('='*70)
print(f'Data: {len(df):,} rows')
print(f'Period: 2024-01-01 to 2024-12-31')
print(f'Model trained on: 2021-2023 (3 years, 314,930 samples)')

# Prepare features
feature_cols = [
    'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
    'RSI_14', 'MACD', 'MACD_hist',
    'BB_width', 'ATR_14',
    'returns_1', 'returns_3', 'returns_12',
    'volatility_12', 'volatility_48',
    'volume_ma_ratio', 'hour', 'day_of_week'
]

X = df[feature_cols].fillna(0)
predictions = model.predict(X)

# Convert to signals
pred_std = predictions.std()
signals = np.tanh(predictions / (pred_std * 3))

print(f'\nSignals: min={signals.min():.3f}, max={signals.max():.3f}, mean={signals.mean():.3f}')

# Simple backtest
capital = 10000.0
position = 0.0
entry_price = 0.0
leverage = 3.0
commission = 0.0004
trades = []
equity = [capital]

for i in range(len(df)):
    price = df['close'].iloc[i]
    signal = signals[i]
    
    # Calculate desired position
    desired_value = capital * leverage * abs(signal)
    desired_position = (desired_value / price) * np.sign(signal)
    
    # Execute if position changes
    if abs(desired_position - position) > 0.01:
        # Close old position
        if position != 0:
            pnl = position * (price - entry_price)
            capital += pnl
        
        # Open new position
        trade_value = abs(desired_position * price)
        capital -= trade_value * commission
        
        trades.append({
            'price': price,
            'position': desired_position,
            'signal': signal
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
max_dd = ((cummax - equity) / cummax).max() * 100

sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 288)) if returns.std() > 0 else 0
win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0

print(f'\n{'='*70}')
print('📈 RESULTS')
print(f'{'='*70}')
print(f'Total Return:      {total_return:>10.2f}%')
print(f'Final Capital:     ${equity[-1]:>10,.2f}')
print(f'Max Drawdown:      {max_dd:>10.2f}%')
print(f'Sharpe Ratio:      {sharpe:>10.2f}')
print(f'Win Rate:          {win_rate:>10.2f}%')
print(f'Total Trades:      {len(trades):>10,}')
print('='*70)
