#!/bin/bash

# 5분봉 Feature 생성

cd /home/user/webapp

echo "======================================================================="
echo "🎯 FEATURE GENERATION (5-minute bars)"
echo "======================================================================="
echo

for file in data/historical_5min/*.parquet; do
    filename=$(basename "$file")
    echo "Processing: $filename"
    
    python -c "
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, '/home/user/webapp')
from ai.features.technical.indicators import TechnicalIndicators
import numpy as np

# Load
df = pd.read_parquet('$file')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Basic features
df['SMA_10'] = TechnicalIndicators.sma(df['close'], 10)
df['SMA_20'] = TechnicalIndicators.sma(df['close'], 20)
df['SMA_50'] = TechnicalIndicators.sma(df['close'], 50)
df['EMA_12'] = TechnicalIndicators.ema(df['close'], 12)
df['EMA_26'] = TechnicalIndicators.ema(df['close'], 26)
df['RSI_14'] = TechnicalIndicators.rsi(df['close'], 14)

# MACD
macd, macd_sig, macd_hist = TechnicalIndicators.macd(df['close'])
df['MACD'] = macd
df['MACD_signal'] = macd_sig
df['MACD_hist'] = macd_hist

# Stochastic
stoch_k, stoch_d = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
df['Stoch_K'] = stoch_k
df['Stoch_D'] = stoch_d

# Bollinger Bands
bb_up, bb_mid, bb_low = TechnicalIndicators.bollinger_bands(df['close'])
df['BB_upper'] = bb_up
df['BB_middle'] = bb_mid
df['BB_lower'] = bb_low
df['BB_width'] = (bb_up - bb_low) / bb_mid

# ATR
df['ATR_14'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], 14)

# Volume indicators
df['OBV'] = TechnicalIndicators.obv(df['close'], df['volume'])
df['VWAP'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])

# Price features
df['high_low_ratio'] = df['high'] / df['low']
df['close_open_ratio'] = df['close'] / df['open']
df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

# Time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Returns (adjust for 5min bars)
df['returns_1'] = df['close'].pct_change(1)   # 5 min
df['returns_3'] = df['close'].pct_change(3)   # 15 min
df['returns_12'] = df['close'].pct_change(12) # 1 hour
df['returns_60'] = df['close'].pct_change(60) # 5 hours

# Volatility
df['volatility_12'] = df['returns_1'].rolling(12).std()  # 1 hour
df['volatility_48'] = df['returns_1'].rolling(48).std()  # 4 hours

# Drop warmup
df = df.iloc[60:]  # ~5 hours warmup
df = df.ffill().bfill().dropna()

# Save
output = 'data/historical_5min_features/$filename'
Path('data/historical_5min_features').mkdir(parents=True, exist_ok=True)
df.to_parquet(output, index=False)

print(f'✅ {len(df):,} rows, {len(df.columns)} features')
" || echo "❌ Failed: $filename"
    
    echo
done

echo "======================================================================="
echo "✅ ALL FILES PROCESSED!"
echo "======================================================================="
echo

ls -lh data/historical_5min_features/
