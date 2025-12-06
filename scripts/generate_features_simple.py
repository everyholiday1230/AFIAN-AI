"""
간단한 Feature 생성 스크립트 (테스트용)
최소한의 dependency로 빠르게 feature 생성
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai.features.technical.indicators import TechnicalIndicators


def calculate_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """기본 feature 계산"""
    result = df.copy()
    
    print("  📊 Calculating trend indicators...")
    # Trend indicators
    result['SMA_10'] = TechnicalIndicators.sma(df['close'], 10)
    result['SMA_20'] = TechnicalIndicators.sma(df['close'], 20)
    result['SMA_50'] = TechnicalIndicators.sma(df['close'], 50)
    result['EMA_12'] = TechnicalIndicators.ema(df['close'], 12)
    result['EMA_26'] = TechnicalIndicators.ema(df['close'], 26)
    
    # MACD
    macd, macd_signal, macd_hist = TechnicalIndicators.macd(df['close'])
    result['MACD'] = macd
    result['MACD_signal'] = macd_signal
    result['MACD_hist'] = macd_hist
    
    print("  📈 Calculating momentum indicators...")
    # Momentum indicators
    result['RSI_14'] = TechnicalIndicators.rsi(df['close'], 14)
    
    # Stochastic
    stoch_k, stoch_d = TechnicalIndicators.stochastic(
        df['high'], df['low'], df['close']
    )
    result['Stoch_K'] = stoch_k
    result['Stoch_D'] = stoch_d
    
    print("  💨 Calculating volatility indicators...")
    # Volatility indicators
    bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
    result['BB_upper'] = bb_upper
    result['BB_middle'] = bb_middle
    result['BB_lower'] = bb_lower
    result['BB_width'] = (bb_upper - bb_lower) / bb_middle
    
    result['ATR_14'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], 14)
    
    print("  📊 Calculating volume indicators...")
    # Volume indicators
    result['OBV'] = TechnicalIndicators.obv(df['close'], df['volume'])
    result['VWAP'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
    
    print("  🎯 Calculating price features...")
    # Price features
    result['price_change'] = df['close'].pct_change()
    result['high_low_ratio'] = df['high'] / df['low']
    result['close_open_ratio'] = df['close'] / df['open']
    
    # Volume features
    result['volume_change'] = df['volume'].pct_change()
    result['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    print("  ⏰ Calculating time features...")
    # Time features
    result['hour'] = df.index.hour
    result['day_of_week'] = df.index.dayofweek
    result['minute'] = df.index.minute
    
    # Returns
    result['returns_1'] = df['close'].pct_change(1)
    result['returns_5'] = df['close'].pct_change(5)
    result['returns_15'] = df['close'].pct_change(15)
    result['returns_60'] = df['close'].pct_change(60)
    
    # Volatility (rolling std of returns)
    result['volatility_60'] = result['returns_1'].rolling(60).std()
    result['volatility_240'] = result['returns_1'].rolling(240).std()
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Generate features from preprocessed data')
    parser.add_argument('--input-dir', type=str, default='data/test_processed',
                       help='Input directory containing preprocessed data')
    parser.add_argument('--output-dir', type=str, default='data/test_features',
                       help='Output directory for feature data')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("🎯 FEATURE GENERATION (Simple Version)")
    print(f"{'='*70}\n")
    
    print(f"📂 Input directory: {input_dir}")
    print(f"📂 Output directory: {output_dir}\n")
    
    # Find all parquet files
    parquet_files = list(input_dir.glob("*_processed.parquet"))
    
    if not parquet_files:
        print(f"❌ No processed parquet files found in {input_dir}")
        return
    
    print(f"📁 Found {len(parquet_files)} files to process\n")
    
    # Process each file
    for file_path in parquet_files:
        symbol = file_path.stem.replace('_1m_processed', '')
        print(f"\n{'─'*70}")
        print(f"⚙️  Processing: {symbol}")
        print(f"{'─'*70}")
        
        try:
            # Load data
            print(f"  📥 Loading data from {file_path.name}...")
            df = pd.read_parquet(file_path)
            
            # Set timestamp as index if it's not already
            if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            print(f"     Shape: {df.shape}")
            print(f"     Date range: {df.index[0]} to {df.index[-1]}")
            
            # Calculate features
            print(f"\n  🔧 Calculating features...")
            df_features = calculate_basic_features(df)
            
            # Handle NaN values from rolling calculations
            original_len = len(df_features)
            
            # Strategy: Drop initial warmup period (first 300 rows for indicators to stabilize)
            warmup_period = 300
            df_features = df_features.iloc[warmup_period:]
            
            # Then forward fill any remaining NaN values
            df_features = df_features.ffill().bfill()
            
            # Final check: drop any rows that still have NaN
            df_features = df_features.dropna()
            
            dropped = original_len - len(df_features)
            
            print(f"\n  ✅ Feature calculation complete!")
            print(f"     Total features: {len(df_features.columns)}")
            print(f"     Dropped {dropped} rows with NaN")
            print(f"     Final shape: {df_features.shape}")
            
            # Save to parquet
            output_path = output_dir / f"{symbol}_1m_features.parquet"
            df_features.to_parquet(output_path)
            
            file_size = output_path.stat().st_size / 1024 / 1024
            print(f"\n  💾 Saved to: {output_path.name}")
            print(f"     File size: {file_size:.2f} MB")
            
            # Show feature summary
            print(f"\n  📊 Feature Summary:")
            feature_cols = [col for col in df_features.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            print(f"     Total features: {len(feature_cols)}")
            print(f"     Features: {', '.join(feature_cols[:10])}...")
            
        except Exception as e:
            print(f"  ❌ Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("✅ FEATURE GENERATION COMPLETED!")
    print(f"{'='*70}\n")
    
    # Show final summary
    print("📋 Generated files:")
    for output_file in sorted(output_dir.glob("*.parquet")):
        file_size = output_file.stat().st_size / 1024 / 1024
        print(f"   • {output_file.name} ({file_size:.2f} MB)")
    
    print(f"\n🎉 All features generated successfully!")
    print(f"   Next step: Train models or run backtest")
    print(f"   └─ Training: python ai/training/pipelines/<model>_training_pipeline.py")
    print(f"   └─ Backtest: python scripts/run_backtest_simple.py\n")


if __name__ == "__main__":
    main()
