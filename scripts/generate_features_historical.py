"""
Historical 데이터 Feature 생성 (연도별)
메모리 효율적으로 48개 feature 생성
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai.features.technical.indicators import TechnicalIndicators


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """48개 기술 지표 Feature 생성"""
    result = df.copy()
    
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
    
    # Momentum indicators
    result['RSI_14'] = TechnicalIndicators.rsi(df['close'], 14)
    
    # Stochastic
    stoch_k, stoch_d = TechnicalIndicators.stochastic(
        df['high'], df['low'], df['close']
    )
    result['Stoch_K'] = stoch_k
    result['Stoch_D'] = stoch_d
    
    # Volatility indicators
    bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
    result['BB_upper'] = bb_upper
    result['BB_middle'] = bb_middle
    result['BB_lower'] = bb_lower
    result['BB_width'] = (bb_upper - bb_lower) / bb_middle
    
    result['ATR_14'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], 14)
    
    # Volume indicators
    result['OBV'] = TechnicalIndicators.obv(df['close'], df['volume'])
    result['VWAP'] = TechnicalIndicators.vwap(df['high'], df['low'], df['close'], df['volume'])
    
    # Price features
    result['high_low_ratio'] = df['high'] / df['low']
    result['close_open_ratio'] = df['close'] / df['open']
    
    # Volume features
    result['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Time features
    result['hour'] = df['timestamp'].dt.hour
    result['day_of_week'] = df['timestamp'].dt.dayofweek
    result['minute'] = df['timestamp'].dt.minute
    
    # Returns
    result['returns_1'] = df['close'].pct_change(1)
    result['returns_5'] = df['close'].pct_change(5)
    result['returns_15'] = df['close'].pct_change(15)
    result['returns_60'] = df['close'].pct_change(60)
    
    # Volatility (rolling std of returns)
    result['volatility_60'] = result['returns_1'].rolling(60).std()
    result['volatility_240'] = result['returns_1'].rolling(240).std()
    
    return result


def process_file(input_file: Path, output_file: Path) -> dict:
    """단일 파일 Feature 생성"""
    
    # Load data
    df = pd.read_parquet(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    original_rows = len(df)
    
    # Generate features
    df_features = generate_features(df)
    
    # Drop initial rows with NaN (warmup period)
    warmup_period = 300  # 5 hours
    df_features = df_features.iloc[warmup_period:]
    
    # Forward fill any remaining NaN
    df_features = df_features.ffill().bfill()
    
    # Final check: drop any rows with NaN
    df_features = df_features.dropna()
    
    final_rows = len(df_features)
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_file, index=False)
    
    return {
        'file': input_file.name,
        'original_rows': original_rows,
        'final_rows': final_rows,
        'dropped_rows': original_rows - final_rows,
        'features': len(df_features.columns),
        'file_size_mb': output_file.stat().st_size / 1024 / 1024
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='data/historical_processed')
    parser.add_argument('--output-dir', type=str, default='data/historical_features')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("🎯 FEATURE GENERATION")
    print(f"{'='*70}\n")
    
    print(f"📂 Input:  {input_dir}")
    print(f"📂 Output: {output_dir}\n")
    
    # Find all files
    files = sorted(input_dir.glob("*.parquet"))
    
    if not files:
        print(f"❌ No files found in {input_dir}")
        return
    
    print(f"📁 Found {len(files)} files\n")
    
    results = []
    
    # Process each file
    for file in tqdm(files, desc="Generating features"):
        output_file = output_dir / file.name
        
        try:
            result = process_file(file, output_file)
            results.append(result)
            
            print(f"\n✅ {result['file']}")
            print(f"   Rows: {result['original_rows']:,} → {result['final_rows']:,}")
            print(f"   Features: {result['features']}")
            print(f"   Size: {result['file_size_mb']:.2f} MB")
            
        except Exception as e:
            print(f"\n❌ Error: {file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("✅ FEATURE GENERATION COMPLETED!")
    print(f"{'='*70}\n")
    
    # Summary
    total_original = sum(r['original_rows'] for r in results)
    total_final = sum(r['final_rows'] for r in results)
    total_dropped = total_original - total_final
    
    print(f"📊 Summary:")
    print(f"   Files: {len(results)}")
    print(f"   Total rows: {total_final:,}")
    print(f"   Dropped warmup: {total_dropped:,} ({total_dropped/total_original*100:.2f}%)")
    print(f"   Features per file: {results[0]['features'] if results else 0}")
    
    print(f"\n📁 Output files:")
    for file in sorted(output_dir.glob("*.parquet")):
        size_mb = file.stat().st_size / 1024 / 1024
        print(f"   • {file.name} ({size_mb:.2f} MB)")
    
    print(f"\n🎉 Features ready for training!")
    print(f"   Next: Train AI models\n")


if __name__ == "__main__":
    main()
