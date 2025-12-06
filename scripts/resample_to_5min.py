"""
1분봉 → 5분봉 리샘플링
메모리 효율적으로 데이터 크기 1/5 축소
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def resample_to_5min(input_file: Path, output_file: Path):
    """1분봉을 5분봉으로 리샘플링"""
    
    # Load
    df = pd.read_parquet(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to 5min
    resampled = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trades': 'sum',
        'taker_buy_base': 'sum',
        'taker_buy_quote': 'sum',
    })
    
    # Remove rows with NaN (market closed periods)
    resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'])
    
    # Reset index
    resampled = resampled.reset_index()
    
    # Add derived features
    resampled['returns'] = resampled['close'].pct_change()
    resampled['log_returns'] = np.log(resampled['close'] / resampled['close'].shift(1))
    resampled['hl_range'] = resampled['high'] - resampled['low']
    resampled['hl_range_pct'] = resampled['hl_range'] / resampled['close']
    resampled['volume_change'] = resampled['volume'].pct_change()
    
    # Drop first row with NaN
    resampled = resampled.dropna()
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_parquet(output_file, index=False)
    
    return {
        'original_rows': len(df),
        'resampled_rows': len(resampled),
        'reduction': len(df) / len(resampled),
        'file_size_mb': output_file.stat().st_size / 1024 / 1024
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='data/historical_processed')
    parser.add_argument('--output-dir', type=str, default='data/historical_5min')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("⏱️  RESAMPLING TO 5-MINUTE BARS")
    print(f"{'='*70}\n")
    
    print(f"📂 Input:  {input_dir}")
    print(f"📂 Output: {output_dir}\n")
    
    files = sorted(input_dir.glob("*.parquet"))
    
    if not files:
        print(f"❌ No files in {input_dir}")
        return
    
    print(f"📁 Found {len(files)} files\n")
    
    results = []
    
    for file in tqdm(files, desc="Resampling"):
        output_file = output_dir / file.name
        
        try:
            result = resample_to_5min(file, output_file)
            results.append(result)
            
            print(f"\n✅ {file.name}")
            print(f"   {result['original_rows']:,} → {result['resampled_rows']:,} rows")
            print(f"   Reduction: {result['reduction']:.1f}x")
            print(f"   Size: {result['file_size_mb']:.2f} MB")
            
        except Exception as e:
            print(f"\n❌ Error: {file.name}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("✅ RESAMPLING COMPLETED!")
    print(f"{'='*70}\n")
    
    total_original = sum(r['original_rows'] for r in results)
    total_resampled = sum(r['resampled_rows'] for r in results)
    
    print(f"📊 Summary:")
    print(f"   Files: {len(results)}")
    print(f"   Original rows: {total_original:,}")
    print(f"   Resampled rows: {total_resampled:,}")
    print(f"   Reduction: {total_original/total_resampled:.1f}x")
    
    print(f"\n📁 Output files:")
    for file in sorted(output_dir.glob("*.parquet")):
        size = file.stat().st_size / 1024 / 1024
        print(f"   • {file.name} ({size:.2f} MB)")
    
    print(f"\n🎉 Ready for feature generation!")
    print(f"   Next: bash scripts/generate_features_batch.sh\n")


if __name__ == "__main__":
    main()
