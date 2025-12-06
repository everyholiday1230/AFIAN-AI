"""
Historical 데이터 전처리 (연도별)
메모리 효율적으로 6년 치 데이터 처리
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse


def preprocess_file(input_file: Path, output_file: Path) -> dict:
    """
    단일 파일 전처리
    
    Returns:
        처리 결과 통계
    """
    # Load data
    df = pd.read_parquet(input_file)
    original_rows = len(df)
    
    # 1. 시간 정렬
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 2. 중복 제거
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    
    # 3. 결측값 확인
    missing_before = df.isnull().sum().sum()
    
    # 4. 숫자형 컬럼 확인 및 이상치 제거
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Zero/negative 가격 제거
    for col in ['open', 'high', 'low', 'close']:
        df = df[df[col] > 0]
    
    # Zero volume은 허용 (시장 비활동 시간)
    
    # 5. OHLC 일관성 체크
    # high >= max(open, close) and low <= min(open, close)
    df = df[
        (df['high'] >= df[['open', 'close']].max(axis=1)) &
        (df['low'] <= df[['open', 'close']].min(axis=1))
    ]
    
    # 6. 극단적 가격 변동 제거 (1분 내 50% 이상 변동은 비정상)
    df['price_change_pct'] = df['close'].pct_change().abs()
    df = df[df['price_change_pct'] < 0.5]  # 50% threshold
    
    # 7. 추가 컬럼 생성
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['hl_range'] = df['high'] - df['low']
    df['hl_range_pct'] = df['hl_range'] / df['close']
    df['volume_change'] = df['volume'].pct_change()
    
    # 8. 첫 행의 NaN 제거 (pct_change로 인한)
    df = df.dropna()
    
    # 9. 저장
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    
    # 통계 반환
    return {
        'input_file': input_file.name,
        'output_file': output_file.name,
        'original_rows': original_rows,
        'processed_rows': len(df),
        'removed_rows': original_rows - len(df),
        'missing_values_before': missing_before,
        'missing_values_after': df.isnull().sum().sum(),
        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        'file_size_mb': output_file.stat().st_size / 1024 / 1024
    }


def main():
    parser = argparse.ArgumentParser(description='Preprocess historical data')
    parser.add_argument('--input-dir', type=str, default='data/historical',
                       help='Input directory with yearly files')
    parser.add_argument('--output-dir', type=str, default='data/historical_processed',
                       help='Output directory')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("📊 DATA PREPROCESSING")
    print(f"{'='*70}\n")
    
    print(f"📂 Input:  {input_dir}")
    print(f"📂 Output: {output_dir}\n")
    
    # Find all parquet files
    files = sorted(input_dir.glob("*.parquet"))
    
    if not files:
        print(f"❌ No parquet files found in {input_dir}")
        return
    
    print(f"📁 Found {len(files)} files to process\n")
    
    results = []
    
    # Process each file
    for file in tqdm(files, desc="Processing files"):
        output_file = output_dir / file.name
        
        try:
            result = preprocess_file(file, output_file)
            results.append(result)
            
            print(f"\n✅ {result['input_file']}")
            print(f"   Original: {result['original_rows']:,} rows")
            print(f"   Processed: {result['processed_rows']:,} rows")
            print(f"   Removed: {result['removed_rows']:,} rows ({result['removed_rows']/result['original_rows']*100:.2f}%)")
            print(f"   Size: {result['file_size_mb']:.2f} MB")
            
        except Exception as e:
            print(f"\n❌ Error processing {file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("✅ PREPROCESSING COMPLETED!")
    print(f"{'='*70}\n")
    
    # Summary statistics
    total_original = sum(r['original_rows'] for r in results)
    total_processed = sum(r['processed_rows'] for r in results)
    total_removed = total_original - total_processed
    
    print(f"📊 Summary:")
    print(f"   Files processed: {len(results)}")
    print(f"   Total original rows: {total_original:,}")
    print(f"   Total processed rows: {total_processed:,}")
    print(f"   Total removed rows: {total_removed:,} ({total_removed/total_original*100:.2f}%)")
    
    print(f"\n📁 Output files:")
    for file in sorted(output_dir.glob("*.parquet")):
        size_mb = file.stat().st_size / 1024 / 1024
        print(f"   • {file.name} ({size_mb:.2f} MB)")
    
    print(f"\n🎉 Preprocessing complete!")
    print(f"   Next step: Generate features")
    print(f"   └─ python scripts/generate_features_simple.py --input-dir {output_dir}\n")


if __name__ == "__main__":
    main()
