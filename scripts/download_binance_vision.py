"""
Binance Vision 공개 데이터 다운로드
https://data.binance.vision/ 사용
"""

import os
import sys
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def download_month_data(symbol: str, year: int, month: int, data_type: str = "spot") -> pd.DataFrame:
    """
    Download data for a specific month from Binance Vision
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        year: Year
        month: Month (1-12)
        data_type: 'spot' or 'futures'
    
    Returns:
        DataFrame with OHLCV data
    """
    base_url = "https://data.binance.vision/data"
    
    # Format: data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2019-01.zip
    month_str = f"{month:02d}"
    filename = f"{symbol}-1m-{year}-{month_str}.zip"
    
    if data_type == "spot":
        url = f"{base_url}/spot/monthly/klines/{symbol}/1m/{filename}"
    else:
        url = f"{base_url}/futures/um/monthly/klines/{symbol}/1m/{filename}"
    
    try:
        response = requests.get(url, timeout=60)
        
        if response.status_code == 200:
            # Extract zip file
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df = pd.read_csv(
                        f,
                        header=None,
                        names=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades',
                            'taker_buy_base', 'taker_buy_quote', 'ignore'
                        ]
                    )
                    
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                    
                    return df
        else:
            print(f"  ⚠️  No data for {year}-{month_str} (Status: {response.status_code})")
            return None
            
    except Exception as e:
        print(f"  ❌ Error downloading {year}-{month_str}: {e}")
        return None


def download_symbol_data(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    data_type: str = "spot",
    max_workers: int = 4
) -> None:
    """
    Download all data for a symbol
    
    Args:
        symbol: Trading pair
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory
        data_type: 'spot' or 'futures'
        max_workers: Number of parallel downloads
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate list of months to download
    months = []
    current = start
    while current <= end:
        months.append((current.year, current.month))
        current = current + timedelta(days=32)
        current = current.replace(day=1)
    
    print(f"\n{'─'*70}")
    print(f"📊 Downloading {symbol}")
    print(f"{'─'*70}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Months: {len(months)}")
    print(f"  Type: {data_type}")
    print(f"  Workers: {max_workers}\n")
    
    all_data = []
    
    # Download months in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_month = {
            executor.submit(download_month_data, symbol, year, month, data_type): (year, month)
            for year, month in months
        }
        
        # Process completed downloads with progress bar
        with tqdm(total=len(months), desc=f"  Downloading", unit="month") as pbar:
            for future in as_completed(future_to_month):
                year, month = future_to_month[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        all_data.append(df)
                    pbar.update(1)
                except Exception as e:
                    print(f"  ❌ Error processing {year}-{month:02d}: {e}")
                    pbar.update(1)
    
    if not all_data:
        print(f"  ❌ No data downloaded for {symbol}")
        return
    
    # Combine all dataframes
    print(f"\n  🔄 Combining data...")
    df = pd.concat(all_data, ignore_index=True)
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    
    print(f"  ✅ Total rows: {len(df):,}")
    print(f"  📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Save to parquet
    output_file = output_dir / f"{symbol}_1m.parquet"
    df.to_parquet(output_file, index=False)
    
    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"  💾 Saved: {output_file.name} ({file_size:.2f} MB)\n")


def main():
    parser = argparse.ArgumentParser(description='Download historical data from Binance Vision')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'],
                       help='Symbols to download')
    parser.add_argument('--start-date', type=str, default='2019-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-11-30',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='data/historical',
                       help='Output directory')
    parser.add_argument('--data-type', type=str, default='spot',
                       choices=['spot', 'futures'],
                       help='Data type: spot or futures')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel download workers')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("📥 BINANCE VISION DATA DOWNLOADER")
    print(f"{'='*70}\n")
    
    print(f"📂 Output directory: {output_dir}")
    print(f"📊 Symbols: {', '.join(args.symbols)}")
    print(f"📅 Period: {args.start_date} to {args.end_date}")
    print(f"🔧 Type: {args.data_type}")
    print(f"⚡ Workers: {args.workers}")
    
    start_time = datetime.now()
    
    # Download each symbol
    for symbol in args.symbols:
        try:
            download_symbol_data(
                symbol=symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                output_dir=output_dir,
                data_type=args.data_type,
                max_workers=args.workers
            )
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")
            continue
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print(f"{'='*70}")
    print("✅ DOWNLOAD COMPLETED!")
    print(f"{'='*70}\n")
    
    print(f"⏱️  Total time: {duration:.1f} minutes")
    print(f"📁 Output files:")
    for file in sorted(output_dir.glob("*.parquet")):
        file_size = file.stat().st_size / 1024 / 1024
        print(f"   • {file.name} ({file_size:.2f} MB)")
    
    print(f"\n🎉 Data download complete!")
    print(f"   Next step: Preprocess data")
    print(f"   └─ python scripts/preprocess_data.py --input-dir {output_dir}\n")


if __name__ == "__main__":
    main()
