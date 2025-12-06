"""
Year-by-year 다운로드로 메모리 효율성 개선
"""

import os
import sys
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime
from tqdm import tqdm
import zipfile
import io
import argparse


def download_month_data(symbol: str, year: int, month: int) -> pd.DataFrame:
    """Download one month of data"""
    base_url = "https://data.binance.vision/data/spot/monthly/klines"
    
    month_str = f"{month:02d}"
    filename = f"{symbol}-1m-{year}-{month_str}.zip"
    url = f"{base_url}/{symbol}/1m/{filename}"
    
    try:
        response = requests.get(url, timeout=60)
        
        if response.status_code == 200:
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
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                    
                    return df
        return None
            
    except Exception as e:
        return None


def download_year(symbol: str, year: int, output_dir: Path) -> None:
    """Download one year and save immediately"""
    print(f"\n📅 Downloading {symbol} - {year}")
    
    year_data = []
    
    for month in tqdm(range(1, 13), desc=f"  {year}"):
        df = download_month_data(symbol, year, month)
        if df is not None and len(df) > 0:
            year_data.append(df)
    
    if not year_data:
        print(f"  ❌ No data for {year}")
        return
    
    # Combine and save
    df_year = pd.concat(year_data, ignore_index=True)
    df_year = df_year.sort_values('timestamp').reset_index(drop=True)
    df_year = df_year.drop_duplicates(subset=['timestamp'], keep='first')
    
    output_file = output_dir / f"{symbol}_{year}_1m.parquet"
    df_year.to_parquet(output_file, index=False)
    
    file_size = output_file.stat().st_size / 1024 / 1024
    print(f"  ✅ Saved {len(df_year):,} rows ({file_size:.2f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--start-year', type=int, default=2019)
    parser.add_argument('--end-year', type=int, default=2024)
    parser.add_argument('--output-dir', type=str, default='data/historical')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"📥 DOWNLOADING {args.symbol} ({args.start_year}-{args.end_year})")
    print(f"{'='*70}")
    
    for year in range(args.start_year, args.end_year + 1):
        download_year(args.symbol, year, output_dir)
    
    print(f"\n✅ All years downloaded!")
    print(f"\n📁 Files:")
    for file in sorted(output_dir.glob(f"{args.symbol}_*.parquet")):
        file_size = file.stat().st_size / 1024 / 1024
        print(f"   • {file.name} ({file_size:.2f} MB)")


if __name__ == "__main__":
    main()
