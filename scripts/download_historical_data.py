"""
Historical Data Downloader - Binance Futures

목적: Binance Futures API에서 5년치 OHLCV 데이터 다운로드

특징:
- 여러 심볼 동시 다운로드
- Rate limiting 준수
- 체크포인트 지원 (중단 시 재개)
- 자동 재시도
- 진행 상황 표시

사용법:
    python scripts/download_historical_data.py \
        --symbols BTCUSDT ETHUSDT \
        --start-date 2019-01-01 \
        --end-date 2024-12-01 \
        --interval 1m \
        --output-dir data/raw
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import logging
from tqdm import tqdm
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceDataDownloader:
    """
    Binance Futures 데이터 다운로더
    
    Args:
        output_dir: 데이터 저장 디렉토리
        checkpoint_dir: 체크포인트 저장 디렉토리
    """
    
    BASE_URL = "https://fapi.binance.com"
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 1200
    REQUEST_WEIGHT_LIMIT = 2400
    
    # 데이터 제한
    MAX_LIMIT_PER_REQUEST = 1500  # Binance API limit
    
    def __init__(
        self,
        output_dir: str = "data/raw",
        checkpoint_dir: str = "data/checkpoints"
    ):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.request_count = 0
        self.last_request_time = time.time()
    
    def _rate_limit(self):
        """Rate limiting 적용"""
        self.request_count += 1
        
        # 1200 requests/minute 제한
        if self.request_count >= self.MAX_REQUESTS_PER_MINUTE:
            elapsed = time.time() - self.last_request_time
            if elapsed < 60:
                sleep_time = 60 - elapsed
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
            
            self.request_count = 0
            self.last_request_time = time.time()
        
        # Request 간 최소 간격 (안전 마진)
        time.sleep(0.05)
    
    def _get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
        limit: int = 1500
    ) -> List[List]:
        """
        Binance Futures Kline 데이터 가져오기
        
        Args:
            symbol: 심볼 (예: BTCUSDT)
            interval: 시간 간격 (1m, 5m, 15m, 1h, 4h, 1d)
            start_time: 시작 시간 (밀리초 타임스탬프)
            end_time: 종료 시간 (밀리초 타임스탬프)
            limit: 가져올 데이터 수
            
        Returns:
            List of klines
        """
        self._rate_limit()
        
        endpoint = f"{self.BASE_URL}/fapi/v1/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching klines: {e}")
            return []
    
    def download_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1m'
    ) -> pd.DataFrame:
        """
        특정 심볼의 데이터 다운로드
        
        Args:
            symbol: 심볼 (예: BTCUSDT)
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            interval: 시간 간격
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Downloading {symbol} from {start_date} to {end_date}")
        
        # 체크포인트 확인
        checkpoint_file = self.checkpoint_dir / f"{symbol}_{interval}_checkpoint.json"
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 체크포인트가 있으면 이어서 다운로드
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                last_timestamp = checkpoint.get('last_timestamp', 0)
                if last_timestamp > 0:
                    start_dt = datetime.fromtimestamp(last_timestamp / 1000)
                    logger.info(f"Resuming from checkpoint: {start_dt}")
        
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        
        all_klines = []
        current_start = start_ms
        
        # 진행 상황 표시
        total_days = (end_dt - start_dt).days
        pbar = tqdm(total=total_days, desc=f"Downloading {symbol}")
        
        retry_count = 0
        max_retries = 3
        
        while current_start < end_ms:
            try:
                klines = self._get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    end_time=end_ms,
                    limit=self.MAX_LIMIT_PER_REQUEST
                )
                
                if not klines:
                    logger.warning(f"No data received for {symbol} at {current_start}")
                    break
                
                all_klines.extend(klines)
                
                # 다음 시작 시간
                last_timestamp = klines[-1][0]
                current_start = last_timestamp + 1
                
                # 진행률 업데이트
                current_dt = datetime.fromtimestamp(last_timestamp / 1000)
                days_done = (current_dt - start_dt).days
                pbar.n = min(days_done, total_days)
                pbar.refresh()
                
                # 체크포인트 저장
                with open(checkpoint_file, 'w') as f:
                    json.dump({'last_timestamp': last_timestamp}, f)
                
                retry_count = 0  # 성공 시 재시도 카운트 리셋
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Error downloading {symbol}: {e} (retry {retry_count}/{max_retries})")
                
                if retry_count >= max_retries:
                    logger.error(f"Max retries reached for {symbol}. Saving progress...")
                    break
                
                # Exponential backoff
                sleep_time = 2 ** retry_count
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        
        pbar.close()
        
        if not all_klines:
            logger.error(f"No data downloaded for {symbol}")
            return pd.DataFrame()
        
        # DataFrame 변환
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # 데이터 타입 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                    'taker_buy_base', 'taker_buy_quote']:
            df[col] = df[col].astype(float)
        
        df['trades'] = df['trades'].astype(int)
        
        # 중복 제거
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp')
        df = df.reset_index(drop=True)
        
        logger.info(f"Downloaded {len(df)} candles for {symbol}")
        
        # 체크포인트 삭제 (완료)
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        
        return df
    
    def save_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        format: str = 'parquet'
    ):
        """
        데이터 저장
        
        Args:
            df: 저장할 DataFrame
            symbol: 심볼
            interval: 시간 간격
            format: 저장 형식 (parquet, csv)
        """
        if df.empty:
            logger.warning(f"No data to save for {symbol}")
            return
        
        filename = f"{symbol}_{interval}"
        
        if format == 'parquet':
            filepath = self.output_dir / f"{filename}.parquet"
            df.to_parquet(filepath, index=False, compression='snappy')
        elif format == 'csv':
            filepath = self.output_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {symbol} data to {filepath} ({file_size_mb:.2f} MB)")
    
    def download_multiple_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1m',
        save_format: str = 'parquet'
    ):
        """
        여러 심볼 다운로드
        
        Args:
            symbols: 심볼 리스트
            start_date: 시작 날짜
            end_date: 종료 날짜
            interval: 시간 간격
            save_format: 저장 형식
        """
        logger.info(f"Downloading {len(symbols)} symbols from {start_date} to {end_date}")
        
        for symbol in symbols:
            try:
                df = self.download_symbol(symbol, start_date, end_date, interval)
                self.save_data(df, symbol, interval, save_format)
            
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                continue
        
        logger.info("All downloads completed!")


def main():
    parser = argparse.ArgumentParser(description='Download historical data from Binance Futures')
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=['BTCUSDT', 'ETHUSDT'],
        help='Symbols to download (default: BTCUSDT ETHUSDT)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD, default: today)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        default='1m',
        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
        help='Candle interval (default: 1m)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory (default: data/raw)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='parquet',
        choices=['parquet', 'csv'],
        help='Output format (default: parquet)'
    )
    
    args = parser.parse_args()
    
    # 다운로더 생성
    downloader = BinanceDataDownloader(output_dir=args.output_dir)
    
    # 다운로드 시작
    logger.info(f"Starting download...")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Period: {args.start_date} to {args.end_date}")
    logger.info(f"Interval: {args.interval}")
    
    start_time = time.time()
    
    downloader.download_multiple_symbols(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        save_format=args.format
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Total download time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
