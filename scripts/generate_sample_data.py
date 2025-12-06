"""
Generate Sample Market Data

목적: 실제와 유사한 샘플 OHLCV 데이터 생성 (테스트용)

특징:
- Geometric Brownian Motion 기반 가격 생성
- 실제 시장과 유사한 변동성
- 트렌드 + 노이즈 + 사이클 조합
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_gbm_prices(
    initial_price: float,
    num_periods: int,
    drift: float = 0.0001,  # 일일 상승률
    volatility: float = 0.02,  # 일일 변동성
    dt: float = 1.0 / (24 * 60)  # 1분 = 1/(24*60) 일
) -> np.ndarray:
    """
    Geometric Brownian Motion으로 가격 생성
    
    dS = μ * S * dt + σ * S * dW
    """
    prices = np.zeros(num_periods)
    prices[0] = initial_price
    
    for i in range(1, num_periods):
        drift_component = drift * dt
        random_component = volatility * np.sqrt(dt) * np.random.randn()
        
        prices[i] = prices[i-1] * (1 + drift_component + random_component)
    
    return prices


def generate_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_price: float,
    base_volume: float = 1000.0
) -> pd.DataFrame:
    """
    OHLCV 데이터 생성
    
    Args:
        symbol: 심볼 (예: BTCUSDT)
        start_date: 시작 날짜
        end_date: 종료 날짜
        initial_price: 초기 가격
        base_volume: 기본 거래량
        
    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Generating sample data for {symbol}")
    
    # 시간 범위 생성
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    num_minutes = int((end_dt - start_dt).total_seconds() / 60)
    timestamps = [start_dt + timedelta(minutes=i) for i in range(num_minutes)]
    
    logger.info(f"Generating {num_minutes:,} candles ({len(timestamps):,} minutes)")
    
    # Close 가격 생성 (GBM)
    close_prices = generate_gbm_prices(
        initial_price=initial_price,
        num_periods=num_minutes,
        drift=0.00005,  # 약간의 상승 트렌드
        volatility=0.015  # 1.5% 일일 변동성
    )
    
    # OHLC 생성
    high_prices = close_prices * (1 + np.abs(np.random.randn(num_minutes)) * 0.002)
    low_prices = close_prices * (1 - np.abs(np.random.randn(num_minutes)) * 0.002)
    
    # Open 가격 (이전 Close와 유사)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = initial_price
    
    # High/Low 범위 보정
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
    
    # Volume 생성 (변동성 있게)
    volumes = base_volume * (1 + np.random.randn(num_minutes) * 0.3)
    volumes = np.abs(volumes)  # 양수로
    
    # 추가 메타데이터
    close_times = [ts + timedelta(minutes=1) for ts in timestamps]
    quote_volumes = volumes * close_prices
    num_trades = (volumes * np.random.uniform(0.5, 1.5, num_minutes)).astype(int)
    taker_buy_base = volumes * np.random.uniform(0.4, 0.6, num_minutes)
    taker_buy_quote = taker_buy_base * close_prices
    
    # DataFrame 생성
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes,
        'close_time': close_times,
        'quote_volume': quote_volumes,
        'trades': num_trades,
        'taker_buy_base': taker_buy_base,
        'taker_buy_quote': taker_buy_quote,
        'ignore': 0
    })
    
    logger.info(f"Generated {len(df):,} candles")
    logger.info(f"Price range: ${low_prices.min():.2f} - ${high_prices.max():.2f}")
    logger.info(f"Avg volume: {volumes.mean():.2f}")
    
    return df


def main():
    """샘플 데이터 생성"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample market data')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'])
    parser.add_argument('--start-date', type=str, required=True)
    parser.add_argument('--end-date', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='data/test')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 심볼별 초기 가격
    initial_prices = {
        'BTCUSDT': 95000.0,
        'ETHUSDT': 3500.0,
        'BNBUSDT': 650.0,
        'SOLUSDT': 220.0
    }
    
    base_volumes = {
        'BTCUSDT': 50.0,
        'ETHUSDT': 500.0,
        'BNBUSDT': 1000.0,
        'SOLUSDT': 2000.0
    }
    
    # 각 심볼 생성
    for symbol in args.symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*60}")
        
        initial_price = initial_prices.get(symbol, 100.0)
        base_volume = base_volumes.get(symbol, 100.0)
        
        df = generate_ohlcv(
            symbol=symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_price=initial_price,
            base_volume=base_volume
        )
        
        # 저장
        output_file = output_dir / f"{symbol}_1m.parquet"
        df.to_parquet(output_file, index=False, compression='snappy')
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Saved to {output_file} ({file_size_mb:.2f} MB)")
    
    logger.info(f"\n🎉 All sample data generated in {output_dir}")


if __name__ == "__main__":
    main()
