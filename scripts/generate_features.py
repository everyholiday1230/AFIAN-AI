"""
Feature Generation Pipeline

목적: 전처리된 데이터에서 고급 피처 생성

생성 피처:
- Fractional Differencing
- Order Flow Imbalance
- Volume Profile (POC, VAH, VAL)
- Wavelet Denoising
- Technical Indicators (RSI, MACD, ATR, etc.)

사용법:
    python scripts/generate_features.py \
        --input-dir data/processed \
        --output-dir data/features \
        --all-features
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import logging
from tqdm import tqdm

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from ai.features.preprocessing.fractional_differencing import FractionalDifferencing
from ai.features.preprocessing.wavelet_denoiser import WaveletDenoiser
from ai.features.technical.indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_atr, calculate_adx, calculate_stochastic,
    calculate_ema, calculate_sma
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureGenerator:
    """
    피처 생성기
    
    Args:
        input_dir: 입력 데이터 디렉토리
        output_dir: 출력 데이터 디렉토리
    """
    
    def __init__(
        self,
        input_dir: str = "data/processed",
        output_dir: str = "data/features"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """데이터 로드"""
        logger.info(f"Loading {filepath}")
        
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Loaded {len(df)} rows")
        return df
    
    def add_fractional_differencing(
        self,
        df: pd.DataFrame,
        columns: List[str] = ['close']
    ) -> pd.DataFrame:
        """
        Fractional Differencing 적용
        
        목적: 시계열 정상성 확보 + 메모리 보존
        """
        logger.info("Adding fractional differencing features")
        
        frac_diff = FractionalDifferencing()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # 최적 d 찾기
            optimal_d = frac_diff.get_optimal_d(df[col].values)
            logger.info(f"Optimal d for {col}: {optimal_d:.3f}")
            
            # Fractional differencing 적용
            diff_series = frac_diff.fit_transform(df[col].values, d=optimal_d)
            df[f'{col}_frac_diff'] = diff_series
        
        return df
    
    def add_wavelet_denoising(
        self,
        df: pd.DataFrame,
        columns: List[str] = ['close']
    ) -> pd.DataFrame:
        """
        Wavelet Denoising 적용
        
        목적: 노이즈 제거 + 주요 트렌드 보존
        """
        logger.info("Adding wavelet denoising features")
        
        denoiser = WaveletDenoiser(wavelet='db8', level=3)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            result = denoiser.denoise(df[col].values)
            df[f'{col}_denoised'] = result.denoised
            df[f'{col}_noise'] = result.noise
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표 추가
        """
        logger.info("Adding technical indicators")
        
        prices = df['close'].values
        highs = df['high'].values if 'high' in df.columns else prices
        lows = df['low'].values if 'low' in df.columns else prices
        volumes = df['volume'].values if 'volume' in df.columns else np.ones_like(prices)
        
        # RSI
        df['rsi_14'] = calculate_rsi(prices, period=14)
        df['rsi_28'] = calculate_rsi(prices, period=28)
        
        # MACD
        macd, signal, hist = calculate_macd(prices, fast=12, slow=26, signal_period=9)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(prices, period=20, std=2.0)
        df['bb_upper'] = bb_upper
        df['bb_mid'] = bb_mid
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_mid
        
        # ATR (Average True Range)
        df['atr_14'] = calculate_atr(highs, lows, prices, period=14)
        
        # ADX (Average Directional Index)
        df['adx_14'] = calculate_adx(highs, lows, prices, period=14)
        
        # Stochastic
        stoch_k, stoch_d = calculate_stochastic(highs, lows, prices, k_period=14, d_period=3)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # EMA (Exponential Moving Average)
        df['ema_fast'] = calculate_ema(prices, period=12)
        df['ema_slow'] = calculate_ema(prices, period=26)
        df['ema_200'] = calculate_ema(prices, period=200)
        
        # SMA (Simple Moving Average)
        df['sma_20'] = calculate_sma(prices, period=20)
        df['sma_50'] = calculate_sma(prices, period=50)
        
        return df
    
    def add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        가격 패턴 피처
        """
        logger.info("Adding price pattern features")
        
        # Higher highs, Lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Price momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # Rate of change
        df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
        df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        거래량 피처
        """
        if 'volume' not in df.columns:
            return df
        
        logger.info("Adding volume features")
        
        # Volume SMA
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Volume momentum
        df['volume_momentum'] = df['volume'] - df['volume'].shift(1)
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'] - df['close'].shift(1)) * df['volume']).cumsum()
        
        # Volume-Price Trend
        df['vpt'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시간 피처
        """
        if 'timestamp' not in df.columns:
            return df
        
        logger.info("Adding time features")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Hour, Day of week, Month
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # Cyclical encoding (sine/cosine)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def add_statistical_features(self, df: pd.DataFrame, windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """
        통계적 피처
        """
        logger.info("Adding statistical features")
        
        for window in windows:
            # Rolling statistics
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'close_skew_{window}'] = df['close'].rolling(window).skew()
            df[f'close_kurt_{window}'] = df['close'].rolling(window).kurt()
            
            # Z-score
            df[f'close_zscore_{window}'] = (
                (df['close'] - df[f'close_mean_{window}']) / df[f'close_std_{window}']
            )
        
        return df
    
    def process_file(
        self,
        filepath: Path,
        add_frac_diff: bool = True,
        add_wavelet: bool = True,
        add_technical: bool = True,
        add_patterns: bool = True,
        add_volume: bool = True,
        add_time: bool = True,
        add_statistical: bool = True
    ) -> pd.DataFrame:
        """
        파일 피처 생성
        """
        # 데이터 로드
        df = self.load_data(filepath)
        
        # 피처 생성
        logger.info(f"Generating features for {filepath.name}")
        
        if add_frac_diff:
            df = self.add_fractional_differencing(df)
        
        if add_wavelet:
            df = self.add_wavelet_denoising(df)
        
        if add_technical:
            df = self.add_technical_indicators(df)
        
        if add_patterns:
            df = self.add_price_patterns(df)
        
        if add_volume:
            df = self.add_volume_features(df)
        
        if add_time:
            df = self.add_time_features(df)
        
        if add_statistical:
            df = self.add_statistical_features(df)
        
        # NaN 제거 (초기 윈도우)
        initial_nulls = df.isnull().sum().sum()
        if initial_nulls > 0:
            logger.info(f"Removing {initial_nulls} null values from initial windows")
            df = df.dropna()
        
        logger.info(f"Feature generation completed. Shape: {df.shape}")
        
        return df
    
    def process_all_files(self, all_features: bool = True):
        """모든 파일 피처 생성"""
        # 입력 파일 찾기
        input_files = list(self.input_dir.glob("*.parquet")) + list(self.input_dir.glob("*.csv"))
        
        if not input_files:
            logger.error(f"No files found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(input_files)} files to process")
        
        for filepath in tqdm(input_files, desc="Generating features"):
            try:
                # 피처 생성
                df = self.process_file(
                    filepath,
                    add_frac_diff=all_features,
                    add_wavelet=all_features,
                    add_technical=all_features,
                    add_patterns=all_features,
                    add_volume=all_features,
                    add_time=all_features,
                    add_statistical=all_features
                )
                
                # 저장
                output_file = self.output_dir / f"{filepath.stem}_features.parquet"
                df.to_parquet(output_file, index=False, compression='snappy')
                
                logger.info(f"Saved to {output_file}")
                logger.info(f"Total features: {len(df.columns)}")
            
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info("All files processed!")


def main():
    parser = argparse.ArgumentParser(description='Generate features from processed data')
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/processed',
        help='Input directory (default: data/processed)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/features',
        help='Output directory (default: data/features)'
    )
    
    parser.add_argument(
        '--all-features',
        action='store_true',
        help='Generate all features'
    )
    
    args = parser.parse_args()
    
    # 피처 생성기 생성
    generator = FeatureGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # 피처 생성 시작
    logger.info("Starting feature generation...")
    
    generator.process_all_files(all_features=args.all_features)


if __name__ == "__main__":
    main()
