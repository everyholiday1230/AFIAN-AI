"""
Data Preprocessing Pipeline

목적: 원시 데이터를 정제하여 학습 가능한 형태로 변환

주요 기능:
- 결측값 처리
- 이상치 제거
- 데이터 정규화
- 시간 정렬
- 중복 제거

사용법:
    python scripts/preprocess_data.py \
        --input-dir data/raw \
        --output-dir data/processed \
        --clean-outliers \
        --fill-missing
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    데이터 전처리기
    
    Args:
        input_dir: 입력 데이터 디렉토리
        output_dir: 출력 데이터 디렉토리
    """
    
    def __init__(
        self,
        input_dir: str = "data/raw",
        output_dir: str = "data/processed"
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
    
    def check_data_quality(self, df: pd.DataFrame) -> dict:
        """데이터 품질 검사"""
        quality_report = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'time_gaps': 0,
            'zero_volume_rows': (df['volume'] == 0).sum() if 'volume' in df.columns else 0
        }
        
        # 시간 간격 체크 (1분봉 기준)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            time_diffs = df['timestamp'].diff().dt.total_seconds()
            expected_interval = 60  # 1분
            
            # 예상보다 긴 간격 (누락)
            quality_report['time_gaps'] = (time_diffs > expected_interval * 1.5).sum()
        
        return quality_report
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """중복 제거"""
        before = len(df)
        
        if 'timestamp' in df.columns:
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
        else:
            df = df.drop_duplicates()
        
        after = len(df)
        
        if before != after:
            logger.info(f"Removed {before - after} duplicate rows")
        
        return df
    
    def sort_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간순 정렬"""
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            df = df.reset_index(drop=True)
        
        return df
    
    def fill_missing_values(
        self,
        df: pd.DataFrame,
        method: str = 'forward'
    ) -> pd.DataFrame:
        """
        결측값 처리
        
        Args:
            df: DataFrame
            method: 'forward' (ffill), 'interpolate', 'drop'
        """
        missing_before = df.isnull().sum().sum()
        
        if missing_before == 0:
            return df
        
        logger.info(f"Filling {missing_before} missing values using {method}")
        
        if method == 'forward':
            # Forward fill (이전 값으로 채우기)
            df = df.fillna(method='ffill')
            # Backward fill (첫 행 결측값 처리)
            df = df.fillna(method='bfill')
        
        elif method == 'interpolate':
            # 선형 보간
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        
        elif method == 'drop':
            # 결측값 제거
            df = df.dropna()
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values after filling: {missing_after}")
        
        return df
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: list = ['open', 'high', 'low', 'close', 'volume'],
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        이상치 제거
        
        Args:
            df: DataFrame
            columns: 확인할 컬럼
            method: 'iqr' (IQR 방법), 'zscore' (Z-score 방법)
            threshold: 임계값
        """
        before = len(df)
        mask = pd.Series([True] * len(df), index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'iqr':
                # IQR (Interquartile Range) 방법
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            
            elif method == 'zscore':
                # Z-score 방법
                mean = df[col].mean()
                std = df[col].std()
                
                z_scores = np.abs((df[col] - mean) / std)
                col_mask = z_scores < threshold
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # 마스크 결합
            mask = mask & col_mask
        
        df = df[mask]
        after = len(df)
        
        if before != after:
            logger.info(f"Removed {before - after} outlier rows using {method} method")
        
        return df
    
    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        기본 피처 추가
        
        - Returns (수익률)
        - Log Returns
        - Price changes
        """
        logger.info("Adding basic features")
        
        if 'close' in df.columns:
            # Returns
            df['returns'] = df['close'].pct_change()
            
            # Log returns
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Price changes
            df['price_change'] = df['close'] - df['open']
            df['price_change_pct'] = (df['close'] - df['open']) / df['open']
        
        if 'high' in df.columns and 'low' in df.columns:
            # High-Low range
            df['hl_range'] = df['high'] - df['low']
            df['hl_range_pct'] = (df['high'] - df['low']) / df['close']
        
        if 'volume' in df.columns:
            # Volume changes
            df['volume_change'] = df['volume'].pct_change()
        
        # 첫 행 NaN 제거
        df = df.fillna(method='bfill', limit=5)
        
        return df
    
    def validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        OHLC 데이터 유효성 검증
        
        - High >= Low
        - High >= Open, Close
        - Low <= Open, Close
        """
        logger.info("Validating OHLC data")
        
        # 유효하지 않은 행 찾기
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid OHLC rows. Removing...")
            df = df[~invalid_mask]
        
        return df
    
    def process_file(
        self,
        filepath: Path,
        clean_outliers: bool = True,
        fill_missing: bool = True,
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        파일 전처리
        
        Args:
            filepath: 입력 파일 경로
            clean_outliers: 이상치 제거 여부
            fill_missing: 결측값 처리 여부
            add_features: 기본 피처 추가 여부
        """
        # 데이터 로드
        df = self.load_data(filepath)
        
        # 품질 검사
        quality_report = self.check_data_quality(df)
        logger.info(f"Data quality report: {quality_report}")
        
        # 전처리 시작
        logger.info("Starting preprocessing...")
        
        # 1. 중복 제거
        df = self.remove_duplicates(df)
        
        # 2. 시간순 정렬
        df = self.sort_by_time(df)
        
        # 3. OHLC 유효성 검증
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            df = self.validate_ohlc(df)
        
        # 4. 이상치 제거
        if clean_outliers:
            df = self.remove_outliers(df, method='iqr', threshold=3.0)
        
        # 5. 결측값 처리
        if fill_missing:
            df = self.fill_missing_values(df, method='forward')
        
        # 6. 기본 피처 추가
        if add_features:
            df = self.add_basic_features(df)
        
        logger.info(f"Preprocessing completed. Final shape: {df.shape}")
        
        return df
    
    def process_all_files(
        self,
        clean_outliers: bool = True,
        fill_missing: bool = True,
        add_features: bool = True
    ):
        """모든 파일 전처리"""
        # 입력 파일 찾기
        input_files = list(self.input_dir.glob("*.parquet")) + list(self.input_dir.glob("*.csv"))
        
        if not input_files:
            logger.error(f"No files found in {self.input_dir}")
            return
        
        logger.info(f"Found {len(input_files)} files to process")
        
        for filepath in tqdm(input_files, desc="Processing files"):
            try:
                # 전처리
                df = self.process_file(
                    filepath,
                    clean_outliers=clean_outliers,
                    fill_missing=fill_missing,
                    add_features=add_features
                )
                
                # 저장
                output_file = self.output_dir / f"{filepath.stem}_processed.parquet"
                df.to_parquet(output_file, index=False, compression='snappy')
                
                logger.info(f"Saved to {output_file}")
            
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                continue
        
        logger.info("All files processed!")


def main():
    parser = argparse.ArgumentParser(description='Preprocess raw data')
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw',
        help='Input directory (default: data/raw)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory (default: data/processed)'
    )
    
    parser.add_argument(
        '--clean-outliers',
        action='store_true',
        help='Remove outliers'
    )
    
    parser.add_argument(
        '--fill-missing',
        action='store_true',
        help='Fill missing values'
    )
    
    parser.add_argument(
        '--add-features',
        action='store_true',
        help='Add basic features'
    )
    
    args = parser.parse_args()
    
    # 전처리기 생성
    preprocessor = DataPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # 전처리 시작
    logger.info("Starting data preprocessing...")
    
    preprocessor.process_all_files(
        clean_outliers=args.clean_outliers,
        fill_missing=args.fill_missing,
        add_features=args.add_features
    )


if __name__ == "__main__":
    main()
