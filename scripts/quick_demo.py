"""
Quick Demo - 빠른 시스템 테스트

목적: 1일치 데이터로 전체 파이프라인 빠르게 테스트

테스트 내용:
1. 데이터 다운로드 (1일)
2. 데이터 전처리
3. 피처 생성
4. 간단한 백테스팅

사용법:
    python scripts/quick_demo.py
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuickDemo:
    """빠른 데모 실행기"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.demo_dir = self.project_root / "data" / "demo"
        self.demo_dir.mkdir(parents=True, exist_ok=True)
    
    def step1_download_data(self):
        """Step 1: 1일치 데이터 다운로드"""
        logger.info("="*60)
        logger.info("STEP 1: Downloading 1-day data")
        logger.info("="*60)
        
        # 어제 날짜
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        today = datetime.now().strftime('%Y-%m-%d')
        
        cmd = [
            "python", "scripts/download_historical_data.py",
            "--symbols", "BTCUSDT",
            "--start-date", yesterday,
            "--end-date", today,
            "--interval", "1m",
            "--output-dir", str(self.demo_dir / "raw")
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode != 0:
            logger.error("❌ Data download failed")
            return False
        
        logger.info("✅ Data download completed")
        return True
    
    def step2_preprocess(self):
        """Step 2: 데이터 전처리"""
        logger.info("="*60)
        logger.info("STEP 2: Preprocessing data")
        logger.info("="*60)
        
        cmd = [
            "python", "scripts/preprocess_data.py",
            "--input-dir", str(self.demo_dir / "raw"),
            "--output-dir", str(self.demo_dir / "processed"),
            "--clean-outliers",
            "--fill-missing",
            "--add-features"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode != 0:
            logger.error("❌ Preprocessing failed")
            return False
        
        logger.info("✅ Preprocessing completed")
        return True
    
    def step3_generate_features(self):
        """Step 3: 피처 생성"""
        logger.info("="*60)
        logger.info("STEP 3: Generating features")
        logger.info("="*60)
        
        cmd = [
            "python", "scripts/generate_features.py",
            "--input-dir", str(self.demo_dir / "processed"),
            "--output-dir", str(self.demo_dir / "features"),
            "--all-features"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode != 0:
            logger.error("❌ Feature generation failed")
            return False
        
        logger.info("✅ Feature generation completed")
        return True
    
    def step4_summary(self):
        """Step 4: 결과 요약"""
        logger.info("="*60)
        logger.info("STEP 4: Summary")
        logger.info("="*60)
        
        # 생성된 파일 확인
        raw_files = list((self.demo_dir / "raw").glob("*.parquet"))
        processed_files = list((self.demo_dir / "processed").glob("*.parquet"))
        feature_files = list((self.demo_dir / "features").glob("*.parquet"))
        
        logger.info(f"Raw data files: {len(raw_files)}")
        logger.info(f"Processed files: {len(processed_files)}")
        logger.info(f"Feature files: {len(feature_files)}")
        
        if feature_files:
            import pandas as pd
            
            # 피처 파일 로드
            df = pd.read_parquet(feature_files[0])
            
            logger.info(f"\n📊 Feature Data Info:")
            logger.info(f"   - Rows: {len(df):,}")
            logger.info(f"   - Columns: {len(df.columns)}")
            logger.info(f"   - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            logger.info(f"\n   - Sample columns:")
            for i, col in enumerate(df.columns[:10], 1):
                logger.info(f"     {i}. {col}")
            logger.info(f"     ... and {len(df.columns) - 10} more")
        
        logger.info("\n✅ Quick demo completed successfully!")
        logger.info(f"\n📁 Demo data saved in: {self.demo_dir}")
        
        return True
    
    def run(self):
        """전체 데모 실행"""
        logger.info("\n" + "="*60)
        logger.info("🚀 QUANTUM ALPHA - Quick Demo")
        logger.info("="*60)
        logger.info("\nThis will test the entire pipeline with 1-day data")
        logger.info("Estimated time: 3-5 minutes\n")
        
        # Step 1: 데이터 다운로드
        if not self.step1_download_data():
            logger.error("Demo failed at step 1")
            return
        
        # Step 2: 전처리
        if not self.step2_preprocess():
            logger.error("Demo failed at step 2")
            return
        
        # Step 3: 피처 생성
        if not self.step3_generate_features():
            logger.error("Demo failed at step 3")
            return
        
        # Step 4: 요약
        self.step4_summary()
        
        logger.info("\n" + "="*60)
        logger.info("🎉 Next Steps:")
        logger.info("="*60)
        logger.info("1. Review generated features in data/demo/features/")
        logger.info("2. Start full 5-year data download: python scripts/download_historical_data.py")
        logger.info("3. Train AI models: python ai/training/pipelines/tft_training_pipeline.py")
        logger.info("4. See docs/NEXT_STEPS.md for complete guide")


def main():
    demo = QuickDemo()
    demo.run()


if __name__ == "__main__":
    main()
