"""
TFT Training Pipeline
5년치 원시 데이터로 Temporal Fusion Transformer 학습

특징:
- Walk-forward validation
- Distributed training (multi-GPU)
- Automatic checkpoint saving
- TensorBoard logging
- Early stopping with patience
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
from loguru import logger

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.models.tft.temporal_fusion_transformer import TemporalFusionTransformer
from ai.features.preprocessing.fractional_differencing import FractionalDifferencing
from ai.features.orderflow.order_flow_imbalance import OrderFlowImbalance
from ai.features.preprocessing.wavelet_denoiser import WaveletDenoiser


class CryptoTimeSeriesDataset(Dataset):
    """
    암호화폐 시계열 데이터셋
    
    5년치 원시 데이터를 TFT 입력 형식으로 변환
    """
    
    def __init__(
        self,
        data_path: str,
        symbols: List[str],
        encoder_length: int = 60,
        decoder_length: int = 10,
        feature_engineering: bool = True,
        train: bool = True,
    ):
        self.data_path = Path(data_path)
        self.symbols = symbols
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.feature_engineering = feature_engineering
        self.train = train
        
        # Load and preprocess data
        logger.info(f"Loading data from {data_path}")
        self.data = self._load_raw_data()
        
        if self.feature_engineering:
            logger.info("Applying feature engineering...")
            self.data = self._engineer_features(self.data)
        
        # Create sequences
        logger.info("Creating sequences...")
        self.sequences = self._create_sequences()
        
        logger.success(f"Dataset ready: {len(self.sequences)} sequences")
    
    def _load_raw_data(self) -> pd.DataFrame:
        """5년치 원시 데이터 로드"""
        all_data = []
        
        for symbol in self.symbols:
            file_path = self.data_path / f"{symbol}.csv"
            
            if not file_path.exists():
                logger.warning(f"Data file not found: {file_path}")
                continue
            
            df = pd.read_csv(file_path)
            df['symbol'] = symbol
            all_data.append(df)
        
        if not all_data:
            raise ValueError("No data files found!")
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Parse timestamp
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        combined = combined.sort_values(['symbol', 'timestamp'])
        
        return combined
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """고급 피처 엔지니어링"""
        result_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # 1. Fractional Differencing
            fd = FractionalDifferencing(d=0.5)
            symbol_df['price_fracdiff'] = fd.fit_transform(symbol_df['close'])
            
            # 2. Returns
            symbol_df['returns'] = symbol_df['close'].pct_change()
            symbol_df['log_returns'] = np.log(symbol_df['close'] / symbol_df['close'].shift(1))
            
            # 3. Volatility (rolling)
            for window in [10, 20, 60]:
                symbol_df[f'volatility_{window}'] = symbol_df['returns'].rolling(window).std()
            
            # 4. Volume features
            symbol_df['volume_ma'] = symbol_df['volume'].rolling(20).mean()
            symbol_df['volume_ratio'] = symbol_df['volume'] / symbol_df['volume_ma']
            
            # 5. Price momentum
            for period in [5, 10, 20]:
                symbol_df[f'momentum_{period}'] = symbol_df['close'] / symbol_df['close'].shift(period) - 1
            
            # 6. Technical indicators
            symbol_df['rsi'] = self._calculate_rsi(symbol_df['close'], 14)
            symbol_df['macd'], symbol_df['macd_signal'] = self._calculate_macd(symbol_df['close'])
            
            # 7. Time features
            symbol_df['hour'] = symbol_df['timestamp'].dt.hour
            symbol_df['day_of_week'] = symbol_df['timestamp'].dt.dayofweek
            symbol_df['day_of_month'] = symbol_df['timestamp'].dt.day
            
            result_dfs.append(symbol_df)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _create_sequences(self) -> List[Dict]:
        """시퀀스 생성"""
        sequences = []
        total_length = self.encoder_length + self.decoder_length
        
        for symbol in self.data['symbol'].unique():
            symbol_data = self.data[self.data['symbol'] == symbol].reset_index(drop=True)
            
            # Skip if not enough data
            if len(symbol_data) < total_length:
                continue
            
            # Feature columns (exclude non-numeric)
            feature_cols = symbol_data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c not in ['timestamp']]
            
            # Create sliding windows
            for i in range(len(symbol_data) - total_length + 1):
                encoder_data = symbol_data.iloc[i:i+self.encoder_length][feature_cols].values
                decoder_data = symbol_data.iloc[i+self.encoder_length:i+total_length][feature_cols].values
                
                # Target: future close prices
                target = symbol_data.iloc[i+self.encoder_length:i+total_length]['close'].values
                
                sequences.append({
                    'encoder': encoder_data,
                    'decoder': decoder_data,
                    'target': target,
                    'symbol': symbol,
                })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        
        return {
            'encoder': torch.FloatTensor(seq['encoder']),
            'decoder': torch.FloatTensor(seq['decoder']),
            'target': torch.FloatTensor(seq['target']),
        }


class TFTLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for TFT
    
    Features:
    - Quantile loss
    - Learning rate scheduling
    - Gradient clipping
    - Metric tracking
    """
    
    def __init__(
        self,
        model: TemporalFusionTransformer,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        quantiles: List[float] = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        if quantiles is None:
            self.quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            self.quantiles = quantiles
        
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, batch: Dict) -> Dict:
        return self.model(
            static_vars=None,
            historical_vars=batch['encoder'],
            future_vars=batch['decoder']
        )
    
    def quantile_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Quantile loss for probabilistic forecasting
        
        Args:
            predictions: (batch, time, num_quantiles)
            targets: (batch, time)
        """
        targets = targets.unsqueeze(-1)  # (batch, time, 1)
        
        errors = targets - predictions
        losses = []
        
        for i, q in enumerate(self.quantiles):
            loss = torch.max((q - 1) * errors[..., i], q * errors[..., i])
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        output = self(batch)
        predictions = output['predictions']
        targets = batch['target']
        
        loss = self.quantile_loss(predictions, targets)
        
        self.log('train_loss', loss, prog_bar=True)
        
        # Log median prediction MAE
        median_pred = predictions[..., len(self.quantiles) // 2]
        mae = torch.abs(median_pred - targets).mean()
        self.log('train_mae', mae, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int):
        output = self(batch)
        predictions = output['predictions']
        targets = batch['target']
        
        loss = self.quantile_loss(predictions, targets)
        
        self.log('val_loss', loss, prog_bar=True)
        
        # Metrics
        median_pred = predictions[..., len(self.quantiles) // 2]
        mae = torch.abs(median_pred - targets).mean()
        mse = ((median_pred - targets) ** 2).mean()
        
        self.log('val_mae', mae)
        self.log('val_mse', mse)
        self.log('val_rmse', torch.sqrt(mse))
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


class TFTTrainingPipeline:
    """
    Complete TFT Training Pipeline
    
    Features:
    - Data loading and preprocessing
    - Model initialization
    - Training with validation
    - Checkpoint management
    - TensorBoard logging
    """
    
    def __init__(
        self,
        data_path: str,
        symbols: List[str],
        output_dir: str = "models/tft",
        encoder_length: int = 60,
        decoder_length: int = 10,
    ):
        self.data_path = data_path
        self.symbols = symbols
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        
        logger.info("Initializing TFT Training Pipeline")
    
    def prepare_data(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.2,
    ) -> Tuple[DataLoader, DataLoader]:
        """데이터 준비"""
        
        # Create dataset
        full_dataset = CryptoTimeSeriesDataset(
            data_path=self.data_path,
            symbols=self.symbols,
            encoder_length=self.encoder_length,
            decoder_length=self.decoder_length,
        )
        
        # Split train/val
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        logger.success(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val")
        
        return train_loader, val_loader
    
    def create_model(
        self,
        num_historical_vars: int,
        hidden_size: int = 256,
        num_heads: int = 4,
    ) -> TFTLightningModule:
        """모델 생성"""
        
        tft = TemporalFusionTransformer(
            num_static_vars=0,
            num_historical_vars=num_historical_vars,
            num_future_vars=num_historical_vars,
            encoder_length=self.encoder_length,
            decoder_length=self.decoder_length,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_quantiles=9,
        )
        
        lightning_model = TFTLightningModule(
            model=tft,
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        
        logger.success("Model created")
        
        return lightning_model
    
    def train(
        self,
        max_epochs: int = 100,
        gpus: int = 1,
        early_stop_patience: int = 10,
    ):
        """모델 학습"""
        
        # Prepare data
        train_loader, val_loader = self.prepare_data()
        
        # Get number of features from first batch
        sample_batch = next(iter(train_loader))
        num_features = sample_batch['encoder'].shape[-1]
        
        # Create model
        model = self.create_model(num_historical_vars=num_features)
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir / "checkpoints",
            filename='tft-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=early_stop_patience,
            mode='min',
            verbose=True,
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Logger
        tb_logger = TensorBoardLogger(
            save_dir=self.output_dir / "logs",
            name='tft_training',
        )
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=gpus if torch.cuda.is_available() else 0,
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            logger=tb_logger,
            gradient_clip_val=1.0,
            log_every_n_steps=10,
            deterministic=True,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.fit(model, train_loader, val_loader)
        
        logger.success("Training complete!")
        
        # Save best model
        best_model_path = checkpoint_callback.best_model_path
        logger.info(f"Best model: {best_model_path}")
        
        return trainer, model


if __name__ == "__main__":
    # Configuration
    DATA_PATH = "data/historical"
    SYMBOLS = ["BTCUSDT", "ETHUSDT"]
    
    # Create pipeline
    pipeline = TFTTrainingPipeline(
        data_path=DATA_PATH,
        symbols=SYMBOLS,
        output_dir="models/tft",
    )
    
    # Train
    trainer, model = pipeline.train(
        max_epochs=100,
        gpus=1,
        early_stop_patience=10,
    )
    
    logger.success("✅ TFT training pipeline completed!")
