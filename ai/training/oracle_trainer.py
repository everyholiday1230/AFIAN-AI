"""
Oracle (TFT) Trainer - 최고 성능 버전
Temporal Fusion Transformer를 사용한 가격 예측 모델 학습
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class OracleDataset(Dataset):
    """TFT용 시계열 데이터셋"""
    
    def __init__(self, data_dir: str, symbols: List[str], years: List[int], 
                 encoder_length: int, decoder_length: int):
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        
        # Load all data
        print(f'\n📥 Loading data for years: {years}')
        dfs = []
        for symbol in symbols:
            for year in years:
                file_path = Path(data_dir) / f'{symbol}_{year}_1m.parquet'
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    df['symbol'] = symbol
                    dfs.append(df)
                    print(f'   ✅ {symbol} {year}: {len(df):,} rows')
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f'\n✅ Total data: {len(self.data):,} rows')
        
        # Prepare features
        self.feature_cols = [
            'close', 'volume', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
            'RSI_14', 'MACD', 'MACD_hist', 'BB_width', 'ATR_14',
            'returns_1', 'returns_3', 'volatility_12', 'hour', 'day_of_week'
        ]
        
        # Create target (future returns)
        self.data['target'] = self.data.groupby('symbol')['close'].pct_change(decoder_length).shift(-decoder_length) * 100
        
        # Drop NaN
        self.data = self.data.dropna()
        
        # Normalize
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.data[self.feature_cols] = self.scaler.fit_transform(self.data[self.feature_cols])
        
        print(f'   After preprocessing: {len(self.data):,} rows')
        
    def __len__(self):
        return len(self.data) - self.encoder_length - self.decoder_length
    
    def __getitem__(self, idx):
        # Encoder input (past)
        encoder_data = self.data.iloc[idx:idx+self.encoder_length][self.feature_cols].values
        
        # Decoder target (future)
        target_idx = idx + self.encoder_length + self.decoder_length - 1
        target = self.data.iloc[target_idx]['target']
        
        return {
            'encoder_input': torch.FloatTensor(encoder_data),
            'target': torch.FloatTensor([target])
        }


class SimpleTFT(pl.LightningModule):
    """간소화된 TFT 모델 (PyTorch Lightning)"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        input_size = 16  # feature_cols 개수
        hidden_size = config['hidden_size']
        
        # LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            batch_first=True
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=config['attention_heads'],
            dropout=config['dropout'],
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        # Encode
        encoded, _ = self.encoder(x)
        
        # Attention
        attended, _ = self.attention(encoded, encoded, encoded)
        
        # Decode (use last timestep)
        out = self.decoder(attended[:, -1, :])
        
        return out
    
    def training_step(self, batch, batch_idx):
        x = batch['encoder_input']
        y = batch['target']
        
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['encoder_input']
        y = batch['target']
        
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }


class OracleTrainer:
    """Oracle (TFT) 학습 관리자"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def train(self):
        print('\n🔧 Preparing datasets...')
        
        # Train dataset
        train_dataset = OracleDataset(
            data_dir=self.config['data_dir'],
            symbols=self.config['symbols'],
            years=self.config['train_years'],
            encoder_length=self.config['encoder_length'],
            decoder_length=self.config['decoder_length']
        )
        
        # Val dataset
        val_dataset = OracleDataset(
            data_dir=self.config['data_dir'],
            symbols=self.config['symbols'],
            years=[self.config['test_year']],
            encoder_length=self.config['encoder_length'],
            decoder_length=self.config['decoder_length']
        )
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'] * 2,
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        print(f'   Train batches: {len(train_loader)}')
        print(f'   Val batches: {len(val_loader)}')
        
        # Model
        print('\n🏗️  Building Oracle (TFT) model...')
        model = SimpleTFT(self.config)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f'   Parameters: {n_params:,}')
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config['output_dir'],
            filename='oracle-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=self.config.get('save_top_k', 3),
            save_last=True
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.config.get('early_stopping_patience', 15),
            mode='min',
            verbose=True
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Logger
        logger = TensorBoardLogger(
            save_dir=self.config['output_dir'],
            name='tensorboard_logs'
        )
        
        # Trainer
        print('\n🎓 Starting training...')
        print(f'   Max epochs: {self.config["max_epochs"]}')
        print(f'   Batch size: {self.config["batch_size"]}')
        print(f'   Learning rate: {self.config["learning_rate"]}')
        
        trainer = pl.Trainer(
            max_epochs=self.config['max_epochs'],
            accelerator='gpu' if self.config.get('use_gpu') and torch.cuda.is_available() else 'cpu',
            devices=1,
            precision=self.config.get('precision', 32),
            gradient_clip_val=self.config.get('gradient_clip_val', 0.1),
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            logger=logger,
            log_every_n_steps=self.config.get('log_every_n_steps', 50),
        )
        
        # Train!
        trainer.fit(model, train_loader, val_loader)
        
        # Save best model
        best_model_path = Path(self.config['output_dir']) / 'best_model.ckpt'
        import shutil
        shutil.copy(checkpoint_callback.best_model_path, best_model_path)
        
        print(f'\n✅ Training completed!')
        print(f'   Best model: {best_model_path}')
        print(f'   Best val_loss: {checkpoint_callback.best_model_score:.6f}')
