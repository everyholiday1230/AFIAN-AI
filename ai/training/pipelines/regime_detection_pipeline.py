"""
Regime Detection Training Pipeline (Guardian 학습)

목적: Contrastive VAE를 활용한 시장 상태(Regime) 자동 탐지 모델 학습

핵심 기술:
- Contrastive Learning: 유사한 시장 상태끼리 가까이, 다른 상태끼리 멀리
- VAE (Variational Autoencoder): 시장 상태의 잠재 표현 학습
- K-Means Clustering: 잠재 공간에서 자동 regime 분류

목표 Regime:
1. Bull Market (상승장)
2. Bear Market (하락장)
3. Sideways (횡보장)
4. High Volatility (고변동성)

학습 전략:
- 5년치 데이터에서 다양한 시장 상태 학습
- 자기지도학습 (Self-supervised): 레이블 불필요
- 온라인 학습: 새로운 시장 패턴 지속적 학습
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 프로젝트 경로 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from ai.models.regime_detection.contrastive_vae import ContrastiveVAE

logger = logging.getLogger(__name__)


class RegimeDataset(Dataset):
    """
    Regime Detection 학습용 데이터셋
    
    시장 데이터를 윈도우 단위로 잘라서 모델 입력으로 변환
    """
    
    def __init__(
        self,
        data_path: str,
        window_size: int = 60,
        feature_cols: Optional[List[str]] = None,
        stride: int = 1
    ):
        """
        Args:
            data_path: 데이터 파일 경로 (CSV 또는 Parquet)
            window_size: 시계열 윈도우 크기
            feature_cols: 사용할 피처 컬럼 리스트
            stride: 윈도우 이동 간격
        """
        self.window_size = window_size
        self.stride = stride
        
        # 데이터 로드
        if data_path.endswith('.parquet'):
            self.data = pd.read_parquet(data_path)
        else:
            self.data = pd.read_csv(data_path)
        
        # 피처 선택
        if feature_cols is None:
            # 기본 피처: OHLCV + 기술적 지표
            feature_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'bb_upper', 'bb_lower',
                'atr', 'adx', 'ema_fast', 'ema_slow'
            ]
        
        available_cols = [col for col in feature_cols if col in self.data.columns]
        self.feature_cols = available_cols
        
        logger.info(f"Using {len(self.feature_cols)} features: {self.feature_cols}")
        
        # 정규화
        self.scaler = StandardScaler()
        self.normalized_data = self.scaler.fit_transform(
            self.data[self.feature_cols].fillna(0).values
        )
        
        # 윈도우 인덱스 생성
        self.indices = list(range(
            0,
            len(self.normalized_data) - window_size,
            stride
        ))
        
        logger.info(f"Dataset created: {len(self.indices)} samples")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        윈도우 샘플 반환
        
        Returns:
            (window_size, num_features) 크기의 텐서
        """
        start_idx = self.indices[idx]
        end_idx = start_idx + self.window_size
        
        window = self.normalized_data[start_idx:end_idx]
        
        return torch.FloatTensor(window)


class RegimeDetectionModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Regime Detection
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: List[int] = [128, 64],
        learning_rate: float = 1e-3,
        beta: float = 0.1,  # VAE β (KL divergence weight)
        tau: float = 0.07,  # Contrastive temperature
        n_clusters: int = 4  # regime 개수
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Contrastive VAE 모델
        self.model = ContrastiveVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        )
        
        self.learning_rate = learning_rate
        self.beta = beta
        self.tau = tau
        self.n_clusters = n_clusters
        
        # K-Means (regime 분류용)
        self.kmeans = None
        
        # 메트릭 추적
        self.validation_latents = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """학습 스텝"""
        x = batch
        
        # Forward pass
        recon, mu, logvar = self.model(x)
        
        # Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL Divergence Loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        
        # Contrastive Loss (InfoNCE)
        contrastive_loss = self._contrastive_loss(mu)
        
        # Total Loss
        loss = recon_loss + self.beta * kl_loss + 0.1 * contrastive_loss
        
        # Logging
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_recon_loss', recon_loss)
        self.log('train_kl_loss', kl_loss)
        self.log('train_contrastive_loss', contrastive_loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """검증 스텝"""
        x = batch
        
        # Forward pass
        recon, mu, logvar = self.model(x)
        
        # Losses
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        contrastive_loss = self._contrastive_loss(mu)
        
        loss = recon_loss + self.beta * kl_loss + 0.1 * contrastive_loss
        
        # Logging
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_recon_loss', recon_loss)
        self.log('val_kl_loss', kl_loss)
        
        # Latent 수집 (클러스터링용)
        self.validation_latents.append(mu.detach().cpu().numpy())
        
        return loss
    
    def on_validation_epoch_end(self):
        """검증 에포크 종료 시 클러스터링"""
        if len(self.validation_latents) > 0:
            # 모든 latent 합치기
            all_latents = np.concatenate(self.validation_latents, axis=0)
            
            # K-Means 클러스터링
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            cluster_labels = self.kmeans.fit_predict(all_latents)
            
            # 클러스터 분포 로깅
            unique, counts = np.unique(cluster_labels, return_counts=True)
            cluster_dist = dict(zip(unique, counts))
            
            logger.info(f"Cluster distribution: {cluster_dist}")
            
            # 초기화
            self.validation_latents.clear()
    
    def _contrastive_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Contrastive Loss (InfoNCE)
        
        유사한 latent끼리는 가깝게, 다른 latent끼리는 멀게
        """
        batch_size = z.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=z.device)
        
        # L2 정규화
        z_norm = F.normalize(z, p=2, dim=1)
        
        # Similarity matrix
        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.tau
        
        # Positive pairs: 자기 자신 제외
        mask = torch.eye(batch_size, device=z.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # InfoNCE Loss
        loss = -torch.log(
            F.softmax(sim_matrix, dim=1).diagonal() + 1e-8
        ).mean()
        
        return loss
    
    def configure_optimizers(self):
        """Optimizer 설정"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning Rate Scheduler
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
                'monitor': 'val_loss'
            }
        }
    
    def predict_regime(self, x: torch.Tensor) -> Dict[str, any]:
        """
        Regime 예측
        
        Args:
            x: 입력 시계열 (batch_size, window_size, features)
            
        Returns:
            {
                'regime': regime 번호 (0-3),
                'latent': latent representation,
                'confidence': 신뢰도
            }
        """
        self.eval()
        with torch.no_grad():
            _, mu, _ = self.model(x)
            
            # K-Means 예측
            if self.kmeans is not None:
                latent_np = mu.cpu().numpy()
                regime_labels = self.kmeans.predict(latent_np)
                
                # 신뢰도: 가장 가까운 중심까지의 거리 역수
                distances = self.kmeans.transform(latent_np)
                min_distances = np.min(distances, axis=1)
                confidence = 1.0 / (1.0 + min_distances)
                
                return {
                    'regime': regime_labels,
                    'latent': mu,
                    'confidence': confidence
                }
            else:
                return {
                    'regime': np.zeros(mu.size(0), dtype=int),
                    'latent': mu,
                    'confidence': np.zeros(mu.size(0))
                }


def train_regime_detection(
    train_data_path: str,
    val_data_path: str,
    output_dir: str = '/home/user/webapp/data/models',
    input_dim: int = 13,
    latent_dim: int = 32,
    batch_size: int = 128,
    max_epochs: int = 50,
    learning_rate: float = 1e-3
):
    """
    Regime Detection 모델 학습
    
    Args:
        train_data_path: 학습 데이터 경로
        val_data_path: 검증 데이터 경로
        output_dir: 모델 저장 경로
        input_dim: 입력 피처 차원
        latent_dim: Latent 차원
        batch_size: 배치 크기
        max_epochs: 최대 에포크
        learning_rate: 학습률
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터셋 생성
    train_dataset = RegimeDataset(
        train_data_path,
        window_size=60,
        stride=1
    )
    
    val_dataset = RegimeDataset(
        val_data_path,
        window_size=60,
        stride=10  # 검증은 stride 크게
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 모델 생성
    model = RegimeDetectionModule(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=[128, 64],
        learning_rate=learning_rate,
        beta=0.1,
        tau=0.07,
        n_clusters=4
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='regime-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    # 학습 시작
    logger.info("🚀 Starting Regime Detection training...")
    trainer.fit(model, train_loader, val_loader)
    
    # 최종 모델 저장
    best_model_path = os.path.join(output_dir, 'regime_detection_final.pt')
    torch.save(model.state_dict(), best_model_path)
    
    logger.info(f"✅ Training completed! Model saved to {best_model_path}")
    
    return model


if __name__ == "__main__":
    print("🧪 Testing Regime Detection Pipeline...")
    
    # 더미 데이터 생성 (테스트용)
    dummy_train_data = pd.DataFrame({
        'open': np.random.randn(10000) * 100 + 50000,
        'high': np.random.randn(10000) * 100 + 50100,
        'low': np.random.randn(10000) * 100 + 49900,
        'close': np.random.randn(10000) * 100 + 50000,
        'volume': np.random.rand(10000) * 1000,
        'rsi': np.random.rand(10000) * 100,
        'macd': np.random.randn(10000) * 10,
        'bb_upper': np.random.randn(10000) * 100 + 50200,
        'bb_lower': np.random.randn(10000) * 100 + 49800,
        'atr': np.random.rand(10000) * 100,
        'adx': np.random.rand(10000) * 100,
        'ema_fast': np.random.randn(10000) * 100 + 50000,
        'ema_slow': np.random.randn(10000) * 100 + 50000,
    })
    
    train_path = '/home/user/webapp/data/dummy_train.parquet'
    val_path = '/home/user/webapp/data/dummy_val.parquet'
    
    os.makedirs('/home/user/webapp/data', exist_ok=True)
    dummy_train_data.to_parquet(train_path, index=False)
    dummy_train_data.iloc[:1000].to_parquet(val_path, index=False)
    
    # Dataset 테스트
    dataset = RegimeDataset(train_path, window_size=60, stride=1)
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    sample = dataset[0]
    print(f"   - Sample shape: {sample.shape}")
    
    # 모델 테스트
    model = RegimeDetectionModule(input_dim=13, latent_dim=32)
    
    # Forward pass
    batch = torch.stack([dataset[i] for i in range(4)])
    recon, mu, logvar = model(batch)
    
    print(f"\n✅ Model forward pass:")
    print(f"   - Input shape: {batch.shape}")
    print(f"   - Recon shape: {recon.shape}")
    print(f"   - Latent (mu) shape: {mu.shape}")
    print(f"   - Logvar shape: {logvar.shape}")
    
    print("\n🎉 Regime Detection Pipeline test completed!")
