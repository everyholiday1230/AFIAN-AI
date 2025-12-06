"""
Decision Transformer Training Pipeline
강화학습 + 오프라인 학습으로 최적 트레이딩 행동 학습

특징:
- Offline RL with historical trajectories
- Return-conditioned action generation
- Experience replay buffer
- Multi-step returns
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.models.decision_transformer.decision_transformer import (
    DecisionTransformer,
    TradingEnvironmentWrapper
)


class TradingTrajectoryDataset(Dataset):
    """
    Trading Trajectory Dataset
    
    5년치 거래 기록을 trajectory 형식으로 변환
    """
    
    def __init__(
        self,
        trajectory_file: str,
        max_length: int = 100,
        state_dim: int = 50,
        action_dim: int = 4,
        reward_scale: float = 1.0,
    ):
        self.trajectory_file = Path(trajectory_file)
        self.max_length = max_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        
        logger.info(f"Loading trajectories from {trajectory_file}")
        self.trajectories = self._load_trajectories()
        logger.success(f"Loaded {len(self.trajectories)} trajectories")
    
    def _load_trajectories(self) -> List[Dict]:
        """Load pre-generated trajectories"""
        if not self.trajectory_file.exists():
            logger.warning(f"Trajectory file not found: {self.trajectory_file}")
            logger.info("Generating sample trajectories...")
            return self._generate_sample_trajectories()
        
        with open(self.trajectory_file, 'rb') as f:
            trajectories = pickle.load(f)
        
        return trajectories
    
    def _generate_sample_trajectories(self, num_trajectories: int = 1000) -> List[Dict]:
        """Generate sample trajectories for testing"""
        trajectories = []
        
        for _ in range(num_trajectories):
            length = np.random.randint(50, self.max_length)
            
            states = np.random.randn(length, self.state_dim)
            actions = np.random.randn(length, self.action_dim) * 0.5
            rewards = np.random.randn(length) * 10
            
            # Calculate returns-to-go
            returns_to_go = np.zeros(length)
            cumulative = 0
            for i in range(length - 1, -1, -1):
                cumulative += rewards[i] * self.reward_scale
                returns_to_go[i] = cumulative
            
            trajectories.append({
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'returns_to_go': returns_to_go,
            })
        
        return trajectories
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx]
        
        # Pad or truncate to max_length
        length = min(len(traj['states']), self.max_length)
        
        states = np.zeros((self.max_length, self.state_dim))
        actions = np.zeros((self.max_length, self.action_dim))
        returns_to_go = np.zeros((self.max_length, 1))
        timesteps = np.zeros(self.max_length, dtype=np.int64)
        mask = np.zeros(self.max_length, dtype=bool)
        
        states[:length] = traj['states'][:length]
        actions[:length] = traj['actions'][:length]
        returns_to_go[:length, 0] = traj['returns_to_go'][:length]
        timesteps[:length] = np.arange(length)
        mask[:length] = True
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.FloatTensor(actions),
            'returns_to_go': torch.FloatTensor(returns_to_go),
            'timesteps': torch.LongTensor(timesteps),
            'mask': torch.BoolTensor(mask),
        }


class DecisionTransformerLightning(pl.LightningModule):
    """
    Lightning wrapper for Decision Transformer
    """
    
    def __init__(
        self,
        model: DecisionTransformer,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, batch: Dict) -> Dict:
        return self.model(
            states=batch['states'],
            actions=batch['actions'],
            returns_to_go=batch['returns_to_go'],
            timesteps=batch['timesteps'],
        )
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        output = self(batch)
        
        # Action prediction loss
        action_preds = output['action_preds']
        action_targets = batch['actions']
        mask = batch['mask'].unsqueeze(-1)
        
        action_loss = nn.functional.mse_loss(
            action_preds * mask,
            action_targets * mask,
            reduction='sum'
        ) / mask.sum()
        
        # State prediction loss (auxiliary)
        state_preds = output['state_preds']
        state_targets = batch['states']
        
        state_loss = nn.functional.mse_loss(
            state_preds * mask[:, :, 0:1],
            state_targets * mask[:, :, 0:1],
            reduction='sum'
        ) / mask.sum()
        
        # Total loss
        loss = action_loss + 0.1 * state_loss
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_action_loss', action_loss)
        self.log('train_state_loss', state_loss)
        
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int):
        output = self(batch)
        
        action_preds = output['action_preds']
        action_targets = batch['actions']
        mask = batch['mask'].unsqueeze(-1)
        
        action_loss = nn.functional.mse_loss(
            action_preds * mask,
            action_targets * mask,
            reduction='sum'
        ) / mask.sum()
        
        state_preds = output['state_preds']
        state_targets = batch['states']
        
        state_loss = nn.functional.mse_loss(
            state_preds * mask[:, :, 0:1],
            state_targets * mask[:, :, 0:1],
            reduction='sum'
        ) / mask.sum()
        
        loss = action_loss + 0.1 * state_loss
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_action_loss', action_loss)
        self.log('val_state_loss', state_loss)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Warmup + cosine annealing
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / max(1, self.trainer.max_steps - self.warmup_steps)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }


class DecisionTransformerTrainingPipeline:
    """
    Complete DT Training Pipeline
    """
    
    def __init__(
        self,
        trajectory_file: str,
        output_dir: str = "models/decision_transformer",
        state_dim: int = 50,
        action_dim: int = 4,
    ):
        self.trajectory_file = trajectory_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        logger.info("Initializing Decision Transformer Training Pipeline")
    
    def prepare_data(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.2,
    ) -> Tuple[DataLoader, DataLoader]:
        """데이터 준비"""
        
        # Create dataset
        full_dataset = TradingTrajectoryDataset(
            trajectory_file=self.trajectory_file,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )
        
        # Split
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
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
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
    ) -> DecisionTransformerLightning:
        """모델 생성"""
        
        dt = DecisionTransformer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=1000,
            action_range=(-1.0, 1.0),
        )
        
        lightning_model = DecisionTransformerLightning(
            model=dt,
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        
        logger.success("Model created")
        
        return lightning_model
    
    def train(
        self,
        max_epochs: int = 50,
        gpus: int = 1,
        early_stop_patience: int = 10,
    ):
        """모델 학습"""
        
        # Prepare data
        train_loader, val_loader = self.prepare_data()
        
        # Create model
        model = self.create_model()
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.output_dir / "checkpoints",
            filename='dt-{epoch:02d}-{val_loss:.4f}',
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
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=gpus if torch.cuda.is_available() else 0,
            callbacks=[checkpoint_callback, early_stop_callback],
            gradient_clip_val=1.0,
            log_every_n_steps=10,
            deterministic=True,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.fit(model, train_loader, val_loader)
        
        logger.success("Training complete!")
        
        best_model_path = checkpoint_callback.best_model_path
        logger.info(f"Best model: {best_model_path}")
        
        return trainer, model


if __name__ == "__main__":
    # Configuration
    TRAJECTORY_FILE = "data/trajectories/trading_trajectories.pkl"
    
    # Create pipeline
    pipeline = DecisionTransformerTrainingPipeline(
        trajectory_file=TRAJECTORY_FILE,
        output_dir="models/decision_transformer",
        state_dim=50,
        action_dim=4,
    )
    
    # Train
    trainer, model = pipeline.train(
        max_epochs=50,
        gpus=1,
        early_stop_patience=10,
    )
    
    logger.success("✅ Decision Transformer training completed!")
