"""
Decision Transformer for Trading

핵심 아이디어:
- RL을 시퀀스 모델링 문제로 변환
- Return-to-go를 조건으로 최적 행동 생성
- GPT 스타일 아키텍처

Reference:
- "Decision Transformer: Reinforcement Learning via Sequence Modeling" (Chen et al., 2021)
- "Offline Reinforcement Learning as One Big Sequence Modeling Problem" (Janner et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """사인-코사인 위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class TrajectoryGPT(nn.Module):
    """
    Trajectory GPT Block
    
    GPT-style Transformer for trajectory modeling
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_length: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_length)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN like GPT
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            mask: (seq_len, seq_len) causal mask
            
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer with causal mask
        output = self.transformer(x, mask=mask, is_causal=(mask is None))
        output = self.layer_norm(output)
        
        return output


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for Trading
    
    시퀀스: [R_t, s_t, a_t, R_(t+1), s_(t+1), a_(t+1), ...]
    
    여기서:
    - R_t: Return-to-go at timestep t
    - s_t: State (market features) at timestep t
    - a_t: Action (position/size/etc) at timestep t
    
    Args:
        state_dim: State 차원 (market features)
        action_dim: Action 차원 (예: [position, size, stop_loss, take_profit])
        hidden_size: Transformer 은닉 크기
        num_layers: Transformer 레이어 개수
        num_heads: Attention 헤드 개수
        max_length: 최대 시퀀스 길이 (trajectory steps)
        action_range: Action 범위 (예: [-1, 1] for position)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_length: int = 1000,
        action_range: Tuple[float, float] = (-1.0, 1.0),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.action_range = action_range
        
        # Embedding layers
        self.embed_timestep = nn.Embedding(max_length, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(action_dim, hidden_size)
        
        # Layer normalization for embeddings
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # Transformer
        self.transformer = TrajectoryGPT(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=3 * max_length,  # [rtg, state, action] * max_length
            dropout=dropout
        )
        
        # Prediction heads
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        self.predict_state = nn.Linear(hidden_size, state_dim)
        self.predict_return = nn.Linear(hidden_size, 1)
        
        # Action scaling (from [-1,1] to action_range)
        self.action_scale = (action_range[1] - action_range[0]) / 2.0
        self.action_bias = (action_range[1] + action_range[0]) / 2.0
    
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)
            returns_to_go: (batch, seq_len, 1)
            timesteps: (batch, seq_len) - absolute timestep indices
            
        Returns:
            Dictionary with predictions
        """
        batch_size, seq_len = states.shape[:2]
        
        # Embed each component
        time_emb = self.embed_timestep(timesteps)  # (batch, seq_len, hidden)
        return_emb = self.embed_return(returns_to_go)  # (batch, seq_len, hidden)
        state_emb = self.embed_state(states)  # (batch, seq_len, hidden)
        action_emb = self.embed_action(actions)  # (batch, seq_len, hidden)
        
        # Add time embedding to each
        return_emb = return_emb + time_emb
        state_emb = state_emb + time_emb
        action_emb = action_emb + time_emb
        
        # Normalize
        return_emb = self.embed_ln(return_emb)
        state_emb = self.embed_ln(state_emb)
        action_emb = self.embed_ln(action_emb)
        
        # Stack in sequence: [rtg_1, s_1, a_1, rtg_2, s_2, a_2, ...]
        sequence = torch.stack([return_emb, state_emb, action_emb], dim=2)  # (batch, seq_len, 3, hidden)
        sequence = sequence.reshape(batch_size, 3 * seq_len, self.hidden_size)
        
        # Create causal mask
        mask_size = 3 * seq_len
        mask = self._generate_causal_mask(mask_size).to(states.device)
        
        # Apply transformer
        transformer_out = self.transformer(sequence, mask=mask)
        
        # Extract state positions (every 3rd position starting from index 1)
        # Indices: [rtg_1:0, s_1:1, a_1:2, rtg_2:3, s_2:4, a_2:5, ...]
        state_indices = torch.arange(1, 3 * seq_len, 3).to(states.device)
        state_out = transformer_out[:, state_indices, :]  # (batch, seq_len, hidden)
        
        # Make predictions from state positions
        action_preds = self.predict_action(state_out)
        state_preds = self.predict_state(state_out)
        return_preds = self.predict_return(state_out)
        
        # Scale actions to actual range
        action_preds = action_preds * self.action_scale + self.action_bias
        
        return {
            'action_preds': action_preds,
            'state_preds': state_preds,
            'return_preds': return_preds
        }
    
    @staticmethod
    def _generate_causal_mask(size: int) -> torch.Tensor:
        """
        Generate causal mask for autoregressive modeling
        
        Args:
            size: sequence length
            
        Returns:
            mask: (size, size) upper triangular mask
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
    
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        추론용: 다음 action 선택
        
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)
            returns_to_go: (batch, seq_len, 1)
            timesteps: (batch, seq_len)
            temperature: sampling temperature (not used for deterministic)
            
        Returns:
            next_action: (batch, action_dim)
        """
        with torch.no_grad():
            preds = self.forward(states, actions, returns_to_go, timesteps)
            # Return action for last timestep
            return preds['action_preds'][:, -1]
    
    def configure_optimizers(
        self, 
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999)
    ):
        """
        Configure optimizer (AdamW)
        
        Separate weight decay for different parameter groups
        """
        # Separate parameters
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'LayerNorm' in name or 'layer_norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=learning_rate, betas=betas)
        
        return optimizer


class TradingEnvironmentWrapper:
    """
    Trading Environment Wrapper for Decision Transformer
    
    시장 데이터를 DT가 이해할 수 있는 형식으로 변환
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        reward_scale: float = 1.0
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        
        # Trajectory buffer
        self.reset_trajectory()
    
    def reset_trajectory(self):
        """Reset trajectory buffers"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.returns_to_go = []
        self.timesteps = []
        self.current_return = 0.0
        self.current_step = 0
    
    def add_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float
    ):
        """Add a transition to the trajectory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward * self.reward_scale)
        self.timesteps.append(self.current_step)
        self.current_return += reward * self.reward_scale
        self.current_step += 1
    
    def get_trajectory(
        self,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get trajectory in DT format
        
        Returns:
            Dictionary with trajectory tensors
        """
        if len(self.states) == 0:
            return None
        
        # Calculate returns-to-go
        rtg = []
        cumulative = 0.0
        for r in reversed(self.rewards):
            cumulative += r
            rtg.append(cumulative)
        rtg = list(reversed(rtg))
        
        # Truncate if needed
        if max_length is not None and len(self.states) > max_length:
            start_idx = len(self.states) - max_length
            states = self.states[start_idx:]
            actions = self.actions[start_idx:]
            rtg = rtg[start_idx:]
            timesteps = self.timesteps[start_idx:]
        else:
            states = self.states
            actions = self.actions
            timesteps = self.timesteps
        
        return {
            'states': torch.stack(states),
            'actions': torch.stack(actions),
            'returns_to_go': torch.tensor(rtg).unsqueeze(-1).float(),
            'timesteps': torch.tensor(timesteps).long()
        }


if __name__ == "__main__":
    print("🧪 Testing Decision Transformer...")
    
    # Model configuration
    state_dim = 50  # Market features
    action_dim = 4  # [position, size, stop_loss, take_profit]
    batch_size = 4
    seq_len = 20
    
    # Create model
    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        max_length=1000,
        action_range=(-1.0, 1.0)
    )
    
    # Test forward pass
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randn(batch_size, seq_len, action_dim) * 0.5
    returns_to_go = torch.randn(batch_size, seq_len, 1) * 10
    timesteps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    
    print(f"\n✅ Input shapes:")
    print(f"   States: {states.shape}")
    print(f"   Actions: {actions.shape}")
    print(f"   Returns-to-go: {returns_to_go.shape}")
    print(f"   Timesteps: {timesteps.shape}")
    
    # Forward pass
    output = model(states, actions, returns_to_go, timesteps)
    
    print(f"\n✅ Output shapes:")
    print(f"   Action predictions: {output['action_preds'].shape}")
    print(f"   State predictions: {output['state_preds'].shape}")
    print(f"   Return predictions: {output['return_preds'].shape}")
    
    # Test action selection
    next_action = model.get_action(states, actions, returns_to_go, timesteps)
    print(f"\n✅ Next action shape: {next_action.shape}")
    print(f"   Action range: [{next_action.min():.4f}, {next_action.max():.4f}]")
    
    # Test optimizer configuration
    optimizer = model.configure_optimizers()
    print(f"\n✅ Optimizer configured: {type(optimizer).__name__}")
    print(f"   Number of parameter groups: {len(optimizer.param_groups)}")
    
    # Test environment wrapper
    print(f"\n✅ Testing Environment Wrapper...")
    env_wrapper = TradingEnvironmentWrapper(
        state_dim=state_dim,
        action_dim=action_dim,
        reward_scale=0.1
    )
    
    # Simulate trajectory
    for t in range(10):
        state = torch.randn(state_dim)
        action = torch.randn(action_dim) * 0.5
        reward = torch.randn(1).item()
        env_wrapper.add_transition(state, action, reward)
    
    trajectory = env_wrapper.get_trajectory()
    print(f"   Trajectory length: {len(trajectory['states'])}")
    print(f"   Returns-to-go: {trajectory['returns_to_go'].squeeze()}")
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    print("\n🎉 Decision Transformer test completed!")
