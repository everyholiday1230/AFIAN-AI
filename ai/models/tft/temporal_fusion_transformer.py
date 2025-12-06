"""
Temporal Fusion Transformer (TFT) Implementation
Google Research 완전 구현

특징:
- Variable Selection Networks (중요 변수 자동 선택)
- Multi-Head Attention (장기 의존성)
- Quantile Regression (불확실성 추정)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class GatedLinearUnit(nn.Module):
    """게이트된 선형 유닛 (GLU)"""
    
    def __init__(self, input_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.fc_gate = nn.Linear(input_size, input_size)
        self.fc_activation = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.sigmoid(self.fc_gate(x))
        activation = self.fc_activation(x)
        return self.dropout(gate * activation)


class GatedResidualNetwork(nn.Module):
    """게이트된 잔차 네트워크 (GRN)"""
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int, 
        dropout: float = 0.1,
        context_size: Optional[int] = None
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        
        # Main layers
        if context_size is not None:
            self.fc1 = nn.Linear(input_size + context_size, hidden_size)
        else:
            self.fc1 = nn.Linear(input_size, hidden_size)
            
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = GatedLinearUnit(hidden_size, dropout)
        
        # Skip connection
        if output_size != input_size:
            self.skip = nn.Linear(input_size, output_size)
        else:
            self.skip = None
            
        self.layer_norm = nn.LayerNorm(output_size)
        
    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Residual connection
        residual = self.skip(x) if self.skip else x
        
        # Main path
        if context is not None:
            x = torch.cat([x, context], dim=-1)
            
        hidden = self.elu(self.fc1(x))
        hidden = self.fc2(hidden)
        gated = self.gate(hidden)
        
        # Residual + LayerNorm
        return self.layer_norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    """변수 선택 네트워크 (VSN)"""
    
    def __init__(
        self, 
        input_size: int, 
        num_vars: int,
        hidden_size: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_vars = num_vars
        self.hidden_size = hidden_size
        
        # 각 변수별 개별 GRN
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
            for _ in range(num_vars)
        ])
        
        # 변수 중요도 계산을 위한 GRN
        self.selection_grn = GatedResidualNetwork(
            input_size, 
            hidden_size, 
            num_vars, 
            dropout
        )
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, num_vars) 또는 (batch, num_vars)
            
        Returns:
            weighted_emb: (batch, [time,] hidden_size)
            weights: (batch, [time,] num_vars)
        """
        # 차원 처리
        has_time = x.dim() == 3
        if not has_time:
            x = x.unsqueeze(1)  # (batch, 1, num_vars)
            
        batch_size, time_steps, num_vars = x.shape
        
        # 각 변수 개별 처리
        processed_vars = []
        for i, grn in enumerate(self.variable_grns):
            var = x[..., i:i+1]  # (batch, time, 1)
            processed = grn(var)  # (batch, time, hidden_size)
            processed_vars.append(processed)
        
        # 변수 선택 가중치
        flattened = x.reshape(batch_size * time_steps, num_vars)
        weights = F.softmax(
            self.selection_grn(flattened).reshape(batch_size, time_steps, num_vars), 
            dim=-1
        )
        
        # 가중 합
        processed_stack = torch.stack(processed_vars, dim=-2)  # (batch, time, num_vars, hidden)
        weighted_emb = torch.sum(processed_stack * weights.unsqueeze(-1), dim=-2)
        
        if not has_time:
            weighted_emb = weighted_emb.squeeze(1)
            weights = weights.squeeze(1)
            
        return weighted_emb, weights


class InterpretableMultiHeadAttention(nn.Module):
    """해석 가능한 멀티헤드 어텐션"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch, query_len, hidden_size)
            key: (batch, key_len, hidden_size)
            value: (batch, value_len, hidden_size)
            mask: (batch, query_len, key_len) or None
            
        Returns:
            output: (batch, query_len, hidden_size)
            attention_weights: (batch, num_heads, query_len, key_len)
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_proj(query)  # (batch, query_len, hidden_size)
        K = self.k_proj(key)    # (batch, key_len, hidden_size)
        V = self.v_proj(value)  # (batch, value_len, hidden_size)
        
        # Reshape to multi-head
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, -1, self.hidden_size)
        output = self.out_proj(attended)
        
        return output, attention_weights


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT)
    
    Args:
        num_static_vars: 정적 변수 개수 (예: 거래쌍 ID)
        num_historical_vars: 과거 시계열 변수 개수
        num_future_vars: 미래 입력 변수 개수 (예: 예정된 이벤트)
        encoder_length: 인코더 시퀀스 길이
        decoder_length: 디코더 시퀀스 길이 (예측 범위)
        hidden_size: 은닉층 크기
        num_heads: 어텐션 헤드 개수
        num_quantiles: 예측 분위수 개수
        dropout: 드롭아웃 비율
    """
    
    def __init__(
        self,
        num_static_vars: int,
        num_historical_vars: int, 
        num_future_vars: int,
        encoder_length: int,
        decoder_length: int,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_quantiles: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.hidden_size = hidden_size
        self.num_quantiles = num_quantiles
        
        # 예측 분위수
        self.quantiles = torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9][:num_quantiles]
        )
        
        # Variable Selection Networks
        self.static_vsn = VariableSelectionNetwork(
            num_static_vars, num_static_vars, hidden_size, dropout
        ) if num_static_vars > 0 else None
        
        self.historical_vsn = VariableSelectionNetwork(
            num_historical_vars, num_historical_vars, hidden_size, dropout
        )
        
        self.future_vsn = VariableSelectionNetwork(
            num_future_vars, num_future_vars, hidden_size, dropout
        ) if num_future_vars > 0 else None
        
        # LSTM Encoder-Decoder
        self.lstm_encoder = nn.LSTM(
            hidden_size, hidden_size, num_layers=2,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True
        )
        self.lstm_decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers=2,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True
        )
        
        # Gated Residual Networks for post-LSTM processing
        self.gate_encoder = GatedLinearUnit(hidden_size, dropout)
        self.gate_decoder = GatedLinearUnit(hidden_size, dropout)
        self.gate_attn = GatedLinearUnit(hidden_size, dropout)
        
        # Layer Normalization
        self.norm_encoder = nn.LayerNorm(hidden_size)
        self.norm_decoder = nn.LayerNorm(hidden_size)
        self.norm_attn = nn.LayerNorm(hidden_size)
        
        # Multi-Head Attention
        self.multihead_attn = InterpretableMultiHeadAttention(
            hidden_size, num_heads, dropout
        )
        
        # Output Heads (분위수별)
        self.output_layer = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_quantiles)
        ])
        
    def forward(
        self,
        static_vars: Optional[torch.Tensor],
        historical_vars: torch.Tensor,
        future_vars: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            static_vars: (batch, num_static_vars) or None
            historical_vars: (batch, encoder_length, num_historical_vars)
            future_vars: (batch, decoder_length, num_future_vars) or None
            
        Returns:
            Dictionary containing:
                - predictions: (batch, decoder_length, num_quantiles)
                - attention_weights: attention weights
                - variable_importance: variable selection weights
        """
        batch_size = historical_vars.size(0)
        
        # 1. Variable Selection
        if self.static_vsn is not None and static_vars is not None:
            static_emb, static_weights = self.static_vsn(static_vars)
            static_emb = static_emb.unsqueeze(1).repeat(1, self.encoder_length, 1)
        else:
            static_emb = None
            static_weights = None
        
        hist_emb, hist_weights = self.historical_vsn(historical_vars)
        
        if self.future_vsn is not None and future_vars is not None:
            future_emb, future_weights = self.future_vsn(future_vars)
        else:
            # No future variables - use zeros
            future_emb = torch.zeros(
                batch_size, self.decoder_length, self.hidden_size,
                device=historical_vars.device
            )
            future_weights = None
        
        # 2. LSTM Processing
        encoder_input = hist_emb
        if static_emb is not None:
            encoder_input = encoder_input + static_emb
            
        encoder_out, (h_n, c_n) = self.lstm_encoder(encoder_input)
        decoder_out, _ = self.lstm_decoder(future_emb, (h_n, c_n))
        
        # 3. Gating & Normalization
        gated_encoder = self.gate_encoder(encoder_out)
        gated_encoder = self.norm_encoder(encoder_input + gated_encoder)
        
        gated_decoder = self.gate_decoder(decoder_out)
        gated_decoder = self.norm_decoder(future_emb + gated_decoder)
        
        # 4. Multi-Head Attention
        attn_out, attn_weights = self.multihead_attn(
            query=gated_decoder,
            key=gated_encoder,
            value=gated_encoder
        )
        
        attn_out = self.gate_attn(attn_out)
        final_out = self.norm_attn(gated_decoder + attn_out)
        
        # 5. Quantile Predictions
        quantile_preds = torch.stack([
            head(final_out).squeeze(-1) for head in self.output_layer
        ], dim=-1)
        
        return {
            'predictions': quantile_preds,
            'attention_weights': attn_weights,
            'variable_importance': {
                'static': static_weights,
                'historical': hist_weights,
                'future': future_weights
            }
        }


if __name__ == "__main__":
    # 테스트
    batch_size = 8
    encoder_length = 60
    decoder_length = 10
    
    model = TemporalFusionTransformer(
        num_static_vars=5,
        num_historical_vars=20,
        num_future_vars=10,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        hidden_size=128,
        num_heads=4,
        num_quantiles=9
    )
    
    static = torch.randn(batch_size, 5)
    historical = torch.randn(batch_size, encoder_length, 20)
    future = torch.randn(batch_size, decoder_length, 10)
    
    output = model(static, historical, future)
    
    print("✅ TFT Model Test")
    print(f"Predictions shape: {output['predictions'].shape}")
    print(f"Attention weights shape: {output['attention_weights'].shape}")
    print(f"Historical variable importance shape: {output['variable_importance']['historical'].shape}")
