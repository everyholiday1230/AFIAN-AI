"""
Lookahead Optimizer

목적: Slow weights와 Fast weights를 결합하여 더 안정적인 학습

핵심 개념:
- Fast weights: 일반 optimizer로 빠르게 학습
- Slow weights: Fast weights를 주기적으로 보간하여 안정화

Reference:
- "Lookahead Optimizer: k steps forward, 1 step back" (Zhang et al., 2019)
- https://arxiv.org/abs/1907.08610

수식:
θ_slow = θ_slow + α * (θ_fast - θ_slow)

특징:
- 학습 안정성 향상
- Generalization 성능 개선
- Adam, SGD 등 모든 optimizer와 결합 가능
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Dict, Any
from collections import defaultdict


class Lookahead(Optimizer):
    """
    Lookahead Optimizer
    
    Args:
        optimizer: 기본 optimizer (Adam, SGD 등)
        k: Fast weights 업데이트 주기 (기본: 5)
        alpha: Slow weights 보간 비율 (기본: 0.5)
    
    Example:
        >>> base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
        >>> 
        >>> for epoch in range(epochs):
        >>>     for batch in dataloader:
        >>>         loss = criterion(model(batch), target)
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        k: int = 5,
        alpha: float = 0.5
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not k >= 1:
            raise ValueError(f"Invalid k: {k}")
        
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.step_counter = 0
        
        # Slow weights 초기화
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['slow_weights'] = torch.zeros_like(p.data)
                param_state['slow_weights'].copy_(p.data)
    
    def __getstate__(self) -> Dict[str, Any]:
        return {
            'optimizer': self.optimizer,
            'k': self.k,
            'alpha': self.alpha,
            'step_counter': self.step_counter,
            'state': self.state
        }
    
    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)
    
    def step(self, closure=None):
        """
        Single optimization step
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        
        # k번마다 slow weights 업데이트
        if self.step_counter % self.k == 0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    param_state = self.state[p]
                    slow_weights = param_state['slow_weights']
                    
                    # Slow weights 보간
                    # θ_slow = θ_slow + α * (θ_fast - θ_slow)
                    slow_weights.add_(
                        p.data - slow_weights,
                        alpha=self.alpha
                    )
                    
                    # Fast weights를 slow weights로 복사
                    p.data.copy_(slow_weights)
        
        return loss
    
    def zero_grad(self):
        """Clear gradients"""
        self.optimizer.zero_grad()
    
    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a dict"""
        return {
            'optimizer': self.optimizer.state_dict(),
            'k': self.k,
            'alpha': self.alpha,
            'step_counter': self.step_counter,
            'state': self.state
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the optimizer state"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.k = state_dict['k']
        self.alpha = state_dict['alpha']
        self.step_counter = state_dict['step_counter']
        self.state = state_dict['state']


if __name__ == "__main__":
    print("🧪 Testing Lookahead Optimizer...")
    
    # 더미 모델
    model = torch.nn.Linear(10, 1)
    
    # Base optimizer
    base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Lookahead
    optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
    
    # 더미 학습
    for i in range(20):
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 5 == 0:
            print(f"   Step {i}: Loss = {loss.item():.4f}")
    
    print("\n✅ Lookahead Optimizer test completed!")