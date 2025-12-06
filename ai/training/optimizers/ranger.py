"""
Ranger Optimizer (RAdam + Lookahead)

목적: RAdam의 안정성과 Lookahead의 일반화 성능을 결합

핵심 구성:
- RAdam: Variance-adaptive learning rate
- Lookahead: Slow/Fast weights interpolation

Reference:
- "On the Variance of the Adaptive Learning Rate and Beyond" (Liu et al., 2020)
- https://arxiv.org/abs/1908.03265

특징:
- 학습 초기 단계 안정성 (RAdam)
- Generalization 성능 향상 (Lookahead)
- Hyperparameter에 덜 민감
"""

import math
import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class Ranger(Optimizer):
    """
    Ranger Optimizer = RAdam + Lookahead
    
    Args:
        params: 최적화할 파라미터
        lr: Learning rate (default: 1e-3)
        betas: Adam betas (default: (0.9, 0.999))
        eps: Numerical stability (default: 1e-8)
        weight_decay: L2 regularization (default: 0)
        k: Lookahead step (default: 6)
        alpha: Lookahead interpolation factor (default: 0.5)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        k: int = 6,
        alpha: float = 0.5
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            k=k,
            alpha=alpha
        )
        super(Ranger, self).__init__(params, defaults)
        
        # Lookahead 상태 초기화
        for group in self.param_groups:
            group['step_counter'] = 0
    
    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)
    
    def step(self, closure=None):
        """
        Single optimization step
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Ranger does not support sparse gradients')
                
                state = self.state[p]
                
                # State 초기화
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Slow weights (Lookahead)
                    state['slow_buffer'] = torch.zeros_like(p.data)
                    state['slow_buffer'].copy_(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Exponential moving average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # RAdam adaptive learning rate
                buffered = [[None, None, None] for _ in range(10)]
                
                step = state['step']
                
                # 분산 추정
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * step * (beta2 ** step) / (1 - beta2 ** step)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                if rho_t > 4:
                    # Adaptive learning rate 계산
                    r_t = math.sqrt(
                        (rho_t - 4) * (rho_t - 2) * rho_inf /
                        ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    )
                    step_size = group['lr'] * r_t / bias_correction1
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    # 초기 단계: 단순 momentum만 사용
                    step_size = group['lr'] / bias_correction1
                    p.data.add_(exp_avg, alpha=-step_size)
            
            # Lookahead step
            group['step_counter'] += 1
            if group['step_counter'] >= group['k']:
                group['step_counter'] = 0
                
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    state = self.state[p]
                    slow_buffer = state['slow_buffer']
                    
                    # Slow weights 보간
                    slow_buffer.add_(
                        p.data - slow_buffer,
                        alpha=group['alpha']
                    )
                    
                    # Fast weights를 slow weights로 복사
                    p.data.copy_(slow_buffer)
        
        return loss


if __name__ == "__main__":
    print("🧪 Testing Ranger Optimizer...")
    
    # 더미 모델
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1)
    )
    
    # Ranger optimizer
    optimizer = Ranger(model.parameters(), lr=1e-3, k=6, alpha=0.5)
    
    # 더미 학습
    for i in range(30):
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"   Step {i}: Loss = {loss.item():.4f}")
    
    print("\n✅ Ranger Optimizer test completed!")
