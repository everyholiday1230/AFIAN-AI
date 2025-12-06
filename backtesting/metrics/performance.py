"""
백테스팅 성능 메트릭 계산

목적: 트레이딩 전략의 성능을 정량적으로 평가

핵심 메트릭:
- Sharpe Ratio: 위험 대비 수익률
- Sortino Ratio: 하락 위험 대비 수익률
- Calmar Ratio: MDD 대비 수익률
- Maximum Drawdown (MDD): 최대 손실
- Win Rate: 승률
- Profit Factor: 총 수익 / 총 손실

Reference:
- "The Sharpe Ratio" (William Sharpe, 1966)
- "A New Measure of Risk-Adjusted Performance" (Sortino & Van der Meer, 1991)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numba


@dataclass
class PerformanceMetrics:
    """성능 메트릭 결과"""
    # 수익률 메트릭
    total_return: float
    annual_return: float
    monthly_return: float
    daily_return: float
    
    # 위험 메트릭
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # 거래 메트릭
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    
    # 추가 메트릭
    volatility: float
    downside_volatility: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR


class PerformanceAnalyzer:
    """
    트레이딩 성능 분석기
    
    Args:
        returns: 수익률 시계열 (pandas Series 또는 numpy array)
        benchmark_returns: 벤치마크 수익률 (선택사항)
        risk_free_rate: 무위험 이자율 (연율, 기본: 0.02)
        trading_days_per_year: 연간 거래일 수 (기본: 365 for crypto)
    """
    
    def __init__(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 365
    ):
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
    
    def calculate_all_metrics(self) -> PerformanceMetrics:
        """모든 성능 메트릭 계산"""
        # 수익률 메트릭
        total_return = self._total_return()
        annual_return = self._annualized_return()
        monthly_return = annual_return / 12
        daily_return = np.mean(self.returns)
        
        # 위험 메트릭
        sharpe = self._sharpe_ratio()
        sortino = self._sortino_ratio()
        calmar = self._calmar_ratio()
        mdd, mdd_duration = self._max_drawdown()
        
        # 거래 메트릭
        win_rate = self._win_rate()
        profit_factor = self._profit_factor()
        num_trades = len(self.returns)
        avg_trade_return = np.mean(self.returns)
        avg_win = np.mean(self.returns[self.returns > 0]) if np.any(self.returns > 0) else 0
        avg_loss = np.mean(self.returns[self.returns < 0]) if np.any(self.returns < 0) else 0
        
        # 추가 메트릭
        volatility = self._volatility()
        downside_vol = self._downside_volatility()
        var_95 = self._value_at_risk(0.95)
        cvar_95 = self._conditional_var(0.95)
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            monthly_return=monthly_return,
            daily_return=daily_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=mdd,
            max_drawdown_duration=mdd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades,
            avg_trade_return=avg_trade_return,
            avg_win=avg_win,
            avg_loss=avg_loss,
            volatility=volatility,
            downside_volatility=downside_vol,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def _total_return(self) -> float:
        """총 수익률"""
        cumulative_return = np.prod(1 + self.returns) - 1
        return cumulative_return
    
    def _annualized_return(self) -> float:
        """연율화 수익률"""
        total_return = self._total_return()
        n_periods = len(self.returns)
        
        if n_periods == 0:
            return 0.0
        
        annual_return = (1 + total_return) ** (self.trading_days_per_year / n_periods) - 1
        return annual_return
    
    def _volatility(self) -> float:
        """연율화 변동성"""
        if len(self.returns) == 0:
            return 0.0
        
        std = np.std(self.returns, ddof=1)
        annual_vol = std * np.sqrt(self.trading_days_per_year)
        return annual_vol
    
    def _sharpe_ratio(self) -> float:
        """
        Sharpe Ratio
        
        SR = (R_p - R_f) / σ_p
        
        R_p: 포트폴리오 수익률
        R_f: 무위험 이자율
        σ_p: 포트폴리오 변동성
        """
        annual_return = self._annualized_return()
        annual_vol = self._volatility()
        
        if annual_vol == 0:
            return 0.0
        
        sharpe = (annual_return - self.risk_free_rate) / annual_vol
        return sharpe
    
    def _downside_volatility(self, target_return: float = 0.0) -> float:
        """
        하락 변동성 (Downside Volatility)
        
        목표 수익률 이하의 수익률만 고려
        """
        downside_returns = self.returns[self.returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns, ddof=1)
        annual_downside_vol = downside_std * np.sqrt(self.trading_days_per_year)
        
        return annual_downside_vol
    
    def _sortino_ratio(self, target_return: float = 0.0) -> float:
        """
        Sortino Ratio
        
        SR = (R_p - R_t) / σ_downside
        
        R_t: 목표 수익률
        σ_downside: 하락 변동성
        """
        annual_return = self._annualized_return()
        downside_vol = self._downside_volatility(target_return)
        
        if downside_vol == 0:
            return 0.0
        
        sortino = (annual_return - target_return) / downside_vol
        return sortino
    
    def _max_drawdown(self) -> Tuple[float, int]:
        """
        Maximum Drawdown (MDD)
        
        Returns:
            (mdd, duration) - MDD 크기와 기간
        """
        cumulative = np.cumprod(1 + self.returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        mdd = np.min(drawdown)
        
        # MDD 기간 계산
        mdd_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                mdd_duration = max(mdd_duration, current_duration)
            else:
                current_duration = 0
        
        return abs(mdd), mdd_duration
    
    def _calmar_ratio(self) -> float:
        """
        Calmar Ratio
        
        CR = Annual Return / |MDD|
        """
        annual_return = self._annualized_return()
        mdd, _ = self._max_drawdown()
        
        if mdd == 0:
            return 0.0
        
        calmar = annual_return / mdd
        return calmar
    
    def _win_rate(self) -> float:
        """승률"""
        if len(self.returns) == 0:
            return 0.0
        
        wins = np.sum(self.returns > 0)
        total = len(self.returns)
        
        return wins / total
    
    def _profit_factor(self) -> float:
        """
        Profit Factor
        
        PF = Total Profit / |Total Loss|
        """
        profits = np.sum(self.returns[self.returns > 0])
        losses = abs(np.sum(self.returns[self.returns < 0]))
        
        if losses == 0:
            return np.inf if profits > 0 else 0.0
        
        return profits / losses
    
    def _value_at_risk(self, confidence_level: float = 0.95) -> float:
        """
        Value at Risk (VaR)
        
        주어진 신뢰수준에서 최대 손실
        """
        if len(self.returns) == 0:
            return 0.0
        
        var = np.percentile(self.returns, (1 - confidence_level) * 100)
        return abs(var)
    
    def _conditional_var(self, confidence_level: float = 0.95) -> float:
        """
        Conditional Value at Risk (CVaR / Expected Shortfall)
        
        VaR를 초과하는 손실의 평균
        """
        var = self._value_at_risk(confidence_level)
        threshold = -var
        
        tail_losses = self.returns[self.returns <= threshold]
        
        if len(tail_losses) == 0:
            return var
        
        cvar = abs(np.mean(tail_losses))
        return cvar


@numba.jit(nopython=True)
def calculate_drawdown_fast(cumulative_returns: np.ndarray) -> np.ndarray:
    """
    Drawdown 계산 (Numba 최적화)
    
    Args:
        cumulative_returns: 누적 수익률
        
    Returns:
        drawdown 배열
    """
    n = len(cumulative_returns)
    drawdown = np.zeros(n)
    running_max = cumulative_returns[0]
    
    for i in range(n):
        if cumulative_returns[i] > running_max:
            running_max = cumulative_returns[i]
        
        drawdown[i] = (cumulative_returns[i] - running_max) / running_max
    
    return drawdown


def print_performance_report(metrics: PerformanceMetrics):
    """성능 메트릭 보고서 출력"""
    print("\n" + "="*60)
    print("           PERFORMANCE METRICS REPORT")
    print("="*60)
    
    print("\n📈 RETURN METRICS:")
    print(f"   Total Return:        {metrics.total_return*100:>8.2f}%")
    print(f"   Annual Return:       {metrics.annual_return*100:>8.2f}%")
    print(f"   Monthly Return:      {metrics.monthly_return*100:>8.2f}%")
    print(f"   Daily Return:        {metrics.daily_return*100:>8.2f}%")
    
    print("\n⚠️  RISK METRICS:")
    print(f"   Sharpe Ratio:        {metrics.sharpe_ratio:>8.2f}")
    print(f"   Sortino Ratio:       {metrics.sortino_ratio:>8.2f}")
    print(f"   Calmar Ratio:        {metrics.calmar_ratio:>8.2f}")
    print(f"   Max Drawdown:        {metrics.max_drawdown*100:>8.2f}%")
    print(f"   MDD Duration:        {metrics.max_drawdown_duration:>8} periods")
    print(f"   Volatility:          {metrics.volatility*100:>8.2f}%")
    print(f"   Downside Vol:        {metrics.downside_volatility*100:>8.2f}%")
    print(f"   VaR (95%):           {metrics.var_95*100:>8.2f}%")
    print(f"   CVaR (95%):          {metrics.cvar_95*100:>8.2f}%")
    
    print("\n💰 TRADE METRICS:")
    print(f"   Win Rate:            {metrics.win_rate*100:>8.2f}%")
    print(f"   Profit Factor:       {metrics.profit_factor:>8.2f}")
    print(f"   Num Trades:          {metrics.num_trades:>8}")
    print(f"   Avg Trade Return:    {metrics.avg_trade_return*100:>8.2f}%")
    print(f"   Avg Win:             {metrics.avg_win*100:>8.2f}%")
    print(f"   Avg Loss:            {metrics.avg_loss*100:>8.2f}%")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("🧪 Testing Performance Metrics...")
    
    # 샘플 수익률 생성 (랜덤 워크)
    np.random.seed(42)
    
    # Bull market with drawdowns
    returns = np.random.normal(0.001, 0.02, 1000)  # 평균 0.1% 수익, 2% 변동성
    returns[200:250] = np.random.normal(-0.005, 0.03, 50)  # Drawdown 구간
    
    # 분석
    analyzer = PerformanceAnalyzer(
        returns=returns,
        risk_free_rate=0.02,
        trading_days_per_year=365
    )
    
    metrics = analyzer.calculate_all_metrics()
    
    # 보고서 출력
    print_performance_report(metrics)
    
    # Numba 최적화 테스트
    print("\n🚀 Testing Numba optimization...")
    cumulative = np.cumprod(1 + returns)
    
    import time
    start = time.perf_counter()
    dd_fast = calculate_drawdown_fast(cumulative)
    end = time.perf_counter()
    
    print(f"   Numba drawdown calculation: {(end-start)*1000:.2f}ms")
    print(f"   Max drawdown: {abs(np.min(dd_fast))*100:.2f}%")
    
    print("\n✅ Performance Metrics test completed!")
