"""
Order Flow Imbalance (OFI)

목적: 호가창 변화를 통한 단기 가격 압력 측정

Reference:
- "High-Frequency Trading and Price Discovery" (Hasbrouck & Saar, 2013)
- "The High-Frequency Trading Arms Race" (Budish et al., 2015)

수학적 정의:
OFI_t = Σ(i=1 to N) [ΔBidSize_i - ΔAskSize_i]

여기서:
- ΔBidSize_i: i번째 가격 레벨의 매수 호가 크기 변화
- ΔAskSize_i: i번째 가격 레벨의 매도 호가 크기 변화
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
import numba


class OrderFlowImbalance:
    """
    Order Flow Imbalance (OFI) 계산기
    
    OFI는 호가창의 변화를 통해 단기적 가격 압력을 측정하는 지표입니다.
    
    특징:
    - 호가창 깊이별 유동성 변화 추적
    - 매수/매도 압력의 불균형 측정
    - 고빈도 트레이딩의 핵심 지표
    
    Args:
        depth: 분석할 호가창 깊이 (기본: 10 레벨)
        weighted: 가격 레벨별 가중치 적용 여부
        normalize: 정규화 여부
    """
    
    def __init__(
        self, 
        depth: int = 10,
        weighted: bool = True,
        normalize: bool = True
    ):
        self.depth = depth
        self.weighted = weighted
        self.normalize = normalize
        self.prev_orderbook: Optional[Dict] = None
        self.history: deque = deque(maxlen=1000)
        
    def calculate(
        self, 
        orderbook: Dict[str, List[List[float]]]
    ) -> Dict[str, float]:
        """
        OFI 계산
        
        Args:
            orderbook: {
                'bids': [[price, size], ...],  # 매수 호가 (높은 가격 순)
                'asks': [[price, size], ...]   # 매도 호가 (낮은 가격 순)
            }
            
        Returns:
            {
                'ofi': OFI 값,
                'bid_ofi': 매수측 OFI,
                'ask_ofi': 매도측 OFI,
                'ofi_ratio': OFI 비율 (-1 ~ 1),
                'liquidity_imbalance': 유동성 불균형
            }
        """
        if self.prev_orderbook is None:
            self.prev_orderbook = orderbook
            return self._zero_result()
        
        # 매수/매도 호가별 OFI 계산
        bid_ofi = self._calculate_side_ofi(
            self.prev_orderbook['bids'][:self.depth],
            orderbook['bids'][:self.depth],
            'bid'
        )
        
        ask_ofi = self._calculate_side_ofi(
            self.prev_orderbook['asks'][:self.depth],
            orderbook['asks'][:self.depth],
            'ask'
        )
        
        # 전체 OFI
        ofi = bid_ofi - ask_ofi
        
        # 유동성 불균형
        total_bid_liquidity = sum(level[1] for level in orderbook['bids'][:self.depth])
        total_ask_liquidity = sum(level[1] for level in orderbook['asks'][:self.depth])
        liquidity_imbalance = (total_bid_liquidity - total_ask_liquidity) / (
            total_bid_liquidity + total_ask_liquidity + 1e-10
        )
        
        # 정규화
        if self.normalize:
            ofi_std = np.std([x['ofi'] for x in self.history]) if len(self.history) > 10 else 1.0
            ofi = ofi / (ofi_std + 1e-10)
        
        # OFI 비율 (-1 ~ 1)
        total_ofi = abs(bid_ofi) + abs(ask_ofi)
        ofi_ratio = ofi / (total_ofi + 1e-10) if total_ofi > 0 else 0.0
        ofi_ratio = np.clip(ofi_ratio, -1, 1)
        
        result = {
            'ofi': ofi,
            'bid_ofi': bid_ofi,
            'ask_ofi': ask_ofi,
            'ofi_ratio': ofi_ratio,
            'liquidity_imbalance': liquidity_imbalance,
            'total_bid_liquidity': total_bid_liquidity,
            'total_ask_liquidity': total_ask_liquidity
        }
        
        self.history.append(result)
        self.prev_orderbook = orderbook
        
        return result
    
    def _calculate_side_ofi(
        self, 
        prev_levels: List[List[float]], 
        curr_levels: List[List[float]], 
        side: str
    ) -> float:
        """
        한쪽 호가의 OFI 계산
        
        Args:
            prev_levels: 이전 호가 레벨 [[price, size], ...]
            curr_levels: 현재 호가 레벨 [[price, size], ...]
            side: 'bid' 또는 'ask'
            
        Returns:
            side_ofi: 해당 사이드의 OFI
        """
        prev_dict = {level[0]: level[1] for level in prev_levels}
        curr_dict = {level[0]: level[1] for level in curr_levels}
        
        ofi = 0.0
        all_prices = sorted(set(prev_dict.keys()) | set(curr_dict.keys()), reverse=(side == 'bid'))
        
        for idx, price in enumerate(all_prices):
            prev_size = prev_dict.get(price, 0)
            curr_size = curr_dict.get(price, 0)
            size_change = curr_size - prev_size
            
            # 가중치 계산 (상위 레벨일수록 높은 가중치)
            if self.weighted:
                weight = 1.0 / (idx + 1)  # 1, 1/2, 1/3, ...
            else:
                weight = 1.0
            
            # OFI 계산 로직
            if price not in prev_dict and price in curr_dict:
                # 새로운 유동성 추가
                ofi += size_change * weight
            elif price in prev_dict and price not in curr_dict:
                # 유동성 제거
                ofi -= prev_size * weight
            else:
                # 기존 레벨 변화
                ofi += size_change * weight
        
        return ofi
    
    def _zero_result(self) -> Dict[str, float]:
        """초기 결과"""
        return {
            'ofi': 0.0,
            'bid_ofi': 0.0,
            'ask_ofi': 0.0,
            'ofi_ratio': 0.0,
            'liquidity_imbalance': 0.0,
            'total_bid_liquidity': 0.0,
            'total_ask_liquidity': 0.0
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """OFI 통계"""
        if len(self.history) < 2:
            return {}
        
        ofi_values = [x['ofi'] for x in self.history]
        ofi_ratios = [x['ofi_ratio'] for x in self.history]
        
        return {
            'ofi_mean': np.mean(ofi_values),
            'ofi_std': np.std(ofi_values),
            'ofi_min': np.min(ofi_values),
            'ofi_max': np.max(ofi_values),
            'ofi_ratio_mean': np.mean(ofi_ratios),
            'positive_ofi_pct': np.mean([1 if x > 0 else 0 for x in ofi_values])
        }
    
    def reset(self):
        """상태 초기화"""
        self.prev_orderbook = None
        self.history.clear()


class VolumeWeightedOFI(OrderFlowImbalance):
    """
    거래량 가중 OFI
    
    일반 OFI에 거래량 정보를 추가하여 더 정확한 압력 측정
    """
    
    def __init__(self, depth: int = 10, volume_window: int = 100):
        super().__init__(depth=depth, weighted=True, normalize=True)
        self.volume_window = volume_window
        self.recent_volumes = deque(maxlen=volume_window)
    
    def calculate_with_trades(
        self,
        orderbook: Dict[str, List[List[float]]],
        recent_trades: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        거래 정보를 포함한 OFI 계산
        
        Args:
            orderbook: 호가창 정보
            recent_trades: 최근 거래 [{price, size, side}, ...]
            
        Returns:
            enhanced_ofi: 향상된 OFI 메트릭
        """
        # 기본 OFI 계산
        ofi_result = self.calculate(orderbook)
        
        # 거래량 분석
        if recent_trades:
            buy_volume = sum(t['size'] for t in recent_trades if t['side'] == 'buy')
            sell_volume = sum(t['size'] for t in recent_trades if t['side'] == 'sell')
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                volume_imbalance = (buy_volume - sell_volume) / total_volume
                self.recent_volumes.append(volume_imbalance)
            else:
                volume_imbalance = 0.0
        else:
            volume_imbalance = 0.0
        
        # OFI와 거래량 불균형 결합
        if len(self.recent_volumes) > 10:
            avg_volume_imbalance = np.mean(list(self.recent_volumes))
            combined_signal = 0.6 * ofi_result['ofi_ratio'] + 0.4 * avg_volume_imbalance
        else:
            combined_signal = ofi_result['ofi_ratio']
        
        ofi_result['volume_imbalance'] = volume_imbalance
        ofi_result['combined_signal'] = combined_signal
        
        return ofi_result


@numba.jit(nopython=True)
def calculate_microprice(
    best_bid: float,
    best_ask: float,
    bid_size: float,
    ask_size: float
) -> float:
    """
    마이크로프라이스 계산 (호가창 가중 중간 가격)
    
    microprice = (bid_size * ask + ask_size * bid) / (bid_size + ask_size)
    
    Args:
        best_bid: 최우선 매수호가
        best_ask: 최우선 매도호가
        bid_size: 매수호가 수량
        ask_size: 매도호가 수량
        
    Returns:
        microprice: 마이크로프라이스
    """
    total_size = bid_size + ask_size
    if total_size == 0:
        return (best_bid + best_ask) / 2
    
    return (bid_size * best_ask + ask_size * best_bid) / total_size


def calculate_spread_metrics(orderbook: Dict[str, List[List[float]]]) -> Dict[str, float]:
    """
    스프레드 관련 메트릭 계산
    
    Args:
        orderbook: 호가창 정보
        
    Returns:
        spread_metrics: 스프레드 관련 지표들
    """
    if not orderbook['bids'] or not orderbook['asks']:
        return {}
    
    best_bid = orderbook['bids'][0][0]
    best_ask = orderbook['asks'][0][0]
    bid_size = orderbook['bids'][0][1]
    ask_size = orderbook['asks'][0][1]
    
    # 스프레드 계산
    spread = best_ask - best_bid
    spread_bps = (spread / best_bid) * 10000  # basis points
    mid_price = (best_bid + best_ask) / 2
    
    # 마이크로프라이스
    microprice = calculate_microprice(best_bid, best_ask, bid_size, ask_size)
    
    # 가격 압력 (microprice가 mid_price보다 높으면 매수 압력)
    price_pressure = (microprice - mid_price) / mid_price
    
    return {
        'spread': spread,
        'spread_bps': spread_bps,
        'mid_price': mid_price,
        'microprice': microprice,
        'price_pressure': price_pressure,
        'bid_ask_ratio': bid_size / (ask_size + 1e-10)
    }


if __name__ == "__main__":
    print("🧪 Testing Order Flow Imbalance...")
    
    # 샘플 호가창 생성
    orderbook_t0 = {
        'bids': [
            [50000.0, 2.5],
            [49999.0, 1.8],
            [49998.0, 3.2],
            [49997.0, 1.5],
            [49996.0, 2.0],
        ],
        'asks': [
            [50001.0, 2.0],
            [50002.0, 1.5],
            [50003.0, 2.8],
            [50004.0, 1.2],
            [50005.0, 1.8],
        ]
    }
    
    orderbook_t1 = {
        'bids': [
            [50000.0, 3.5],  # 매수 압력 증가
            [49999.0, 2.0],
            [49998.0, 3.0],
            [49997.0, 1.5],
            [49996.0, 2.0],
        ],
        'asks': [
            [50001.0, 1.5],  # 매도 압력 감소
            [50002.0, 1.2],
            [50003.0, 2.5],
            [50004.0, 1.0],
            [50005.0, 1.5],
        ]
    }
    
    # OFI 계산
    ofi_calc = OrderFlowImbalance(depth=5, weighted=True, normalize=False)
    
    result_t0 = ofi_calc.calculate(orderbook_t0)
    print(f"✅ T0 OFI: {result_t0}")
    
    result_t1 = ofi_calc.calculate(orderbook_t1)
    print(f"✅ T1 OFI: {result_t1['ofi']:.4f}")
    print(f"   - Bid OFI: {result_t1['bid_ofi']:.4f}")
    print(f"   - Ask OFI: {result_t1['ask_ofi']:.4f}")
    print(f"   - OFI Ratio: {result_t1['ofi_ratio']:.4f}")
    print(f"   - Liquidity Imbalance: {result_t1['liquidity_imbalance']:.4f}")
    
    # 스프레드 메트릭
    spread_metrics = calculate_spread_metrics(orderbook_t1)
    print(f"\n✅ Spread Metrics:")
    for key, value in spread_metrics.items():
        print(f"   - {key}: {value:.6f}")
    
    # 거래량 가중 OFI 테스트
    print(f"\n✅ Testing Volume-Weighted OFI...")
    vw_ofi = VolumeWeightedOFI(depth=5, volume_window=50)
    
    recent_trades = [
        {'price': 50000.5, 'size': 0.5, 'side': 'buy'},
        {'price': 50000.3, 'size': 0.8, 'side': 'buy'},
        {'price': 50000.1, 'size': 0.3, 'side': 'sell'},
    ]
    
    vw_result = vw_ofi.calculate_with_trades(orderbook_t1, recent_trades)
    print(f"   - Combined Signal: {vw_result['combined_signal']:.4f}")
    print(f"   - Volume Imbalance: {vw_result['volume_imbalance']:.4f}")
    
    # 통계
    for _ in range(20):
        # 시뮬레이션: 랜덤 호가창 변화
        for i in range(len(orderbook_t1['bids'])):
            orderbook_t1['bids'][i][1] += np.random.randn() * 0.1
            orderbook_t1['asks'][i][1] += np.random.randn() * 0.1
        ofi_calc.calculate(orderbook_t1)
    
    stats = ofi_calc.get_statistics()
    print(f"\n✅ OFI Statistics:")
    for key, value in stats.items():
        print(f"   - {key}: {value:.4f}")
    
    print("\n🎉 Order Flow Imbalance test completed!")
