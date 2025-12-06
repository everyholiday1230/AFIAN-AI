"""
ONNX 추론 엔진 - 초저지연 AI 모델 추론

목적: PyTorch 모델을 ONNX로 변환하여 추론 속도 3-10배 향상

핵심 기술:
- ONNX Runtime: C++로 구현된 고성능 추론 엔진
- Quantization: INT8/FP16 양자화로 모델 크기 및 추론 시간 감소
- Graph Optimization: 불필요한 연산 제거 및 Operator Fusion

Reference:
- ONNX Runtime: https://onnxruntime.ai/
- Model Optimization: https://onnxruntime.ai/docs/performance/model-optimizations.html

성능 목표:
- TFT 추론: < 5ms (P99)
- Decision Transformer 추론: < 3ms (P99)
- Guardian 추론: < 2ms (P99)
"""

import os
import time
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Optional, Union, Tuple
import torch
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """추론 설정"""
    model_path: str
    provider: str = 'CPUExecutionProvider'  # 'CUDAExecutionProvider' for GPU
    inter_op_num_threads: int = 4
    intra_op_num_threads: int = 4
    graph_optimization_level: str = 'ORT_ENABLE_ALL'  # 최고 수준 최적화
    execution_mode: str = 'ORT_SEQUENTIAL'


class ONNXInferenceEngine:
    """
    ONNX 추론 엔진
    
    특징:
    - 멀티스레드 추론 지원
    - Dynamic Batching
    - Warm-up 자동화
    - Latency Tracking
    
    Example:
        >>> engine = ONNXInferenceEngine('models/tft.onnx')
        >>> result = engine.predict({'encoder_input': X})
        >>> print(f"Latency: {result['latency_ms']:.2f}ms")
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        warmup_iterations: int = 10
    ):
        self.config = config
        self.warmup_iterations = warmup_iterations
        
        # ONNX Runtime Session 생성
        self.session = self._create_session()
        
        # 입출력 메타데이터
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]
        
        logger.info(f"✅ ONNX Model loaded: {config.model_path}")
        logger.info(f"   - Inputs: {self.input_names}")
        logger.info(f"   - Outputs: {self.output_names}")
        logger.info(f"   - Provider: {config.provider}")
        
        # Warm-up
        self._warmup()
        
        # Latency 추적
        self.latencies: List[float] = []
        
    def _create_session(self) -> ort.InferenceSession:
        """ONNX Runtime Session 생성"""
        sess_options = ort.SessionOptions()
        
        # Thread 설정
        sess_options.inter_op_num_threads = self.config.inter_op_num_threads
        sess_options.intra_op_num_threads = self.config.intra_op_num_threads
        
        # Graph Optimization
        if self.config.graph_optimization_level == 'ORT_ENABLE_ALL':
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        elif self.config.graph_optimization_level == 'ORT_ENABLE_EXTENDED':
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        else:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        
        # Execution Mode
        if self.config.execution_mode == 'ORT_PARALLEL':
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        else:
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Provider 설정
        providers = [self.config.provider]
        if self.config.provider == 'CUDAExecutionProvider':
            providers.append('CPUExecutionProvider')  # Fallback
        
        session = ort.InferenceSession(
            self.config.model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        return session
    
    def _warmup(self):
        """Warm-up 실행 - 첫 추론은 느릴 수 있으므로 사전 실행"""
        logger.info(f"🔥 Warming up model ({self.warmup_iterations} iterations)...")
        
        # 더미 입력 생성
        dummy_inputs = {}
        for inp in self.session.get_inputs():
            shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
            dummy_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
        
        # Warm-up 실행
        for i in range(self.warmup_iterations):
            self.session.run(self.output_names, dummy_inputs)
        
        logger.info("✅ Warm-up completed")
    
    def predict(
        self,
        inputs: Dict[str, np.ndarray],
        return_latency: bool = True
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        추론 실행
        
        Args:
            inputs: 입력 딕셔너리 {input_name: np.ndarray}
            return_latency: 지연시간 반환 여부
            
        Returns:
            결과 딕셔너리 {output_name: np.ndarray, 'latency_ms': float}
        """
        # 입력 타입 변환 (torch.Tensor -> np.ndarray)
        processed_inputs = {}
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            processed_inputs[name] = value.astype(np.float32)
        
        # 추론 실행
        start_time = time.perf_counter()
        outputs = self.session.run(self.output_names, processed_inputs)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        self.latencies.append(latency_ms)
        
        # 결과 구성
        result = {name: output for name, output in zip(self.output_names, outputs)}
        
        if return_latency:
            result['latency_ms'] = latency_ms
        
        return result
    
    def batch_predict(
        self,
        batch_inputs: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        """
        배치 추론
        
        Args:
            batch_inputs: 입력 배치 리스트
            
        Returns:
            결과 배치 리스트
        """
        results = []
        for inputs in batch_inputs:
            result = self.predict(inputs, return_latency=False)
            results.append(result)
        
        return results
    
    def get_latency_stats(self) -> Dict[str, float]:
        """
        지연시간 통계
        
        Returns:
            {
                'mean_ms': 평균 지연시간,
                'p50_ms': 중앙값,
                'p95_ms': 95 백분위수,
                'p99_ms': 99 백분위수,
                'max_ms': 최대 지연시간
            }
        """
        if not self.latencies:
            return {}
        
        latencies = np.array(self.latencies)
        
        return {
            'mean_ms': float(np.mean(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'max_ms': float(np.max(latencies)),
            'count': len(latencies)
        }
    
    def reset_latency_stats(self):
        """지연시간 통계 리셋"""
        self.latencies.clear()


class TrinityONNXEnsemble:
    """
    Trinity 모델 앙상블 추론 엔진
    
    세 가지 모델을 효율적으로 추론:
    1. Oracle (TFT): 시장 예측
    2. Strategist (Decision Transformer): 액션 결정
    3. Guardian (Contrastive VAE): 시장 상태 감지
    """
    
    def __init__(
        self,
        oracle_path: str,
        strategist_path: str,
        guardian_path: str,
        provider: str = 'CPUExecutionProvider'
    ):
        logger.info("🚀 Initializing Trinity ONNX Ensemble...")
        
        # Oracle (TFT)
        self.oracle = ONNXInferenceEngine(
            InferenceConfig(
                model_path=oracle_path,
                provider=provider,
                inter_op_num_threads=4,
                intra_op_num_threads=4
            )
        )
        
        # Strategist (Decision Transformer)
        self.strategist = ONNXInferenceEngine(
            InferenceConfig(
                model_path=strategist_path,
                provider=provider,
                inter_op_num_threads=2,
                intra_op_num_threads=2
            )
        )
        
        # Guardian (Contrastive VAE)
        self.guardian = ONNXInferenceEngine(
            InferenceConfig(
                model_path=guardian_path,
                provider=provider,
                inter_op_num_threads=2,
                intra_op_num_threads=2
            )
        )
        
        logger.info("✅ Trinity Ensemble ready")
    
    def predict_full_pipeline(
        self,
        market_data: Dict[str, np.ndarray],
        current_state: Dict[str, np.ndarray]
    ) -> Dict[str, any]:
        """
        전체 Trinity 파이프라인 실행
        
        Args:
            market_data: 시장 데이터
            current_state: 현재 상태 (포지션, PnL 등)
            
        Returns:
            {
                'oracle_prediction': TFT 예측 (가격, 변동성 등),
                'strategist_action': 최적 액션 (buy/sell/hold),
                'guardian_regime': 시장 상태 (bull/bear/sideways/high_vol),
                'total_latency_ms': 전체 지연시간
            }
        """
        start_time = time.perf_counter()
        
        # 1. Guardian: 시장 상태 감지
        guardian_result = self.guardian.predict(market_data, return_latency=True)
        
        # 2. Oracle: 가격 예측
        oracle_result = self.oracle.predict(market_data, return_latency=True)
        
        # 3. Strategist: 액션 결정
        strategist_inputs = {
            **current_state,
            'market_prediction': oracle_result[self.oracle.output_names[0]]
        }
        strategist_result = self.strategist.predict(strategist_inputs, return_latency=True)
        
        end_time = time.perf_counter()
        total_latency_ms = (end_time - start_time) * 1000
        
        return {
            'oracle_prediction': oracle_result,
            'strategist_action': strategist_result,
            'guardian_regime': guardian_result,
            'total_latency_ms': total_latency_ms,
            'individual_latencies': {
                'oracle_ms': oracle_result.get('latency_ms', 0),
                'strategist_ms': strategist_result.get('latency_ms', 0),
                'guardian_ms': guardian_result.get('latency_ms', 0)
            }
        }
    
    def get_all_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """모든 모델의 지연시간 통계"""
        return {
            'oracle': self.oracle.get_latency_stats(),
            'strategist': self.strategist.get_latency_stats(),
            'guardian': self.guardian.get_latency_stats()
        }


def export_pytorch_to_onnx(
    model: torch.nn.Module,
    dummy_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    output_path: str,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 14
):
    """
    PyTorch 모델을 ONNX로 변환
    
    Args:
        model: PyTorch 모델
        dummy_input: 더미 입력 (모델 구조 추론용)
        output_path: ONNX 파일 저장 경로
        input_names: 입력 노드 이름 리스트
        output_names: 출력 노드 이름 리스트
        dynamic_axes: 동적 차원 설정 (배치 크기 등)
        opset_version: ONNX opset 버전
    """
    model.eval()
    
    # ONNX 변환
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    logger.info(f"✅ PyTorch model exported to ONNX: {output_path}")
    
    # 모델 크기 확인
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"   - Model size: {model_size_mb:.2f} MB")


if __name__ == "__main__":
    print("🧪 Testing ONNX Inference Engine...")
    
    # 더미 ONNX 모델 생성 (테스트용)
    import torch.nn as nn
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 3)
        
        def forward(self, x):
            return self.fc(x)
    
    model = DummyModel()
    dummy_input = torch.randn(1, 10)
    
    # ONNX 변환
    onnx_path = "/home/user/webapp/data/models/dummy_model.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    export_pytorch_to_onnx(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # 추론 테스트
    config = InferenceConfig(model_path=onnx_path)
    engine = ONNXInferenceEngine(config, warmup_iterations=5)
    
    # 100번 추론 실행
    for i in range(100):
        test_input = {'input': np.random.randn(1, 10).astype(np.float32)}
        result = engine.predict(test_input)
        if i == 0:
            print(f"✅ First inference result: {result['output'].shape}")
            print(f"   - Latency: {result['latency_ms']:.2f}ms")
    
    # 지연시간 통계
    stats = engine.get_latency_stats()
    print(f"\n✅ Latency Statistics (100 inferences):")
    for key, value in stats.items():
        if key != 'count':
            print(f"   - {key}: {value:.2f}ms")
    
    print("\n🎉 ONNX Inference Engine test completed!")
