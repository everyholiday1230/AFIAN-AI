"""
PyTorch to ONNX Converter

목적: 학습된 PyTorch 모델을 ONNX 형식으로 변환하여 추론 최적화

특징:
- TFT, Decision Transformer, Guardian 모델 변환
- 자동 입력 shape 추론
- ONNX 최적화 적용
- 검증 및 벤치마크

사용법:
    python scripts/convert_to_onnx.py \
        --model-dir data/models \
        --output-dir data/models/onnx \
        --verify
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Tuple
import logging
import time
import numpy as np

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from ai.models.tft.temporal_fusion_transformer import TemporalFusionTransformer
from ai.models.decision_transformer.decision_transformer import DecisionTransformer
from ai.models.regime_detection.contrastive_vae import ContrastiveVAE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ONNXConverter:
    """
    PyTorch → ONNX 변환기
    
    Args:
        model_dir: PyTorch 모델 디렉토리
        output_dir: ONNX 모델 출력 디렉토리
    """
    
    def __init__(
        self,
        model_dir: str = "data/models",
        output_dir: str = "data/models/onnx"
    ):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_tft(
        self,
        checkpoint_path: str,
        output_name: str = "tft_oracle.onnx",
        batch_size: int = 1,
        encoder_length: int = 60,
        num_features: int = 20
    ):
        """
        TFT 모델 ONNX 변환
        
        Args:
            checkpoint_path: PyTorch 체크포인트 경로
            output_name: ONNX 파일 이름
            batch_size: 배치 크기
            encoder_length: 인코더 길이
            num_features: 피처 수
        """
        logger.info(f"Converting TFT model from {checkpoint_path}")
        
        # 모델 로드
        model = TemporalFusionTransformer(
            input_size=num_features,
            hidden_size=160,
            num_attention_heads=4,
            dropout=0.1
        )
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # 더미 입력 생성
        dummy_input = torch.randn(batch_size, encoder_length, num_features)
        
        # ONNX 변환
        output_path = self.output_dir / output_name
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['encoder_input'],
            output_names=['predictions', 'attention_weights'],
            dynamic_axes={
                'encoder_input': {0: 'batch_size'},
                'predictions': {0: 'batch_size'},
                'attention_weights': {0: 'batch_size'}
            }
        )
        
        logger.info(f"TFT model converted to {output_path}")
        
        return output_path
    
    def convert_decision_transformer(
        self,
        checkpoint_path: str,
        output_name: str = "decision_transformer.onnx",
        batch_size: int = 1,
        context_length: int = 20,
        state_dim: int = 10,
        action_dim: int = 3
    ):
        """
        Decision Transformer 모델 ONNX 변환
        
        Args:
            checkpoint_path: PyTorch 체크포인트 경로
            output_name: ONNX 파일 이름
            batch_size: 배치 크기
            context_length: 컨텍스트 길이
            state_dim: 상태 차원
            action_dim: 액션 차원
        """
        logger.info(f"Converting Decision Transformer from {checkpoint_path}")
        
        # 모델 로드
        model = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=128,
            max_length=context_length
        )
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # 더미 입력 생성
        states = torch.randn(batch_size, context_length, state_dim)
        actions = torch.randn(batch_size, context_length, action_dim)
        rewards = torch.randn(batch_size, context_length, 1)
        timesteps = torch.arange(context_length).unsqueeze(0).repeat(batch_size, 1)
        
        # ONNX 변환
        output_path = self.output_dir / output_name
        
        torch.onnx.export(
            model,
            (states, actions, rewards, timesteps),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['states', 'actions', 'rewards', 'timesteps'],
            output_names=['action_preds'],
            dynamic_axes={
                'states': {0: 'batch_size'},
                'actions': {0: 'batch_size'},
                'rewards': {0: 'batch_size'},
                'timesteps': {0: 'batch_size'},
                'action_preds': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Decision Transformer converted to {output_path}")
        
        return output_path
    
    def convert_guardian(
        self,
        checkpoint_path: str,
        output_name: str = "regime_detector.onnx",
        batch_size: int = 1,
        input_dim: int = 60
    ):
        """
        Guardian (Contrastive VAE) 모델 ONNX 변환
        
        Args:
            checkpoint_path: PyTorch 체크포인트 경로
            output_name: ONNX 파일 이름
            batch_size: 배치 크기
            input_dim: 입력 차원
        """
        logger.info(f"Converting Guardian model from {checkpoint_path}")
        
        # 모델 로드
        model = ContrastiveVAE(
            input_dim=input_dim,
            latent_dim=32,
            hidden_dims=[128, 64]
        )
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # 더미 입력 생성
        dummy_input = torch.randn(batch_size, input_dim)
        
        # ONNX 변환
        output_path = self.output_dir / output_name
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['market_state'],
            output_names=['reconstruction', 'mu', 'logvar'],
            dynamic_axes={
                'market_state': {0: 'batch_size'},
                'reconstruction': {0: 'batch_size'},
                'mu': {0: 'batch_size'},
                'logvar': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Guardian model converted to {output_path}")
        
        return output_path
    
    def verify_onnx_model(self, onnx_path: Path) -> bool:
        """
        ONNX 모델 검증
        
        Args:
            onnx_path: ONNX 모델 경로
            
        Returns:
            검증 성공 여부
        """
        logger.info(f"Verifying ONNX model: {onnx_path}")
        
        try:
            # ONNX 모델 로드
            onnx_model = onnx.load(str(onnx_path))
            
            # 모델 검사
            onnx.checker.check_model(onnx_model)
            
            logger.info("✅ ONNX model is valid")
            
            # ONNX Runtime으로 추론 테스트
            session = ort.InferenceSession(str(onnx_path))
            
            # 입력/출력 정보
            logger.info("Input info:")
            for inp in session.get_inputs():
                logger.info(f"  - {inp.name}: {inp.shape} ({inp.type})")
            
            logger.info("Output info:")
            for out in session.get_outputs():
                logger.info(f"  - {out.name}: {out.shape} ({out.type})")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ ONNX model verification failed: {e}")
            return False
    
    def benchmark_model(
        self,
        onnx_path: Path,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        ONNX 모델 벤치마크
        
        Args:
            onnx_path: ONNX 모델 경로
            num_runs: 실행 횟수
            
        Returns:
            벤치마크 결과
        """
        logger.info(f"Benchmarking ONNX model: {onnx_path}")
        
        # ONNX Runtime 세션 생성
        session = ort.InferenceSession(str(onnx_path))
        
        # 더미 입력 생성
        inputs = {}
        for inp in session.get_inputs():
            shape = [dim if isinstance(dim, int) else 1 for dim in inp.shape]
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
        
        # Warm-up
        for _ in range(10):
            session.run(None, inputs)
        
        # 벤치마크
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
        
        latencies = np.array(latencies)
        
        results = {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies))
        }
        
        logger.info("Benchmark results:")
        for key, value in results.items():
            logger.info(f"  - {key}: {value:.2f}")
        
        return results
    
    def convert_all_models(self, verify: bool = True, benchmark: bool = True):
        """모든 모델 변환"""
        logger.info("Converting all PyTorch models to ONNX...")
        
        converted_models = []
        
        # TFT 변환
        tft_checkpoint = self.model_dir / "oracle" / "tft_best.pt"
        if tft_checkpoint.exists():
            try:
                onnx_path = self.convert_tft(str(tft_checkpoint))
                converted_models.append(('TFT', onnx_path))
            except Exception as e:
                logger.error(f"Failed to convert TFT: {e}")
        else:
            logger.warning(f"TFT checkpoint not found: {tft_checkpoint}")
        
        # Decision Transformer 변환
        dt_checkpoint = self.model_dir / "strategist" / "decision_transformer_best.pt"
        if dt_checkpoint.exists():
            try:
                onnx_path = self.convert_decision_transformer(str(dt_checkpoint))
                converted_models.append(('Decision Transformer', onnx_path))
            except Exception as e:
                logger.error(f"Failed to convert Decision Transformer: {e}")
        else:
            logger.warning(f"Decision Transformer checkpoint not found: {dt_checkpoint}")
        
        # Guardian 변환
        guardian_checkpoint = self.model_dir / "guardian" / "regime_detector_best.pt"
        if guardian_checkpoint.exists():
            try:
                onnx_path = self.convert_guardian(str(guardian_checkpoint))
                converted_models.append(('Guardian', onnx_path))
            except Exception as e:
                logger.error(f"Failed to convert Guardian: {e}")
        else:
            logger.warning(f"Guardian checkpoint not found: {guardian_checkpoint}")
        
        # 검증 및 벤치마크
        if verify or benchmark:
            for model_name, onnx_path in converted_models:
                logger.info(f"\n{'='*60}")
                logger.info(f"Testing {model_name}")
                logger.info(f"{'='*60}")
                
                if verify:
                    is_valid = self.verify_onnx_model(onnx_path)
                    if not is_valid:
                        continue
                
                if benchmark:
                    self.benchmark_model(onnx_path)
        
        logger.info(f"\n✅ Converted {len(converted_models)} models to ONNX")


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch models to ONNX')
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='data/models',
        help='PyTorch models directory (default: data/models)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/models/onnx',
        help='ONNX output directory (default: data/models/onnx)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify ONNX models after conversion'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark ONNX models'
    )
    
    args = parser.parse_args()
    
    # 변환기 생성
    converter = ONNXConverter(
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    
    # 변환 시작
    logger.info("Starting ONNX conversion...")
    
    converter.convert_all_models(
        verify=args.verify,
        benchmark=args.benchmark
    )


if __name__ == "__main__":
    main()
