"""
Contrastive VAE for Market Regime Detection

대조 학습 기반 VAE로 시장 국면을 비지도 학습으로 발견

Reference:
- "Contrastive Learning for Unsupervised Clustering" (Li et al., 2021)
- "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (Higgins et al., 2017)
- "Understanding disentangling in β-VAE" (Burgess et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class ContrastiveVAE(nn.Module):
    """
    대조 학습 기반 VAE
    
    목적: 시장 국면을 비지도 학습으로 발견
    
    구조:
    1. Encoder: market features -> latent distribution (μ, σ)
    2. Decoder: latent vector -> reconstructed features
    3. Contrastive projection head: latent -> contrastive space
    
    Args:
        input_dim: 입력 피처 차원
        latent_dim: 잠재 공간 차원
        hidden_dims: 인코더/디코더 히든 레이어 차원들
        beta: KL divergence 가중치 (β-VAE)
        contrastive_dim: 대조 학습 프로젝션 차원
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: list = None,
        beta: float = 1.0,
        contrastive_dim: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.contrastive_dim = contrastive_dim
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Contrastive projection head
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, contrastive_dim)
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters
        
        Args:
            x: (batch, input_dim)
            
        Returns:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick
        
        z = μ + σ * ε, where ε ~ N(0, I)
        
        Args:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
            
        Returns:
            z: (batch, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction
        
        Args:
            z: (batch, latent_dim)
            
        Returns:
            x_recon: (batch, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: (batch, input_dim)
            
        Returns:
            Dictionary with reconstruction, latent params, and projections
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z)
        
        # Project for contrastive learning
        z_proj = self.projection_head(z)
        
        return {
            'reconstruction': x_recon,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'z_projected': z_proj
        }
    
    def loss_function(
        self,
        x: torch.Tensor,
        output: Dict[str, torch.Tensor],
        contrastive_pairs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        temperature: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE + Contrastive loss
        
        Loss = Reconstruction + β * KL + λ * Contrastive
        
        Args:
            x: (batch, input_dim) original input
            output: forward pass output
            contrastive_pairs: (positive_indices, negative_indices) for contrastive learning
            temperature: temperature for contrastive loss
            
        Returns:
            Dictionary with loss components
        """
        x_recon = output['reconstruction']
        mu = output['mu']
        logvar = output['logvar']
        z_proj = output['z_projected']
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        # KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_loss.mean()
        
        # Total VAE loss
        vae_loss = recon_loss + self.beta * kl_loss
        
        # Contrastive loss (optional)
        contrastive_loss = torch.tensor(0.0, device=x.device)
        if contrastive_pairs is not None:
            pos_pairs, neg_pairs = contrastive_pairs
            contrastive_loss = self._contrastive_loss(
                z_proj, pos_pairs, neg_pairs, temperature
            )
        
        # Total loss
        total_loss = vae_loss + 0.1 * contrastive_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'contrastive_loss': contrastive_loss
        }
    
    def _contrastive_loss(
        self,
        z_proj: torch.Tensor,
        pos_pairs: torch.Tensor,
        neg_pairs: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
        
        Args:
            z_proj: (batch, contrastive_dim) projected embeddings
            pos_pairs: (num_pairs, 2) positive pair indices
            neg_pairs: (num_pairs, 2) negative pair indices
            temperature: temperature parameter
            
        Returns:
            contrastive_loss: scalar
        """
        # Normalize embeddings
        z_norm = F.normalize(z_proj, dim=-1)
        
        # Compute similarities
        pos_sim = torch.sum(
            z_norm[pos_pairs[:, 0]] * z_norm[pos_pairs[:, 1]], 
            dim=-1
        ) / temperature
        
        neg_sim = torch.sum(
            z_norm[neg_pairs[:, 0]] * z_norm[neg_pairs[:, 1]], 
            dim=-1
        ) / temperature
        
        # InfoNCE loss
        logits = torch.cat([pos_sim, neg_sim])
        labels = torch.zeros(len(logits), device=z_proj.device)
        labels[:len(pos_sim)] = 1
        
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return loss


class MarketRegimeDetector:
    """
    시장 국면 감지 시스템
    
    VAE의 잠재 공간에서 K-means 클러스터링을 수행하여
    시장 국면을 자동으로 발견하고 분류
    
    Args:
        vae_model: 학습된 ContrastiveVAE 모델
        n_regimes: 국면 개수 (K-means 클러스터 수)
        regime_names: 각 국면의 이름 (선택적)
    """
    
    def __init__(
        self,
        vae_model: ContrastiveVAE,
        n_regimes: int = 4,
        regime_names: Optional[Dict[int, str]] = None
    ):
        self.vae = vae_model
        self.n_regimes = n_regimes
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        self.is_fitted = False
        
        if regime_names is None:
            self.regime_names = {
                0: 'bull_trend',
                1: 'bear_trend',
                2: 'sideways',
                3: 'high_volatility'
            }
        else:
            self.regime_names = regime_names
        
        # 통계
        self.regime_statistics = {}
    
    def fit(self, market_data: torch.Tensor):
        """
        VAE 잠재 공간에서 클러스터링 학습
        
        Args:
            market_data: (num_samples, input_dim) 시장 데이터
        """
        self.vae.eval()
        
        with torch.no_grad():
            # Extract latent vectors
            latent_vectors = []
            batch_size = 256
            
            for i in range(0, len(market_data), batch_size):
                batch = market_data[i:i+batch_size]
                output = self.vae(batch)
                latent_vectors.append(output['mu'].cpu().numpy())
            
            all_latents = np.concatenate(latent_vectors, axis=0)
        
        # K-means clustering
        self.kmeans.fit(all_latents)
        self.is_fitted = True
        
        # Calculate regime statistics
        self._calculate_regime_statistics(market_data, all_latents)
    
    def predict_regime(
        self, 
        market_features: torch.Tensor
    ) -> Dict[str, any]:
        """
        현재 시장 국면 예측
        
        Args:
            market_features: (1, input_dim) or (input_dim,)
            
        Returns:
            Dictionary with regime info
        """
        if not self.is_fitted:
            return {
                'regime': 'unknown',
                'regime_id': -1,
                'confidence': 0.0,
                'probabilities': {}
            }
        
        self.vae.eval()
        
        # Ensure batch dimension
        if market_features.dim() == 1:
            market_features = market_features.unsqueeze(0)
        
        with torch.no_grad():
            output = self.vae(market_features)
            latent = output['mu'].cpu().numpy()
        
        # Predict regime
        regime_id = self.kmeans.predict(latent)[0]
        
        # Calculate confidence (based on distance to cluster centers)
        distances = cdist(latent, self.kmeans.cluster_centers_, metric='euclidean')[0]
        
        # Softmax over negative distances for probabilities
        probs = np.exp(-distances) / np.sum(np.exp(-distances))
        confidence = probs[regime_id]
        
        # Probabilities for all regimes
        regime_probs = {
            self.regime_names.get(i, f'regime_{i}'): float(probs[i])
            for i in range(self.n_regimes)
        }
        
        return {
            'regime': self.regime_names.get(regime_id, f'regime_{regime_id}'),
            'regime_id': int(regime_id),
            'confidence': float(confidence),
            'probabilities': regime_probs,
            'latent_vector': latent[0].tolist(),
            'distances_to_centers': distances.tolist()
        }
    
    def _calculate_regime_statistics(
        self, 
        market_data: torch.Tensor, 
        latent_vectors: np.ndarray
    ):
        """각 국면의 통계 계산"""
        labels = self.kmeans.labels_
        
        for regime_id in range(self.n_regimes):
            regime_mask = labels == regime_id
            regime_count = np.sum(regime_mask)
            
            if regime_count > 0:
                self.regime_statistics[regime_id] = {
                    'name': self.regime_names.get(regime_id, f'regime_{regime_id}'),
                    'count': int(regime_count),
                    'frequency': float(regime_count / len(labels)),
                    'center': self.kmeans.cluster_centers_[regime_id].tolist()
                }
    
    def get_regime_transitions(
        self, 
        regime_sequence: list
    ) -> Dict[Tuple[str, str], int]:
        """
        국면 전환 분석
        
        Args:
            regime_sequence: 시간순 국면 리스트
            
        Returns:
            전환 빈도 딕셔너리
        """
        transitions = {}
        
        for i in range(len(regime_sequence) - 1):
            curr_regime = regime_sequence[i]
            next_regime = regime_sequence[i + 1]
            transition = (curr_regime, next_regime)
            
            if transition in transitions:
                transitions[transition] += 1
            else:
                transitions[transition] = 1
        
        return transitions
    
    def visualize_latent_space(
        self, 
        market_data: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """잠재 공간 시각화 (2D PCA)"""
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        
        self.vae.eval()
        
        with torch.no_grad():
            output = self.vae(market_data)
            latent_vectors = output['mu'].cpu().numpy()
        
        # PCA for visualization
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_vectors)
        
        # Get labels
        labels = self.kmeans.predict(latent_vectors)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        for regime_id in range(self.n_regimes):
            mask = labels == regime_id
            plt.scatter(
                latent_2d[mask, 0], 
                latent_2d[mask, 1],
                label=self.regime_names.get(regime_id, f'Regime {regime_id}'),
                alpha=0.6,
                s=20
            )
        
        # Plot cluster centers
        centers_2d = pca.transform(self.kmeans.cluster_centers_)
        plt.scatter(
            centers_2d[:, 0], 
            centers_2d[:, 1],
            marker='X',
            s=200,
            c='black',
            edgecolors='white',
            linewidths=2,
            label='Centers'
        )
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Market Regime Latent Space (2D PCA)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Latent space visualization saved to {save_path}")
        
        return plt.gcf()


if __name__ == "__main__":
    print("🧪 Testing Contrastive VAE & Market Regime Detection...")
    
    # Generate synthetic market data
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 1000
    input_dim = 50  # Market features
    
    # Simulate 4 different market regimes
    regime1 = torch.randn(250, input_dim) + torch.tensor([2.0] * input_dim)  # Bull
    regime2 = torch.randn(250, input_dim) + torch.tensor([-2.0] * input_dim)  # Bear
    regime3 = torch.randn(250, input_dim) * 0.5  # Low volatility sideways
    regime4 = torch.randn(250, input_dim) * 3.0  # High volatility
    
    market_data = torch.cat([regime1, regime2, regime3, regime4], dim=0)
    
    # Shuffle
    perm = torch.randperm(n_samples)
    market_data = market_data[perm]
    
    print(f"\n✅ Synthetic market data: {market_data.shape}")
    
    # Create VAE model
    vae = ContrastiveVAE(
        input_dim=input_dim,
        latent_dim=32,
        hidden_dims=[256, 128, 64],
        beta=1.0,
        contrastive_dim=16
    )
    
    print(f"\n✅ VAE Model created")
    print(f"   Input dim: {input_dim}")
    print(f"   Latent dim: 32")
    print(f"   Parameters: {sum(p.numel() for p in vae.parameters()):,}")
    
    # Test forward pass
    batch = market_data[:32]
    output = vae(batch)
    
    print(f"\n✅ Forward pass:")
    print(f"   Reconstruction: {output['reconstruction'].shape}")
    print(f"   Latent (z): {output['z'].shape}")
    print(f"   Projected (z_proj): {output['z_projected'].shape}")
    
    # Test loss
    loss_dict = vae.loss_function(batch, output)
    print(f"\n✅ Loss computation:")
    print(f"   Total loss: {loss_dict['loss'].item():.4f}")
    print(f"   Recon loss: {loss_dict['recon_loss'].item():.4f}")
    print(f"   KL loss: {loss_dict['kl_loss'].item():.4f}")
    
    # Simulate training (mini version)
    print(f"\n✅ Simulating training...")
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    vae.train()
    for epoch in range(5):
        total_loss = 0
        for i in range(0, len(market_data), 64):
            batch = market_data[i:i+64]
            
            optimizer.zero_grad()
            output = vae(batch)
            loss_dict = vae.loss_function(batch, output)
            loss_dict['loss'].backward()
            optimizer.step()
            
            total_loss += loss_dict['loss'].item()
        
        avg_loss = total_loss / (len(market_data) / 64)
        print(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Market Regime Detection
    print(f"\n✅ Testing Market Regime Detector...")
    
    detector = MarketRegimeDetector(
        vae_model=vae,
        n_regimes=4,
        regime_names={
            0: 'bull_trend',
            1: 'bear_trend',
            2: 'sideways',
            3: 'high_volatility'
        }
    )
    
    # Fit detector
    detector.fit(market_data)
    print(f"   ✅ Detector fitted with {detector.n_regimes} regimes")
    
    # Print regime statistics
    print(f"\n✅ Regime Statistics:")
    for regime_id, stats in detector.regime_statistics.items():
        print(f"   {stats['name']}: {stats['count']} samples ({stats['frequency']:.2%})")
    
    # Predict regime for new data
    test_sample = market_data[0]
    prediction = detector.predict_regime(test_sample)
    
    print(f"\n✅ Regime Prediction:")
    print(f"   Predicted regime: {prediction['regime']}")
    print(f"   Confidence: {prediction['confidence']:.4f}")
    print(f"   Probabilities:")
    for regime, prob in prediction['probabilities'].items():
        print(f"      {regime}: {prob:.4f}")
    
    # Test regime transitions
    regime_sequence = []
    for i in range(0, min(100, len(market_data)), 10):
        sample = market_data[i]
        pred = detector.predict_regime(sample)
        regime_sequence.append(pred['regime'])
    
    transitions = detector.get_regime_transitions(regime_sequence)
    print(f"\n✅ Regime Transitions (sample):")
    for (from_regime, to_regime), count in sorted(transitions.items(), key=lambda x: -x[1])[:5]:
        print(f"   {from_regime} -> {to_regime}: {count} times")
    
    # Visualization
    print(f"\n✅ Generating latent space visualization...")
    try:
        fig = detector.visualize_latent_space(
            market_data,
            save_path='/home/user/webapp/docs/regime_detection_latent_space.png'
        )
    except Exception as e:
        print(f"   ⚠️  Visualization skipped: {e}")
    
    print("\n🎉 Contrastive VAE & Regime Detection test completed!")
