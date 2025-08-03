"""
Cross-Modal Attention Mechanisms

Implements attention-based fusion for multi-modal spiking neural networks
with temporal dynamics and neuromorphic-optimized computations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for multi-modal sensor fusion.
    
    Implements scaled dot-product attention with temporal integration
    and modality-specific projections for neuromorphic architectures.
    """
    
    def __init__(
        self,
        modalities: List[str],
        modality_dims: Dict[str, int],
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
        temporal_window: int = 10,
        enable_temporal_attention: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            modalities: List of modality names
            modality_dims: Feature dimensions for each modality
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout rate
            temperature: Attention temperature scaling
            temporal_window: Window size for temporal attention
            enable_temporal_attention: Enable temporal attention mechanism
            device: Computation device
        """
        super().__init__()
        
        self.modalities = modalities
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        self.temporal_window = temporal_window
        self.enable_temporal_attention = enable_temporal_attention
        self.device = device or torch.device('cpu')
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Modality-specific projections to common space
        self.modality_projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(modality_dims[modality], hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for modality in modalities
        })
        
        # Multi-head attention components
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Temporal attention (if enabled)
        if enable_temporal_attention:
            self.temporal_attention = TemporalAttention(
                hidden_dim=hidden_dim,
                num_heads=2,
                window_size=temporal_window,
                device=device
            )
        
        # Modality importance weighting
        self.modality_importance = nn.Parameter(
            torch.ones(len(modalities), device=self.device)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self.to(self.device)
    
    def forward(
        self,
        modality_outputs: Dict[str, torch.Tensor],
        available_modalities: Optional[List[str]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through cross-modal attention.
        
        Args:
            modality_outputs: Dictionary of modality features
                {modality: [batch_size, feature_dim]}
            available_modalities: List of available modalities
            attention_mask: Optional attention mask
            
        Returns:
            fused_output: Fused representation [batch_size, hidden_dim]
            attention_weights: Attention weights for each modality pair
        """
        if available_modalities is None:
            available_modalities = list(modality_outputs.keys())
        
        batch_size = next(iter(modality_outputs.values())).shape[0]
        
        # Project modalities to common space
        projected_modalities = {}
        for modality in available_modalities:
            if modality in modality_outputs:
                projected = self.modality_projections[modality](modality_outputs[modality])
                projected_modalities[modality] = projected
        
        if not projected_modalities:
            raise ValueError("No available modalities for attention")
        
        # Stack modalities for attention computation
        modality_features = []
        modality_names = []
        active_importance_weights = []
        
        for i, modality in enumerate(self.modalities):
            if modality in projected_modalities:
                modality_features.append(projected_modalities[modality])
                modality_names.append(modality)
                active_importance_weights.append(self.modality_importance[i])
        
        # Stack features: [batch_size, num_modalities, hidden_dim]
        features = torch.stack(modality_features, dim=1)
        
        # Apply modality importance weighting
        importance_weights = torch.stack(active_importance_weights)
        importance_weights = F.softmax(importance_weights / self.temperature, dim=0)
        features = features * importance_weights.view(1, -1, 1)
        
        # Multi-head self-attention across modalities
        fused_features, attention_weights = self._multi_head_attention(
            features, attention_mask
        )
        
        # Temporal attention (if enabled)
        if self.enable_temporal_attention and hasattr(self, 'temporal_attention'):
            # Add temporal dimension for temporal attention
            temporal_features = fused_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            temporal_output, _ = self.temporal_attention(temporal_features)
            fused_features = temporal_output.squeeze(1)
        
        # Layer normalization and residual connection
        if len(modality_features) == 1:
            # Skip residual if only one modality
            output = self.layer_norm(fused_features)
        else:
            # Residual connection with mean of input modalities
            residual = torch.mean(features, dim=1)
            output = self.layer_norm(fused_features + residual)
        
        # Final projection
        output = self.output_projection(output)
        
        # Create attention weights dictionary
        attention_dict = {
            'cross_modal_attention': attention_weights,
            'modality_importance': importance_weights.detach(),
            'available_modalities': modality_names,
        }
        
        return output, attention_dict
    
    def _multi_head_attention(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention computation.
        
        Args:
            features: Input features [batch_size, num_modalities, hidden_dim]
            attention_mask: Optional attention mask
            
        Returns:
            attended_features: Attended features [batch_size, hidden_dim]
            attention_weights: Attention weights [batch_size, num_heads, num_modalities, num_modalities]
        """
        batch_size, num_modalities, hidden_dim = features.shape
        
        # Linear projections for Q, K, V
        queries = self.query_projection(features)  # [batch_size, num_modalities, hidden_dim]
        keys = self.key_projection(features)
        values = self.value_projection(features)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, num_modalities, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_modalities, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_modalities, self.num_heads, self.head_dim)
        
        # Transpose for attention computation: [batch_size, num_heads, num_modalities, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply temperature scaling
        attention_scores = attention_scores / self.temperature
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(1) == 0, 
                float('-inf')
            )
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)
        
        # Concatenate heads: [batch_size, num_modalities, hidden_dim]
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, num_modalities, hidden_dim)
        
        # Pool across modalities (weighted average)
        modality_weights = attention_weights.mean(dim=1).mean(dim=-1)  # [batch_size, num_modalities]
        modality_weights = F.softmax(modality_weights, dim=-1)
        
        # Weighted sum across modalities
        fused_output = torch.sum(
            attended_values * modality_weights.unsqueeze(-1), 
            dim=1
        )  # [batch_size, hidden_dim]
        
        return fused_output, attention_weights
    
    def get_attention_statistics(self) -> Dict[str, Any]:
        """Get statistics about attention patterns."""
        with torch.no_grad():
            importance_weights = F.softmax(self.modality_importance, dim=0)
            
            return {
                'modality_importance': {
                    modality: importance_weights[i].item()
                    for i, modality in enumerate(self.modalities)
                },
                'temperature': self.temperature,
                'num_heads': self.num_heads,
                'hidden_dim': self.hidden_dim,
            }


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for processing sequences.
    
    Handles temporal dependencies in multi-modal fusion with
    sliding window attention for neuromorphic efficiency.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 2,
        window_size: int = 10,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize temporal attention.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            window_size: Temporal window size
            device: Computation device
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.device = device or torch.device('cpu')
        
        # Temporal position encoding
        self.position_encoding = PositionalEncoding(
            hidden_dim, max_length=window_size * 2
        )
        
        # Attention components
        self.temporal_query = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_key = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_value = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Memory buffer for temporal context
        self.register_buffer('memory_buffer', 
                           torch.zeros(window_size, hidden_dim))
        self.memory_index = 0
        
        self.to(self.device)
    
    def forward(
        self,
        x: torch.Tensor,
        update_memory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal attention.
        
        Args:
            x: Input features [batch_size, seq_len, hidden_dim]
            update_memory: Whether to update memory buffer
            
        Returns:
            output: Temporally attended features
            attention_weights: Temporal attention weights
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Add positional encoding
        x_pos = self.position_encoding(x)
        
        # Update memory buffer
        if update_memory and seq_len > 0:
            with torch.no_grad():
                # Add latest features to memory
                latest_features = x[:, -1, :].mean(dim=0)  # Average across batch
                self.memory_buffer[self.memory_index] = latest_features
                self.memory_index = (self.memory_index + 1) % self.window_size
        
        # Create temporal context by combining current input with memory
        memory_context = self.memory_buffer.unsqueeze(0).expand(batch_size, -1, -1)
        full_context = torch.cat([memory_context, x_pos], dim=1)  # [batch_size, window_size + seq_len, hidden_dim]
        
        # Temporal attention
        queries = self.temporal_query(x_pos)
        keys = self.temporal_key(full_context)
        values = self.temporal_value(full_context)
        
        # Multi-head attention computation
        attended_output, attention_weights = self._compute_attention(
            queries, keys, values
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(attended_output + x_pos)
        output = self.output_linear(output)
        
        return output, attention_weights
    
    def _compute_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head temporal attention."""
        batch_size, seq_len, _ = queries.shape
        context_len = keys.shape[1]
        
        # Reshape for multi-head attention
        Q = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = keys.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = values.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.hidden_dim)
        
        return attended, attention_weights
    
    def reset_memory(self):
        """Reset the temporal memory buffer."""
        self.memory_buffer.zero_()
        self.memory_index = 0


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequences.
    """
    
    def __init__(self, hidden_dim: int, max_length: int = 1000):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, hidden_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           -(math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention mechanism that adjusts attention patterns
    based on input characteristics and modality availability.
    """
    
    def __init__(
        self,
        modalities: List[str],
        hidden_dim: int = 128,
        adaptation_rate: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.modalities = modalities
        self.hidden_dim = hidden_dim
        self.adaptation_rate = adaptation_rate
        self.device = device or torch.device('cpu')
        
        # Adaptive temperature parameter
        self.adaptive_temperature = nn.Parameter(torch.ones(1, device=self.device))
        
        # Modality reliability estimation
        self.reliability_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Adaptation history
        self.register_buffer('adaptation_history', 
                           torch.zeros(100, len(modalities)))
        self.history_index = 0
        
        self.to(self.device)
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        base_attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adapt attention weights based on modality reliability.
        
        Args:
            modality_features: Dictionary of modality features
            base_attention_weights: Base attention weights
            
        Returns:
            adapted_weights: Adapted attention weights
        """
        # Estimate modality reliability
        reliabilities = []
        for modality in self.modalities:
            if modality in modality_features:
                reliability = self.reliability_estimator(modality_features[modality])
                reliabilities.append(reliability.mean())
            else:
                reliabilities.append(torch.tensor(0.0, device=self.device))
        
        reliability_scores = torch.stack(reliabilities)
        
        # Update adaptation history
        with torch.no_grad():
            self.adaptation_history[self.history_index] = reliability_scores
            self.history_index = (self.history_index + 1) % 100
        
        # Adaptive temperature based on reliability variance
        reliability_variance = reliability_scores.var()
        adaptive_temp = self.adaptive_temperature * (1.0 + reliability_variance)
        
        # Adapt attention weights
        adapted_weights = base_attention_weights * reliability_scores.unsqueeze(0)
        adapted_weights = F.softmax(adapted_weights / adaptive_temp, dim=-1)
        
        return adapted_weights
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        with torch.no_grad():
            history_mean = self.adaptation_history.mean(dim=0)
            history_std = self.adaptation_history.std(dim=0)
            
            return {
                'reliability_mean': {
                    modality: history_mean[i].item()
                    for i, modality in enumerate(self.modalities)
                },
                'reliability_std': {
                    modality: history_std[i].item()
                    for i, modality in enumerate(self.modalities)
                },
                'adaptive_temperature': self.adaptive_temperature.item(),
            }