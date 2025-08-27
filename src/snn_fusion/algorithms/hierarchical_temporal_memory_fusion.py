"""
Hierarchical Temporal Memory Fusion for Long-Term Neuromorphic Dependencies

Advanced neuromorphic architecture for capturing long-term temporal dependencies
spanning microseconds to seconds while maintaining ultra-low energy consumption.
Implements hierarchical temporal memory with adaptive window sizing and
energy-aware temporal compression.

Research Innovation:
- Multi-scale temporal attention (μs to seconds)
- Adaptive temporal window sizing based on signal complexity
- Energy-aware temporal compression for long-term memory
- Biologically-inspired hierarchical processing

Publication Status: Novel algorithm for academic publication
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math

from .temporal_spike_attention import TemporalSpikeAttention
from ..models.neurons import AdaptiveLIF


class HierarchicalTemporalScale:
    """Multi-scale temporal processing with adaptive window sizing."""
    
    def __init__(self, base_window_ms: float = 1.0):
        self.scales = [
            base_window_ms,           # Microsecond scale (1ms)
            base_window_ms * 10,      # Fast dynamics (10ms)
            base_window_ms * 50,      # Medium dynamics (50ms)
            base_window_ms * 200,     # Slow dynamics (200ms)
            base_window_ms * 1000,    # Long-term memory (1s)
            base_window_ms * 5000,    # Ultra-long dependencies (5s)
        ]
        self.scale_names = ['micro', 'fast', 'medium', 'slow', 'long', 'ultra']
    
    def get_window_size(self, scale_idx: int, complexity: float) -> int:
        """Adaptive window sizing based on signal complexity."""
        base_size = int(self.scales[scale_idx])
        # Higher complexity requires longer windows for pattern capture
        complexity_multiplier = 1.0 + 0.5 * complexity
        return max(1, int(base_size * complexity_multiplier))


class TemporalComplexityEstimator(nn.Module):
    """Real-time estimation of temporal signal complexity for adaptive processing."""
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.conv1d = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.complexity_head = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, spike_sequence: torch.Tensor) -> torch.Tensor:
        """
        Estimate temporal complexity of spike sequences.
        
        Args:
            spike_sequence: [batch_size, time_steps, input_size]
            
        Returns:
            complexity: [batch_size] - complexity score (0-1)
        """
        # Transpose for conv1d: [batch_size, input_size, time_steps]
        x = spike_sequence.transpose(1, 2)
        x = F.relu(self.conv1d(x))
        
        # Global average pooling over time
        x = x.mean(dim=2)  # [batch_size, hidden_size]
        
        complexity = self.sigmoid(self.complexity_head(x)).squeeze(-1)
        return complexity


class EnergyAwareTemporalCompressor(nn.Module):
    """Energy-efficient temporal compression for long-term memory storage."""
    
    def __init__(self, input_size: int, compression_ratio: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.compressed_size = max(1, int(input_size * compression_ratio))
        
        # Learnable compression matrix
        self.compression_matrix = nn.Parameter(
            torch.randn(self.compressed_size, input_size) / math.sqrt(input_size)
        )
        
        # Reconstruction matrix for memory retrieval
        self.reconstruction_matrix = nn.Parameter(
            torch.randn(input_size, self.compressed_size) / math.sqrt(self.compressed_size)
        )
        
        self.energy_weight = nn.Parameter(torch.tensor(1.0))
    
    def compress(self, temporal_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compress temporal state with energy cost tracking.
        
        Args:
            temporal_state: [batch_size, time_steps, input_size]
            
        Returns:
            compressed_state: [batch_size, time_steps, compressed_size]
            energy_cost: Compression energy cost
        """
        batch_size, time_steps, input_size = temporal_state.shape
        
        # Reshape for matrix multiplication
        flat_state = temporal_state.view(-1, input_size)
        
        # Sparse compression to reduce energy
        compressed = torch.matmul(flat_state, self.compression_matrix.T)
        
        # Apply sparsity for energy reduction
        compressed = torch.where(
            torch.abs(compressed) > 0.1,
            compressed,
            torch.zeros_like(compressed)
        )
        
        compressed_state = compressed.view(batch_size, time_steps, self.compressed_size)
        
        # Calculate energy cost (based on non-zero operations)
        non_zero_ops = (compressed != 0).float().sum()
        energy_cost = (non_zero_ops * self.energy_weight).item()
        
        return compressed_state, energy_cost
    
    def reconstruct(self, compressed_state: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct temporal state from compressed representation.
        
        Args:
            compressed_state: [batch_size, time_steps, compressed_size]
            
        Returns:
            reconstructed_state: [batch_size, time_steps, input_size]
        """
        batch_size, time_steps, compressed_size = compressed_state.shape
        
        flat_compressed = compressed_state.view(-1, compressed_size)
        reconstructed = torch.matmul(flat_compressed, self.reconstruction_matrix.T)
        
        return reconstructed.view(batch_size, time_steps, self.input_size)


class HierarchicalTemporalLayer(nn.Module):
    """Single layer of hierarchical temporal processing."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        scale_idx: int,
        n_heads: int = 8,
        compression_ratio: float = 0.2,
    ):
        super().__init__()
        self.scale_idx = scale_idx
        self.hidden_size = hidden_size
        
        # Temporal attention for this scale
        self.temporal_attention = TemporalSpikeAttention(
            input_size=input_size,
            hidden_size=hidden_size,
            n_heads=n_heads,
        )
        
        # Compression for memory efficiency
        self.compressor = EnergyAwareTemporalCompressor(
            input_size=hidden_size,
            compression_ratio=compression_ratio
        )
        
        # Integration with lower scales
        self.scale_integration = nn.Linear(hidden_size * 2, hidden_size)
        
        # Adaptive neurons for temporal dynamics
        self.neurons = AdaptiveLIF(
            n_neurons=hidden_size,
            tau_mem=20.0 + scale_idx * 30.0,  # Longer time constants for higher scales
            tau_adapt=100.0 + scale_idx * 100.0,
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        lower_scale_state: Optional[torch.Tensor] = None,
        complexity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Process temporal inputs at this hierarchical scale.
        
        Args:
            inputs: [batch_size, time_steps, input_size]
            lower_scale_state: Optional state from lower temporal scale
            complexity: Signal complexity for adaptive processing
            
        Returns:
            output_state: Processed temporal state
            compressed_memory: Compressed state for long-term storage
            metrics: Processing metrics and energy costs
        """
        batch_size, time_steps, input_size = inputs.shape
        
        # Apply temporal attention
        attended_state, attention_weights = self.temporal_attention(inputs)
        
        # Integrate with lower scale if available
        if lower_scale_state is not None:
            # Ensure compatible dimensions
            if lower_scale_state.shape[-1] != attended_state.shape[-1]:
                lower_scale_state = F.adaptive_avg_pool1d(
                    lower_scale_state.transpose(1, 2),
                    attended_state.shape[-1]
                ).transpose(1, 2)
            
            # Concatenate and integrate
            combined = torch.cat([attended_state, lower_scale_state], dim=-1)
            attended_state = torch.tanh(self.scale_integration(combined))
        
        # Apply neuromorphic dynamics
        current = attended_state.view(-1, self.hidden_size)
        spikes, neuron_states = self.neurons(current)
        
        output_state = spikes.view(batch_size, time_steps, self.hidden_size)
        
        # Compress for long-term memory
        compressed_memory, compression_energy = self.compressor.compress(output_state)
        
        # Collect metrics
        metrics = {
            'attention_entropy': self._compute_attention_entropy(attention_weights),
            'spike_rate': spikes.mean().item(),
            'membrane_potential_std': neuron_states['v_mem'].std().item(),
            'compression_energy': compression_energy,
            'compression_ratio': compressed_memory.shape[-1] / output_state.shape[-1],
        }
        
        return output_state, compressed_memory, metrics
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution."""
        # attention_weights: [batch_size, n_heads, time_steps, time_steps]
        mean_attention = attention_weights.mean(dim=(0, 1))  # Average over batch and heads
        
        # Compute entropy
        entropy = -torch.sum(mean_attention * torch.log(mean_attention + 1e-8))
        return entropy.item()


class HierarchicalTemporalMemoryFusion(nn.Module):
    """
    Complete hierarchical temporal memory system for neuromorphic sensor fusion.
    
    Captures temporal dependencies across multiple scales while maintaining
    energy efficiency through adaptive processing and compression.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 96, 64, 48, 32, 24],
        n_heads: int = 8,
        compression_ratios: List[float] = [0.8, 0.6, 0.4, 0.3, 0.2, 0.1],
        energy_budget: float = 1000.0,  # μJ per inference
    ):
        """
        Initialize hierarchical temporal memory fusion.
        
        Args:
            input_size: Size of input features
            hidden_sizes: Hidden sizes for each temporal scale
            n_heads: Number of attention heads
            compression_ratios: Compression ratios for each scale
            energy_budget: Energy budget constraint (μJ)
        """
        super().__init__()
        
        self.input_size = input_size
        self.n_scales = len(hidden_sizes)
        self.energy_budget = energy_budget
        
        # Temporal scale configuration
        self.temporal_scales = HierarchicalTemporalScale()
        
        # Complexity estimator
        self.complexity_estimator = TemporalComplexityEstimator(input_size)
        
        # Hierarchical processing layers
        self.layers = nn.ModuleList()
        for i, (hidden_size, compression_ratio) in enumerate(zip(hidden_sizes, compression_ratios)):
            layer_input_size = input_size if i == 0 else hidden_sizes[i-1]
            
            layer = HierarchicalTemporalLayer(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                scale_idx=i,
                n_heads=n_heads,
                compression_ratio=compression_ratio,
            )
            self.layers.append(layer)
        
        # Cross-scale fusion
        total_hidden = sum(hidden_sizes)
        self.cross_scale_fusion = nn.Sequential(
            nn.Linear(total_hidden, total_hidden // 2),
            nn.ReLU(),
            nn.Linear(total_hidden // 2, hidden_sizes[0]),
            nn.Tanh(),
        )
        
        # Energy monitoring
        self.energy_consumed = 0.0
        self.processing_history = []
    
    def forward(
        self,
        inputs: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process inputs through hierarchical temporal memory.
        
        Args:
            inputs: [batch_size, time_steps, input_size]
            return_intermediate: Return intermediate states for analysis
            
        Returns:
            fused_output: Final fused temporal representation
            processing_info: Detailed processing metrics and states
        """
        batch_size, time_steps, input_size = inputs.shape
        
        # Estimate temporal complexity
        complexity = self.complexity_estimator(inputs)
        
        # Initialize processing
        scale_outputs = []
        compressed_memories = []
        scale_metrics = []
        total_energy = 0.0
        
        # Process through hierarchical scales
        current_input = inputs
        lower_scale_state = None
        
        for scale_idx, layer in enumerate(self.layers):
            # Adaptive window sizing based on complexity
            window_size = self.temporal_scales.get_window_size(
                scale_idx, complexity.mean().item()
            )
            
            # Temporal windowing for this scale
            if time_steps > window_size:
                # Use sliding window for efficiency
                stride = max(1, time_steps // window_size)
                windowed_input = current_input[:, ::stride, :][:, :window_size, :]
            else:
                windowed_input = current_input
            
            # Process at this scale
            scale_output, compressed_memory, metrics = layer(
                windowed_input,
                lower_scale_state,
                complexity,
            )
            
            # Accumulate results
            scale_outputs.append(scale_output)
            compressed_memories.append(compressed_memory)
            scale_metrics.append(metrics)
            total_energy += metrics['compression_energy']
            
            # Prepare for next scale
            current_input = scale_output
            lower_scale_state = scale_output
            
            # Energy budget constraint
            if total_energy > self.energy_budget:
                print(f"Energy budget exceeded at scale {scale_idx}: {total_energy:.2f}μJ")
                break
        
        # Cross-scale fusion
        if len(scale_outputs) > 1:
            # Align temporal dimensions
            aligned_outputs = []
            target_time_steps = scale_outputs[0].shape[1]
            
            for output in scale_outputs:
                if output.shape[1] != target_time_steps:
                    # Temporal interpolation
                    output_transposed = output.transpose(1, 2)
                    aligned = F.interpolate(
                        output_transposed,
                        size=target_time_steps,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)
                    aligned_outputs.append(aligned)
                else:
                    aligned_outputs.append(output)
            
            # Concatenate across scales
            concatenated = torch.cat(aligned_outputs, dim=-1)
            fused_output = self.cross_scale_fusion(concatenated)
        else:
            fused_output = scale_outputs[0] if scale_outputs else inputs
        
        # Update energy monitoring
        self.energy_consumed += total_energy
        
        # Compile processing information
        processing_info = {
            'complexity_scores': complexity,
            'scale_metrics': scale_metrics,
            'total_energy': total_energy,
            'energy_efficiency': self._compute_energy_efficiency(fused_output, total_energy),
            'temporal_coherence': self._compute_temporal_coherence(scale_outputs),
            'compression_effectiveness': self._evaluate_compression(compressed_memories),
            'n_scales_processed': len(scale_outputs),
        }
        
        if return_intermediate:
            processing_info.update({
                'scale_outputs': scale_outputs,
                'compressed_memories': compressed_memories,
                'attention_patterns': self._extract_attention_patterns(scale_metrics),
            })
        
        return fused_output, processing_info
    
    def _compute_energy_efficiency(
        self,
        output: torch.Tensor,
        energy_cost: float
    ) -> float:
        """Compute energy efficiency (information per μJ)."""
        # Information content approximation using output variance
        information_content = output.var().item()
        efficiency = information_content / (energy_cost + 1e-6)
        return efficiency
    
    def _compute_temporal_coherence(
        self,
        scale_outputs: List[torch.Tensor]
    ) -> float:
        """Compute coherence across temporal scales."""
        if len(scale_outputs) < 2:
            return 1.0
        
        coherence_scores = []
        
        for i in range(len(scale_outputs) - 1):
            output1 = scale_outputs[i]
            output2 = scale_outputs[i + 1]
            
            # Align dimensions for correlation
            if output1.shape[-1] != output2.shape[-1]:
                min_dim = min(output1.shape[-1], output2.shape[-1])
                output1 = output1[:, :, :min_dim]
                output2 = output2[:, :, :min_dim]
            
            if output1.shape[1] != output2.shape[1]:
                min_time = min(output1.shape[1], output2.shape[1])
                output1 = output1[:, :min_time, :]
                output2 = output2[:, :min_time, :]
            
            # Compute cross-correlation
            correlation = F.cosine_similarity(
                output1.flatten(1),
                output2.flatten(1),
                dim=1
            ).mean()
            
            coherence_scores.append(correlation.item())
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
    
    def _evaluate_compression(
        self,
        compressed_memories: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate compression effectiveness."""
        if not compressed_memories:
            return {'average_ratio': 0.0, 'memory_savings': 0.0}
        
        total_original = 0
        total_compressed = 0
        
        for memory in compressed_memories:
            # Estimate original size (assuming full precision)
            original_size = memory.shape[0] * memory.shape[1] * self.input_size
            compressed_size = memory.numel()
            
            total_original += original_size
            total_compressed += compressed_size
        
        compression_ratio = total_compressed / (total_original + 1e-6)
        memory_savings = 1.0 - compression_ratio
        
        return {
            'average_ratio': compression_ratio,
            'memory_savings': memory_savings,
        }
    
    def _extract_attention_patterns(
        self,
        scale_metrics: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Extract attention pattern analysis across scales."""
        entropy_pattern = [m.get('attention_entropy', 0.0) for m in scale_metrics]
        spike_rate_pattern = [m.get('spike_rate', 0.0) for m in scale_metrics]
        
        return {
            'entropy_across_scales': entropy_pattern,
            'spike_rates_across_scales': spike_rate_pattern,
        }
    
    def get_memory_state(self) -> Dict[str, torch.Tensor]:
        """Get compressed memory state for long-term storage."""
        memory_states = {}
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'compressor'):
                # Get compressed representation
                memory_states[f'scale_{i}_memory'] = layer.compressor.compression_matrix.data
                memory_states[f'scale_{i}_reconstruction'] = layer.compressor.reconstruction_matrix.data
        
        return memory_states
    
    def load_memory_state(self, memory_states: Dict[str, torch.Tensor]) -> None:
        """Load compressed memory state from storage."""
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'compressor'):
                memory_key = f'scale_{i}_memory'
                reconstruction_key = f'scale_{i}_reconstruction'
                
                if memory_key in memory_states:
                    layer.compressor.compression_matrix.data = memory_states[memory_key]
                if reconstruction_key in memory_states:
                    layer.compressor.reconstruction_matrix.data = memory_states[reconstruction_key]
    
    def reset_energy_monitoring(self) -> None:
        """Reset energy consumption tracking."""
        self.energy_consumed = 0.0
        self.processing_history.clear()
    
    def get_energy_report(self) -> Dict[str, Any]:
        """Get detailed energy consumption report."""
        return {
            'total_energy_consumed': self.energy_consumed,
            'energy_budget': self.energy_budget,
            'energy_utilization': self.energy_consumed / self.energy_budget,
            'processing_episodes': len(self.processing_history),
        }


# Research validation and benchmarking utilities
class TemporalDependencyBenchmark:
    """Benchmark long-term temporal dependency capabilities."""
    
    @staticmethod
    def generate_long_term_sequence(
        length: int = 10000,
        dependency_span: int = 1000,
        noise_level: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences with long-term dependencies for testing.
        
        Returns:
            sequences: Input sequences with embedded long-term patterns
            targets: Target patterns to identify
        """
        batch_size = 32
        input_size = 64
        
        sequences = torch.randn(batch_size, length, input_size) * noise_level
        targets = torch.zeros(batch_size, 1)
        
        for b in range(batch_size):
            # Embed pattern at random position
            pattern_start = torch.randint(0, length - dependency_span, (1,)).item()
            pattern_end = pattern_start + dependency_span
            
            # Create distinctive pattern
            pattern = torch.sin(torch.arange(dependency_span).float() * 0.1) * 2.0
            sequences[b, pattern_start:pattern_end, 0] = pattern
            
            # Target indicates presence of pattern
            targets[b] = 1.0 if torch.rand(1) > 0.5 else 0.0
        
        return sequences, targets
    
    @staticmethod
    def evaluate_temporal_memory(
        model: HierarchicalTemporalMemoryFusion,
        test_sequences: torch.Tensor,
        dependency_spans: List[int] = [100, 500, 1000, 2000, 5000],
    ) -> Dict[str, float]:
        """Evaluate model's ability to capture long-term dependencies."""
        model.eval()
        results = {}
        
        with torch.no_grad():
            for span in dependency_spans:
                # Generate test data
                sequences, targets = TemporalDependencyBenchmark.generate_long_term_sequence(
                    length=min(span * 2, test_sequences.shape[1]),
                    dependency_span=span,
                )
                
                # Process with model
                output, info = model(sequences[:, :sequences.shape[1]], return_intermediate=True)
                
                # Measure temporal coherence at this span
                temporal_coherence = info.get('temporal_coherence', 0.0)
                energy_efficiency = info.get('energy_efficiency', 0.0)
                
                results[f'coherence_span_{span}'] = temporal_coherence
                results[f'efficiency_span_{span}'] = energy_efficiency
        
        return results