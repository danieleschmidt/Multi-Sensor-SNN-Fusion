"""
Hierarchical Fusion Network for Multi-Modal Spiking Neural Networks

This module implements a hierarchical multi-level fusion architecture that processes
different sensory modalities at multiple scales before combining them for final
decision making. The architecture supports adaptive fusion strategies and can
handle missing or corrupted modalities.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .neurons import AdaptiveLIF
from .attention import CrossModalAttention
from .readouts import LinearReadout, TemporalReadout


class HierarchicalLevel(nn.Module):
    """
    Single level in the hierarchical fusion network.
    
    Each level processes modality-specific features and performs cross-modal
    interactions before passing information to the next level.
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int,
        num_neurons: int,
        fusion_type: str = "attention",
        dropout: float = 0.1,
        tau_mem: float = 20.0,
        tau_adapt: float = 100.0
    ):
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        self.modalities = list(input_dims.keys())
        
        # Modality-specific processing layers
        self.modality_processors = nn.ModuleDict()
        for modality, input_dim in input_dims.items():
            self.modality_processors[modality] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                AdaptiveLIF(
                    threshold=1.0,
                    tau_mem=tau_mem,
                    tau_adapt=tau_adapt,
                    reset_mechanism="subtract"
                ),
                nn.Dropout(dropout)
            )
        
        # Cross-modal fusion mechanism
        if fusion_type == "attention":
            self.fusion_layer = CrossModalAttention(
                hidden_dim=hidden_dim,
                num_heads=4,
                modalities=self.modalities,
                temperature=1.0
            )
        elif fusion_type == "concatenation":
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_dim * len(self.modalities), hidden_dim),
                AdaptiveLIF(
                    threshold=1.0,
                    tau_mem=tau_mem,
                    tau_adapt=tau_adapt
                )
            )
        elif fusion_type == "weighted_sum":
            self.fusion_layer = nn.Linear(len(self.modalities), 1)
            self.weight_activation = nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            AdaptiveLIF(
                threshold=1.0,
                tau_mem=tau_mem,
                tau_adapt=tau_adapt
            )
        )
        
        # Residual connection
        self.use_residual = True
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        states: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through a single hierarchical level.
        
        Args:
            inputs: Dictionary mapping modality names to input tensors
            states: Optional previous states for recurrent processing
            
        Returns:
            fused_output: Fused multi-modal representation
            new_states: Updated states for next time step
        """
        batch_size = next(iter(inputs.values())).shape[0]
        device = next(iter(inputs.values())).device
        
        # Process each modality
        processed_modalities = {}
        new_states = states if states is not None else {}
        
        for modality in self.modalities:
            if modality in inputs and inputs[modality] is not None:
                # Process modality-specific input
                processed = self.modality_processors[modality](inputs[modality])
                processed_modalities[modality] = processed
            else:
                # Handle missing modality with zeros
                processed_modalities[modality] = torch.zeros(
                    batch_size, self.hidden_dim, device=device
                )
        
        # Cross-modal fusion
        if self.fusion_type == "attention":
            fused_output = self.fusion_layer(processed_modalities)
        elif self.fusion_type == "concatenation":
            concatenated = torch.cat(list(processed_modalities.values()), dim=-1)
            fused_output = self.fusion_layer(concatenated)
        elif self.fusion_type == "weighted_sum":
            # Stack modality features
            stacked = torch.stack(list(processed_modalities.values()), dim=-1)
            
            # Compute adaptive weights
            weights = self.weight_activation(
                self.fusion_layer.weight.view(1, 1, -1).expand(
                    batch_size, self.hidden_dim, -1
                )
            )
            
            # Weighted combination
            fused_output = torch.sum(stacked * weights, dim=-1)
        
        # Output projection with residual connection
        projected = self.output_projection(fused_output)
        
        if self.use_residual and fused_output.shape == projected.shape:
            output = projected + fused_output
        else:
            output = projected
        
        return output, new_states


class HierarchicalFusionSNN(nn.Module):
    """
    Hierarchical Spiking Neural Network for Multi-Modal Sensor Fusion.
    
    This network implements a multi-level processing architecture where each level
    performs increasingly abstract fusion of sensory modalities. The hierarchy
    enables both fine-grained and coarse-grained multi-modal interactions.
    
    Architecture:
    - Level 1: Low-level feature processing per modality
    - Level 2: Cross-modal feature interactions  
    - Level 3: High-level abstract representations
    - Final: Task-specific readouts
    """
    
    def __init__(
        self,
        input_shapes: Dict[str, Tuple[int, ...]],
        level_configs: List[Dict],
        task_heads: Dict[str, int],
        global_fusion: bool = True,
        temporal_integration: int = 10,
        device: str = "cpu"
    ):
        """
        Initialize the hierarchical fusion network.
        
        Args:
            input_shapes: Dictionary mapping modality names to input shapes
            level_configs: List of configuration dictionaries for each level
            task_heads: Dictionary mapping task names to output dimensions
            global_fusion: Whether to use global fusion at the top level
            temporal_integration: Number of time steps for temporal integration
            device: Device to run computations on
        """
        super().__init__()
        
        self.input_shapes = input_shapes
        self.modalities = list(input_shapes.keys())
        self.num_levels = len(level_configs)
        self.task_heads = task_heads
        self.global_fusion = global_fusion
        self.temporal_integration = temporal_integration
        self.device = device
        
        # Validate configuration
        self._validate_config(level_configs)
        
        # Input projection layers for each modality
        self.input_projections = nn.ModuleDict()
        for modality, shape in input_shapes.items():
            input_dim = np.prod(shape) if isinstance(shape, (list, tuple)) else shape
            self.input_projections[modality] = nn.Linear(
                input_dim, level_configs[0]["input_dims"][modality]
            )
        
        # Hierarchical levels
        self.levels = nn.ModuleList()
        for i, config in enumerate(level_configs):
            level = HierarchicalLevel(**config)
            self.levels.append(level)
        
        # Global fusion layer (if enabled)
        if global_fusion:
            final_dim = level_configs[-1]["hidden_dim"]
            self.global_fusion_layer = CrossModalAttention(
                hidden_dim=final_dim,
                num_heads=8,
                modalities=["hierarchical_features"],
                temperature=0.1
            )
        
        # Task-specific readout heads
        self.readout_heads = nn.ModuleDict()
        final_dim = level_configs[-1]["hidden_dim"]
        
        for task_name, output_dim in task_heads.items():
            if task_name == "classification":
                self.readout_heads[task_name] = LinearReadout(
                    input_size=final_dim,
                    output_size=output_dim,
                    integration_method="exponential",
                    tau_integration=20.0
                )
            elif task_name == "regression":
                self.readout_heads[task_name] = LinearReadout(
                    input_size=final_dim,
                    output_size=output_dim,
                    integration_method="sliding_window",
                    window_size=temporal_integration
                )
            elif task_name == "sequence":
                self.readout_heads[task_name] = TemporalReadout(
                    input_size=final_dim,
                    output_size=output_dim,
                    sequence_length=temporal_integration,
                    attention_dim=64
                )
            else:
                # Default linear readout
                self.readout_heads[task_name] = LinearReadout(
                    input_size=final_dim,
                    output_size=output_dim
                )
        
        # Temporal buffer for integration
        self.temporal_buffer = []
        self.buffer_size = temporal_integration
        
        # Initialize states
        self.reset_states()
        
    def _validate_config(self, level_configs: List[Dict]) -> None:
        """Validate the hierarchical configuration."""
        if not level_configs:
            raise ValueError("At least one level configuration required")
        
        # Check that input dimensions match across levels
        for i in range(1, len(level_configs)):
            prev_hidden = level_configs[i-1]["hidden_dim"]
            curr_inputs = level_configs[i]["input_dims"]
            
            # For hierarchical levels, input should match previous output
            expected_input = {f"level_{i-1}": prev_hidden}
            if "hierarchical" in curr_inputs:
                if curr_inputs["hierarchical"] != prev_hidden:
                    raise ValueError(
                        f"Level {i} hierarchical input dim {curr_inputs['hierarchical']} "
                        f"doesn't match level {i-1} output dim {prev_hidden}"
                    )
    
    def reset_states(self) -> None:
        """Reset all internal states."""
        self.level_states = [{} for _ in range(self.num_levels)]
        self.temporal_buffer = []
        
        # Reset neuron states in all levels
        for level in self.levels:
            for processor in level.modality_processors.values():
                for module in processor:
                    if hasattr(module, 'reset_state'):
                        module.reset_state()
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_intermediates: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]]:
        """
        Forward pass through the hierarchical fusion network.
        
        Args:
            inputs: Dictionary mapping modality names to input tensors
            return_intermediates: Whether to return intermediate representations
            
        Returns:
            outputs: Dictionary mapping task names to output tensors
            intermediates: (Optional) List of intermediate representations from each level
        """
        batch_size = next(iter(inputs.values())).shape[0]
        intermediates = []
        
        # Project inputs to first level dimensions
        current_inputs = {}
        for modality, tensor in inputs.items():
            if modality in self.input_projections:
                # Flatten if necessary
                if len(tensor.shape) > 2:
                    tensor = tensor.view(batch_size, -1)
                current_inputs[modality] = self.input_projections[modality](tensor)
        
        # Forward through hierarchical levels
        for i, level in enumerate(self.levels):
            level_output, self.level_states[i] = level(
                current_inputs, self.level_states[i]
            )
            
            intermediates.append(level_output)
            
            # Prepare input for next level (if not the last level)
            if i < len(self.levels) - 1:
                # Next level takes output from current level as hierarchical input
                current_inputs = {"hierarchical": level_output}
                
                # Add any direct modality inputs for the next level if configured
                next_level_inputs = self.levels[i + 1].input_dims
                for modality in self.modalities:
                    if modality in next_level_inputs and modality in inputs:
                        if modality in self.input_projections:
                            tensor = inputs[modality]
                            if len(tensor.shape) > 2:
                                tensor = tensor.view(batch_size, -1)
                            current_inputs[modality] = self.input_projections[modality](tensor)
        
        # Global fusion (if enabled)
        final_features = level_output
        if self.global_fusion:
            # Apply global attention to final features
            global_input = {"hierarchical_features": final_features}
            final_features = self.global_fusion_layer(global_input)
        
        # Temporal integration
        self.temporal_buffer.append(final_features)
        if len(self.temporal_buffer) > self.buffer_size:
            self.temporal_buffer.pop(0)
        
        # Compute task-specific outputs
        outputs = {}
        for task_name, readout in self.readout_heads.items():
            outputs[task_name] = readout(final_features)
        
        if return_intermediates:
            return outputs, intermediates
        else:
            return outputs
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get attention weights from all levels for visualization."""
        attention_weights = {}
        
        for i, level in enumerate(self.levels):
            if hasattr(level.fusion_layer, 'get_attention_weights'):
                attention_weights[f"level_{i}"] = level.fusion_layer.get_attention_weights()
        
        if self.global_fusion and hasattr(self.global_fusion_layer, 'get_attention_weights'):
            attention_weights["global"] = self.global_fusion_layer.get_attention_weights()
        
        return attention_weights
    
    def get_modality_importance(self) -> Dict[str, float]:
        """Compute importance scores for each modality."""
        attention_weights = self.get_attention_weights()
        importance_scores = {modality: 0.0 for modality in self.modalities}
        
        # Aggregate attention weights across levels
        total_weight = 0.0
        for level_weights in attention_weights.values():
            if isinstance(level_weights, dict):
                for modality, weight in level_weights.items():
                    if modality in importance_scores:
                        importance_scores[modality] += weight.mean().item()
                        total_weight += 1.0
        
        # Normalize scores
        if total_weight > 0:
            for modality in importance_scores:
                importance_scores[modality] /= total_weight
        
        return importance_scores
    
    def enable_adaptation(self, adaptation_rate: float = 0.01) -> None:
        """Enable adaptive fusion weights based on modality reliability."""
        for level in self.levels:
            if hasattr(level.fusion_layer, 'enable_adaptation'):
                level.fusion_layer.enable_adaptation(adaptation_rate)
    
    def get_fusion_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get detailed fusion statistics for monitoring."""
        stats = {}
        
        for i, level in enumerate(self.levels):
            level_stats = {}
            
            # Neuron firing rates
            for modality, processor in level.modality_processors.items():
                for j, module in enumerate(processor):
                    if isinstance(module, AdaptiveLIF):
                        firing_rate = module.get_firing_rate()
                        level_stats[f"{modality}_firing_rate"] = firing_rate
            
            # Fusion layer statistics
            if hasattr(level.fusion_layer, 'get_statistics'):
                fusion_stats = level.fusion_layer.get_statistics()
                level_stats.update(fusion_stats)
            
            stats[f"level_{i}"] = level_stats
        
        return stats


def create_default_hierarchical_config(
    input_shapes: Dict[str, Tuple[int, ...]],
    num_levels: int = 3,
    base_hidden_dim: int = 256,
    fusion_types: List[str] = None
) -> List[Dict]:
    """
    Create a default hierarchical configuration.
    
    Args:
        input_shapes: Dictionary mapping modality names to input shapes
        num_levels: Number of hierarchical levels
        base_hidden_dim: Base hidden dimension (increases per level)
        fusion_types: List of fusion types for each level
        
    Returns:
        List of configuration dictionaries for each level
    """
    if fusion_types is None:
        fusion_types = ["attention"] * num_levels
    
    if len(fusion_types) != num_levels:
        raise ValueError("Number of fusion types must match number of levels")
    
    modalities = list(input_shapes.keys())
    level_configs = []
    
    for i in range(num_levels):
        hidden_dim = base_hidden_dim * (2 ** i)  # Double size each level
        
        if i == 0:
            # First level: process raw modality inputs
            input_dims = {}
            for modality, shape in input_shapes.items():
                input_dims[modality] = np.prod(shape) if isinstance(shape, (list, tuple)) else shape
        else:
            # Higher levels: take hierarchical input from previous level
            prev_hidden = base_hidden_dim * (2 ** (i - 1))
            input_dims = {"hierarchical": prev_hidden}
        
        config = {
            "input_dims": input_dims,
            "hidden_dim": hidden_dim,
            "num_neurons": hidden_dim,
            "fusion_type": fusion_types[i],
            "dropout": 0.1,
            "tau_mem": 20.0,
            "tau_adapt": 100.0
        }
        
        level_configs.append(config)
    
    return level_configs