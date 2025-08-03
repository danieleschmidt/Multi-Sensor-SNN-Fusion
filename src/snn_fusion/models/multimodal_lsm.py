"""
Multi-Modal Liquid State Machine

Implements specialized LSM architecture for processing multiple asynchronous
sensor modalities with cross-modal fusion and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
import warnings

from .lsm import LiquidStateMachine
from .attention import CrossModalAttention
from .readouts import LinearReadout, TemporalReadout


class MultiModalLSM(nn.Module):
    """
    Multi-Modal Liquid State Machine for sensor fusion.
    
    Processes multiple asynchronous sensor modalities through dedicated LSMs
    with cross-modal attention and hierarchical fusion for real-time applications.
    """
    
    def __init__(
        self,
        modality_configs: Dict[str, Dict[str, Any]],
        fusion_config: Optional[Dict[str, Any]] = None,
        n_outputs: int = 10,
        fusion_type: str = "attention",
        global_reservoir_size: int = 500,
        enable_cross_modal_plasticity: bool = False,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize Multi-Modal LSM.
        
        Args:
            modality_configs: Configuration for each modality LSM
                {
                    'audio': {'n_inputs': 64, 'n_reservoir': 300, ...},
                    'vision': {'n_inputs': 1024, 'n_reservoir': 400, ...},
                    'tactile': {'n_inputs': 6, 'n_reservoir': 200, ...}
                }
            fusion_config: Cross-modal fusion configuration
            n_outputs: Number of output classes/dimensions
            fusion_type: Type of fusion mechanism
            global_reservoir_size: Size of global fusion reservoir
            enable_cross_modal_plasticity: Enable cross-modal learning
            device: Computation device
        """
        super().__init__()
        
        self.modality_configs = modality_configs
        self.fusion_config = fusion_config or {}
        self.n_outputs = n_outputs
        self.fusion_type = fusion_type
        self.global_reservoir_size = global_reservoir_size
        self.enable_cross_modal_plasticity = enable_cross_modal_plasticity
        self.device = device or torch.device('cpu')
        
        self.modalities = list(modality_configs.keys())
        
        # Initialize modality-specific LSMs
        self.modality_lsms = self._create_modality_lsms()
        
        # Cross-modal fusion components
        self.fusion_layer = self._create_fusion_layer()
        
        # Global processing reservoir (optional)
        if global_reservoir_size > 0:
            self.global_lsm = self._create_global_lsm()
        else:
            self.global_lsm = None
        
        # Final readout
        self.readout = self._create_readout()
        
        # State tracking
        self.modality_states = {}
        self.fusion_states = {}
        
        self.to(self.device)
    
    def _create_modality_lsms(self) -> nn.ModuleDict:
        """Create LSM for each sensory modality."""
        lsms = nn.ModuleDict()
        
        for modality, config in self.modality_configs.items():
            # Default LSM parameters
            lsm_config = {
                'n_inputs': config['n_inputs'],
                'n_reservoir': config.get('n_reservoir', 300),
                'n_outputs': config.get('n_outputs', 32),  # Intermediate representation
                'connectivity': config.get('connectivity', 0.1),
                'spectral_radius': config.get('spectral_radius', 0.9),
                'tau_mem': config.get('tau_mem', 20.0),
                'tau_adapt': config.get('tau_adapt', 100.0),
                'device': self.device,
            }
            
            lsms[modality] = LiquidStateMachine(**lsm_config)
        
        return lsms
    
    def _create_fusion_layer(self) -> nn.Module:
        """Create cross-modal fusion mechanism."""
        if self.fusion_type == "attention":
            return CrossModalAttention(
                modalities=self.modalities,
                modality_dims={
                    mod: self.modality_configs[mod].get('n_outputs', 32) 
                    for mod in self.modalities
                },
                hidden_dim=self.fusion_config.get('hidden_dim', 128),
                num_heads=self.fusion_config.get('num_heads', 4),
                dropout=self.fusion_config.get('dropout', 0.1),
                device=self.device,
            )
        elif self.fusion_type == "concatenation":
            return ConcatenationFusion(self.modalities, self.device)
        elif self.fusion_type == "weighted_sum":
            return WeightedSumFusion(
                modalities=self.modalities,
                feature_dims={
                    mod: self.modality_configs[mod].get('n_outputs', 32)
                    for mod in self.modalities
                },
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
    
    def _create_global_lsm(self) -> LiquidStateMachine:
        """Create global LSM for processing fused representations."""
        # Compute input size based on fusion output
        if self.fusion_type == "attention":
            fusion_output_size = self.fusion_config.get('hidden_dim', 128)
        elif self.fusion_type == "concatenation":
            fusion_output_size = sum(
                self.modality_configs[mod].get('n_outputs', 32) 
                for mod in self.modalities
            )
        elif self.fusion_type == "weighted_sum":
            fusion_output_size = max(
                self.modality_configs[mod].get('n_outputs', 32)
                for mod in self.modalities
            )
        
        return LiquidStateMachine(
            n_inputs=fusion_output_size,
            n_reservoir=self.global_reservoir_size,
            n_outputs=self.n_outputs,
            connectivity=0.15,  # Slightly higher connectivity for global processing
            spectral_radius=0.95,
            device=self.device,
        )
    
    def _create_readout(self) -> nn.Module:
        """Create final readout layer."""
        if self.global_lsm is not None:
            # Use global LSM output size
            input_size = self.global_reservoir_size * 4  # Liquid state representation
        else:
            # Use fusion layer output size
            if self.fusion_type == "attention":
                input_size = self.fusion_config.get('hidden_dim', 128)
            elif self.fusion_type == "concatenation":
                input_size = sum(
                    self.modality_configs[mod].get('n_outputs', 32)
                    for mod in self.modalities
                )
            elif self.fusion_type == "weighted_sum":
                input_size = max(
                    self.modality_configs[mod].get('n_outputs', 32)
                    for mod in self.modalities
                )
        
        return LinearReadout(
            input_size=input_size,
            output_size=self.n_outputs,
            dropout=0.2,
            temporal_integration=True,
            device=self.device,
        )
    
    def forward(
        self, 
        inputs: Dict[str, torch.Tensor],
        return_states: bool = False,
        missing_modalities: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass through multi-modal LSM.
        
        Args:
            inputs: Dictionary of modality inputs
                {
                    'audio': torch.Tensor,  # [batch_size, time_steps, audio_features]
                    'vision': torch.Tensor, # [batch_size, time_steps, vision_features]
                    'tactile': torch.Tensor # [batch_size, time_steps, tactile_features]
                }
            return_states: Whether to return internal states
            missing_modalities: List of unavailable modalities for robustness testing
            
        Returns:
            outputs: Final predictions [batch_size, n_outputs]
            states: Optional dictionary of internal states
        """
        # Handle missing modalities
        available_modalities = set(inputs.keys())
        if missing_modalities:
            available_modalities -= set(missing_modalities)
        
        # Process each available modality
        modality_outputs = {}
        modality_states_dict = {}
        
        for modality in self.modalities:
            if modality in available_modalities and modality in inputs:
                modality_input = inputs[modality]
                
                # Process through modality-specific LSM
                mod_output, mod_states = self.modality_lsms[modality](
                    modality_input, 
                    return_states=return_states
                )
                
                modality_outputs[modality] = mod_output
                if return_states:
                    modality_states_dict[modality] = mod_states
        
        if not modality_outputs:
            raise ValueError("No available modalities for processing")
        
        # Cross-modal fusion
        fused_representation, fusion_attention = self.fusion_layer(
            modality_outputs, 
            available_modalities=list(available_modalities)
        )
        
        # Global processing (if enabled)
        if self.global_lsm is not None:
            # Reshape fused representation for temporal processing
            if fused_representation.dim() == 2:
                fused_representation = fused_representation.unsqueeze(1)  # Add time dimension
            
            global_output, global_states = self.global_lsm(
                fused_representation,
                return_states=return_states
            )
            final_representation = global_output
            
            if return_states:
                self.fusion_states['global_lsm'] = global_states
        else:
            final_representation = fused_representation
        
        # Final readout
        outputs = self.readout(final_representation)
        
        # Compile states if requested
        states = None
        if return_states:
            states = {
                'modality_states': modality_states_dict,
                'fusion_attention': fusion_attention,
                'fusion_states': self.fusion_states,
                'fused_representation': fused_representation,
                'available_modalities': list(available_modalities),
            }
        
        return outputs, states
    
    def forward_single_modality(
        self, 
        modality: str, 
        modality_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process single modality for analysis.
        
        Args:
            modality: Modality name
            modality_input: Input tensor for the modality
            
        Returns:
            outputs: Single modality predictions
        """
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Process through modality LSM
        mod_output, _ = self.modality_lsms[modality](modality_input)
        
        # Direct readout (skip fusion)
        single_mod_readout = LinearReadout(
            input_size=mod_output.shape[-1],
            output_size=self.n_outputs,
            device=self.device,
        )
        
        outputs = single_mod_readout(mod_output)
        return outputs
    
    def reset_all_states(self) -> None:
        """Reset all internal states across modalities."""
        for lsm in self.modality_lsms.values():
            lsm.reset_state()
        
        if self.global_lsm is not None:
            self.global_lsm.reset_state()
        
        self.readout.reset_history()
        self.modality_states.clear()
        self.fusion_states.clear()
    
    def get_modality_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each modality LSM."""
        stats = {}
        for modality, lsm in self.modality_lsms.items():
            stats[modality] = lsm.get_reservoir_statistics()
        return stats
    
    def enable_modality_plasticity(self, modalities: Optional[List[str]] = None) -> None:
        """Enable plasticity for specific modalities."""
        target_modalities = modalities or self.modalities
        
        for modality in target_modalities:
            if modality in self.modality_lsms:
                self.modality_lsms[modality].enable_plasticity()
    
    def disable_modality_plasticity(self, modalities: Optional[List[str]] = None) -> None:
        """Disable plasticity for specific modalities."""
        target_modalities = modalities or self.modalities
        
        for modality in target_modalities:
            if modality in self.modality_lsms:
                self.modality_lsms[modality].disable_plasticity()


class ConcatenationFusion(nn.Module):
    """Simple concatenation-based fusion."""
    
    def __init__(self, modalities: List[str], device: torch.device):
        super().__init__()
        self.modalities = modalities
        self.device = device
        
    def forward(
        self, 
        modality_outputs: Dict[str, torch.Tensor],
        available_modalities: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Concatenate available modality outputs."""
        if available_modalities is None:
            available_modalities = list(modality_outputs.keys())
        
        # Concatenate available modality features
        features = []
        for modality in self.modalities:
            if modality in available_modalities:
                features.append(modality_outputs[modality])
        
        if not features:
            raise ValueError("No features to concatenate")
        
        concatenated = torch.cat(features, dim=-1)
        return concatenated, None  # No attention weights


class WeightedSumFusion(nn.Module):
    """Learnable weighted sum fusion."""
    
    def __init__(
        self, 
        modalities: List[str], 
        feature_dims: Dict[str, int],
        device: torch.device,
    ):
        super().__init__()
        self.modalities = modalities
        self.feature_dims = feature_dims
        self.device = device
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(modalities), device=device))
        
        # Project all modalities to same dimension
        max_dim = max(feature_dims.values())
        self.projections = nn.ModuleDict({
            modality: nn.Linear(feature_dims[modality], max_dim)
            for modality in modalities
        })
        
    def forward(
        self, 
        modality_outputs: Dict[str, torch.Tensor],
        available_modalities: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute weighted sum of available modalities."""
        if available_modalities is None:
            available_modalities = list(modality_outputs.keys())
        
        # Project features to same dimension
        projected_features = []
        active_weights = []
        
        for i, modality in enumerate(self.modalities):
            if modality in available_modalities:
                projected = self.projections[modality](modality_outputs[modality])
                projected_features.append(projected)
                active_weights.append(self.fusion_weights[i])
        
        if not projected_features:
            raise ValueError("No features to fuse")
        
        # Normalize weights
        active_weights = torch.stack(active_weights)
        normalized_weights = F.softmax(active_weights, dim=0)
        
        # Weighted sum
        weighted_features = []
        for i, features in enumerate(projected_features):
            weighted_features.append(normalized_weights[i] * features)
        
        fused = torch.stack(weighted_features).sum(dim=0)
        
        return fused, normalized_weights