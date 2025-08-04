"""
Loss Functions for Spiking Neural Networks

This module implements specialized loss functions for training spiking neural networks,
including temporal losses, spike-based losses, and multi-modal fusion losses optimized
for neuromorphic computing applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
import math


class SpikeLoss(nn.Module):
    """
    Base class for spike-based loss functions.
    
    Provides common functionality for handling spike trains and temporal
    sequences in spiking neural networks.
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        time_weighting: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ):
        """
        Initialize spike loss base class.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
            time_weighting: Temporal weighting for different time steps
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        if time_weighting is not None:
            self.register_buffer('time_weighting', time_weighting)
        else:
            self.time_weighting = None
    
    def _apply_reduction(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor."""
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
    
    def _apply_time_weighting(self, loss: torch.Tensor, time_dim: int = 1) -> torch.Tensor:
        """Apply temporal weighting to loss."""
        if self.time_weighting is not None:
            # Expand time weighting to match loss dimensions
            weight_shape = [1] * loss.dim()
            weight_shape[time_dim] = -1
            weights = self.time_weighting.view(weight_shape)
            loss = loss * weights
        return loss


class TemporalLoss(SpikeLoss):
    """
    Temporal loss function for spike-based sequence learning.
    
    Computes loss over temporal sequences with support for different
    temporal objectives including first-spike timing, spike count,
    and temporal pattern matching.
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        temporal_objective: str = "rate",
        time_window: Optional[float] = None,
        tau_integration: float = 20.0,
        **kwargs
    ):
        """
        Initialize temporal loss.
        
        Args:
            loss_type: Base loss function ('mse', 'cross_entropy', 'cosine')
            temporal_objective: Type of temporal learning ('rate', 'timing', 'pattern')
            time_window: Time window for integration (ms)
            tau_integration: Time constant for exponential integration
        """
        super().__init__(**kwargs)
        
        self.loss_type = loss_type
        self.temporal_objective = temporal_objective
        self.time_window = time_window
        self.tau_integration = tau_integration
        
        # Initialize base loss function
        if loss_type == "mse":
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == "cross_entropy":
            self.base_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        elif loss_type == "cosine":
            self.base_loss = nn.CosineEmbeddingLoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _compute_spike_rate(self, spikes: torch.Tensor, time_dim: int = 1) -> torch.Tensor:
        """Compute spike rates from spike trains."""
        if self.time_window is not None:
            # Use sliding window integration
            window_size = int(self.time_window)
            return F.avg_pool1d(
                spikes.transpose(1, 2), 
                kernel_size=window_size,
                stride=1,
                padding=window_size//2
            ).transpose(1, 2)
        else:
            # Use exponential integration
            alpha = math.exp(-1.0 / self.tau_integration)
            rates = torch.zeros_like(spikes)
            
            for t in range(spikes.shape[time_dim]):
                if t == 0:
                    rates[:, t] = spikes[:, t]
                else:
                    rates[:, t] = alpha * rates[:, t-1] + spikes[:, t]
            
            return rates
    
    def _compute_first_spike_time(self, spikes: torch.Tensor, time_dim: int = 1) -> torch.Tensor:
        """Compute first spike timing for each neuron."""
        # Find first spike indices
        first_spike_mask = spikes > 0
        first_spike_times = torch.argmax(first_spike_mask.float(), dim=time_dim)
        
        # Handle neurons that never spike
        never_spiked = ~torch.any(first_spike_mask, dim=time_dim)
        first_spike_times[never_spiked] = spikes.shape[time_dim]  # Set to max time
        
        return first_spike_times.float()
    
    def _compute_spike_pattern(self, spikes: torch.Tensor, time_dim: int = 1) -> torch.Tensor:
        """Compute spike pattern representation."""
        # Normalize spike trains to create pattern representation
        spike_counts = spikes.sum(dim=time_dim, keepdim=True)
        spike_counts = torch.clamp(spike_counts, min=1e-6)  # Avoid division by zero
        patterns = spikes / spike_counts
        return patterns
    
    def forward(
        self,
        spike_output: torch.Tensor,
        target: torch.Tensor,
        target_spikes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute temporal loss.
        
        Args:
            spike_output: Network spike output [batch, time, features]
            target: Target values or class indices
            target_spikes: Target spike trains (for pattern matching)
            
        Returns:
            loss: Computed temporal loss
        """
        batch_size, time_steps, features = spike_output.shape
        
        if self.temporal_objective == "rate":
            # Rate-based learning
            output_rates = self._compute_spike_rate(spike_output)
            
            if self.loss_type == "cross_entropy":
                # Average over time for classification
                avg_rates = output_rates.mean(dim=1)  # [batch, features]
                loss = self.base_loss(avg_rates, target.long())
            else:
                if target.dim() == 1:
                    # Expand target to match temporal dimension
                    target = target.unsqueeze(1).expand(-1, time_steps)
                if target.shape[-1] != features:
                    # One-hot encode if necessary
                    target = F.one_hot(target.long(), num_classes=features).float()
                
                loss = self.base_loss(output_rates, target)
        
        elif self.temporal_objective == "timing":
            # Timing-based learning using first spike times
            output_times = self._compute_first_spike_time(spike_output)
            
            if target_spikes is not None:
                target_times = self._compute_first_spike_time(target_spikes)
            else:
                # Convert target to timing representation
                target_times = target.float()
            
            loss = self.base_loss(output_times, target_times)
        
        elif self.temporal_objective == "pattern":
            # Pattern-based learning
            output_patterns = self._compute_spike_pattern(spike_output)
            
            if target_spikes is not None:
                target_patterns = self._compute_spike_pattern(target_spikes)
            else:
                # Create target patterns from class labels
                target_patterns = torch.zeros_like(output_patterns)
                for i, label in enumerate(target):
                    target_patterns[i, :, label] = 1.0 / time_steps
            
            if self.loss_type == "cosine":
                # Flatten for cosine similarity
                output_flat = output_patterns.view(batch_size, -1)
                target_flat = target_patterns.view(batch_size, -1)
                ones = torch.ones(batch_size, device=output_flat.device)
                loss = self.base_loss(output_flat, target_flat, ones)
            else:
                loss = self.base_loss(output_patterns, target_patterns)
        
        else:
            raise ValueError(f"Unknown temporal objective: {self.temporal_objective}")
        
        # Apply temporal weighting
        if self.temporal_objective != "timing" and loss.dim() > 1:
            loss = self._apply_time_weighting(loss)
        
        return self._apply_reduction(loss)


class VanRossumLoss(SpikeLoss):
    """
    Van Rossum distance loss for spike train similarity.
    
    Computes the Van Rossum distance between spike trains, which measures
    the similarity between spike patterns using convolution with an
    exponential kernel.
    """
    
    def __init__(
        self,
        tau: float = 20.0,
        dt: float = 1.0,
        **kwargs
    ):
        """
        Initialize Van Rossum loss.
        
        Args:
            tau: Time constant for exponential kernel (ms)
            dt: Time step size (ms)
        """
        super().__init__(**kwargs)
        
        self.tau = tau
        self.dt = dt
    
    def _create_kernel(self, length: int, device: torch.device) -> torch.Tensor:
        """Create exponential convolution kernel."""
        t = torch.arange(length, device=device, dtype=torch.float) * self.dt
        kernel = torch.exp(-t / self.tau) / self.tau
        return kernel
    
    def forward(
        self,
        spike_output: torch.Tensor,
        target_spikes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Van Rossum distance loss.
        
        Args:
            spike_output: Output spike trains [batch, time, neurons]
            target_spikes: Target spike trains [batch, time, neurons]
            
        Returns:
            loss: Van Rossum distance loss
        """
        batch_size, time_steps, num_neurons = spike_output.shape
        device = spike_output.device
        
        # Create exponential kernel
        kernel = self._create_kernel(time_steps, device)
        kernel = kernel.view(1, 1, -1)  # [1, 1, time]
        
        # Compute filtered spike trains
        def filter_spikes(spikes):
            # Reshape for convolution: [batch * neurons, 1, time]
            spikes_reshaped = spikes.transpose(1, 2).contiguous().view(-1, 1, time_steps)
            
            # Apply causal convolution
            filtered = F.conv1d(spikes_reshaped, kernel, padding=time_steps-1)
            filtered = filtered[:, :, :time_steps]  # Trim to original length
            
            # Reshape back: [batch, time, neurons]
            filtered = filtered.view(batch_size, num_neurons, time_steps).transpose(1, 2)
            return filtered
        
        # Filter both spike trains
        filtered_output = filter_spikes(spike_output)
        filtered_target = filter_spikes(target_spikes)
        
        # Compute L2 distance between filtered spike trains
        diff = filtered_output - filtered_target
        loss = torch.sum(diff ** 2, dim=(1, 2))  # Sum over time and neurons
        
        return self._apply_reduction(loss)


class CrossModalLoss(nn.Module):
    """
    Cross-modal loss for multi-modal fusion learning.
    
    Combines losses from multiple modalities with adaptive weighting
    and supports various fusion strategies.
    """
    
    def __init__(
        self,
        modality_weights: Optional[Dict[str, float]] = None,
        fusion_loss_weight: float = 0.1,
        consistency_weight: float = 0.05,
        adaptive_weighting: bool = True,
        temperature: float = 1.0
    ):
        """
        Initialize cross-modal loss.
        
        Args:
            modality_weights: Initial weights for each modality
            fusion_loss_weight: Weight for fusion-specific loss
            consistency_weight: Weight for cross-modal consistency
            adaptive_weighting: Whether to use adaptive weighting
            temperature: Temperature for adaptive weighting
        """
        super().__init__()
        
        self.modality_weights = modality_weights or {}
        self.fusion_loss_weight = fusion_loss_weight
        self.consistency_weight = consistency_weight
        self.adaptive_weighting = adaptive_weighting
        self.temperature = temperature
        
        # Base loss functions
        self.temporal_loss = TemporalLoss(loss_type="cross_entropy")
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        # Adaptive weights
        if adaptive_weighting:
            self.register_buffer('weight_history', torch.tensor([]))
            self.register_buffer('loss_history', torch.tensor([]))
    
    def _compute_adaptive_weights(
        self,
        modality_losses: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute adaptive weights based on loss magnitudes."""
        if not self.adaptive_weighting:
            return self.modality_weights
        
        # Convert losses to weights (inverse relationship)
        loss_values = torch.tensor([loss.item() for loss in modality_losses.values()])
        
        # Apply temperature scaling and softmax
        weights = F.softmax(-loss_values / self.temperature, dim=0)
        
        # Create weight dictionary
        adaptive_weights = {}
        for i, modality in enumerate(modality_losses.keys()):
            adaptive_weights[modality] = weights[i].item()
        
        return adaptive_weights
    
    def _compute_consistency_loss(
        self,
        modality_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute cross-modal consistency loss."""
        if len(modality_outputs) < 2:
            return torch.tensor(0.0, device=next(iter(modality_outputs.values())).device)
        
        consistency_loss = 0.0
        num_pairs = 0
        
        modalities = list(modality_outputs.keys())
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                out1, out2 = modality_outputs[mod1], modality_outputs[mod2]
                
                # Compute cosine similarity loss
                batch_size = out1.shape[0]
                out1_flat = out1.view(batch_size, -1)
                out2_flat = out2.view(batch_size, -1)
                
                # Normalize for cosine similarity
                out1_norm = F.normalize(out1_flat, p=2, dim=1)
                out2_norm = F.normalize(out2_flat, p=2, dim=1)
                
                # Cosine similarity (we want this to be high, so minimize negative)
                cosine_sim = F.cosine_similarity(out1_norm, out2_norm, dim=1)
                consistency_loss += (1 - cosine_sim).mean()
                num_pairs += 1
        
        return consistency_loss / num_pairs if num_pairs > 0 else consistency_loss
    
    def forward(
        self,
        modality_outputs: Dict[str, torch.Tensor],
        fusion_output: torch.Tensor,
        targets: torch.Tensor,
        modality_targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute cross-modal loss.
        
        Args:
            modality_outputs: Dictionary of modality-specific outputs
            fusion_output: Fused multi-modal output
            targets: Ground truth targets
            modality_targets: Optional modality-specific targets
            
        Returns:
            total_loss: Combined cross-modal loss
            loss_components: Dictionary of individual loss components
        """
        device = fusion_output.device
        loss_components = {}
        
        # Compute individual modality losses
        modality_losses = {}
        for modality, output in modality_outputs.items():
            if modality_targets and modality in modality_targets:
                target = modality_targets[modality]
            else:
                target = targets
            
            if output.dim() > 2:
                # Temporal output
                modality_loss = self.temporal_loss(output, target)
            else:
                # Regular output
                modality_loss = F.cross_entropy(output, target.long())
            
            modality_losses[modality] = modality_loss
            loss_components[f"{modality}_loss"] = modality_loss
        
        # Compute fusion loss
        if fusion_output.dim() > 2:
            fusion_loss = self.temporal_loss(fusion_output, targets)
        else:
            fusion_loss = F.cross_entropy(fusion_output, targets.long())
        
        loss_components["fusion_loss"] = fusion_loss
        
        # Compute consistency loss
        consistency_loss = self._compute_consistency_loss(modality_outputs)
        loss_components["consistency_loss"] = consistency_loss
        
        # Compute adaptive weights
        weights = self._compute_adaptive_weights(modality_losses)
        
        # Combine losses
        total_loss = self.fusion_loss_weight * fusion_loss
        
        # Add weighted modality losses
        for modality, loss in modality_losses.items():
            weight = weights.get(modality, self.modality_weights.get(modality, 1.0))
            total_loss += weight * loss
        
        # Add consistency loss
        total_loss += self.consistency_weight * consistency_loss
        
        # Store loss components for monitoring
        loss_components["total_loss"] = total_loss
        loss_components["adaptive_weights"] = weights
        
        return total_loss, loss_components
    
    def get_loss_statistics(self) -> Dict[str, float]:
        """Get loss statistics for monitoring."""
        stats = {
            "fusion_loss_weight": self.fusion_loss_weight,
            "consistency_weight": self.consistency_weight,
            "adaptive_weighting": self.adaptive_weighting,
        }
        
        # Add current adaptive weights
        if hasattr(self, 'current_weights'):
            for modality, weight in self.current_weights.items():
                stats[f"{modality}_weight"] = weight
        
        return stats


def create_loss_function(
    loss_type: str,
    num_classes: Optional[int] = None,
    modalities: Optional[List[str]] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create appropriate loss function.
    
    Args:
        loss_type: Type of loss ('temporal', 'van_rossum', 'cross_modal')
        num_classes: Number of output classes (for classification)
        modalities: List of modality names (for cross-modal loss)
        **kwargs: Additional arguments for loss function
        
    Returns:
        Configured loss function
    """
    if loss_type == "temporal":
        return TemporalLoss(**kwargs)
    elif loss_type == "van_rossum":
        return VanRossumLoss(**kwargs)
    elif loss_type == "cross_modal":
        return CrossModalLoss(**kwargs)
    elif loss_type == "spike":
        return SpikeLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Utility functions for loss computation

def compute_spike_rate_accuracy(
    spike_output: torch.Tensor,
    targets: torch.Tensor,
    time_window: Optional[int] = None
) -> float:
    """
    Compute accuracy based on spike rates.
    
    Args:
        spike_output: Spike output [batch, time, classes]
        targets: Target class indices [batch]
        time_window: Time window for rate computation
        
    Returns:
        accuracy: Classification accuracy
    """
    if time_window is not None:
        # Use sliding window
        rates = F.avg_pool1d(
            spike_output.transpose(1, 2),
            kernel_size=time_window,
            stride=time_window
        ).transpose(1, 2).mean(dim=1)
    else:
        # Average over all time
        rates = spike_output.mean(dim=1)
    
    predictions = torch.argmax(rates, dim=1)
    accuracy = (predictions == targets).float().mean().item()
    
    return accuracy


def compute_first_spike_latency(
    spike_output: torch.Tensor,
    targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute first spike latency for each class.
    
    Args:
        spike_output: Spike output [batch, time, classes]
        targets: Target class indices [batch]
        
    Returns:
        latencies: First spike latencies [batch]
        accuracies: Accuracy based on first spike timing
    """
    batch_size, time_steps, num_classes = spike_output.shape
    
    # Find first spike times for each class
    first_spike_times = torch.full((batch_size, num_classes), time_steps, 
                                  device=spike_output.device, dtype=torch.float)
    
    for b in range(batch_size):
        for c in range(num_classes):
            spike_indices = torch.nonzero(spike_output[b, :, c] > 0)
            if len(spike_indices) > 0:
                first_spike_times[b, c] = spike_indices[0].float()
    
    # Predictions based on earliest spike
    predictions = torch.argmin(first_spike_times, dim=1)
    accuracies = (predictions == targets).float()
    
    # Get latencies for target classes
    target_latencies = first_spike_times[torch.arange(batch_size), targets]
    
    return target_latencies, accuracies