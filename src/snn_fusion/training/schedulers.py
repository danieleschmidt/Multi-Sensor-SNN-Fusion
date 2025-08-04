"""
Learning Rate and Threshold Schedulers for Spiking Neural Networks

This module implements specialized schedulers for SNN training, including
adaptive threshold schedulers, plasticity schedulers, and learning rate
schedulers that account for temporal dynamics and spike statistics.
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings


class SpikingLRScheduler(_LRScheduler):
    """
    Base learning rate scheduler for spiking neural networks.
    
    Provides common functionality for SNN-specific learning rate scheduling
    based on spike statistics and temporal dynamics.
    """
    
    def __init__(
        self,
        optimizer,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.spike_stats = {}
        self.adaptation_history = []
        super().__init__(optimizer, last_epoch, verbose)
    
    def update_spike_statistics(self, spike_outputs: torch.Tensor) -> None:
        """
        Update spike statistics for scheduler adaptation.
        
        Args:
            spike_outputs: Model spike outputs [batch, time, neurons]
        """
        if spike_outputs.dim() >= 3:
            # Temporal spike data
            firing_rate = spike_outputs.mean().item()
            spike_variance = spike_outputs.var().item()
            temporal_correlation = self._compute_temporal_correlation(spike_outputs)
        else:
            # Non-temporal data
            firing_rate = spike_outputs.mean().item()
            spike_variance = spike_outputs.var().item()
            temporal_correlation = 0.0
        
        self.spike_stats.update({
            'firing_rate': firing_rate,
            'spike_variance': spike_variance,
            'temporal_correlation': temporal_correlation,
            'step': self.last_epoch + 1
        })
    
    def _compute_temporal_correlation(self, spike_outputs: torch.Tensor) -> float:
        """Compute temporal correlation in spike trains."""
        if spike_outputs.dim() < 3:
            return 0.0
        
        # Compute autocorrelation across time dimension
        batch_size, time_steps, num_neurons = spike_outputs.shape
        correlations = []
        
        for lag in range(1, min(10, time_steps // 2)):
            corr = torch.corrcoef(torch.stack([
                spike_outputs[:, :-lag, :].flatten(),
                spike_outputs[:, lag:, :].flatten()
            ]))[0, 1]
            
            if not torch.isnan(corr):
                correlations.append(corr.item())
        
        return np.mean(correlations) if correlations else 0.0
    
    def get_scheduler_state(self) -> Dict[str, Any]:
        """Get current scheduler state for monitoring."""
        return {
            'current_lr': [group['lr'] for group in self.optimizer.param_groups],
            'spike_stats': self.spike_stats,
            'last_epoch': self.last_epoch,
            'adaptation_history': self.adaptation_history[-10:]  # Last 10 steps
        }


class AdaptiveThresholdScheduler:
    """
    Adaptive threshold scheduler for spiking neurons.
    
    Dynamically adjusts neuron firing thresholds based on firing rate
    statistics to maintain target activity levels.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_firing_rate: float = 0.1,
        adaptation_rate: float = 0.01,
        threshold_bounds: Tuple[float, float] = (0.1, 10.0),
        adaptation_window: int = 100,
        momentum: float = 0.9
    ):
        """
        Initialize adaptive threshold scheduler.
        
        Args:
            model: SNN model with adaptive neurons
            target_firing_rate: Target average firing rate
            adaptation_rate: Rate of threshold adaptation
            threshold_bounds: (min_threshold, max_threshold) bounds
            adaptation_window: Window size for firing rate estimation
            momentum: Momentum factor for threshold updates
        """
        self.model = model
        self.target_firing_rate = target_firing_rate
        self.adaptation_rate = adaptation_rate
        self.threshold_bounds = threshold_bounds
        self.adaptation_window = adaptation_window
        self.momentum = momentum
        
        # Find all adaptive neurons in the model
        self.adaptive_neurons = []
        self._find_adaptive_neurons()
        
        # Initialize tracking
        self.firing_rate_history = {id(neuron): [] for neuron in self.adaptive_neurons}
        self.threshold_history = {id(neuron): [] for neuron in self.adaptive_neurons}
        self.momentum_buffer = {id(neuron): 0.0 for neuron in self.adaptive_neurons}
        
        # Statistics
        self.adaptation_step = 0
        self.total_adaptations = 0
    
    def _find_adaptive_neurons(self) -> None:
        """Find all adaptive neurons in the model."""
        for module in self.model.modules():
            if hasattr(module, 'threshold') and hasattr(module, 'get_firing_rate'):
                self.adaptive_neurons.append(module)
    
    def step(self, model_outputs: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Perform threshold adaptation step.
        
        Args:
            model_outputs: Optional model outputs for global firing rate
            
        Returns:
            Dictionary of adaptation statistics
        """
        adaptation_stats = {}
        total_threshold_change = 0.0
        
        for neuron in self.adaptive_neurons:
            neuron_id = id(neuron)
            
            # Get current firing rate
            if hasattr(neuron, 'get_firing_rate'):
                current_firing_rate = neuron.get_firing_rate()
            else:
                # Fallback: estimate from recent activity
                current_firing_rate = getattr(neuron, 'last_firing_rate', 0.1)
            
            # Update firing rate history
            self.firing_rate_history[neuron_id].append(current_firing_rate)
            if len(self.firing_rate_history[neuron_id]) > self.adaptation_window:
                self.firing_rate_history[neuron_id].pop(0)
            
            # Compute adaptation signal
            if len(self.firing_rate_history[neuron_id]) >= 10:  # Need minimum history
                avg_firing_rate = np.mean(self.firing_rate_history[neuron_id][-10:])
                firing_rate_error = self.target_firing_rate - avg_firing_rate
                
                # Compute threshold adjustment
                current_threshold = neuron.threshold.item() if hasattr(neuron.threshold, 'item') else neuron.threshold
                
                # Proportional adaptation with momentum
                threshold_adjustment = self.adaptation_rate * firing_rate_error * current_threshold
                
                # Apply momentum
                self.momentum_buffer[neuron_id] = (
                    self.momentum * self.momentum_buffer[neuron_id] + 
                    (1 - self.momentum) * threshold_adjustment
                )
                
                # Update threshold
                new_threshold = current_threshold - self.momentum_buffer[neuron_id]
                
                # Apply bounds
                new_threshold = max(
                    self.threshold_bounds[0], 
                    min(self.threshold_bounds[1], new_threshold)
                )
                
                # Apply threshold update
                if hasattr(neuron.threshold, 'data'):
                    neuron.threshold.data.fill_(new_threshold)
                else:
                    neuron.threshold = new_threshold
                
                # Track statistics
                threshold_change = abs(new_threshold - current_threshold)
                total_threshold_change += threshold_change
                
                self.threshold_history[neuron_id].append(new_threshold)
                if len(self.threshold_history[neuron_id]) > self.adaptation_window:
                    self.threshold_history[neuron_id].pop(0)
                
                # Store per-neuron stats
                adaptation_stats[f'neuron_{neuron_id}_firing_rate'] = avg_firing_rate
                adaptation_stats[f'neuron_{neuron_id}_threshold'] = new_threshold
                adaptation_stats[f'neuron_{neuron_id}_error'] = firing_rate_error
                
                self.total_adaptations += 1
        
        # Global statistics
        adaptation_stats.update({
            'total_threshold_change': total_threshold_change,
            'adaptation_step': self.adaptation_step,
            'total_adaptations': self.total_adaptations,
            'num_adaptive_neurons': len(self.adaptive_neurons)
        })
        
        self.adaptation_step += 1
        return adaptation_stats
    
    def get_firing_rate_statistics(self) -> Dict[str, Any]:
        """Get detailed firing rate statistics."""
        stats = {}
        
        for neuron in self.adaptive_neurons:
            neuron_id = id(neuron)
            history = self.firing_rate_history[neuron_id]
            
            if history:
                stats[f'neuron_{neuron_id}'] = {
                    'current_firing_rate': history[-1],
                    'average_firing_rate': np.mean(history),
                    'firing_rate_std': np.std(history),
                    'target_firing_rate': self.target_firing_rate,
                    'history_length': len(history)
                }
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset all adaptation statistics."""
        for neuron_id in self.firing_rate_history:
            self.firing_rate_history[neuron_id] = []
            self.threshold_history[neuron_id] = []
            self.momentum_buffer[neuron_id] = 0.0
        
        self.adaptation_step = 0
        self.total_adaptations = 0


class PlasticityScheduler:
    """
    Scheduler for controlling synaptic plasticity parameters.
    
    Manages STDP learning rates, eligibility traces, and other plasticity
    parameters throughout training.
    """
    
    def __init__(
        self,
        plasticity_rules: Dict[str, Any],
        schedule_type: str = "exponential",
        decay_rate: float = 0.95,
        schedule_interval: int = 100,
        min_learning_rate: float = 1e-6
    ):
        """
        Initialize plasticity scheduler.
        
        Args:
            plasticity_rules: Dictionary of plasticity rules to schedule
            schedule_type: Type of scheduling ('exponential', 'linear', 'cosine')
            decay_rate: Decay rate for exponential scheduling
            schedule_interval: Interval between schedule updates (steps)
            min_learning_rate: Minimum learning rate
        """
        self.plasticity_rules = plasticity_rules
        self.schedule_type = schedule_type
        self.decay_rate = decay_rate
        self.schedule_interval = schedule_interval
        self.min_learning_rate = min_learning_rate
        
        # Store initial values
        self.initial_values = {}
        for name, rule in plasticity_rules.items():
            self.initial_values[name] = {
                'A_plus': getattr(rule, 'A_plus', 0.01),
                'A_minus': getattr(rule, 'A_minus', 0.012),
                'tau_pre': getattr(rule, 'tau_pre', 20.0),
                'tau_post': getattr(rule, 'tau_post', 20.0)
            }
        
        self.step_count = 0
        self.schedule_history = []
    
    def step(self, performance_metric: Optional[float] = None) -> Dict[str, Any]:
        """
        Update plasticity parameters according to schedule.
        
        Args:
            performance_metric: Optional performance metric for adaptive scheduling
            
        Returns:
            Dictionary of updated parameters
        """
        if self.step_count % self.schedule_interval != 0:
            self.step_count += 1
            return {}
        
        updated_params = {}
        schedule_factor = self._compute_schedule_factor(performance_metric)
        
        for name, rule in self.plasticity_rules.items():
            initial_vals = self.initial_values[name]
            
            # Update A_plus (LTP rate)
            if hasattr(rule, 'A_plus'):
                new_A_plus = max(
                    self.min_learning_rate,
                    initial_vals['A_plus'] * schedule_factor
                )
                rule.A_plus = new_A_plus
                updated_params[f'{name}_A_plus'] = new_A_plus
            
            # Update A_minus (LTD rate)
            if hasattr(rule, 'A_minus'):
                new_A_minus = max(
                    self.min_learning_rate,
                    initial_vals['A_minus'] * schedule_factor
                )
                rule.A_minus = new_A_minus
                updated_params[f'{name}_A_minus'] = new_A_minus
            
            # Optionally update time constants (less common)
            if hasattr(rule, 'tau_pre') and self.schedule_type == "adaptive":
                # Adaptive time constant based on performance
                if performance_metric is not None:
                    tau_factor = 1.0 + (performance_metric - 0.5) * 0.1  # Adjust based on performance
                    new_tau_pre = initial_vals['tau_pre'] * tau_factor
                    rule.tau_pre = max(5.0, min(100.0, new_tau_pre))  # Bounded
                    updated_params[f'{name}_tau_pre'] = rule.tau_pre
        
        # Store history
        self.schedule_history.append({
            'step': self.step_count,
            'schedule_factor': schedule_factor,
            'updated_params': updated_params.copy()
        })
        
        self.step_count += 1
        return updated_params
    
    def _compute_schedule_factor(self, performance_metric: Optional[float] = None) -> float:
        """Compute scheduling factor based on schedule type."""
        schedule_step = self.step_count // self.schedule_interval
        
        if self.schedule_type == "exponential":
            return self.decay_rate ** schedule_step
        
        elif self.schedule_type == "linear":
            return max(0.1, 1.0 - 0.1 * schedule_step)
        
        elif self.schedule_type == "cosine":
            # Cosine annealing
            return 0.5 * (1 + math.cos(math.pi * schedule_step / 100))
        
        elif self.schedule_type == "adaptive":
            # Adaptive based on performance metric
            if performance_metric is not None:
                # If performance is good, reduce plasticity; if poor, increase it
                base_factor = self.decay_rate ** schedule_step
                performance_factor = 2.0 - performance_metric  # Higher perf -> lower factor
                return base_factor * performance_factor
            else:
                return self.decay_rate ** schedule_step
        
        else:
            return 1.0  # No scheduling
    
    def get_current_parameters(self) -> Dict[str, Dict[str, float]]:
        """Get current plasticity parameters."""
        current_params = {}
        
        for name, rule in self.plasticity_rules.items():
            current_params[name] = {
                'A_plus': getattr(rule, 'A_plus', 0.0),
                'A_minus': getattr(rule, 'A_minus', 0.0),
                'tau_pre': getattr(rule, 'tau_pre', 0.0),
                'tau_post': getattr(rule, 'tau_post', 0.0)
            }
        
        return current_params


class FiringRateScheduler(SpikingLRScheduler):
    """
    Learning rate scheduler based on firing rate statistics.
    
    Adapts learning rate based on average firing rates, aiming to maintain
    optimal activity levels for effective learning.
    """
    
    def __init__(
        self,
        optimizer,
        target_firing_rate: float = 0.1,
        adaptation_strength: float = 0.1,
        rate_window: int = 50,
        **kwargs
    ):
        """
        Initialize firing rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            target_firing_rate: Target average firing rate
            adaptation_strength: Strength of learning rate adaptation
            rate_window: Window size for firing rate averaging
        """
        self.target_firing_rate = target_firing_rate
        self.adaptation_strength = adaptation_strength
        self.rate_window = rate_window
        self.firing_rate_buffer = []
        
        super().__init__(optimizer, **kwargs)
    
    def get_lr(self) -> List[float]:
        """Compute learning rates based on firing rate statistics."""
        if not self.firing_rate_buffer or len(self.firing_rate_buffer) < 10:
            # Not enough data, use base learning rate
            return [group['lr'] for group in self.optimizer.param_groups]
        
        # Compute average firing rate
        avg_firing_rate = np.mean(self.firing_rate_buffer[-10:])
        
        # Compute adaptation factor
        rate_error = self.target_firing_rate - avg_firing_rate
        adaptation_factor = 1.0 + self.adaptation_strength * rate_error
        
        # Apply to all parameter groups
        adapted_lrs = []
        for group in self.optimizer.param_groups:
            base_lr = group.get('initial_lr', group['lr'])  
            adapted_lr = base_lr * adaptation_factor
            adapted_lr = max(1e-6, min(1e-1, adapted_lr))  # Bound learning rate
            adapted_lrs.append(adapted_lr)
        
        return adapted_lrs
    
    def step(self, epoch: Optional[int] = None, spike_outputs: Optional[torch.Tensor] = None):
        """Step with optional spike outputs for firing rate computation."""
        if spike_outputs is not None:
            self.update_spike_statistics(spike_outputs)
            
            # Update firing rate buffer
            firing_rate = self.spike_stats.get('firing_rate', 0.0)
            self.firing_rate_buffer.append(firing_rate)
            
            if len(self.firing_rate_buffer) > self.rate_window:
                self.firing_rate_buffer.pop(0)
        
        # Store adaptation info
        if len(self.firing_rate_buffer) > 0:
            self.adaptation_history.append({
                'epoch': self.last_epoch + 1,
                'firing_rate': self.firing_rate_buffer[-1],
                'target_rate': self.target_firing_rate,
                'lr_adaptation': self.get_lr()[0] / self.optimizer.param_groups[0].get('initial_lr', 1e-3)
            })
        
        super().step(epoch)


class CosineAnnealingSpikingLR(SpikingLRScheduler):
    """
    Cosine annealing learning rate scheduler for spiking neural networks.
    
    Implements cosine annealing with warm restarts and spike-aware adaptations.
    """
    
    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 0.0,
        T_mult: int = 1,
        restart_decay: float = 1.0,
        spike_aware: bool = True,
        **kwargs
    ):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate
            T_mult: Factor for increasing T_max after restart
            restart_decay: Decay factor for learning rate after restart
            spike_aware: Whether to incorporate spike statistics
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.T_mult = T_mult
        self.restart_decay = restart_decay
        self.spike_aware = spike_aware
        
        self.T_cur = 0
        self.T_i = T_max
        self.eta_max = None
        
        super().__init__(optimizer, **kwargs)
        
        # Store initial learning rates
        if self.eta_max is None:
            self.eta_max = [group['lr'] for group in self.optimizer.param_groups]
    
    def get_lr(self) -> List[float]:
        """Compute cosine annealing learning rates."""
        lrs = []
        
        for i, (group, eta_max_val) in enumerate(zip(self.optimizer.param_groups, self.eta_max)):
            # Basic cosine annealing
            lr = self.eta_min + (eta_max_val - self.eta_min) * (
                1 + math.cos(math.pi * self.T_cur / self.T_i)
            ) / 2
            
            # Spike-aware adaptation
            if self.spike_aware and self.spike_stats:
                firing_rate = self.spike_stats.get('firing_rate', 0.1)
                spike_variance = self.spike_stats.get('spike_variance', 0.01)
                
                # Adapt based on spike statistics
                activity_factor = 1.0 + 0.1 * (0.1 - firing_rate)  # Adjust if firing rate deviates
                variance_factor = 1.0 + 0.05 * spike_variance      # Increase LR with higher variance
                
                lr *= activity_factor * variance_factor
            
            lrs.append(max(self.eta_min, lr))
        
        return lrs
    
    def step(self, epoch: Optional[int] = None, spike_outputs: Optional[torch.Tensor] = None):
        """Step with optional restart logic."""
        if spike_outputs is not None:
            self.update_spike_statistics(spike_outputs)
        
        # Check for restart
        if self.T_cur >= self.T_i:
            # Restart
            self.T_cur = 0
            self.T_i = int(self.T_i * self.T_mult)
            
            # Decay maximum learning rates
            for i in range(len(self.eta_max)):
                self.eta_max[i] *= self.restart_decay
        
        self.T_cur += 1
        super().step(epoch)


# Factory functions

def create_threshold_scheduler(
    model: nn.Module,
    scheduler_type: str = "adaptive",
    **kwargs
) -> Union[AdaptiveThresholdScheduler, None]:
    """
    Create threshold scheduler for SNN model.
    
    Args:
        model: SNN model
        scheduler_type: Type of scheduler ('adaptive')
        **kwargs: Additional scheduler arguments
        
    Returns:
        Configured threshold scheduler
    """
    if scheduler_type == "adaptive":
        return AdaptiveThresholdScheduler(model, **kwargs)
    else:
        warnings.warn(f"Unknown threshold scheduler type: {scheduler_type}")
        return None


def create_lr_scheduler(
    optimizer,
    scheduler_type: str,
    **kwargs
) -> SpikingLRScheduler:
    """
    Create learning rate scheduler for SNN optimizer.
    
    Args:
        optimizer: SNN optimizer
        scheduler_type: Type of scheduler ('firing_rate', 'cosine', 'base')
        **kwargs: Additional scheduler arguments
        
    Returns:
        Configured learning rate scheduler
    """
    if scheduler_type == "firing_rate":
        return FiringRateScheduler(optimizer, **kwargs)
    elif scheduler_type == "cosine":
        return CosineAnnealingSpikingLR(optimizer, **kwargs)
    elif scheduler_type == "base":
        return SpikingLRScheduler(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def create_plasticity_scheduler(
    plasticity_rules: Dict[str, Any],
    scheduler_type: str = "exponential",
    **kwargs
) -> PlasticityScheduler:
    """
    Create plasticity parameter scheduler.
    
    Args:
        plasticity_rules: Dictionary of plasticity rules
        scheduler_type: Type of scheduling
        **kwargs: Additional scheduler arguments
        
    Returns:
        Configured plasticity scheduler
    """
    return PlasticityScheduler(plasticity_rules, scheduler_type, **kwargs)