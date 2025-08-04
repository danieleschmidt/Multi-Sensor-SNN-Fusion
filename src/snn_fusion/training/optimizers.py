"""
Specialized Optimizers for Spiking Neural Networks

This module implements optimizers specifically designed for training spiking neural networks,
including spike-aware optimizers, neuromorphic-friendly optimizers, and adaptive optimizers
that account for the sparse and temporal nature of spike-based computation.
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import math
from typing import Dict, List, Optional, Tuple, Any, Iterator
import numpy as np


class SpikingOptimizer(Optimizer):
    """
    Base optimizer for spiking neural networks.
    
    Provides common functionality for handling sparse gradients and
    temporal dynamics in spiking neural networks.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        spike_threshold_adaptation: bool = True,
        gradient_clipping: Optional[float] = None
    ):
        """
        Initialize spiking optimizer base class.
        
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            weight_decay: L2 weight decay coefficient
            spike_threshold_adaptation: Whether to adapt spike thresholds
            gradient_clipping: Gradient clipping value (optional)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            spike_threshold_adaptation=spike_threshold_adaptation,
            gradient_clipping=gradient_clipping
        )
        super().__init__(params, defaults)
        
        # Statistics tracking
        self.gradient_stats = {}
        self.weight_stats = {}
        self.spike_stats = {}
    
    def _clip_gradients(self, params: Iterator[torch.Tensor]) -> float:
        """Clip gradients and return gradient norm."""
        total_norm = 0.0
        
        for p in params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** (1. / 2)
        
        if self.defaults['gradient_clipping'] is not None:
            clip_value = self.defaults['gradient_clipping']
            torch.nn.utils.clip_grad_norm_(params, clip_value)
        
        return total_norm
    
    def _update_statistics(self, group: Dict[str, Any]) -> None:
        """Update optimizer statistics for monitoring."""
        for p in group['params']:
            if p.grad is not None:
                # Gradient statistics
                grad_norm = p.grad.norm().item()
                grad_sparsity = (p.grad == 0).float().mean().item()
                
                # Weight statistics
                weight_norm = p.data.norm().item()
                weight_mean = p.data.mean().item()
                weight_std = p.data.std().item()
                
                param_id = id(p)
                self.gradient_stats[param_id] = {
                    'norm': grad_norm,
                    'sparsity': grad_sparsity
                }
                self.weight_stats[param_id] = {
                    'norm': weight_norm,
                    'mean': weight_mean,
                    'std': weight_std
                }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics for monitoring."""
        return {
            'gradient_stats': self.gradient_stats,
            'weight_stats': self.weight_stats,
            'spike_stats': self.spike_stats
        }


class NeuromorphicOptimizer(SpikingOptimizer):
    """
    Neuromorphic-friendly optimizer with sparse gradient handling.
    
    Designed for neuromorphic hardware deployment with optimizations for
    sparse computations and event-driven processing.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        sparsity_threshold: float = 0.01,
        momentum_sparsification: bool = True,
        **kwargs
    ):
        """
        Initialize neuromorphic optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            betas: Exponential decay rates for moment estimates
            eps: Numerical stability constant
            weight_decay: L2 weight decay
            sparsity_threshold: Threshold for sparsifying momentum
            momentum_sparsification: Whether to sparsify momentum updates
        """
        self.betas = betas
        self.eps = eps
        self.sparsity_threshold = sparsity_threshold
        self.momentum_sparsification = momentum_sparsification
        
        super().__init__(params, lr, weight_decay, **kwargs)
    
    def step(self, closure=None):
        """Perform optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas'] if 'betas' in group else self.betas
                
                state['step'] += 1
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Update exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Sparsify momentum if enabled
                if self.momentum_sparsification:
                    # Zero out small momentum values
                    momentum_mask = torch.abs(exp_avg) > self.sparsity_threshold
                    exp_avg.mul_(momentum_mask.float())
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute step size
                step_size = group['lr'] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)
                
                # Update parameters
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(self.eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Apply sparsification to weights if threshold is set
                if hasattr(group, 'weight_sparsity_threshold'):
                    weight_mask = torch.abs(p.data) > group['weight_sparsity_threshold']
                    p.data.mul_(weight_mask.float())
            
            # Update statistics
            self._update_statistics(group)
        
        return loss


class AdaptiveSpikingOptimizer(SpikingOptimizer):
    """
    Adaptive optimizer that adjusts learning rates based on spike statistics.
    
    Dynamically adapts learning rates based on firing rates, spike patterns,
    and temporal dynamics of the spiking neural network.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        adaptation_rate: float = 0.01,
        target_firing_rate: float = 0.1,
        firing_rate_window: int = 100,
        lr_bounds: Tuple[float, float] = (1e-6, 1e-1),
        **kwargs
    ):
        """
        Initialize adaptive spiking optimizer.
        
        Args:
            params: Model parameters
            lr: Initial learning rate
            adaptation_rate: Rate of learning rate adaptation
            target_firing_rate: Target average firing rate
            firing_rate_window: Window size for firing rate estimation
            lr_bounds: (min_lr, max_lr) bounds for learning rate
        """
        self.adaptation_rate = adaptation_rate
        self.target_firing_rate = target_firing_rate
        self.firing_rate_window = firing_rate_window
        self.lr_bounds = lr_bounds
        
        super().__init__(params, lr, **kwargs)
        
        # Initialize firing rate tracking
        self.firing_rate_history = []
        self.lr_history = []
        self.adaptation_step = 0
    
    def update_firing_rates(self, model_outputs: torch.Tensor) -> None:
        """
        Update firing rate statistics from model outputs.
        
        Args:
            model_outputs: Model spike outputs [batch, time, neurons]
        """
        if model_outputs.dim() >= 3:
            # Compute average firing rate across batch and time
            firing_rate = model_outputs.mean().item()
        else:
            # For non-temporal outputs, use mean activation
            firing_rate = model_outputs.mean().item()
        
        self.firing_rate_history.append(firing_rate)
        
        # Keep only recent history
        if len(self.firing_rate_history) > self.firing_rate_window:
            self.firing_rate_history.pop(0)
    
    def _adapt_learning_rate(self) -> None:
        """Adapt learning rate based on firing rate statistics."""
        if len(self.firing_rate_history) < 10:  # Need minimum history
            return
        
        # Compute average firing rate
        avg_firing_rate = np.mean(self.firing_rate_history[-10:])
        
        # Compute adaptation signal
        firing_rate_error = self.target_firing_rate - avg_firing_rate
        
        # Adapt learning rate
        for group in self.param_groups:
            current_lr = group['lr']
            
            # Proportional adaptation
            lr_adjustment = self.adaptation_rate * firing_rate_error
            new_lr = current_lr * (1 + lr_adjustment)
            
            # Apply bounds
            new_lr = max(self.lr_bounds[0], min(self.lr_bounds[1], new_lr))
            
            group['lr'] = new_lr
            
        self.lr_history.append(group['lr'])
        self.adaptation_step += 1
    
    def step(self, closure=None, model_outputs=None):
        """
        Perform optimization step with adaptive learning rate.
        
        Args:
            closure: Optional closure for computing loss
            model_outputs: Model outputs for firing rate adaptation
        """
        # Update firing rates if outputs provided
        if model_outputs is not None:
            self.update_firing_rates(model_outputs)
            self._adapt_learning_rate()
        
        # Perform standard optimization step
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            # Clip gradients
            params_with_grad = [p for p in group['params'] if p.grad is not None]
            if params_with_grad:
                grad_norm = self._clip_gradients(params_with_grad)
            
            # Update parameters
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Simple gradient descent with adaptive learning rate
                p.data.add_(grad, alpha=-group['lr'])
            
            # Update statistics
            self._update_statistics(group)
        
        return loss
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptation-specific statistics."""
        return {
            'current_lr': self.param_groups[0]['lr'] if self.param_groups else 0.0,
            'firing_rate_history': self.firing_rate_history[-10:],
            'lr_history': self.lr_history[-10:],
            'average_firing_rate': np.mean(self.firing_rate_history) if self.firing_rate_history else 0.0,
            'target_firing_rate': self.target_firing_rate,
            'adaptation_step': self.adaptation_step
        }


class SurrogateGradientOptimizer(SpikingOptimizer):
    """
    Optimizer designed for surrogate gradient training of SNNs.
    
    Handles the specific challenges of training with surrogate gradients,
    including gradient scaling and surrogate function adaptation.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        surrogate_scale: float = 1.0,
        surrogate_beta: float = 1.0,
        gradient_scaling: str = "adaptive",
        **kwargs
    ):
        """
        Initialize surrogate gradient optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            surrogate_scale: Scaling factor for surrogate gradients
            surrogate_beta: Beta parameter for surrogate function
            gradient_scaling: Gradient scaling method ('fixed', 'adaptive', 'none')
        """
        self.surrogate_scale = surrogate_scale
        self.surrogate_beta = surrogate_beta
        self.gradient_scaling = gradient_scaling
        
        super().__init__(params, lr, **kwargs)
        
        # Initialize surrogate adaptation
        self.surrogate_adaptation_history = []
    
    def _scale_surrogate_gradients(self, group: Dict[str, Any]) -> None:
        """Scale surrogate gradients based on spike statistics."""
        if self.gradient_scaling == "none":
            return
        
        for p in group['params']:
            if p.grad is None:
                continue
            
            if self.gradient_scaling == "adaptive":
                # Adaptive scaling based on gradient magnitude
                grad_norm = p.grad.norm()
                if grad_norm > 0:
                    scale_factor = self.surrogate_scale / (1 + grad_norm)
                    p.grad.mul_(scale_factor)
            elif self.gradient_scaling == "fixed":
                # Fixed scaling
                p.grad.mul_(self.surrogate_scale)
    
    def step(self, closure=None):
        """Perform optimization step with surrogate gradient handling."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            # Scale surrogate gradients
            self._scale_surrogate_gradients(group)
            
            # Clip gradients
            params_with_grad = [p for p in group['params'] if p.grad is not None]
            if params_with_grad:
                grad_norm = self._clip_gradients(params_with_grad)
            
            # Update parameters (simple SGD for now)
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Update parameters
                p.data.add_(grad, alpha=-group['lr'])
            
            # Update statistics
            self._update_statistics(group)
        
        return loss


# Factory function for creating optimizers

def create_snn_optimizer(
    optimizer_type: str,
    model_parameters,
    lr: float = 1e-3,
    **kwargs
) -> SpikingOptimizer:
    """
    Factory function to create SNN-specific optimizers.
    
    Args:
        optimizer_type: Type of optimizer ('neuromorphic', 'adaptive', 'surrogate')
        model_parameters: Model parameters to optimize
        lr: Learning rate
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Configured SNN optimizer
    """
    if optimizer_type == "neuromorphic":
        return NeuromorphicOptimizer(model_parameters, lr=lr, **kwargs)
    elif optimizer_type == "adaptive":
        return AdaptiveSpikingOptimizer(model_parameters, lr=lr, **kwargs)
    elif optimizer_type == "surrogate":
        return SurrogateGradientOptimizer(model_parameters, lr=lr, **kwargs)
    elif optimizer_type == "base":
        return SpikingOptimizer(model_parameters, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


# Utility functions

def compute_gradient_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient sparsity for all model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping parameter names to sparsity values
    """
    sparsity_dict = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_elements = param.grad.numel()
            zero_elements = (param.grad == 0).sum().item()
            sparsity = zero_elements / total_elements
            sparsity_dict[name] = sparsity
    
    return sparsity_dict


def get_optimizer_recommendations(
    model_type: str,
    hardware_target: str = "gpu",
    temporal_length: int = 100
) -> Dict[str, Any]:
    """
    Get optimizer recommendations based on model and hardware constraints.
    
    Args:
        model_type: Type of SNN model
        hardware_target: Target hardware ('gpu', 'loihi', 'akida')
        temporal_length: Length of temporal sequences
        
    Returns:
        Dictionary of recommended optimizer settings
    """
    recommendations = {}
    
    if hardware_target == "loihi":
        recommendations.update({
            'optimizer_type': 'neuromorphic',
            'lr': 1e-3,
            'sparsity_threshold': 0.01,
            'momentum_sparsification': True,
            'gradient_clipping': 1.0
        })
    elif hardware_target == "akida":
        recommendations.update({
            'optimizer_type': 'surrogate',
            'lr': 5e-4,
            'surrogate_scale': 0.5,
            'gradient_scaling': 'adaptive'
        })
    else:  # GPU
        recommendations.update({
            'optimizer_type': 'adaptive',
            'lr': 1e-3,
            'adaptation_rate': 0.01,
            'firing_rate_window': min(temporal_length, 100)
        })
    
    # Adjust for model type
    if "hierarchical" in model_type.lower():
        recommendations['lr'] *= 0.5  # More conservative for complex models
    
    return recommendations