"""
Plasticity and Learning Rules for Spiking Neural Networks

This module implements various forms of synaptic plasticity including STDP
(Spike-Timing Dependent Plasticity), reward-modulated plasticity, and other
bio-inspired learning mechanisms for neuromorphic computing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
import math


class STDPPlasticity(nn.Module):
    """
    Spike-Timing Dependent Plasticity (STDP) implementation.
    
    STDP modifies synaptic weights based on the relative timing of pre- and
    post-synaptic spikes. This implementation supports both additive and
    multiplicative STDP with customizable learning windows.
    """
    
    def __init__(
        self,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        weight_bounds: Tuple[float, float] = (0.0, 1.0),
        multiplicative: bool = False,
        symmetric: bool = False,
        trace_decay: float = 0.95
    ):
        """
        Initialize STDP plasticity rule.
        
        Args:
            tau_pre: Pre-synaptic trace time constant (ms)
            tau_post: Post-synaptic trace time constant (ms) 
            A_plus: Learning rate for potentiation (LTP)
            A_minus: Learning rate for depression (LTD)
            weight_bounds: (min_weight, max_weight) bounds
            multiplicative: Use multiplicative STDP instead of additive
            symmetric: Use symmetric learning window
            trace_decay: Exponential decay factor for spike traces
        """
        super().__init__()
        
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.weight_bounds = weight_bounds
        self.multiplicative = multiplicative
        self.symmetric = symmetric
        self.trace_decay = trace_decay
        
        # Register buffers for spike traces
        self.register_buffer('pre_trace', None)
        self.register_buffer('post_trace', None)
        self.register_buffer('weight_changes', None)
        
        # Learning window parameters
        self.register_buffer('dt_window', torch.linspace(-100, 100, 201))
        self._compute_learning_window()
        
    def _compute_learning_window(self):
        """Compute the STDP learning window."""
        dt = self.dt_window
        
        if self.symmetric:
            # Symmetric learning window
            window = torch.exp(-torch.abs(dt) / self.tau_pre)
            window = window * torch.sign(dt) * self.A_plus
        else:
            # Asymmetric learning window
            window = torch.zeros_like(dt)
            
            # Potentiation (dt > 0, post after pre)
            pos_mask = dt > 0
            window[pos_mask] = self.A_plus * torch.exp(-dt[pos_mask] / self.tau_post)
            
            # Depression (dt < 0, pre after post)
            neg_mask = dt < 0
            window[neg_mask] = -self.A_minus * torch.exp(dt[neg_mask] / self.tau_pre)
        
        self.register_buffer('learning_window', window)
    
    def reset_traces(self, batch_size: int, n_pre: int, n_post: int, device: torch.device):
        """Reset spike traces for new sequence."""
        self.pre_trace = torch.zeros(batch_size, n_pre, device=device)
        self.post_trace = torch.zeros(batch_size, n_post, device=device)
        self.weight_changes = torch.zeros(batch_size, n_pre, n_post, device=device)
    
    def update_traces(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """
        Update spike traces based on current spikes.
        
        Args:
            pre_spikes: Pre-synaptic spikes [batch, n_pre]
            post_spikes: Post-synaptic spikes [batch, n_post]
        """
        if self.pre_trace is None:
            self.reset_traces(
                pre_spikes.shape[0], pre_spikes.shape[1], 
                post_spikes.shape[1], pre_spikes.device
            )
        
        # Decay traces
        self.pre_trace *= self.trace_decay
        self.post_trace *= self.trace_decay
        
        # Add new spikes
        self.pre_trace += pre_spikes.float()
        self.post_trace += post_spikes.float()
    
    def compute_weight_update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute STDP weight updates.
        
        Args:
            pre_spikes: Pre-synaptic spikes [batch, n_pre]
            post_spikes: Post-synaptic spikes [batch, n_post]
            weights: Current synaptic weights [n_pre, n_post] or [batch, n_pre, n_post]
            
        Returns:
            weight_updates: Weight change matrix [batch, n_pre, n_post]
        """
        batch_size = pre_spikes.shape[0]
        n_pre = pre_spikes.shape[1]  
        n_post = post_spikes.shape[1]
        
        # Initialize traces if needed
        if self.pre_trace is None:
            self.reset_traces(batch_size, n_pre, n_post, pre_spikes.device)
        
        # Update traces
        self.update_traces(pre_spikes, post_spikes)
        
        # Compute weight updates using outer product of traces and spikes
        # LTP: post-synaptic spike with pre-synaptic trace
        ltp_update = torch.bmm(
            self.pre_trace.unsqueeze(-1),  # [batch, n_pre, 1]
            post_spikes.unsqueeze(1)       # [batch, 1, n_post]
        ) * self.A_plus
        
        # LTD: pre-synaptic spike with post-synaptic trace  
        ltd_update = torch.bmm(
            pre_spikes.unsqueeze(-1),      # [batch, n_pre, 1]
            self.post_trace.unsqueeze(1)   # [batch, 1, n_post]
        ) * (-self.A_minus)
        
        # Total update
        weight_update = ltp_update + ltd_update
        
        # Apply multiplicative scaling if enabled
        if self.multiplicative:
            if weights.dim() == 2:
                # Broadcast weights for batch processing
                weights_expanded = weights.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                weights_expanded = weights
                
            # Multiplicative STDP: scale by distance from bounds
            weight_update = torch.where(
                weight_update > 0,
                weight_update * (self.weight_bounds[1] - weights_expanded),
                weight_update * (weights_expanded - self.weight_bounds[0])
            )
        
        return weight_update
    
    def apply_bounds(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply weight bounds after update."""
        return torch.clamp(weights, self.weight_bounds[0], self.weight_bounds[1])
    
    def get_learning_curve(self, dt_range: Tuple[float, float] = (-100, 100)) -> Tuple[np.ndarray, np.ndarray]:
        """Get the STDP learning window for visualization."""
        dt = np.linspace(dt_range[0], dt_range[1], 1000)
        
        if self.symmetric:
            window = np.exp(-np.abs(dt) / self.tau_pre.item())
            window = window * np.sign(dt) * self.A_plus
        else:
            window = np.zeros_like(dt)
            
            # Potentiation
            pos_mask = dt > 0
            window[pos_mask] = self.A_plus * np.exp(-dt[pos_mask] / self.tau_post.item())
            
            # Depression
            neg_mask = dt < 0
            window[neg_mask] = -self.A_minus * np.exp(dt[neg_mask] / self.tau_pre.item())
        
        return dt, window


class RewardModulatedSTDP(STDPPlasticity):
    """
    Reward-Modulated STDP for reinforcement learning in spiking networks.
    
    Extends basic STDP with reward modulation, allowing the network to learn
    from delayed reward signals. Weight updates are modulated by reward
    prediction error.
    """
    
    def __init__(
        self,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        tau_reward: float = 1000.0,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        reward_baseline: float = 0.0,
        eligibility_decay: float = 0.99,
        **kwargs
    ):
        """
        Initialize reward-modulated STDP.
        
        Args:
            tau_reward: Reward trace time constant (ms)
            reward_baseline: Baseline reward for computing prediction error
            eligibility_decay: Decay factor for eligibility traces
            **kwargs: Additional arguments passed to STDPPlasticity
        """
        super().__init__(tau_pre, tau_post, A_plus, A_minus, **kwargs)
        
        self.tau_reward = tau_reward
        self.reward_baseline = reward_baseline
        self.eligibility_decay = eligibility_decay
        
        # Register buffers for eligibility traces and rewards
        self.register_buffer('eligibility_trace', None)
        self.register_buffer('reward_trace', None)
        self.register_buffer('reward_prediction_error', torch.tensor(0.0))
        
    def reset_traces(self, batch_size: int, n_pre: int, n_post: int, device: torch.device):
        """Reset all traces including eligibility traces."""
        super().reset_traces(batch_size, n_pre, n_post, device)
        self.eligibility_trace = torch.zeros(batch_size, n_pre, n_post, device=device)
        self.reward_trace = torch.zeros(batch_size, device=device)
    
    def update_eligibility(self, weight_update: torch.Tensor):
        """Update eligibility traces with STDP weight updates."""
        if self.eligibility_trace is None:
            batch_size, n_pre, n_post = weight_update.shape
            self.eligibility_trace = torch.zeros_like(weight_update)
        
        # Decay eligibility trace
        self.eligibility_trace *= self.eligibility_decay
        
        # Add new STDP updates to eligibility
        self.eligibility_trace += weight_update
    
    def update_reward_trace(self, reward: torch.Tensor):
        """
        Update reward trace and compute prediction error.
        
        Args:
            reward: Current reward signal [batch]
        """
        if self.reward_trace is None:
            self.reward_trace = torch.zeros_like(reward)
        
        # Update reward prediction error  
        self.reward_prediction_error = reward - self.reward_baseline
        
        # Update reward trace with exponential decay
        decay_factor = math.exp(-1.0 / self.tau_reward)
        self.reward_trace = decay_factor * self.reward_trace + reward
    
    def compute_modulated_update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        reward: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reward-modulated weight updates.
        
        Args:
            pre_spikes: Pre-synaptic spikes [batch, n_pre]
            post_spikes: Post-synaptic spikes [batch, n_post]
            weights: Current weights [n_pre, n_post] or [batch, n_pre, n_post]
            reward: Reward signal [batch] (optional)
            
        Returns:
            modulated_weight_update: [batch, n_pre, n_post]
        """
        # Compute basic STDP update
        stdp_update = self.compute_weight_update(pre_spikes, post_spikes, weights)
        
        # Update eligibility traces
        self.update_eligibility(stdp_update)
        
        # Update reward trace if reward provided
        if reward is not None:
            self.update_reward_trace(reward)
        
        # Modulate eligibility traces by reward prediction error
        if self.eligibility_trace is not None:
            # Expand reward prediction error for broadcasting
            rpe = self.reward_prediction_error.view(-1, 1, 1)
            modulated_update = self.eligibility_trace * rpe
        else:
            modulated_update = stdp_update
        
        return modulated_update


class STDPLearner(nn.Module):
    """
    High-level STDP learning coordinator for spiking neural networks.
    
    Manages STDP plasticity across multiple layers and connections in a
    spiking neural network, with support for different plasticity rules
    and learning schedules.
    """
    
    def __init__(
        self,
        network: nn.Module,
        plasticity_rules: Dict[str, Union[STDPPlasticity, RewardModulatedSTDP]],
        learning_rate: float = 1.0,
        enable_plasticity: bool = True
    ):
        """
        Initialize STDP learner.
        
        Args:
            network: The spiking neural network to train
            plasticity_rules: Dictionary mapping layer names to plasticity rules
            learning_rate: Global learning rate multiplier
            enable_plasticity: Whether plasticity is initially enabled
        """
        super().__init__()
        
        self.network = network
        self.plasticity_rules = nn.ModuleDict(plasticity_rules)
        self.learning_rate = learning_rate
        self.enable_plasticity = enable_plasticity
        
        # Track learning statistics
        self.weight_changes = {}
        self.learning_history = []
        
    def update_weights(
        self,
        layer_spikes: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        rewards: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Update network weights using STDP.
        
        Args:
            layer_spikes: Dict mapping layer names to (pre_spikes, post_spikes) tuples
            rewards: Optional reward signals for reward-modulated STDP
            
        Returns:
            weight_updates: Dictionary of weight updates applied
        """
        if not self.enable_plasticity:
            return {}
        
        weight_updates = {}
        
        for layer_name, (pre_spikes, post_spikes) in layer_spikes.items():
            if layer_name in self.plasticity_rules:
                plasticity_rule = self.plasticity_rules[layer_name]
                
                # Get current weights
                weights = None
                for name, param in self.network.named_parameters():
                    if layer_name in name and 'weight' in name:
                        weights = param.data
                        break
                
                if weights is None:
                    continue
                
                # Compute weight update
                if isinstance(plasticity_rule, RewardModulatedSTDP) and rewards is not None:
                    update = plasticity_rule.compute_modulated_update(
                        pre_spikes, post_spikes, weights, rewards
                    )
                else:
                    update = plasticity_rule.compute_weight_update(
                        pre_spikes, post_spikes, weights
                    )
                
                # Apply learning rate
                scaled_update = update * self.learning_rate
                
                # Apply weight update
                if weights.dim() == 2 and update.dim() == 3:
                    # Average across batch dimension
                    weights += scaled_update.mean(dim=0)
                else:
                    weights += scaled_update
                
                # Apply bounds
                weights = plasticity_rule.apply_bounds(weights)
                
                # Store update for monitoring
                weight_updates[layer_name] = scaled_update
                self.weight_changes[layer_name] = scaled_update.abs().mean().item()
        
        # Update learning history
        total_change = sum(self.weight_changes.values())
        self.learning_history.append(total_change)
        
        return weight_updates
    
    def reset_traces(self):
        """Reset all spike traces in plasticity rules."""
        for rule in self.plasticity_rules.values():
            if hasattr(rule, 'pre_trace'):
                rule.pre_trace = None
                rule.post_trace = None
                if hasattr(rule, 'eligibility_trace'):
                    rule.eligibility_trace = None
                    rule.reward_trace = None
    
    def set_learning_rate(self, lr: float):
        """Update the global learning rate."""
        self.learning_rate = lr
    
    def enable_learning(self, enable: bool = True):
        """Enable or disable plasticity."""
        self.enable_plasticity = enable
    
    def get_learning_statistics(self) -> Dict[str, float]:
        """Get current learning statistics."""
        stats = {
            'total_weight_change': sum(self.weight_changes.values()),
            'learning_rate': self.learning_rate,
            'plasticity_enabled': self.enable_plasticity,
        }
        
        # Add per-layer statistics
        for layer_name, change in self.weight_changes.items():
            stats[f'{layer_name}_weight_change'] = change
        
        return stats
    
    def save_learning_state(self, filepath: str):
        """Save plasticity state for resuming training."""
        state = {
            'plasticity_rules': {name: rule.state_dict() 
                               for name, rule in self.plasticity_rules.items()},
            'learning_rate': self.learning_rate,
            'enable_plasticity': self.enable_plasticity,
            'weight_changes': self.weight_changes,
            'learning_history': self.learning_history,
        }
        torch.save(state, filepath)
    
    def load_learning_state(self, filepath: str):
        """Load plasticity state for resuming training."""
        state = torch.load(filepath)
        
        for name, rule_state in state['plasticity_rules'].items():
            if name in self.plasticity_rules:
                self.plasticity_rules[name].load_state_dict(rule_state)
        
        self.learning_rate = state['learning_rate']
        self.enable_plasticity = state['enable_plasticity']
        self.weight_changes = state['weight_changes']
        self.learning_history = state['learning_history']


def create_stdp_learner(
    network: nn.Module,
    layer_names: List[str],
    stdp_config: Optional[Dict] = None,
    reward_modulated: bool = False
) -> STDPLearner:
    """
    Convenience function to create STDP learner for a network.
    
    Args:
        network: The spiking neural network
        layer_names: Names of layers to apply STDP to
        stdp_config: Configuration dictionary for STDP parameters
        reward_modulated: Whether to use reward-modulated STDP
        
    Returns:
        Configured STDPLearner instance
    """
    if stdp_config is None:
        stdp_config = {}
    
    # Default STDP parameters
    default_config = {
        'tau_pre': 20.0,
        'tau_post': 20.0,
        'A_plus': 0.01,
        'A_minus': 0.012,
        'weight_bounds': (0.0, 1.0),
    }
    default_config.update(stdp_config)
    
    # Create plasticity rules for each layer
    plasticity_rules = {}
    for layer_name in layer_names:
        if reward_modulated:
            rule = RewardModulatedSTDP(**default_config)
        else:
            rule = STDPPlasticity(**default_config)
        plasticity_rules[layer_name] = rule
    
    return STDPLearner(network, plasticity_rules)