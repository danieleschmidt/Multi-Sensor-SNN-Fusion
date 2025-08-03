"""
Neuromorphic Neuron Models

Implements biologically-inspired spiking neuron models optimized for 
neuromorphic hardware deployment and multi-modal processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any


class AdaptiveLIF(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire neuron with threshold adaptation.
    
    Implements biological adaptation mechanisms for stable liquid state dynamics
    and improved temporal processing capabilities.
    """
    
    def __init__(
        self,
        n_neurons: int,
        tau_mem: float = 20.0,
        tau_adapt: float = 100.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        beta: float = 0.95,
        adapt_increment: float = 0.05,
        dt: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Adaptive LIF neurons.
        
        Args:
            n_neurons: Number of neurons in population
            tau_mem: Membrane time constant (ms)
            tau_adapt: Adaptation time constant (ms)  
            v_threshold: Spike threshold voltage
            v_reset: Reset voltage after spike
            beta: Leak factor for membrane potential
            adapt_increment: Threshold adaptation increment per spike
            dt: Time step (ms)
            device: Computation device
        """
        super().__init__()
        
        self.n_neurons = n_neurons
        self.tau_mem = tau_mem
        self.tau_adapt = tau_adapt
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.beta = beta
        self.adapt_increment = adapt_increment
        self.dt = dt
        self.device = device or torch.device('cpu')
        
        # Initialize state variables
        self.register_buffer('v_mem', torch.zeros(n_neurons, device=self.device))
        self.register_buffer('v_adapt', torch.zeros(n_neurons, device=self.device))
        self.register_buffer('spike_count', torch.zeros(n_neurons, device=self.device))
        
        # Compute decay factors
        self.alpha_mem = torch.exp(torch.tensor(-dt / tau_mem, device=self.device))
        self.alpha_adapt = torch.exp(torch.tensor(-dt / tau_adapt, device=self.device))
        
    def forward(self, x_input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through adaptive LIF neurons.
        
        Args:
            x_input: Input current [batch_size, n_neurons] or [n_neurons]
            
        Returns:
            spikes: Binary spike tensor [batch_size, n_neurons] 
            states: Dictionary of internal states
        """
        if x_input.dim() == 1:
            x_input = x_input.unsqueeze(0)
            
        batch_size = x_input.shape[0]
        
        # Expand states for batch processing
        if self.v_mem.shape[0] != batch_size * self.n_neurons:
            self.v_mem = self.v_mem.unsqueeze(0).expand(batch_size, -1).contiguous().view(-1)
            self.v_adapt = self.v_adapt.unsqueeze(0).expand(batch_size, -1).contiguous().view(-1)
        
        # Membrane potential dynamics with leak
        self.v_mem = self.alpha_mem * self.v_mem + x_input.view(-1)
        
        # Adaptive threshold
        effective_threshold = self.v_threshold + self.v_adapt
        
        # Generate spikes
        spikes = (self.v_mem >= effective_threshold).float()
        
        # Reset membrane potential for spiking neurons
        reset_mask = spikes.bool()
        self.v_mem[reset_mask] = self.v_reset
        
        # Update adaptation variable
        self.v_adapt = self.alpha_adapt * self.v_adapt
        self.v_adapt[reset_mask] += self.adapt_increment
        
        # Update spike count
        self.spike_count += spikes
        
        # Reshape outputs
        spikes = spikes.view(batch_size, self.n_neurons)
        
        states = {
            'v_mem': self.v_mem.view(batch_size, self.n_neurons).clone(),
            'v_adapt': self.v_adapt.view(batch_size, self.n_neurons).clone(),
            'threshold': effective_threshold.view(batch_size, self.n_neurons).clone(),
            'spike_count': self.spike_count.view(batch_size, self.n_neurons).clone(),
        }
        
        return spikes, states
    
    def reset_state(self) -> None:
        """Reset all neuron states to initial values."""
        self.v_mem.fill_(0.0)
        self.v_adapt.fill_(0.0)
        self.spike_count.fill_(0.0)
        
    def get_firing_rates(self, time_window: float = 100.0) -> torch.Tensor:
        """
        Compute average firing rates over time window.
        
        Args:
            time_window: Time window for rate calculation (ms)
            
        Returns:
            firing_rates: Average firing rates (Hz)
        """
        return (self.spike_count * 1000.0) / time_window


class SpikingNeuron(nn.Module):
    """
    Base class for spiking neuron models with common functionality.
    """
    
    def __init__(
        self,
        n_neurons: int,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        refractory_period: int = 2,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize base spiking neuron.
        
        Args:
            n_neurons: Number of neurons
            v_threshold: Spike threshold
            v_reset: Reset voltage
            refractory_period: Refractory period in time steps
            device: Computation device
        """
        super().__init__()
        
        self.n_neurons = n_neurons
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.refractory_period = refractory_period
        self.device = device or torch.device('cpu')
        
        # Refractory state tracking
        self.register_buffer(
            'refractory_count', 
            torch.zeros(n_neurons, dtype=torch.int, device=self.device)
        )
        
    def apply_refractory(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Apply refractory period constraints.
        
        Args:
            spikes: Input spike tensor
            
        Returns:
            spikes: Spike tensor with refractory period applied
        """
        # Mask spikes during refractory period
        refractory_mask = self.refractory_count > 0
        spikes = spikes & ~refractory_mask
        
        # Update refractory counters
        self.refractory_count[spikes.bool()] = self.refractory_period
        self.refractory_count = torch.clamp(self.refractory_count - 1, min=0)
        
        return spikes
        
    def compute_spike_statistics(
        self, 
        spike_history: torch.Tensor,
        time_window: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive spike statistics.
        
        Args:
            spike_history: Historical spike data [time_steps, batch_size, n_neurons]
            time_window: Analysis window size
            
        Returns:
            stats: Dictionary of spike statistics
        """
        if spike_history.dim() != 3:
            raise ValueError("spike_history must have shape [time_steps, batch_size, n_neurons]")
            
        T, B, N = spike_history.shape
        
        # Firing rates
        firing_rates = spike_history.sum(dim=0) / (T * 0.001)  # Convert to Hz
        
        # Coefficient of variation for spike timing
        spike_times = []
        for b in range(B):
            for n in range(N):
                times = torch.where(spike_history[:, b, n] == 1)[0].float()
                if len(times) > 1:
                    isi = torch.diff(times)  # Inter-spike intervals
                    cv = torch.std(isi) / torch.mean(isi) if torch.mean(isi) > 0 else torch.tensor(0.0)
                    spike_times.append(cv)
                else:
                    spike_times.append(torch.tensor(0.0))
        
        cv_isi = torch.tensor(spike_times, device=self.device).view(B, N)
        
        # Synchrony index (population vector correlation)
        pop_activity = spike_history.sum(dim=2)  # [T, B]
        synchrony = torch.zeros(B, device=self.device)
        
        for b in range(B):
            activity = pop_activity[:, b]
            if activity.std() > 0:
                # Compute autocorrelation at lag 1
                activity_shifted = torch.roll(activity, 1)
                synchrony[b] = torch.corrcoef(torch.stack([activity, activity_shifted]))[0, 1]
        
        return {
            'firing_rates': firing_rates,
            'cv_isi': cv_isi,
            'synchrony': synchrony,
            'total_spikes': spike_history.sum(dim=0),
            'active_neurons': (spike_history.sum(dim=0) > 0).float().sum(dim=1),
        }


class CurrentBasedLIF(SpikingNeuron):
    """
    Current-based Leaky Integrate-and-Fire neuron for precise temporal dynamics.
    """
    
    def __init__(
        self,
        n_neurons: int,
        tau_mem: float = 20.0,
        resistance: float = 1.0,
        capacitance: float = 1.0,
        **kwargs
    ):
        """
        Initialize current-based LIF neuron.
        
        Args:
            n_neurons: Number of neurons
            tau_mem: Membrane time constant
            resistance: Membrane resistance
            capacitance: Membrane capacitance
            **kwargs: Additional arguments for base class
        """
        super().__init__(n_neurons, **kwargs)
        
        self.tau_mem = tau_mem
        self.resistance = resistance
        self.capacitance = capacitance
        
        # Membrane potential state
        self.register_buffer('v_mem', torch.zeros(n_neurons, device=self.device))
        
        # Leak factor
        self.alpha = torch.exp(torch.tensor(-1.0 / tau_mem, device=self.device))
        
    def forward(self, current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with current input.
        
        Args:
            current: Input current [batch_size, n_neurons]
            
        Returns:
            spikes: Output spikes
            v_mem: Membrane potential
        """
        if current.dim() == 1:
            current = current.unsqueeze(0)
            
        batch_size = current.shape[0]
        
        # Expand membrane potential for batch
        if self.v_mem.shape[0] != batch_size * self.n_neurons:
            self.v_mem = self.v_mem.unsqueeze(0).expand(batch_size, -1).contiguous().view(-1)
        
        # Current-based membrane dynamics
        dv = (-self.v_mem + self.resistance * current.view(-1)) / self.tau_mem
        self.v_mem += dv
        
        # Generate spikes
        spikes = (self.v_mem >= self.v_threshold).float()
        
        # Apply refractory period
        spikes = self.apply_refractory(spikes.view(batch_size, self.n_neurons))
        
        # Reset spiking neurons
        reset_mask = spikes.bool().view(-1)
        self.v_mem[reset_mask] = self.v_reset
        
        v_mem_out = self.v_mem.view(batch_size, self.n_neurons).clone()
        
        return spikes, v_mem_out