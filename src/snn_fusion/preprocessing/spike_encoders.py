"""
Spike Encoding Strategies for Neuromorphic Computing

Implements various spike encoding methods for converting continuous signals
to discrete spike trains suitable for neuromorphic processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
import math


class PopulationEncoder(nn.Module):
    """
    Population coding encoder using overlapping receptive fields.
    Each neuron has a preferred value and Gaussian tuning curve.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_neurons: int = 100,
        min_value: float = -1.0,
        max_value: float = 1.0,
        sigma: float = 0.2,
        spike_rate: float = 100.0,  # Hz
        time_steps: int = 100,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.min_value = min_value
        self.max_value = max_value
        self.sigma = sigma
        self.spike_rate = spike_rate
        self.time_steps = time_steps
        
        # Create neuron centers uniformly distributed across value range
        centers = torch.linspace(min_value, max_value, num_neurons)
        self.register_buffer('centers', centers)
        
        # Learnable parameters
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)))
        self.log_max_rate = nn.Parameter(torch.log(torch.tensor(spike_rate)))
    
    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)
    
    @property
    def max_rate(self) -> torch.Tensor:
        return torch.exp(self.log_max_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using population coding.
        
        Args:
            x: Input tensor [B, Features]
            
        Returns:
            Spike tensor [B, Features, Neurons, Time]
        """
        batch_size, features = x.shape
        
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(-1)  # [B, F, 1]
        centers_expanded = self.centers.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
        
        # Gaussian tuning curves
        response = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * self.sigma ** 2))
        
        # Scale by maximum firing rate
        firing_rates = response * self.max_rate / 1000.0  # Convert Hz to prob per ms
        
        # Generate Poisson spike trains
        spikes = []
        for t in range(self.time_steps):
            random_vals = torch.rand_like(response)
            spike_t = (random_vals < firing_rates).float()
            spikes.append(spike_t)
        
        return torch.stack(spikes, dim=-1)  # [B, F, N, T]


class TemporalEncoder(nn.Module):
    """
    Temporal coding encoder where spike timing carries information.
    Lower values spike earlier, higher values spike later.
    """
    
    def __init__(
        self,
        input_dim: int,
        time_steps: int = 100,
        min_value: float = -1.0,
        max_value: float = 1.0,
        encoding_type: str = 'linear',  # 'linear', 'exponential', 'logarithmic'
        spike_precision: int = 1,  # Number of spikes per neuron
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.min_value = min_value
        self.max_value = max_value
        self.encoding_type = encoding_type
        self.spike_precision = spike_precision
    
    def _linear_timing(self, x: torch.Tensor) -> torch.Tensor:
        """Linear mapping from value to spike time."""
        # Normalize to [0, 1]
        x_norm = (x - self.min_value) / (self.max_value - self.min_value)
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        
        # Map to spike times (inverted: low values -> early spikes)
        spike_times = (1.0 - x_norm) * (self.time_steps - 1)
        return spike_times
    
    def _exponential_timing(self, x: torch.Tensor) -> torch.Tensor:
        """Exponential mapping for better precision at extremes."""
        x_norm = (x - self.min_value) / (self.max_value - self.min_value)
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        
        # Exponential mapping
        spike_times = (1.0 - torch.exp(-3 * x_norm)) * (self.time_steps - 1)
        return spike_times
    
    def _logarithmic_timing(self, x: torch.Tensor) -> torch.Tensor:
        """Logarithmic mapping for compressed dynamic range."""
        x_norm = (x - self.min_value) / (self.max_value - self.min_value)
        x_norm = torch.clamp(x_norm, 1e-6, 1.0)
        
        # Logarithmic mapping
        spike_times = (1.0 - torch.log(x_norm + 1e-6) / torch.log(torch.tensor(1e-6))) * (self.time_steps - 1)
        return spike_times
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using temporal coding.
        
        Args:
            x: Input tensor [B, Features]
            
        Returns:
            Spike tensor [B, Features, Time]
        """
        batch_size, features = x.shape
        
        # Choose encoding method
        if self.encoding_type == 'linear':
            spike_times = self._linear_timing(x)
        elif self.encoding_type == 'exponential':
            spike_times = self._exponential_timing(x)
        elif self.encoding_type == 'logarithmic':
            spike_times = self._logarithmic_timing(x)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        
        # Round to discrete time steps
        spike_times = torch.round(spike_times).long()
        spike_times = torch.clamp(spike_times, 0, self.time_steps - 1)
        
        # Create spike trains
        spikes = torch.zeros(batch_size, features, self.time_steps, device=x.device)
        
        # Set spikes at computed times
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, features)
        feature_indices = torch.arange(features).unsqueeze(0).expand(batch_size, -1)
        
        spikes[batch_indices, feature_indices, spike_times] = 1.0
        
        # Add multiple spikes if specified
        for i in range(1, self.spike_precision):
            offset_times = torch.clamp(spike_times + i, 0, self.time_steps - 1)
            spikes[batch_indices, feature_indices, offset_times] = 1.0
        
        return spikes


class RateEncoder(nn.Module):
    """
    Rate coding encoder where spike frequency represents information.
    Higher values produce higher spike rates.
    """
    
    def __init__(
        self,
        input_dim: int,
        time_steps: int = 100,
        max_rate: float = 100.0,  # Hz
        min_rate: float = 0.0,    # Hz
        noise_std: float = 0.0,
        refractory_period: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.noise_std = noise_std
        self.refractory_period = refractory_period
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using rate coding.
        
        Args:
            x: Input tensor [B, Features], assumed to be in [0, 1] range
            
        Returns:
            Spike tensor [B, Features, Time]
        """
        # Normalize input to [0, 1] if needed
        x_norm = torch.sigmoid(x)  # Soft normalization
        
        # Add noise if specified
        if self.noise_std > 0:
            noise = torch.randn_like(x_norm) * self.noise_std
            x_norm = torch.clamp(x_norm + noise, 0.0, 1.0)
        
        # Map to firing rates
        firing_rates = self.min_rate + x_norm * (self.max_rate - self.min_rate)
        spike_probs = firing_rates / 1000.0  # Convert Hz to probability per ms
        
        # Generate Poisson spike trains
        spikes = []
        refractory_counters = torch.zeros_like(x_norm)
        
        for t in range(self.time_steps):
            # Can only spike if not in refractory period
            can_spike = (refractory_counters == 0)
            
            # Generate spikes
            random_vals = torch.rand_like(x_norm)
            spike_t = (random_vals < spike_probs).float() * can_spike.float()
            
            # Update refractory counters
            refractory_counters = torch.maximum(
                refractory_counters - 1,
                torch.zeros_like(refractory_counters)
            )
            refractory_counters += spike_t * self.refractory_period
            
            spikes.append(spike_t)
        
        return torch.stack(spikes, dim=-1)


class LatencyEncoder(nn.Module):
    """
    Latency coding encoder where first spike time encodes information.
    Combines benefits of temporal and rate coding.
    """
    
    def __init__(
        self,
        input_dim: int,
        time_steps: int = 100,
        min_latency: int = 0,
        max_latency: Optional[int] = None,
        threshold: float = 0.5,
        noise_std: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.min_latency = min_latency
        self.max_latency = max_latency if max_latency is not None else time_steps - 1
        self.threshold = threshold
        self.noise_std = noise_std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using latency coding.
        
        Args:
            x: Input tensor [B, Features]
            
        Returns:
            Spike tensor [B, Features, Time]
        """
        batch_size, features = x.shape
        
        # Add noise for stochasticity
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
        else:
            x_noisy = x
        
        # Normalize to [0, 1]
        x_norm = torch.sigmoid(x_noisy)
        
        # Calculate latencies (higher values -> earlier spikes)
        latencies = self.min_latency + (1.0 - x_norm) * (self.max_latency - self.min_latency)
        latencies = torch.round(latencies).long()
        latencies = torch.clamp(latencies, self.min_latency, self.max_latency)
        
        # Only spike if input is above threshold
        should_spike = (x_norm > self.threshold)
        
        # Create spike trains
        spikes = torch.zeros(batch_size, features, self.time_steps, device=x.device)
        
        # Set first spikes at computed latencies
        for b in range(batch_size):
            for f in range(features):
                if should_spike[b, f]:
                    spikes[b, f, latencies[b, f]] = 1.0
        
        return spikes


class DeltaEncoder(nn.Module):
    """
    Delta encoder that generates spikes based on temporal changes.
    Useful for encoding sensor data with temporal dynamics.
    """
    
    def __init__(
        self,
        input_dim: int,
        threshold: float = 0.1,
        decay_factor: float = 0.9,
        spike_height: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.threshold = threshold
        self.decay_factor = decay_factor
        self.spike_height = spike_height
        
        # State variables
        self.register_buffer('prev_input', torch.zeros(1, input_dim))
        self.register_buffer('accumulator', torch.zeros(1, input_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input changes as spikes.
        
        Args:
            x: Input tensor [B, Features, Time] or [B, Features]
            
        Returns:
            Spike tensor [B, Features, Time]
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add time dimension
        
        batch_size, features, time_steps = x.shape
        
        # Initialize state for this batch
        if self.prev_input.size(0) != batch_size:
            self.prev_input = torch.zeros(batch_size, features, device=x.device)
            self.accumulator = torch.zeros(batch_size, features, device=x.device)
        
        spikes = []
        
        for t in range(time_steps):
            current_input = x[:, :, t]
            
            # Calculate difference
            delta = current_input - self.prev_input
            
            # Update accumulator
            self.accumulator = self.accumulator * self.decay_factor + delta
            
            # Generate spikes when threshold is exceeded
            pos_spikes = (self.accumulator > self.threshold).float() * self.spike_height
            neg_spikes = (self.accumulator < -self.threshold).float() * self.spike_height
            
            # Reset accumulator where spikes occurred
            self.accumulator = torch.where(
                torch.abs(self.accumulator) > self.threshold,
                torch.zeros_like(self.accumulator),
                self.accumulator
            )
            
            # Combine positive and negative spikes
            spike_output = pos_spikes - neg_spikes
            spikes.append(spike_output)
            
            # Update previous input
            self.prev_input = current_input.detach()
        
        return torch.stack(spikes, dim=-1)


class AdaptiveEncoder(nn.Module):
    """
    Adaptive spike encoder that adjusts encoding parameters based on input statistics.
    Provides dynamic range adaptation for varying input conditions.
    """
    
    def __init__(
        self,
        input_dim: int,
        base_encoder: str = 'rate',  # 'rate', 'temporal', 'population'
        adaptation_rate: float = 0.01,
        min_rate: float = 1.0,
        max_rate: float = 200.0,
        time_steps: int = 100,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.base_encoder = base_encoder
        self.adaptation_rate = adaptation_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.time_steps = time_steps
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(input_dim))
        self.register_buffer('running_var', torch.ones(input_dim))
        self.register_buffer('num_batches', torch.tensor(0))
        
        # Create base encoder
        if base_encoder == 'rate':
            self.encoder = RateEncoder(input_dim, time_steps, max_rate)
        elif base_encoder == 'temporal':
            self.encoder = TemporalEncoder(input_dim, time_steps)
        elif base_encoder == 'population':
            self.encoder = PopulationEncoder(input_dim, num_neurons=50, time_steps=time_steps)
        else:
            raise ValueError(f"Unknown base encoder: {base_encoder}")
    
    def _update_statistics(self, x: torch.Tensor) -> None:
        """Update running statistics for adaptation."""
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            momentum = 1.0 / (self.num_batches + 1).float()
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
            self.num_batches += 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode with adaptive parameters.
        
        Args:
            x: Input tensor [B, Features]
            
        Returns:
            Spike tensor (shape depends on base encoder)
        """
        # Update statistics
        self._update_statistics(x)
        
        # Normalize input using running statistics
        x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-6)
        
        # Clip to reasonable range
        x_clipped = torch.clamp(x_normalized, -3.0, 3.0)
        
        # Apply adaptive scaling based on input variance
        scale_factor = torch.clamp(1.0 / torch.sqrt(self.running_var + 1e-6), 0.1, 10.0)
        x_scaled = x_clipped * scale_factor.unsqueeze(0)
        
        return self.encoder(x_scaled)