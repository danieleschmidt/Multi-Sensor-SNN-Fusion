"""
Readout Layers for Spiking Neural Networks

Implements various readout mechanisms for converting reservoir states
to task-specific outputs with temporal processing capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math


class LinearReadout(nn.Module):
    """
    Linear readout layer for liquid state machine outputs.
    
    Converts high-dimensional reservoir states to task-specific predictions
    with optional temporal integration and regularization.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        dropout: float = 0.0,
        temporal_integration: bool = False,
        integration_window: int = 10,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize linear readout layer.
        
        Args:
            input_size: Size of input reservoir state
            output_size: Number of output classes/dimensions
            bias: Whether to include bias term
            dropout: Dropout probability for regularization
            temporal_integration: Enable temporal integration
            integration_window: Window size for temporal integration
            device: Computation device
        """
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.temporal_integration = temporal_integration
        self.integration_window = integration_window
        self.device = device or torch.device('cpu')
        
        # Linear transformation
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        
        # Regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Temporal integration buffer
        if temporal_integration:
            self.register_buffer(
                'output_history', 
                torch.zeros(integration_window, output_size, device=self.device)
            )
            self.history_index = 0
        
        # Initialize weights
        self._initialize_weights()
        
        self.to(self.device)
    
    def _initialize_weights(self) -> None:
        """Initialize readout weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, liquid_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through readout layer.
        
        Args:
            liquid_state: Reservoir state [batch_size, input_size]
            
        Returns:
            outputs: Task predictions [batch_size, output_size]
        """
        # Apply dropout for regularization
        liquid_state = self.dropout(liquid_state)
        
        # Linear transformation
        outputs = self.linear(liquid_state)
        
        # Temporal integration if enabled
        if self.temporal_integration and self.training:
            outputs = self._apply_temporal_integration(outputs)
        
        return outputs
    
    def _apply_temporal_integration(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal integration over output history.
        
        Args:
            outputs: Current outputs [batch_size, output_size]
            
        Returns:
            integrated_outputs: Temporally integrated outputs
        """
        batch_size = outputs.shape[0]
        
        # Update history buffer (circular buffer)
        self.output_history[self.history_index] = outputs.mean(dim=0)
        self.history_index = (self.history_index + 1) % self.integration_window
        
        # Compute exponentially weighted average
        weights = torch.exp(-torch.arange(self.integration_window, device=self.device, dtype=torch.float) / 3.0)
        weights = weights / weights.sum()
        
        # Integrate over history
        integrated = torch.sum(self.output_history * weights.view(-1, 1), dim=0)
        
        # Expand to batch size
        integrated_outputs = integrated.unsqueeze(0).expand(batch_size, -1)
        
        # Combine with current output
        alpha = 0.7  # Weight for current output
        outputs = alpha * outputs + (1 - alpha) * integrated_outputs
        
        return outputs
    
    def reset_history(self) -> None:
        """Reset temporal integration history."""
        if self.temporal_integration:
            self.output_history.fill_(0.0)
            self.history_index = 0


class TemporalReadout(nn.Module):
    """
    Temporal readout with attention mechanism for sequence prediction.
    
    Processes temporal sequences of reservoir states with learnable attention
    weights for improved temporal credit assignment.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_sequence_length: int = 100,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize temporal readout with attention.
        
        Args:
            input_size: Size of reservoir state at each timestep
            output_size: Number of output classes/dimensions
            hidden_size: Hidden layer size for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_sequence_length: Maximum sequence length for positional encoding
            device: Computation device
        """
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.device = device or torch.device('cpu')
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Multi-head attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding()
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )
        
        self.to(self.device)
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(self.max_sequence_length, self.hidden_size, device=self.device)
        position = torch.arange(0, self.max_sequence_length, device=self.device).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, device=self.device).float() *
            -(math.log(10000.0) / self.hidden_size)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        return pe
    
    def forward(
        self, 
        reservoir_sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal readout.
        
        Args:
            reservoir_sequence: Reservoir states [batch_size, seq_len, input_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            outputs: Final predictions [batch_size, output_size]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = reservoir_sequence.shape
        
        if seq_len > self.max_sequence_length:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_sequence_length}")
        
        # Project input to hidden dimension
        hidden_states = self.input_projection(reservoir_sequence)
        
        # Add positional encoding
        hidden_states = hidden_states + self.pe[:seq_len].unsqueeze(0)
        
        # Multi-head attention
        attended_states, attention_weights = self.multihead_attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=attention_mask,
            need_weights=True,
        )
        
        # Layer normalization
        attended_states = self.layer_norm(attended_states)
        
        # Global average pooling over sequence
        if attention_mask is not None:
            # Masked pooling
            mask = (~attention_mask).float().unsqueeze(-1)
            pooled_state = (attended_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled_state = attended_states.mean(dim=1)
        
        # Final output prediction
        outputs = self.output_mlp(pooled_state)
        
        return outputs, attention_weights


class SpikeRateReadout(nn.Module):
    """
    Spike rate-based readout for neuromorphic hardware compatibility.
    
    Converts spike trains to rate-based representations suitable for
    traditional machine learning tasks.
    """
    
    def __init__(
        self,
        n_neurons: int,
        n_outputs: int,
        time_window: float = 50.0,
        smoothing_kernel: str = "exponential",
        tau_decay: float = 10.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize spike rate readout.
        
        Args:
            n_neurons: Number of reservoir neurons
            n_outputs: Number of output classes
            time_window: Time window for rate computation (ms)
            smoothing_kernel: Smoothing kernel type
            tau_decay: Decay time constant for exponential kernel
            device: Computation device
        """
        super().__init__()
        
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.time_window = time_window
        self.smoothing_kernel = smoothing_kernel
        self.tau_decay = tau_decay
        self.device = device or torch.device('cpu')
        
        # Rate computation components
        self.rate_computer = self._create_rate_computer()
        
        # Linear classifier on rates
        self.classifier = nn.Linear(n_neurons, n_outputs)
        
        self.to(self.device)
    
    def _create_rate_computer(self) -> nn.Module:
        """Create rate computation module."""
        if self.smoothing_kernel == "exponential":
            return ExponentialSmoother(self.tau_decay, self.device)
        elif self.smoothing_kernel == "gaussian":
            return GaussianSmoother(self.tau_decay, self.device)
        else:
            raise ValueError(f"Unknown smoothing kernel: {self.smoothing_kernel}")
    
    def forward(self, spike_trains: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spike rate readout.
        
        Args:
            spike_trains: Input spikes [batch_size, time_steps, n_neurons]
            
        Returns:
            outputs: Rate-based predictions [batch_size, n_outputs]
        """
        # Compute smoothed firing rates
        firing_rates = self.rate_computer(spike_trains)
        
        # Classify based on rates
        outputs = self.classifier(firing_rates)
        
        return outputs


class ExponentialSmoother(nn.Module):
    """Exponential smoothing for spike rate computation."""
    
    def __init__(self, tau_decay: float, device: torch.device):
        super().__init__()
        self.tau_decay = tau_decay
        self.device = device
        
    def forward(self, spike_trains: torch.Tensor) -> torch.Tensor:
        """Apply exponential smoothing to spike trains."""
        batch_size, time_steps, n_neurons = spike_trains.shape
        
        # Create decay kernel
        decay_weights = torch.exp(-torch.arange(time_steps, device=self.device, dtype=torch.float) / self.tau_decay)
        decay_weights = decay_weights / decay_weights.sum()
        
        # Apply convolution for smoothing
        smoothed_rates = torch.zeros(batch_size, n_neurons, device=self.device)
        
        for b in range(batch_size):
            for n in range(n_neurons):
                spike_train = spike_trains[b, :, n]
                smoothed_rates[b, n] = torch.sum(spike_train * decay_weights)
        
        return smoothed_rates


class GaussianSmoother(nn.Module):
    """Gaussian smoothing for spike rate computation."""
    
    def __init__(self, sigma: float, device: torch.device):
        super().__init__()
        self.sigma = sigma
        self.device = device
        
    def forward(self, spike_trains: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing to spike trains."""
        batch_size, time_steps, n_neurons = spike_trains.shape
        
        # Create Gaussian kernel
        x = torch.arange(time_steps, device=self.device, dtype=torch.float)
        center = time_steps // 2
        gaussian_kernel = torch.exp(-0.5 * ((x - center) / self.sigma) ** 2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        
        # Apply convolution
        smoothed_rates = torch.zeros(batch_size, n_neurons, device=self.device)
        
        for b in range(batch_size):
            for n in range(n_neurons):
                spike_train = spike_trains[b, :, n]
                smoothed_rates[b, n] = torch.sum(spike_train * gaussian_kernel)
        
        return smoothed_rates