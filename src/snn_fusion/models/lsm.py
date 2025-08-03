"""
Liquid State Machine Implementation

Core reservoir computing architecture for temporal processing of multi-modal
sensory data with biologically-inspired dynamics and neuromorphic optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import warnings

from .neurons import AdaptiveLIF
from .readouts import LinearReadout


class LiquidStateMachine(nn.Module):
    """
    Liquid State Machine with adaptive spiking neurons and sparse connectivity.
    
    Implements reservoir computing paradigm optimized for asynchronous multi-modal
    sensor processing and ultra-low latency neuromorphic deployment.
    """
    
    def __init__(
        self,
        n_inputs: int,
        n_reservoir: int = 1000,
        n_outputs: int = 10,
        connectivity: float = 0.1,
        spectral_radius: float = 0.9,
        input_scaling: float = 1.0,
        leak_rate: float = 0.1,
        tau_mem: float = 20.0,
        tau_adapt: float = 100.0,
        neuron_type: str = "adaptive_lif",
        learning_rule: Optional[str] = None,
        plasticity_enabled: bool = False,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        """
        Initialize Liquid State Machine.
        
        Args:
            n_inputs: Number of input channels
            n_reservoir: Number of reservoir neurons
            n_outputs: Number of output classes/dimensions
            connectivity: Reservoir connection probability
            spectral_radius: Network stability parameter
            input_scaling: Input weight scaling factor
            leak_rate: Neuron leak rate
            tau_mem: Membrane time constant (ms)
            tau_adapt: Adaptation time constant (ms) 
            neuron_type: Type of neuron model
            learning_rule: Online learning rule (STDP, etc.)
            plasticity_enabled: Enable synaptic plasticity
            device: Computation device
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.connectivity = connectivity
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.plasticity_enabled = plasticity_enabled
        self.device = device or torch.device('cpu')
        
        # Set random seed for reproducible initialization
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize neuron population
        self.neurons = self._create_neurons(neuron_type, tau_mem, tau_adapt)
        
        # Create connectivity matrices
        self.W_input = self._create_input_weights()
        self.W_reservoir = self._create_reservoir_weights()
        
        # Initialize readout layer
        self.readout = LinearReadout(n_reservoir, n_outputs)
        
        # State tracking
        self.reservoir_states = []
        self.spike_history = []
        self.time_step = 0
        
        # Plasticity components
        if plasticity_enabled:
            self._initialize_plasticity(learning_rule)
            
        self.to(self.device)
        
    def _create_neurons(self, neuron_type: str, tau_mem: float, tau_adapt: float) -> nn.Module:
        """Create neuron population based on specified type."""
        if neuron_type == "adaptive_lif":
            return AdaptiveLIF(
                n_neurons=self.n_reservoir,
                tau_mem=tau_mem,
                tau_adapt=tau_adapt,
                device=self.device,
            )
        else:
            raise ValueError(f"Unsupported neuron type: {neuron_type}")
    
    def _create_input_weights(self) -> torch.Tensor:
        """Create sparse input weight matrix."""
        W_input = torch.zeros(self.n_reservoir, self.n_inputs, device=self.device)
        
        # Random sparse connectivity from inputs to reservoir
        n_connections = int(self.n_reservoir * self.n_inputs * self.connectivity)
        
        # Random indices for connections
        reservoir_indices = torch.randint(0, self.n_reservoir, (n_connections,))
        input_indices = torch.randint(0, self.n_inputs, (n_connections,))
        
        # Random weights with input scaling
        weights = (torch.randn(n_connections) * self.input_scaling).to(self.device)
        
        W_input[reservoir_indices, input_indices] = weights
        
        # Register as parameter for gradient computation
        self.register_parameter('W_input', nn.Parameter(W_input))
        
        return self.W_input
    
    def _create_reservoir_weights(self) -> torch.Tensor:
        """Create sparse recurrent reservoir weight matrix with spectral radius scaling."""
        W_reservoir = torch.zeros(self.n_reservoir, self.n_reservoir, device=self.device)
        
        # Create sparse random connectivity
        n_connections = int(self.n_reservoir * self.n_reservoir * self.connectivity)
        
        # Random indices (avoid self-connections)
        post_indices = torch.randint(0, self.n_reservoir, (n_connections,))
        pre_indices = torch.randint(0, self.n_reservoir, (n_connections,))
        
        # Remove self-connections
        mask = post_indices != pre_indices
        post_indices = post_indices[mask]
        pre_indices = pre_indices[mask]
        
        # Random weights (mix of excitatory and inhibitory)
        weights = torch.randn(len(post_indices), device=self.device)
        
        # Dale's principle: 80% excitatory, 20% inhibitory
        n_inhibitory = int(0.2 * self.n_reservoir)
        inhibitory_mask = pre_indices < n_inhibitory
        weights[inhibitory_mask] = -torch.abs(weights[inhibitory_mask])
        weights[~inhibitory_mask] = torch.abs(weights[~inhibitory_mask])
        
        W_reservoir[post_indices, pre_indices] = weights
        
        # Scale by spectral radius for stability
        eigenvals = torch.linalg.eigvals(W_reservoir).abs()
        current_spectral_radius = torch.max(eigenvals).real
        
        if current_spectral_radius > 0:
            W_reservoir = W_reservoir * (self.spectral_radius / current_spectral_radius)
        
        # Register as parameter
        self.register_parameter('W_reservoir', nn.Parameter(W_reservoir))
        
        return self.W_reservoir
    
    def _initialize_plasticity(self, learning_rule: Optional[str]) -> None:
        """Initialize synaptic plasticity mechanisms."""
        if learning_rule == "stdp":
            from ..training.plasticity import STDPPlasticity
            self.plasticity = STDPPlasticity(
                pre_neurons=self.n_reservoir,
                post_neurons=self.n_reservoir,
                tau_pre=20.0,
                tau_post=20.0,
                A_plus=0.01,
                A_minus=0.012,
            )
        elif learning_rule is not None:
            warnings.warn(f"Learning rule {learning_rule} not implemented, disabling plasticity")
            self.plasticity_enabled = False
    
    def forward(
        self, 
        x_input: torch.Tensor, 
        return_states: bool = False,
        reset_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through liquid state machine.
        
        Args:
            x_input: Input spike trains [batch_size, time_steps, n_inputs] or [time_steps, n_inputs]
            return_states: Whether to return internal states
            reset_state: Whether to reset neuron states before processing
            
        Returns:
            outputs: Readout predictions [batch_size, n_outputs]
            states: Optional dictionary of internal states
        """
        if reset_state:
            self.reset_state()
            
        # Handle input dimensions
        if x_input.dim() == 2:
            x_input = x_input.unsqueeze(0)  # Add batch dimension
            
        batch_size, time_steps, n_inputs = x_input.shape
        
        if n_inputs != self.n_inputs:
            raise ValueError(f"Input size {n_inputs} doesn't match expected {self.n_inputs}")
        
        # Initialize state storage
        reservoir_activities = []
        spike_trains = []
        
        # Process temporal sequence
        for t in range(time_steps):
            # Current input
            x_t = x_input[:, t, :]  # [batch_size, n_inputs]
            
            # Input current to reservoir
            input_current = torch.matmul(x_t, self.W_input.T)  # [batch_size, n_reservoir]
            
            # Previous reservoir activity (if not first time step)
            if t > 0:
                prev_spikes = spike_trains[-1]
                recurrent_current = torch.matmul(prev_spikes, self.W_reservoir.T)
                total_current = input_current + recurrent_current
            else:
                total_current = input_current
            
            # Neuron dynamics
            spikes, neuron_states = self.neurons(total_current)
            
            # Store states
            reservoir_activities.append(neuron_states['v_mem'])
            spike_trains.append(spikes)
            
            # Apply plasticity if enabled
            if self.plasticity_enabled and hasattr(self, 'plasticity'):
                if t > 0:
                    self.W_reservoir = self.plasticity.update_weights(
                        self.W_reservoir, 
                        spike_trains[-2], 
                        spikes
                    )
        
        # Stack temporal activities
        reservoir_states = torch.stack(reservoir_activities, dim=1)  # [batch_size, time_steps, n_reservoir]
        spike_history = torch.stack(spike_trains, dim=1)  # [batch_size, time_steps, n_reservoir]
        
        # Generate liquid state from final states (or temporal pooling)
        liquid_state = self._compute_liquid_state(reservoir_states, spike_history)
        
        # Readout computation
        outputs = self.readout(liquid_state)
        
        # Update internal tracking
        self.reservoir_states.append(reservoir_states.detach())
        self.spike_history.append(spike_history.detach())
        self.time_step += time_steps
        
        # Return states if requested
        states = None
        if return_states:
            states = {
                'reservoir_states': reservoir_states,
                'spike_history': spike_history,
                'liquid_state': liquid_state,
                'input_weights': self.W_input.clone(),
                'reservoir_weights': self.W_reservoir.clone(),
                'neuron_states': neuron_states,
            }
        
        return outputs, states
    
    def _compute_liquid_state(
        self, 
        reservoir_states: torch.Tensor, 
        spike_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute liquid state representation from reservoir dynamics.
        
        Args:
            reservoir_states: Reservoir membrane potentials [batch_size, time_steps, n_reservoir]
            spike_history: Spike trains [batch_size, time_steps, n_reservoir]
            
        Returns:
            liquid_state: Compact state representation [batch_size, n_reservoir]
        """
        # Multiple liquid state extraction strategies
        
        # 1. Final state
        final_state = reservoir_states[:, -1, :]
        
        # 2. Temporal mean of membrane potentials
        mean_potential = reservoir_states.mean(dim=1)
        
        # 3. Spike rate encoding
        spike_rates = spike_history.mean(dim=1)
        
        # 4. Exponentially weighted temporal integration
        time_steps = reservoir_states.shape[1]
        decay_weights = torch.exp(-torch.arange(time_steps, device=self.device, dtype=torch.float) / 10.0)
        decay_weights = decay_weights / decay_weights.sum()
        
        weighted_state = torch.sum(
            reservoir_states * decay_weights.view(1, -1, 1), 
            dim=1
        )
        
        # Combine multiple representations
        liquid_state = torch.cat([
            final_state * 0.4,
            mean_potential * 0.3, 
            spike_rates * 0.2,
            weighted_state * 0.1,
        ], dim=1)
        
        return liquid_state
    
    def reset_state(self) -> None:
        """Reset all internal states."""
        self.neurons.reset_state()
        self.reservoir_states.clear()
        self.spike_history.clear()
        self.time_step = 0
    
    def get_reservoir_statistics(self) -> Dict[str, float]:
        """Compute reservoir connectivity and dynamics statistics."""
        with torch.no_grad():
            # Connectivity statistics
            n_connections = (self.W_reservoir != 0).sum().item()
            actual_connectivity = n_connections / (self.n_reservoir ** 2)
            
            # Weight statistics
            reservoir_weights = self.W_reservoir[self.W_reservoir != 0]
            mean_weight = reservoir_weights.mean().item()
            std_weight = reservoir_weights.std().item()
            
            # Spectral radius
            eigenvals = torch.linalg.eigvals(self.W_reservoir).abs()
            current_spectral_radius = torch.max(eigenvals).real.item()
            
            # Input connectivity
            input_connections = (self.W_input != 0).sum().item()
            input_connectivity = input_connections / (self.n_reservoir * self.n_inputs)
            
            return {
                'actual_connectivity': actual_connectivity,
                'target_connectivity': self.connectivity,
                'mean_weight': mean_weight,
                'std_weight': std_weight,
                'spectral_radius': current_spectral_radius,
                'target_spectral_radius': self.spectral_radius,
                'input_connectivity': input_connectivity,
                'n_reservoir_connections': n_connections,
                'n_input_connections': input_connections,
            }
    
    def enable_plasticity(self, learning_rule: str = "stdp") -> None:
        """Enable synaptic plasticity during runtime."""
        self.plasticity_enabled = True
        if not hasattr(self, 'plasticity'):
            self._initialize_plasticity(learning_rule)
    
    def disable_plasticity(self) -> None:
        """Disable synaptic plasticity."""
        self.plasticity_enabled = False
    
    def save_liquid_states(self, filepath: str) -> None:
        """Save reservoir states and spike history to file."""
        if not self.reservoir_states:
            warnings.warn("No reservoir states to save")
            return
            
        states_data = {
            'reservoir_states': torch.cat(self.reservoir_states, dim=1),
            'spike_history': torch.cat(self.spike_history, dim=1),
            'time_steps': self.time_step,
            'config': {
                'n_reservoir': self.n_reservoir,
                'connectivity': self.connectivity,
                'spectral_radius': self.spectral_radius,
            }
        }
        
        torch.save(states_data, filepath)
    
    def load_liquid_states(self, filepath: str) -> None:
        """Load reservoir states from file."""
        states_data = torch.load(filepath, map_location=self.device)
        
        # Validate compatibility
        config = states_data.get('config', {})
        if config.get('n_reservoir', self.n_reservoir) != self.n_reservoir:
            raise ValueError("Saved states incompatible with current reservoir size")
        
        # Load states
        self.reservoir_states = [states_data['reservoir_states']]
        self.spike_history = [states_data['spike_history']]
        self.time_step = states_data.get('time_steps', 0)