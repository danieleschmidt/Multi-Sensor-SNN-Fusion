"""
Neuromorphic Optimization Algorithms

Implements specialized optimization algorithms for neuromorphic computing,
including spike-based gradient descent, STDP learning, and hardware-aware optimization.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class OptimizationStrategy(Enum):
    """Optimization strategy types."""
    SPIKE_BASED = "spike_based"
    STDP = "stdp"  
    HARDWARE_AWARE = "hardware_aware"
    ENERGY_EFFICIENT = "energy_efficient"
    LATENCY_OPTIMIZED = "latency_optimized"


@dataclass
class OptimizationConfig:
    """Configuration for neuromorphic optimization."""
    strategy: OptimizationStrategy
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 1e-4
    spike_threshold: float = 1.0
    adaptation_rate: float = 0.01
    hardware_constraints: Optional[Dict[str, Any]] = None
    energy_budget: Optional[float] = None
    latency_target: Optional[float] = None


@dataclass
class OptimizationResult:
    """Result of optimization step."""
    loss: float
    gradients: Dict[str, np.ndarray]
    spike_counts: Dict[str, int]
    energy_consumption: float
    latency_ms: float
    convergence_metric: float
    hardware_utilization: Dict[str, float]


class NeuromorphicOptimizer(ABC):
    """
    Abstract base class for neuromorphic optimization algorithms.
    
    Provides common interface for spike-based, STDP, and hardware-aware
    optimization strategies for neuromorphic neural networks.
    """
    
    def __init__(
        self,
        config: OptimizationConfig,
        model_parameters: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize optimizer.
        
        Args:
            config: Optimization configuration
            model_parameters: Model parameters to optimize
        """
        self.config = config
        self.model_parameters = model_parameters or {}
        self.logger = logging.getLogger(__name__)
        
        # Optimization state
        self.step_count = 0
        self.momentum_buffers: Dict[str, np.ndarray] = {}
        self.adaptation_states: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.loss_history: List[float] = []
        self.energy_history: List[float] = []
        self.convergence_history: List[float] = []
        
        self.logger.info(f"Initialized {self.__class__.__name__} with strategy: {config.strategy}")
    
    @abstractmethod
    def compute_gradients(
        self,
        loss: float,
        spike_data: Dict[str, np.ndarray],
        target_spikes: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute gradients based on spike data and loss.
        
        Args:
            loss: Current loss value
            spike_data: Spike activity data
            target_spikes: Target spike patterns (optional)
            
        Returns:
            Gradients for each parameter
        """
        pass
    
    @abstractmethod
    def update_parameters(
        self,
        gradients: Dict[str, np.ndarray],
        learning_rate: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Update model parameters using computed gradients.
        
        Args:
            gradients: Parameter gradients
            learning_rate: Optional learning rate override
            
        Returns:
            Optimization result
        """
        pass
    
    def step(
        self,
        loss: float,
        spike_data: Dict[str, np.ndarray],
        target_spikes: Optional[Dict[str, np.ndarray]] = None,
    ) -> OptimizationResult:
        """
        Perform single optimization step.
        
        Args:
            loss: Current loss value
            spike_data: Spike activity data
            target_spikes: Target spike patterns (optional)
            
        Returns:
            Optimization result
        """
        try:
            # Compute gradients
            gradients = self.compute_gradients(loss, spike_data, target_spikes)
            
            # Update parameters
            result = self.update_parameters(gradients)
            
            # Update tracking
            self.step_count += 1
            self.loss_history.append(loss)
            self.energy_history.append(result.energy_consumption)
            self.convergence_history.append(result.convergence_metric)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization step failed: {e}")
            raise
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics and performance metrics."""
        if not self.loss_history:
            return {}
        
        return {
            'total_steps': self.step_count,
            'current_loss': self.loss_history[-1],
            'loss_reduction': self.loss_history[0] - self.loss_history[-1] if len(self.loss_history) > 1 else 0,
            'average_energy': np.mean(self.energy_history) if self.energy_history else 0,
            'convergence_trend': np.mean(self.convergence_history[-10:]) if len(self.convergence_history) >= 10 else 0,
            'parameter_count': sum(param.size for param in self.model_parameters.values()),
            'strategy': self.config.strategy.value,
        }


class SpikingGradientDescent(NeuromorphicOptimizer):
    """
    Spike-based gradient descent optimizer for neuromorphic networks.
    
    Implements gradient computation based on spike timing and rates,
    optimized for spiking neural network training.
    """
    
    def __init__(
        self,
        config: OptimizationConfig,
        model_parameters: Optional[Dict[str, np.ndarray]] = None,
        spike_window: float = 10.0,  # ms
        trace_decay: float = 0.95,
    ):
        """
        Initialize spiking gradient descent optimizer.
        
        Args:
            config: Optimization configuration
            model_parameters: Model parameters
            spike_window: Spike integration window in ms
            trace_decay: Eligibility trace decay factor
        """
        super().__init__(config, model_parameters)
        
        self.spike_window = spike_window
        self.trace_decay = trace_decay
        
        # Eligibility traces for spike-based learning
        self.eligibility_traces: Dict[str, np.ndarray] = {}
        
        # Spike statistics
        self.spike_statistics: Dict[str, Dict[str, float]] = {}
    
    def compute_gradients(
        self,
        loss: float,
        spike_data: Dict[str, np.ndarray],
        target_spikes: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute spike-based gradients using eligibility traces.
        
        Args:
            loss: Current loss value
            spike_data: Spike timing data for each layer
            target_spikes: Target spike patterns
            
        Returns:
            Gradients for each parameter
        """
        gradients = {}
        
        try:
            for param_name, param_value in self.model_parameters.items():
                # Initialize eligibility trace if needed
                if param_name not in self.eligibility_traces:
                    self.eligibility_traces[param_name] = np.zeros_like(param_value)
                
                # Get spike data for this parameter's layer
                layer_spikes = spike_data.get(param_name, np.array([]))
                
                if layer_spikes.size == 0:
                    gradients[param_name] = np.zeros_like(param_value)
                    continue
                
                # Compute spike-based gradient
                gradient = self._compute_spike_gradient(
                    param_value, layer_spikes, target_spikes
                )
                
                # Update eligibility trace
                self.eligibility_traces[param_name] = (
                    self.trace_decay * self.eligibility_traces[param_name] + 
                    gradient
                )
                
                # Final gradient includes trace and loss signal
                gradients[param_name] = loss * self.eligibility_traces[param_name]
                
                # Update spike statistics
                self._update_spike_statistics(param_name, layer_spikes)
            
            return gradients
            
        except Exception as e:
            self.logger.error(f"Failed to compute spike gradients: {e}")
            raise
    
    def update_parameters(
        self,
        gradients: Dict[str, np.ndarray],
        learning_rate: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Update parameters using spike-based gradients.
        
        Args:
            gradients: Computed gradients
            learning_rate: Learning rate override
            
        Returns:
            Optimization result
        """
        lr = learning_rate or self.config.learning_rate
        total_spike_count = 0
        energy_consumption = 0.0
        
        try:
            for param_name, gradient in gradients.items():
                if param_name not in self.model_parameters:
                    continue
                
                # Apply momentum if configured
                if self.config.momentum > 0:
                    if param_name not in self.momentum_buffers:
                        self.momentum_buffers[param_name] = np.zeros_like(gradient)
                    
                    self.momentum_buffers[param_name] = (
                        self.config.momentum * self.momentum_buffers[param_name] +
                        gradient
                    )
                    update = self.momentum_buffers[param_name]
                else:
                    update = gradient
                
                # Apply weight decay
                if self.config.weight_decay > 0:
                    update += self.config.weight_decay * self.model_parameters[param_name]
                
                # Update parameters
                self.model_parameters[param_name] -= lr * update
                
                # Calculate spike count and energy
                layer_stats = self.spike_statistics.get(param_name, {})
                total_spike_count += layer_stats.get('total_spikes', 0)
                energy_consumption += layer_stats.get('energy', 0)
            
            # Calculate convergence metric
            convergence_metric = self._calculate_convergence_metric(gradients)
            
            # Estimate latency based on spike processing
            latency_ms = self._estimate_latency(total_spike_count)
            
            return OptimizationResult(
                loss=self.loss_history[-1] if self.loss_history else 0,
                gradients=gradients,
                spike_counts={'total': total_spike_count},
                energy_consumption=energy_consumption,
                latency_ms=latency_ms,
                convergence_metric=convergence_metric,
                hardware_utilization={},
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update parameters: {e}")
            raise
    
    def _compute_spike_gradient(
        self,
        param_value: np.ndarray,
        layer_spikes: np.ndarray,
        target_spikes: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute gradient based on spike timing and rates."""
        try:
            # Simple spike-rate based gradient
            if layer_spikes.size == 0:
                return np.zeros_like(param_value)
            
            # Calculate spike rate
            spike_rate = len(layer_spikes) / self.spike_window
            
            # Target spike rate (or adaptive target)
            target_rate = 0.1 if target_spikes is None else len(target_spikes) / self.spike_window
            
            # Gradient proportional to rate difference
            rate_error = spike_rate - target_rate
            
            # Simple gradient approximation
            gradient = rate_error * np.random.normal(0, 0.01, param_value.shape)
            
            return gradient
            
        except Exception as e:
            self.logger.error(f"Failed to compute spike gradient: {e}")
            return np.zeros_like(param_value)
    
    def _update_spike_statistics(self, param_name: str, layer_spikes: np.ndarray) -> None:
        """Update spike statistics for layer."""
        if param_name not in self.spike_statistics:
            self.spike_statistics[param_name] = {}
        
        stats = self.spike_statistics[param_name]
        stats['total_spikes'] = len(layer_spikes)
        stats['spike_rate'] = len(layer_spikes) / self.spike_window
        stats['energy'] = len(layer_spikes) * 1e-12  # pJ per spike
        
        if len(layer_spikes) > 1:
            stats['isi_mean'] = np.mean(np.diff(layer_spikes))
            stats['isi_std'] = np.std(np.diff(layer_spikes))
        else:
            stats['isi_mean'] = 0
            stats['isi_std'] = 0
    
    def _calculate_convergence_metric(self, gradients: Dict[str, np.ndarray]) -> float:
        """Calculate convergence metric based on gradient magnitudes."""
        total_grad_norm = 0.0
        for gradient in gradients.values():
            total_grad_norm += np.linalg.norm(gradient)
        
        return total_grad_norm / max(len(gradients), 1)
    
    def _estimate_latency(self, spike_count: int) -> float:
        """Estimate processing latency based on spike count."""
        # Simple linear model: ~0.1ms per 1000 spikes
        return (spike_count / 1000.0) * 0.1


class STDPOptimizer(NeuromorphicOptimizer):
    """
    Spike-Timing Dependent Plasticity (STDP) optimizer.
    
    Implements STDP learning rules for unsupervised and reinforcement
    learning in neuromorphic networks.
    """
    
    def __init__(
        self,
        config: OptimizationConfig,
        model_parameters: Optional[Dict[str, np.ndarray]] = None,
        stdp_window: float = 20.0,  # ms
        tau_pre: float = 20.0,     # ms
        tau_post: float = 20.0,    # ms
        a_plus: float = 0.1,
        a_minus: float = 0.12,
    ):
        """
        Initialize STDP optimizer.
        
        Args:
            config: Optimization configuration
            model_parameters: Model parameters
            stdp_window: STDP time window in ms
            tau_pre: Pre-synaptic trace time constant
            tau_post: Post-synaptic trace time constant
            a_plus: LTP amplitude
            a_minus: LTD amplitude
        """
        super().__init__(config, model_parameters)
        
        self.stdp_window = stdp_window
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.a_plus = a_plus
        self.a_minus = a_minus
        
        # STDP traces
        self.pre_traces: Dict[str, np.ndarray] = {}
        self.post_traces: Dict[str, np.ndarray] = {}
        
        # Synaptic weights tracking
        self.weight_changes: Dict[str, List[float]] = {}
    
    def compute_gradients(
        self,
        loss: float,
        spike_data: Dict[str, np.ndarray],
        target_spikes: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute STDP-based weight updates.
        
        Args:
            loss: Current loss (used for modulation)
            spike_data: Pre and post-synaptic spike times
            target_spikes: Not used in STDP
            
        Returns:
            STDP weight updates
        """
        gradients = {}
        
        try:
            for param_name, param_value in self.model_parameters.items():
                # Get pre and post-synaptic spikes
                pre_spikes = spike_data.get(f"{param_name}_pre", np.array([]))
                post_spikes = spike_data.get(f"{param_name}_post", np.array([]))
                
                # Compute STDP updates
                weight_update = self._compute_stdp_update(
                    param_value, pre_spikes, post_spikes
                )
                
                # Modulate by loss (for supervised STDP)
                modulation = 1.0 if loss == 0 else (1.0 / (1.0 + loss))
                gradients[param_name] = modulation * weight_update
                
                # Track weight changes
                if param_name not in self.weight_changes:
                    self.weight_changes[param_name] = []
                self.weight_changes[param_name].append(np.mean(np.abs(weight_update)))
            
            return gradients
            
        except Exception as e:
            self.logger.error(f"Failed to compute STDP gradients: {e}")
            raise
    
    def update_parameters(
        self,
        gradients: Dict[str, np.ndarray],
        learning_rate: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Apply STDP weight updates.
        
        Args:
            gradients: STDP weight updates
            learning_rate: Learning rate (adaptation rate for STDP)
            
        Returns:
            Optimization result
        """
        adaptation_rate = learning_rate or self.config.adaptation_rate
        
        try:
            total_updates = 0
            total_energy = 0.0
            
            for param_name, update in gradients.items():
                if param_name not in self.model_parameters:
                    continue
                
                # Apply STDP updates
                self.model_parameters[param_name] += adaptation_rate * update
                
                # Apply weight bounds (biological constraint)
                self.model_parameters[param_name] = np.clip(
                    self.model_parameters[param_name], 0, 1
                )
                
                total_updates += np.count_nonzero(update)
                total_energy += np.sum(np.abs(update)) * 1e-15  # fJ per weight update
            
            # Calculate convergence (weight stability)
            convergence_metric = self._calculate_stdp_convergence()
            
            return OptimizationResult(
                loss=0.0,  # STDP is unsupervised
                gradients=gradients,
                spike_counts={'weight_updates': total_updates},
                energy_consumption=total_energy,
                latency_ms=total_updates * 0.001,  # 1μs per update
                convergence_metric=convergence_metric,
                hardware_utilization={},
            )
            
        except Exception as e:
            self.logger.error(f"Failed to apply STDP updates: {e}")
            raise
    
    def _compute_stdp_update(
        self,
        weights: np.ndarray,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
    ) -> np.ndarray:
        """Compute STDP weight updates based on spike timing."""
        try:
            if len(pre_spikes) == 0 or len(post_spikes) == 0:
                return np.zeros_like(weights)
            
            # Initialize weight updates
            delta_w = np.zeros_like(weights)
            
            # For each pre-post spike pair, compute STDP
            for i, t_pre in enumerate(pre_spikes):
                for j, t_post in enumerate(post_spikes):
                    dt = t_post - t_pre
                    
                    if abs(dt) > self.stdp_window:
                        continue
                    
                    # STDP learning rule
                    if dt > 0:  # Post after pre (LTP)
                        delta_w[i % weights.shape[0], j % weights.shape[1]] += (
                            self.a_plus * np.exp(-dt / self.tau_pre)
                        )
                    else:  # Pre after post (LTD)
                        delta_w[i % weights.shape[0], j % weights.shape[1]] -= (
                            self.a_minus * np.exp(dt / self.tau_post)
                        )
            
            return delta_w
            
        except Exception as e:
            self.logger.error(f"Failed to compute STDP update: {e}")
            return np.zeros_like(weights)
    
    def _calculate_stdp_convergence(self) -> float:
        """Calculate STDP convergence metric based on weight stability."""
        if not self.weight_changes:
            return 0.0
        
        # Average weight change rate over recent history
        recent_changes = []
        for param_changes in self.weight_changes.values():
            if len(param_changes) >= 2:
                recent_changes.extend(param_changes[-10:])  # Last 10 updates
        
        if not recent_changes:
            return 0.0
        
        # Convergence metric: inverse of average change
        avg_change = np.mean(recent_changes)
        return 1.0 / (1.0 + avg_change)


class HardwareAwareOptimizer(NeuromorphicOptimizer):
    """
    Hardware-aware optimizer for neuromorphic processors.
    
    Optimizes for specific neuromorphic hardware constraints such as
    Loihi, Akida, or SpiNNaker core limitations and energy budgets.
    """
    
    def __init__(
        self,
        config: OptimizationConfig,
        model_parameters: Optional[Dict[str, np.ndarray]] = None,
        hardware_type: str = "loihi2",
        core_count: int = 128,
        memory_per_core: int = 1024,  # KB
    ):
        """
        Initialize hardware-aware optimizer.
        
        Args:
            config: Optimization configuration
            model_parameters: Model parameters
            hardware_type: Target hardware (loihi2, akida, spinnaker)
            core_count: Available cores
            memory_per_core: Memory per core in KB
        """
        super().__init__(config, model_parameters)
        
        self.hardware_type = hardware_type
        self.core_count = core_count
        self.memory_per_core = memory_per_core
        
        # Hardware-specific constraints
        self.hardware_constraints = self._get_hardware_constraints()
        
        # Resource utilization tracking
        self.core_utilization: List[float] = [0.0] * core_count
        self.memory_utilization: List[float] = [0.0] * core_count
        self.energy_consumption_per_core: List[float] = [0.0] * core_count
    
    def compute_gradients(
        self,
        loss: float,
        spike_data: Dict[str, np.ndarray],
        target_spikes: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute hardware-aware gradients with resource constraints.
        
        Args:
            loss: Current loss value
            spike_data: Spike activity data
            target_spikes: Target patterns
            
        Returns:
            Hardware-constrained gradients
        """
        gradients = {}
        
        try:
            for param_name, param_value in self.model_parameters.items():
                # Standard gradient computation
                base_gradient = self._compute_base_gradient(
                    loss, spike_data.get(param_name, np.array([])), target_spikes
                )
                
                # Apply hardware constraints
                constrained_gradient = self._apply_hardware_constraints(
                    param_name, base_gradient, spike_data
                )
                
                gradients[param_name] = constrained_gradient
            
            return gradients
            
        except Exception as e:
            self.logger.error(f"Failed to compute hardware-aware gradients: {e}")
            raise
    
    def update_parameters(
        self,
        gradients: Dict[str, np.ndarray],
        learning_rate: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Update parameters with hardware resource management.
        
        Args:
            gradients: Computed gradients
            learning_rate: Learning rate
            
        Returns:
            Optimization result with hardware metrics
        """
        lr = learning_rate or self.config.learning_rate
        
        try:
            # Distribute parameters across cores
            core_assignments = self._assign_parameters_to_cores()
            
            total_energy = 0.0
            max_latency = 0.0
            
            for param_name, gradient in gradients.items():
                if param_name not in self.model_parameters:
                    continue
                
                core_id = core_assignments.get(param_name, 0)
                
                # Update parameters
                self.model_parameters[param_name] -= lr * gradient
                
                # Calculate hardware metrics
                energy, latency = self._calculate_hardware_metrics(
                    core_id, gradient, self.model_parameters[param_name]
                )
                
                total_energy += energy
                max_latency = max(max_latency, latency)
                
                # Update core utilization
                self._update_core_utilization(core_id, param_name)
            
            # Calculate convergence with hardware efficiency
            convergence_metric = self._calculate_hardware_convergence(gradients)
            
            return OptimizationResult(
                loss=self.loss_history[-1] if self.loss_history else 0,
                gradients=gradients,
                spike_counts={},
                energy_consumption=total_energy,
                latency_ms=max_latency,
                convergence_metric=convergence_metric,
                hardware_utilization=self._get_hardware_utilization(),
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update parameters with hardware awareness: {e}")
            raise
    
    def _get_hardware_constraints(self) -> Dict[str, Any]:
        """Get hardware-specific constraints."""
        constraints = {
            "loihi2": {
                "max_neurons_per_core": 1024,
                "max_synapses_per_neuron": 4096,
                "spike_rate_limit": 1000,  # Hz
                "energy_per_spike": 1e-12,  # J
                "latency_per_spike": 1e-6,  # s
            },
            "akida": {
                "max_neurons_per_core": 256,
                "max_synapses_per_neuron": 1024,
                "spike_rate_limit": 10000,  # Hz
                "energy_per_spike": 5e-13,  # J
                "latency_per_spike": 1e-7,  # s
            },
            "spinnaker": {
                "max_neurons_per_core": 256,
                "max_synapses_per_neuron": 2048,
                "spike_rate_limit": 1000,  # Hz
                "energy_per_spike": 1e-11,  # J
                "latency_per_spike": 1e-5,  # s
            }
        }
        
        return constraints.get(self.hardware_type, constraints["loihi2"])
    
    def _compute_base_gradient(
        self,
        loss: float,
        spike_data: np.ndarray,
        target_spikes: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute base gradient before hardware constraints."""
        # Simple gradient approximation
        if len(spike_data) == 0:
            return np.zeros((10, 10))  # Default shape
        
        spike_rate = len(spike_data) / 10.0  # Assuming 10ms window
        target_rate = 0.1 if target_spikes is None else len(target_spikes) / 10.0
        
        rate_error = spike_rate - target_rate
        gradient = rate_error * np.random.normal(0, 0.01, (10, 10))
        
        return gradient
    
    def _apply_hardware_constraints(
        self,
        param_name: str,
        gradient: np.ndarray,
        spike_data: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Apply hardware-specific constraints to gradients."""
        # Quantize gradients for hardware implementation
        if self.hardware_type == "akida":
            # Akida uses 4-bit weights
            gradient = np.round(gradient * 16) / 16
        elif self.hardware_type == "loihi2":
            # Loihi uses 8-bit weights
            gradient = np.round(gradient * 256) / 256
        
        # Apply sparsity constraints
        max_synapses = self.hardware_constraints["max_synapses_per_neuron"]
        if gradient.size > max_synapses:
            # Keep only top-k connections
            flat_grad = gradient.flatten()
            threshold = np.percentile(np.abs(flat_grad), 90)
            gradient[np.abs(gradient) < threshold] = 0
        
        return gradient
    
    def _assign_parameters_to_cores(self) -> Dict[str, int]:
        """Assign parameters to available cores."""
        assignments = {}
        current_core = 0
        
        for i, param_name in enumerate(self.model_parameters.keys()):
            assignments[param_name] = current_core
            current_core = (current_core + 1) % self.core_count
        
        return assignments
    
    def _calculate_hardware_metrics(
        self,
        core_id: int,
        gradient: np.ndarray,
        parameters: np.ndarray,
    ) -> Tuple[float, float]:
        """Calculate energy and latency for hardware operations."""
        # Energy calculation
        num_operations = np.count_nonzero(gradient)
        energy_per_op = self.hardware_constraints["energy_per_spike"]
        energy = num_operations * energy_per_op
        
        # Latency calculation  
        latency_per_op = self.hardware_constraints["latency_per_spike"]
        latency = num_operations * latency_per_op * 1000  # Convert to ms
        
        return energy, latency
    
    def _update_core_utilization(self, core_id: int, param_name: str) -> None:
        """Update core utilization metrics."""
        if core_id < len(self.core_utilization):
            # Simple utilization model
            param_size = self.model_parameters[param_name].size
            memory_usage = param_size * 4 / 1024  # 4 bytes per param, convert to KB
            
            self.memory_utilization[core_id] = min(
                1.0, 
                self.memory_utilization[core_id] + memory_usage / self.memory_per_core
            )
            
            self.core_utilization[core_id] = min(1.0, self.memory_utilization[core_id])
    
    def _calculate_hardware_convergence(self, gradients: Dict[str, np.ndarray]) -> float:
        """Calculate convergence metric including hardware efficiency."""
        # Standard gradient norm
        grad_norm = sum(np.linalg.norm(grad) for grad in gradients.values())
        
        # Hardware efficiency penalty
        avg_core_util = np.mean(self.core_utilization)
        efficiency_bonus = 1.0 - abs(avg_core_util - 0.7)  # Target 70% utilization
        
        return grad_norm * efficiency_bonus
    
    def _get_hardware_utilization(self) -> Dict[str, float]:
        """Get current hardware utilization metrics."""
        return {
            'average_core_utilization': np.mean(self.core_utilization),
            'max_core_utilization': np.max(self.core_utilization),
            'average_memory_utilization': np.mean(self.memory_utilization),
            'max_memory_utilization': np.max(self.memory_utilization),
            'total_cores_used': np.count_nonzero(self.core_utilization),
            'utilization_efficiency': 1.0 - np.std(self.core_utilization),
        }