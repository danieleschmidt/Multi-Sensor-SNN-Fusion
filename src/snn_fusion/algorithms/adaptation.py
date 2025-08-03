"""
Adaptive Control and Plasticity Mechanisms

Implements adaptive controllers, plasticity managers, and dynamic
adjustment mechanisms for neuromorphic computing systems.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum
import time

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AdaptationType(Enum):
    """Types of adaptation mechanisms."""
    THRESHOLD = "threshold"
    SYNAPTIC_SCALING = "synaptic_scaling"
    HOMEOSTATIC = "homeostatic"
    METAPLASTICITY = "metaplasticity"
    NEUROMODULATION = "neuromodulation"


@dataclass
class AdaptationState:
    """State information for adaptation mechanisms."""
    parameter_name: str
    current_value: float
    target_value: float
    adaptation_rate: float
    last_update: float
    update_count: int
    convergence_metric: float
    metadata: Optional[Dict[str, Any]] = None


class AdaptiveController:
    """
    Adaptive controller for neuromorphic system parameters.
    
    Implements various adaptation strategies for maintaining optimal
    system performance and biological plausibility.
    """
    
    def __init__(
        self,
        adaptation_rate: float = 0.01,
        target_activity: float = 0.1,
        time_constant: float = 100.0,  # ms
        stability_threshold: float = 1e-4,
    ):
        """
        Initialize adaptive controller.
        
        Args:
            adaptation_rate: Rate of parameter adaptation
            target_activity: Target neural activity level
            time_constant: Time constant for adaptation
            stability_threshold: Threshold for considering adaptation stable
        """
        self.adaptation_rate = adaptation_rate
        self.target_activity = target_activity
        self.time_constant = time_constant
        self.stability_threshold = stability_threshold
        self.logger = logging.getLogger(__name__)
        
        # Adaptation states
        self.adaptation_states: Dict[str, AdaptationState] = {}
        
        # Activity tracking
        self.activity_history: Dict[str, List[float]] = {}
        self.activity_window_size = 100
        
        # Callbacks for adaptation events
        self.adaptation_callbacks: List[Callable[[str, AdaptationState], None]] = []
        
        self.logger.info("Initialized adaptive controller")
    
    def register_parameter(
        self,
        param_name: str,
        initial_value: float,
        target_value: Optional[float] = None,
        adaptation_type: AdaptationType = AdaptationType.THRESHOLD,
    ) -> None:
        """
        Register parameter for adaptive control.
        
        Args:
            param_name: Parameter name
            initial_value: Initial parameter value
            target_value: Target value (optional)
            adaptation_type: Type of adaptation
        """
        state = AdaptationState(
            parameter_name=param_name,
            current_value=initial_value,
            target_value=target_value or self.target_activity,
            adaptation_rate=self.adaptation_rate,
            last_update=time.time(),
            update_count=0,
            convergence_metric=float('inf'),
            metadata={'adaptation_type': adaptation_type.value}
        )
        
        self.adaptation_states[param_name] = state
        self.activity_history[param_name] = []
        
        self.logger.info(f"Registered parameter for adaptation: {param_name}")
    
    def update_activity(self, param_name: str, activity_value: float) -> None:
        """
        Update activity measurement for parameter.
        
        Args:
            param_name: Parameter name
            activity_value: Current activity measurement
        """
        if param_name not in self.adaptation_states:
            self.logger.warning(f"Parameter not registered: {param_name}")
            return
        
        # Add to activity history
        history = self.activity_history[param_name]
        history.append(activity_value)
        
        # Keep window size
        if len(history) > self.activity_window_size:
            history.pop(0)
    
    def adapt_parameter(self, param_name: str) -> Optional[float]:
        """
        Perform adaptation step for parameter.
        
        Args:
            param_name: Parameter to adapt
            
        Returns:
            New parameter value or None if adaptation failed
        """
        if param_name not in self.adaptation_states:
            self.logger.warning(f"Parameter not registered: {param_name}")
            return None
        
        try:
            state = self.adaptation_states[param_name]
            history = self.activity_history[param_name]
            
            if len(history) < 5:  # Need minimum activity samples
                return state.current_value
            
            # Calculate current average activity
            current_activity = np.mean(history[-10:])  # Last 10 samples
            
            # Calculate adaptation based on type
            adaptation_type = AdaptationType(state.metadata['adaptation_type'])
            
            if adaptation_type == AdaptationType.THRESHOLD:
                new_value = self._adapt_threshold(state, current_activity)
            elif adaptation_type == AdaptationType.SYNAPTIC_SCALING:
                new_value = self._adapt_synaptic_scaling(state, current_activity)
            elif adaptation_type == AdaptationType.HOMEOSTATIC:
                new_value = self._adapt_homeostatic(state, current_activity, history)
            else:
                new_value = self._adapt_default(state, current_activity)
            
            # Update state
            old_value = state.current_value
            state.current_value = new_value
            state.last_update = time.time()
            state.update_count += 1
            
            # Calculate convergence metric
            state.convergence_metric = abs(current_activity - state.target_value)
            
            # Check for stability
            if state.convergence_metric < self.stability_threshold:
                self.logger.debug(f"Parameter {param_name} reached stability")
            
            # Notify callbacks
            self._notify_adaptation_callbacks(param_name, state)
            
            self.logger.debug(
                f"Adapted {param_name}: {old_value:.4f} -> {new_value:.4f} "
                f"(activity: {current_activity:.4f}, target: {state.target_value:.4f})"
            )
            
            return new_value
            
        except Exception as e:
            self.logger.error(f"Failed to adapt parameter {param_name}: {e}")
            return None
    
    def adapt_all_parameters(self) -> Dict[str, float]:
        """
        Perform adaptation for all registered parameters.
        
        Returns:
            Dictionary of parameter names to new values
        """
        results = {}
        
        for param_name in self.adaptation_states.keys():
            new_value = self.adapt_parameter(param_name)
            if new_value is not None:
                results[param_name] = new_value
        
        return results
    
    def get_adaptation_status(self, param_name: str) -> Optional[Dict[str, Any]]:
        """
        Get adaptation status for parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Status information or None
        """
        if param_name not in self.adaptation_states:
            return None
        
        state = self.adaptation_states[param_name]
        history = self.activity_history[param_name]
        
        return {
            'parameter_name': param_name,
            'current_value': state.current_value,
            'target_value': state.target_value,
            'current_activity': np.mean(history[-10:]) if len(history) >= 10 else 0,
            'convergence_metric': state.convergence_metric,
            'update_count': state.update_count,
            'is_stable': state.convergence_metric < self.stability_threshold,
            'activity_trend': self._calculate_activity_trend(history),
            'last_update': state.last_update,
        }
    
    def add_adaptation_callback(self, callback: Callable[[str, AdaptationState], None]) -> None:
        """Add callback for adaptation events."""
        self.adaptation_callbacks.append(callback)
        self.logger.info("Added adaptation callback")
    
    def _adapt_threshold(self, state: AdaptationState, current_activity: float) -> float:
        """Adapt threshold parameter to maintain target activity."""
        error = current_activity - state.target_value
        
        # Threshold adaptation: increase threshold if activity too high
        adaptation = -error * state.adaptation_rate
        
        new_value = state.current_value + adaptation
        
        # Ensure positive threshold
        return max(0.1, new_value)
    
    def _adapt_synaptic_scaling(self, state: AdaptationState, current_activity: float) -> float:
        """Adapt synaptic weights to maintain homeostasis."""
        error = current_activity - state.target_value
        
        # Synaptic scaling: scale weights to maintain activity
        scaling_factor = state.target_value / (current_activity + 1e-8)
        
        # Gradual adaptation
        adaptation = (scaling_factor - 1.0) * state.adaptation_rate
        new_value = state.current_value * (1.0 + adaptation)
        
        # Bound scaling
        return np.clip(new_value, 0.1, 10.0)
    
    def _adapt_homeostatic(
        self,
        state: AdaptationState,
        current_activity: float,
        history: List[float],
    ) -> float:
        """Homeostatic adaptation based on activity history."""
        if len(history) < 20:
            return state.current_value
        
        # Calculate activity variance
        activity_variance = np.var(history[-20:])
        
        # Adapt based on both mean and variance
        mean_error = current_activity - state.target_value
        variance_penalty = activity_variance * 0.1  # Penalize high variance
        
        total_error = mean_error + variance_penalty
        adaptation = -total_error * state.adaptation_rate
        
        new_value = state.current_value + adaptation
        
        return max(0.05, new_value)
    
    def _adapt_default(self, state: AdaptationState, current_activity: float) -> float:
        """Default adaptation mechanism."""
        error = current_activity - state.target_value
        adaptation = -error * state.adaptation_rate
        
        return state.current_value + adaptation
    
    def _calculate_activity_trend(self, history: List[float]) -> str:
        """Calculate trend in activity history."""
        if len(history) < 10:
            return "insufficient_data"
        
        recent = np.mean(history[-5:])
        older = np.mean(history[-10:-5])
        
        diff = recent - older
        
        if abs(diff) < 0.01:
            return "stable"
        elif diff > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _notify_adaptation_callbacks(self, param_name: str, state: AdaptationState) -> None:
        """Notify adaptation callbacks."""
        for callback in self.adaptation_callbacks:
            try:
                callback(param_name, state)
            except Exception as e:
                self.logger.error(f"Adaptation callback error: {e}")


class PlasticityManager:
    """
    Manages synaptic plasticity mechanisms in neuromorphic networks.
    
    Coordinates multiple forms of plasticity including STDP, homeostatic
    scaling, and metaplasticity for biologically plausible learning.
    """
    
    def __init__(
        self,
        stdp_enabled: bool = True,
        homeostatic_enabled: bool = True,
        metaplasticity_enabled: bool = False,
    ):
        """
        Initialize plasticity manager.
        
        Args:
            stdp_enabled: Enable STDP plasticity
            homeostatic_enabled: Enable homeostatic plasticity
            metaplasticity_enabled: Enable metaplasticity
        """
        self.stdp_enabled = stdp_enabled
        self.homeostatic_enabled = homeostatic_enabled
        self.metaplasticity_enabled = metaplasticity_enabled
        self.logger = logging.getLogger(__name__)
        
        # Plasticity states
        self.synaptic_weights: Dict[str, np.ndarray] = {}
        self.plasticity_traces: Dict[str, Dict[str, np.ndarray]] = {}
        self.metaplasticity_state: Dict[str, np.ndarray] = {}
        
        # Plasticity parameters
        self.stdp_params = {
            'tau_pre': 20.0,  # ms
            'tau_post': 20.0,  # ms
            'a_plus': 0.01,
            'a_minus': 0.012,
            'w_max': 1.0,
            'w_min': 0.0,
        }
        
        self.homeostatic_params = {
            'target_rate': 5.0,  # Hz
            'tau_homeostatic': 1000.0,  # ms
            'scaling_factor': 0.001,
        }
        
        self.logger.info("Initialized plasticity manager")
    
    def register_connection(
        self,
        connection_id: str,
        pre_neurons: int,
        post_neurons: int,
        initial_weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Register synaptic connection for plasticity.
        
        Args:
            connection_id: Connection identifier
            pre_neurons: Number of pre-synaptic neurons
            post_neurons: Number of post-synaptic neurons
            initial_weights: Initial weight matrix
        """
        if initial_weights is not None:
            weights = initial_weights.copy()
        else:
            # Initialize with small random weights
            weights = np.random.normal(0.1, 0.02, (pre_neurons, post_neurons))
            weights = np.clip(weights, self.stdp_params['w_min'], self.stdp_params['w_max'])
        
        self.synaptic_weights[connection_id] = weights
        
        # Initialize plasticity traces
        self.plasticity_traces[connection_id] = {
            'pre_trace': np.zeros(pre_neurons),
            'post_trace': np.zeros(post_neurons),
            'homeostatic_trace': np.zeros(post_neurons),
        }
        
        # Initialize metaplasticity state
        if self.metaplasticity_enabled:
            self.metaplasticity_state[connection_id] = np.ones_like(weights)
        
        self.logger.info(f"Registered connection: {connection_id} ({pre_neurons}x{post_neurons})")
    
    def update_plasticity(
        self,
        connection_id: str,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        dt: float = 1.0,  # ms
    ) -> None:
        """
        Update synaptic plasticity based on spike activity.
        
        Args:
            connection_id: Connection identifier
            pre_spikes: Pre-synaptic spike indicators
            post_spikes: Post-synaptic spike indicators
            dt: Time step in milliseconds
        """
        if connection_id not in self.synaptic_weights:
            self.logger.warning(f"Connection not registered: {connection_id}")
            return
        
        try:
            weights = self.synaptic_weights[connection_id]
            traces = self.plasticity_traces[connection_id]
            
            # Update STDP if enabled
            if self.stdp_enabled:
                self._update_stdp(weights, traces, pre_spikes, post_spikes, dt)
            
            # Update homeostatic plasticity if enabled
            if self.homeostatic_enabled:
                self._update_homeostatic(weights, traces, post_spikes, dt)
            
            # Update metaplasticity if enabled
            if self.metaplasticity_enabled:
                self._update_metaplasticity(connection_id, pre_spikes, post_spikes, dt)
            
            # Apply weight bounds
            self.synaptic_weights[connection_id] = np.clip(
                weights,
                self.stdp_params['w_min'],
                self.stdp_params['w_max']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update plasticity for {connection_id}: {e}")
    
    def get_weights(self, connection_id: str) -> Optional[np.ndarray]:
        """Get current synaptic weights."""
        return self.synaptic_weights.get(connection_id)
    
    def get_plasticity_statistics(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get plasticity statistics for connection."""
        if connection_id not in self.synaptic_weights:
            return None
        
        weights = self.synaptic_weights[connection_id]
        traces = self.plasticity_traces[connection_id]
        
        return {
            'connection_id': connection_id,
            'weight_stats': {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'sparsity': np.sum(weights < 0.01) / weights.size,
            },
            'trace_stats': {
                'pre_trace_mean': np.mean(traces['pre_trace']),
                'post_trace_mean': np.mean(traces['post_trace']),
                'homeostatic_trace_mean': np.mean(traces['homeostatic_trace']),
            },
            'weight_shape': weights.shape,
        }
    
    def _update_stdp(
        self,
        weights: np.ndarray,
        traces: Dict[str, np.ndarray],
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        dt: float,
    ) -> None:
        """Update STDP plasticity."""
        tau_pre = self.stdp_params['tau_pre']
        tau_post = self.stdp_params['tau_post']
        a_plus = self.stdp_params['a_plus']
        a_minus = self.stdp_params['a_minus']
        
        # Decay traces
        traces['pre_trace'] *= np.exp(-dt / tau_pre)
        traces['post_trace'] *= np.exp(-dt / tau_post)
        
        # Add spikes to traces
        traces['pre_trace'] += pre_spikes
        traces['post_trace'] += post_spikes
        
        # STDP weight updates
        # LTP: post-synaptic spike increases weights based on pre-synaptic trace
        ltp_update = np.outer(traces['pre_trace'], post_spikes) * a_plus
        
        # LTD: pre-synaptic spike decreases weights based on post-synaptic trace
        ltd_update = np.outer(pre_spikes, traces['post_trace']) * a_minus
        
        # Apply updates
        weights += ltp_update - ltd_update
    
    def _update_homeostatic(
        self,
        weights: np.ndarray,
        traces: Dict[str, np.ndarray],
        post_spikes: np.ndarray,
        dt: float,
    ) -> None:
        """Update homeostatic plasticity."""
        tau_h = self.homeostatic_params['tau_homeostatic']
        target_rate = self.homeostatic_params['target_rate']
        scaling_factor = self.homeostatic_params['scaling_factor']
        
        # Update homeostatic trace (average firing rate)
        traces['homeostatic_trace'] *= np.exp(-dt / tau_h)
        traces['homeostatic_trace'] += post_spikes * (1000.0 / tau_h)  # Convert to Hz
        
        # Homeostatic scaling
        rate_error = traces['homeostatic_trace'] - target_rate
        scaling = 1.0 - rate_error * scaling_factor * dt / 1000.0
        
        # Apply scaling to all incoming weights
        weights *= scaling.reshape(1, -1)
    
    def _update_metaplasticity(
        self,
        connection_id: str,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        dt: float,
    ) -> None:
        """Update metaplasticity state."""
        if connection_id not in self.metaplasticity_state:
            return
        
        # Simple metaplasticity: plasticity depends on recent activity
        meta_state = self.metaplasticity_state[connection_id]
        
        # Calculate activity metric
        activity = np.outer(pre_spikes, post_spikes)
        
        # Update metaplasticity state (tau = 10000 ms)
        tau_meta = 10000.0
        meta_state *= np.exp(-dt / tau_meta)
        meta_state += activity * 0.001
        
        # Metaplasticity modulates STDP learning rate
        # High activity reduces plasticity (prevents runaway)
        plasticity_modulation = 1.0 / (1.0 + meta_state)
        
        # Store for next STDP update
        self.metaplasticity_state[connection_id] = meta_state


class ThresholdAdaptation:
    """
    Adaptive threshold mechanism for neuromorphic neurons.
    
    Implements various threshold adaptation strategies to maintain
    optimal firing rates and prevent pathological activity.
    """
    
    def __init__(
        self,
        base_threshold: float = 1.0,
        adaptation_rate: float = 0.001,
        target_rate: float = 5.0,  # Hz
        time_constant: float = 1000.0,  # ms
    ):
        """
        Initialize threshold adaptation.
        
        Args:
            base_threshold: Base threshold value
            adaptation_rate: Rate of threshold adaptation
            target_rate: Target firing rate in Hz
            time_constant: Adaptation time constant in ms
        """
        self.base_threshold = base_threshold
        self.adaptation_rate = adaptation_rate
        self.target_rate = target_rate
        self.time_constant = time_constant
        self.logger = logging.getLogger(__name__)
        
        # Neuron states
        self.thresholds: Dict[str, np.ndarray] = {}
        self.firing_rates: Dict[str, np.ndarray] = {}
        self.rate_traces: Dict[str, np.ndarray] = {}
        
        self.logger.info("Initialized threshold adaptation")
    
    def register_neurons(self, neuron_group_id: str, n_neurons: int) -> None:
        """
        Register neuron group for threshold adaptation.
        
        Args:
            neuron_group_id: Group identifier
            n_neurons: Number of neurons in group
        """
        self.thresholds[neuron_group_id] = np.full(n_neurons, self.base_threshold)
        self.firing_rates[neuron_group_id] = np.zeros(n_neurons)
        self.rate_traces[neuron_group_id] = np.zeros(n_neurons)
        
        self.logger.info(f"Registered {n_neurons} neurons for threshold adaptation: {neuron_group_id}")
    
    def update_thresholds(
        self,
        neuron_group_id: str,
        spike_train: np.ndarray,
        dt: float = 1.0,  # ms
    ) -> None:
        """
        Update adaptive thresholds based on spiking activity.
        
        Args:
            neuron_group_id: Group identifier
            spike_train: Binary spike indicators for each neuron
            dt: Time step in milliseconds
        """
        if neuron_group_id not in self.thresholds:
            self.logger.warning(f"Neuron group not registered: {neuron_group_id}")
            return
        
        try:
            thresholds = self.thresholds[neuron_group_id]
            rate_traces = self.rate_traces[neuron_group_id]
            
            # Update firing rate traces
            decay_factor = np.exp(-dt / self.time_constant)
            rate_traces *= decay_factor
            rate_traces += spike_train * (1000.0 / self.time_constant)  # Convert to Hz
            
            # Calculate threshold adaptation
            rate_error = rate_traces - self.target_rate
            threshold_adaptation = rate_error * self.adaptation_rate * dt / 1000.0
            
            # Update thresholds
            thresholds += threshold_adaptation
            
            # Ensure positive thresholds
            thresholds = np.maximum(thresholds, 0.1)
            
            # Store updated values
            self.thresholds[neuron_group_id] = thresholds
            self.rate_traces[neuron_group_id] = rate_traces
            
        except Exception as e:
            self.logger.error(f"Failed to update thresholds for {neuron_group_id}: {e}")
    
    def get_thresholds(self, neuron_group_id: str) -> Optional[np.ndarray]:
        """Get current thresholds for neuron group."""
        return self.thresholds.get(neuron_group_id)
    
    def get_firing_rates(self, neuron_group_id: str) -> Optional[np.ndarray]:
        """Get current firing rates for neuron group."""
        return self.rate_traces.get(neuron_group_id)
    
    def get_adaptation_statistics(self, neuron_group_id: str) -> Optional[Dict[str, Any]]:
        """Get adaptation statistics for neuron group."""
        if neuron_group_id not in self.thresholds:
            return None
        
        thresholds = self.thresholds[neuron_group_id]
        rates = self.rate_traces[neuron_group_id]
        
        return {
            'neuron_group_id': neuron_group_id,
            'n_neurons': len(thresholds),
            'threshold_stats': {
                'mean': np.mean(thresholds),
                'std': np.std(thresholds),
                'min': np.min(thresholds),
                'max': np.max(thresholds),
            },
            'rate_stats': {
                'mean': np.mean(rates),
                'std': np.std(rates),
                'min': np.min(rates),
                'max': np.max(rates),
                'target': self.target_rate,
            },
            'adaptation_efficiency': 1.0 - np.mean(np.abs(rates - self.target_rate)) / self.target_rate,
        }


class SynapticScaling:
    """
    Synaptic scaling mechanism for homeostatic plasticity.
    
    Implements global and local synaptic scaling to maintain
    network stability and prevent runaway excitation.
    """
    
    def __init__(
        self,
        target_activity: float = 0.1,
        scaling_rate: float = 0.0001,
        time_constant: float = 3600000.0,  # 1 hour in ms
    ):
        """
        Initialize synaptic scaling.
        
        Args:
            target_activity: Target activity level
            scaling_rate: Rate of synaptic scaling
            time_constant: Time constant for scaling
        """
        self.target_activity = target_activity
        self.scaling_rate = scaling_rate
        self.time_constant = time_constant
        self.logger = logging.getLogger(__name__)
        
        # Scaling states
        self.synaptic_weights: Dict[str, np.ndarray] = {}
        self.activity_traces: Dict[str, np.ndarray] = {}
        self.scaling_factors: Dict[str, np.ndarray] = {}
        
        self.logger.info("Initialized synaptic scaling")
    
    def register_synapses(
        self,
        synapse_group_id: str,
        weights: np.ndarray,
    ) -> None:
        """
        Register synaptic group for scaling.
        
        Args:
            synapse_group_id: Group identifier
            weights: Initial synaptic weights
        """
        self.synaptic_weights[synapse_group_id] = weights.copy()
        
        # Initialize activity traces (one per post-synaptic neuron)
        post_neurons = weights.shape[1]
        self.activity_traces[synapse_group_id] = np.zeros(post_neurons)
        self.scaling_factors[synapse_group_id] = np.ones(post_neurons)
        
        self.logger.info(f"Registered synapses for scaling: {synapse_group_id}")
    
    def update_scaling(
        self,
        synapse_group_id: str,
        post_synaptic_activity: np.ndarray,
        dt: float = 1.0,  # ms
    ) -> None:
        """
        Update synaptic scaling based on post-synaptic activity.
        
        Args:
            synapse_group_id: Group identifier
            post_synaptic_activity: Post-synaptic activity levels
            dt: Time step in milliseconds
        """
        if synapse_group_id not in self.synaptic_weights:
            self.logger.warning(f"Synapse group not registered: {synapse_group_id}")
            return
        
        try:
            weights = self.synaptic_weights[synapse_group_id]
            activity_traces = self.activity_traces[synapse_group_id]
            scaling_factors = self.scaling_factors[synapse_group_id]
            
            # Update activity traces
            decay_factor = np.exp(-dt / self.time_constant)
            activity_traces *= decay_factor
            activity_traces += post_synaptic_activity * (1.0 - decay_factor)
            
            # Calculate scaling factors
            activity_ratio = activity_traces / (self.target_activity + 1e-8)
            
            # Multiplicative scaling
            scaling_change = (1.0 / activity_ratio - 1.0) * self.scaling_rate * dt / 1000.0
            scaling_factors *= (1.0 + scaling_change)
            
            # Bound scaling factors
            scaling_factors = np.clip(scaling_factors, 0.1, 10.0)
            
            # Apply scaling to weights
            scaled_weights = weights * scaling_factors.reshape(1, -1)
            
            # Store updated values
            self.synaptic_weights[synapse_group_id] = scaled_weights
            self.activity_traces[synapse_group_id] = activity_traces
            self.scaling_factors[synapse_group_id] = scaling_factors
            
        except Exception as e:
            self.logger.error(f"Failed to update scaling for {synapse_group_id}: {e}")
    
    def get_scaled_weights(self, synapse_group_id: str) -> Optional[np.ndarray]:
        """Get current scaled weights."""
        return self.synaptic_weights.get(synapse_group_id)
    
    def get_scaling_factors(self, synapse_group_id: str) -> Optional[np.ndarray]:
        """Get current scaling factors."""
        return self.scaling_factors.get(synapse_group_id)
    
    def get_scaling_statistics(self, synapse_group_id: str) -> Optional[Dict[str, Any]]:
        """Get scaling statistics."""
        if synapse_group_id not in self.synaptic_weights:
            return None
        
        weights = self.synaptic_weights[synapse_group_id]
        activity_traces = self.activity_traces[synapse_group_id]
        scaling_factors = self.scaling_factors[synapse_group_id]
        
        return {
            'synapse_group_id': synapse_group_id,
            'weight_stats': {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
            },
            'activity_stats': {
                'mean': np.mean(activity_traces),
                'std': np.std(activity_traces),
                'target': self.target_activity,
            },
            'scaling_stats': {
                'mean': np.mean(scaling_factors),
                'std': np.std(scaling_factors),
                'min': np.min(scaling_factors),
                'max': np.max(scaling_factors),
            },
            'homeostasis_error': np.mean(np.abs(activity_traces - self.target_activity)),
        }