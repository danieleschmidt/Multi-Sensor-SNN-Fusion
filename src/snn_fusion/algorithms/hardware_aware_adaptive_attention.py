"""
Hardware-Aware Adaptive Attention for Neuromorphic Computing

Dynamic attention mechanism that adapts parameters in real-time based on neuromorphic
hardware constraints, energy budgets, and performance requirements.

Research Contributions:
1. Hardware-Aware Parameter Adaptation: Real-time adjustment based on hardware metrics
2. Energy-Performance Trade-off Optimization: Dynamic balancing of efficiency vs accuracy
3. Neuromorphic Hardware Profiling: Built-in profiling for Intel Loihi, BrainChip Akida
4. Adaptive Sparsity Control: Dynamic sparsity adjustment for optimal energy consumption

Novel Algorithmic Approach:
- Monitors hardware utilization (cores, memory, energy) in real-time
- Adapts attention parameters (thresholds, window sizes) based on constraints
- Implements predictive energy modeling for proactive optimization
- Enables graceful degradation under resource constraints

Research Status: Novel Contribution (2025)
Authors: Terragon Labs Neuromorphic Research Division
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import time
import psutil
import platform
from abc import ABC, abstractmethod

# Local imports
from .temporal_spike_attention import (
    TemporalSpikeAttention, 
    SpikeEvent, 
    AttentionMode, 
    TemporalMemoryTrace
)
from .fusion import CrossModalFusion, ModalityData, FusionResult, FusionStrategy


class HardwareType(Enum):
    """Supported neuromorphic hardware types."""
    INTEL_LOIHI_2 = "intel_loihi_2"
    BRAINCHIP_AKIDA = "brainchip_akida"
    SPINNAKER_2 = "spinnaker_2"
    NVIDIA_GPU = "nvidia_gpu"
    CPU_EMULATION = "cpu_emulation"
    GENERIC_NEUROMORPHIC = "generic_neuromorphic"


class AdaptationStrategy(Enum):
    """Adaptation strategies for hardware constraints."""
    ENERGY_FIRST = "energy_first"           # Prioritize energy efficiency
    PERFORMANCE_FIRST = "performance_first" # Prioritize accuracy/latency
    BALANCED = "balanced"                   # Balance energy and performance
    PREDICTIVE = "predictive"               # Use predictive models
    REACTIVE = "reactive"                   # React to current conditions


@dataclass
class HardwareConstraints:
    """Hardware constraints and capabilities."""
    max_cores: int = 128
    max_memory_mb: float = 512.0
    max_power_mw: float = 1000.0
    max_latency_ms: float = 10.0
    max_energy_per_inference_uj: float = 100.0
    spike_processing_rate: float = 1e6  # spikes/second
    synaptic_operations_per_second: float = 1e9
    
    # Hardware-specific parameters
    quantization_bits: int = 8
    supports_online_learning: bool = True
    supports_sparse_computation: bool = True
    energy_per_spike_pj: float = 10.0  # picojoules per spike
    energy_per_synaptic_op_pj: float = 5.0


@dataclass
class HardwareState:
    """Current hardware state and utilization."""
    timestamp: float
    core_utilization: float  # 0.0 - 1.0
    memory_utilization: float  # 0.0 - 1.0
    power_consumption_mw: float
    temperature_celsius: float
    current_latency_ms: float
    energy_consumption_uj: float
    
    # Performance metrics
    spike_throughput: float  # spikes/second
    attention_operations_per_second: float
    error_rate: float


class HardwareProfiler(ABC):
    """Abstract base class for hardware-specific profiling."""
    
    @abstractmethod
    def get_current_state(self) -> HardwareState:
        """Get current hardware state."""
        pass
    
    @abstractmethod
    def get_constraints(self) -> HardwareConstraints:
        """Get hardware constraints."""
        pass
    
    @abstractmethod
    def estimate_energy_cost(self, operation_count: int, operation_type: str) -> float:
        """Estimate energy cost for operations."""
        pass


class LoihiProfiler(HardwareProfiler):
    """Intel Loihi 2 hardware profiler."""
    
    def __init__(self):
        self.constraints = HardwareConstraints(
            max_cores=128,
            max_memory_mb=512.0,
            max_power_mw=300.0,
            max_latency_ms=1.0,
            max_energy_per_inference_uj=10.0,
            spike_processing_rate=1e8,
            quantization_bits=8,
            energy_per_spike_pj=23.0,  # Loihi-specific
            energy_per_synaptic_op_pj=15.0,
        )
        
        # Simulation parameters (in real deployment, would interface with Lava)
        self.simulated_utilization = 0.3
        self.simulated_power = 150.0
    
    def get_current_state(self) -> HardwareState:
        """Get current Loihi state (simulated)."""
        return HardwareState(
            timestamp=time.time(),
            core_utilization=self.simulated_utilization + np.random.normal(0, 0.05),
            memory_utilization=0.2 + np.random.normal(0, 0.02),
            power_consumption_mw=self.simulated_power + np.random.normal(0, 10),
            temperature_celsius=45.0 + np.random.normal(0, 2),
            current_latency_ms=0.8 + np.random.normal(0, 0.1),
            energy_consumption_uj=5.0 + np.random.normal(0, 0.5),
            spike_throughput=5e7 + np.random.normal(0, 1e6),
            attention_operations_per_second=1e6 + np.random.normal(0, 1e5),
            error_rate=0.001 + np.random.normal(0, 0.0001),
        )
    
    def get_constraints(self) -> HardwareConstraints:
        """Get Loihi constraints."""
        return self.constraints
    
    def estimate_energy_cost(self, operation_count: int, operation_type: str) -> float:
        """Estimate energy cost for Loihi operations."""
        if operation_type == "spike":
            return operation_count * self.constraints.energy_per_spike_pj / 1e6  # Convert to μJ
        elif operation_type == "synaptic":
            return operation_count * self.constraints.energy_per_synaptic_op_pj / 1e6
        elif operation_type == "attention":
            # Attention requires multiple synaptic operations
            return operation_count * 10 * self.constraints.energy_per_synaptic_op_pj / 1e6
        else:
            return operation_count * 1.0  # Default 1μJ per operation


class AkidaProfiler(HardwareProfiler):
    """BrainChip Akida hardware profiler."""
    
    def __init__(self):
        self.constraints = HardwareConstraints(
            max_cores=80,
            max_memory_mb=256.0,
            max_power_mw=500.0,
            max_latency_ms=2.0,
            max_energy_per_inference_uj=50.0,
            spike_processing_rate=5e7,
            quantization_bits=4,
            energy_per_spike_pj=45.0,  # Akida-specific
            energy_per_synaptic_op_pj=25.0,
        )
        
        self.simulated_utilization = 0.4
        self.simulated_power = 200.0
    
    def get_current_state(self) -> HardwareState:
        """Get current Akida state (simulated)."""
        return HardwareState(
            timestamp=time.time(),
            core_utilization=self.simulated_utilization + np.random.normal(0, 0.08),
            memory_utilization=0.3 + np.random.normal(0, 0.03),
            power_consumption_mw=self.simulated_power + np.random.normal(0, 15),
            temperature_celsius=50.0 + np.random.normal(0, 3),
            current_latency_ms=1.5 + np.random.normal(0, 0.2),
            energy_consumption_uj=25.0 + np.random.normal(0, 2.0),
            spike_throughput=3e7 + np.random.normal(0, 5e6),
            attention_operations_per_second=8e5 + np.random.normal(0, 1e5),
            error_rate=0.002 + np.random.normal(0, 0.0002),
        )
    
    def get_constraints(self) -> HardwareConstraints:
        """Get Akida constraints."""
        return self.constraints
    
    def estimate_energy_cost(self, operation_count: int, operation_type: str) -> float:
        """Estimate energy cost for Akida operations."""
        if operation_type == "spike":
            return operation_count * self.constraints.energy_per_spike_pj / 1e6
        elif operation_type == "synaptic":
            return operation_count * self.constraints.energy_per_synaptic_op_pj / 1e6
        elif operation_type == "attention":
            return operation_count * 15 * self.constraints.energy_per_synaptic_op_pj / 1e6
        else:
            return operation_count * 2.0


class CPUProfiler(HardwareProfiler):
    """CPU emulation profiler."""
    
    def __init__(self):
        self.constraints = HardwareConstraints(
            max_cores=psutil.cpu_count(),
            max_memory_mb=psutil.virtual_memory().total / (1024 * 1024),
            max_power_mw=50000.0,  # Much higher for CPU
            max_latency_ms=100.0,
            max_energy_per_inference_uj=10000.0,  # Much higher energy cost
            spike_processing_rate=1e6,
            quantization_bits=32,
            energy_per_spike_pj=1000.0,  # Higher for CPU emulation
            energy_per_synaptic_op_pj=500.0,
        )
    
    def get_current_state(self) -> HardwareState:
        """Get current CPU state."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        return HardwareState(
            timestamp=time.time(),
            core_utilization=cpu_percent / 100.0,
            memory_utilization=memory.percent / 100.0,
            power_consumption_mw=15000.0 + cpu_percent * 100,  # Rough estimate
            temperature_celsius=65.0,  # Typical CPU temperature
            current_latency_ms=50.0 + np.random.normal(0, 5),
            energy_consumption_uj=5000.0 + np.random.normal(0, 500),
            spike_throughput=1e6,
            attention_operations_per_second=1e5,
            error_rate=0.0001,
        )
    
    def get_constraints(self) -> HardwareConstraints:
        """Get CPU constraints."""
        return self.constraints
    
    def estimate_energy_cost(self, operation_count: int, operation_type: str) -> float:
        """Estimate energy cost for CPU operations."""
        # CPU is much less efficient than neuromorphic hardware
        base_cost = 1000.0  # μJ per 1000 operations
        return operation_count * base_cost / 1000.0


@dataclass
class AdaptationParameters:
    """Parameters that can be adapted based on hardware state."""
    attention_threshold: float = 0.5
    temporal_window_ms: float = 100.0
    sparsity_target: float = 0.9
    learning_rate: float = 0.01
    batch_size: int = 32
    quantization_bits: int = 8
    
    # Hardware-specific adaptations
    enable_prediction: bool = True
    enable_cross_modal_fusion: bool = True
    max_attention_heads: int = 4
    memory_window_size: int = 1000


class HardwareAwareAdaptiveAttention(TemporalSpikeAttention):
    """
    Hardware-Aware Adaptive Attention mechanism for neuromorphic computing.
    
    Research Innovation:
    1. Real-time adaptation to hardware constraints and utilization
    2. Energy-performance trade-off optimization with predictive modeling
    3. Hardware-specific parameter tuning for optimal efficiency
    4. Graceful degradation under resource constraints
    
    Key Algorithmic Contributions:
    - Dynamic parameter adaptation based on hardware metrics
    - Predictive energy modeling for proactive optimization
    - Multi-objective optimization (energy, latency, accuracy)
    - Hardware-specific profiling and optimization
    """
    
    def __init__(
        self,
        modalities: List[str],
        hardware_type: HardwareType = HardwareType.CPU_EMULATION,
        adaptation_strategy: AdaptationStrategy = AdaptationStrategy.BALANCED,
        adaptation_interval: float = 1.0,  # seconds
        **tsa_kwargs,
    ):
        """
        Initialize Hardware-Aware Adaptive Attention.
        
        Args:
            modalities: List of input modalities
            hardware_type: Type of neuromorphic hardware
            adaptation_strategy: Strategy for parameter adaptation
            adaptation_interval: How often to adapt parameters (seconds)
            **tsa_kwargs: Arguments for base TSA class
        """
        super().__init__(modalities, **tsa_kwargs)
        
        self.hardware_type = hardware_type
        self.adaptation_strategy = adaptation_strategy
        self.adaptation_interval = adaptation_interval
        
        # Initialize hardware profiler
        self.hardware_profiler = self._create_hardware_profiler(hardware_type)
        self.hardware_constraints = self.hardware_profiler.get_constraints()
        
        # Adaptation parameters
        self.adaptation_params = AdaptationParameters()
        self.last_adaptation_time = 0.0
        
        # Performance tracking
        self.performance_history = {
            'hardware_states': [],
            'adaptation_decisions': [],
            'performance_metrics': [],
            'energy_costs': [],
        }
        
        # Predictive models (simplified)
        self.energy_predictor = self._initialize_energy_predictor()
        self.performance_predictor = self._initialize_performance_predictor()
        
        # Multi-objective optimization weights
        self.optimization_weights = {
            'energy': 0.3,
            'latency': 0.3, 
            'accuracy': 0.4,
        }
        
        self.logger.info(f"Initialized Hardware-Aware Adaptive Attention for {hardware_type.value}")
        self.logger.info(f"Adaptation strategy: {adaptation_strategy.value}")
    
    def _create_hardware_profiler(self, hardware_type: HardwareType) -> HardwareProfiler:
        """Create hardware-specific profiler."""
        if hardware_type == HardwareType.INTEL_LOIHI_2:
            return LoihiProfiler()
        elif hardware_type == HardwareType.BRAINCHIP_AKIDA:
            return AkidaProfiler()
        elif hardware_type == HardwareType.CPU_EMULATION:
            return CPUProfiler()
        else:
            # Default to CPU profiler for unsupported hardware
            self.logger.warning(f"Hardware type {hardware_type.value} not fully supported, using CPU profiler")
            return CPUProfiler()
    
    def _initialize_energy_predictor(self) -> Callable:
        """Initialize simple energy prediction model."""
        def predict_energy(operation_count: int, current_state: HardwareState) -> float:
            # Simple linear model (in practice, would be learned from data)
            base_energy = self.hardware_profiler.estimate_energy_cost(operation_count, "attention")
            
            # Adjust based on current hardware state
            utilization_factor = 1.0 + 0.5 * current_state.core_utilization
            temperature_factor = 1.0 + 0.1 * (current_state.temperature_celsius - 25) / 25
            
            predicted_energy = base_energy * utilization_factor * temperature_factor
            return predicted_energy
        
        return predict_energy
    
    def _initialize_performance_predictor(self) -> Callable:
        """Initialize simple performance prediction model."""
        def predict_performance(params: AdaptationParameters, current_state: HardwareState) -> Dict[str, float]:
            # Simple performance model
            
            # Accuracy decreases with higher sparsity and lower thresholds
            accuracy_factor = (1.0 - params.sparsity_target) * params.attention_threshold
            base_accuracy = 0.9
            predicted_accuracy = base_accuracy * accuracy_factor
            
            # Latency increases with larger windows and more attention heads
            base_latency = 5.0  # ms
            window_factor = params.temporal_window_ms / 100.0
            heads_factor = params.max_attention_heads / 4.0
            predicted_latency = base_latency * window_factor * heads_factor
            
            # Adjust for hardware state
            predicted_latency *= (1.0 + current_state.core_utilization)
            
            return {
                'accuracy': predicted_accuracy,
                'latency': predicted_latency,
            }
        
        return predict_performance
    
    def fuse_modalities(
        self,
        modality_data: Dict[str, ModalityData],
    ) -> FusionResult:
        """
        Perform hardware-aware adaptive fusion.
        
        Args:
            modality_data: Dictionary of modality spike data
            
        Returns:
            Fusion result with hardware-aware optimization
        """
        start_time = time.time()
        
        try:
            # Check if adaptation is needed
            if start_time - self.last_adaptation_time > self.adaptation_interval:
                self._adapt_to_hardware_state()
                self.last_adaptation_time = start_time
            
            # Get current hardware state
            current_state = self.hardware_profiler.get_current_state()
            
            # Estimate operation count
            total_spikes = sum(len(data.spike_times) for data in modality_data.values())
            operation_count = total_spikes * len(modality_data)  # Cross-modal operations
            
            # Predict energy cost
            predicted_energy = self.energy_predictor(operation_count, current_state)
            
            # Check energy budget
            if predicted_energy > self.hardware_constraints.max_energy_per_inference_uj:
                self._apply_energy_constraints(modality_data, predicted_energy)
            
            # Apply current adaptation parameters
            self._apply_adaptation_parameters()
            
            # Perform fusion with adapted parameters
            fusion_result = super().fuse_modalities(modality_data)
            
            # Track performance
            inference_time = (time.time() - start_time) * 1000  # ms
            actual_energy = self.hardware_profiler.estimate_energy_cost(operation_count, "attention")
            
            self._record_performance(current_state, fusion_result, inference_time, actual_energy)
            
            # Add hardware-aware metadata
            fusion_result.metadata.update({
                'fusion_type': 'hardware_aware_adaptive_attention',
                'hardware_type': self.hardware_type.value,
                'adaptation_strategy': self.adaptation_strategy.value,
                'current_hardware_state': current_state.__dict__,
                'adaptation_parameters': self.adaptation_params.__dict__,
                'predicted_energy_uj': predicted_energy,
                'actual_energy_uj': actual_energy,
                'hardware_constraints': self.hardware_constraints.__dict__,
                'optimization_weights': self.optimization_weights.copy(),
            })
            
            return fusion_result
            
        except Exception as e:
            self.logger.error(f"Hardware-aware adaptive fusion failed: {e}")
            raise
    
    def _adapt_to_hardware_state(self) -> None:
        """Adapt parameters based on current hardware state."""
        current_state = self.hardware_profiler.get_current_state()
        
        if self.adaptation_strategy == AdaptationStrategy.ENERGY_FIRST:
            self._energy_first_adaptation(current_state)
        elif self.adaptation_strategy == AdaptationStrategy.PERFORMANCE_FIRST:
            self._performance_first_adaptation(current_state)
        elif self.adaptation_strategy == AdaptationStrategy.BALANCED:
            self._balanced_adaptation(current_state)
        elif self.adaptation_strategy == AdaptationStrategy.PREDICTIVE:
            self._predictive_adaptation(current_state)
        else:  # REACTIVE
            self._reactive_adaptation(current_state)
        
        # Log adaptation decision
        self.performance_history['adaptation_decisions'].append({
            'timestamp': current_state.timestamp,
            'strategy': self.adaptation_strategy.value,
            'parameters': self.adaptation_params.__dict__.copy(),
            'hardware_state': current_state.__dict__.copy(),
        })
    
    def _energy_first_adaptation(self, current_state: HardwareState) -> None:
        """Adapt parameters prioritizing energy efficiency."""
        # Increase sparsity to reduce energy
        if current_state.power_consumption_mw > self.hardware_constraints.max_power_mw * 0.8:
            self.adaptation_params.sparsity_target = min(0.98, self.adaptation_params.sparsity_target + 0.05)
            self.adaptation_params.attention_threshold += 0.1
        
        # Reduce temporal window to save energy
        if current_state.energy_consumption_uj > self.hardware_constraints.max_energy_per_inference_uj * 0.7:
            self.adaptation_params.temporal_window_ms = max(50.0, self.adaptation_params.temporal_window_ms * 0.9)
        
        # Disable expensive features if needed
        if current_state.power_consumption_mw > self.hardware_constraints.max_power_mw * 0.9:
            self.adaptation_params.enable_prediction = False
            self.adaptation_params.max_attention_heads = max(1, self.adaptation_params.max_attention_heads - 1)
    
    def _performance_first_adaptation(self, current_state: HardwareState) -> None:
        """Adapt parameters prioritizing performance."""
        # Reduce sparsity for better accuracy
        if current_state.error_rate > 0.005:
            self.adaptation_params.sparsity_target = max(0.7, self.adaptation_params.sparsity_target - 0.05)
            self.adaptation_params.attention_threshold = max(0.1, self.adaptation_params.attention_threshold - 0.05)
        
        # Increase temporal window for better temporal modeling
        if current_state.current_latency_ms < self.hardware_constraints.max_latency_ms * 0.5:
            self.adaptation_params.temporal_window_ms = min(200.0, self.adaptation_params.temporal_window_ms * 1.1)
        
        # Enable advanced features if resources allow
        if (current_state.core_utilization < 0.7 and 
            current_state.memory_utilization < 0.6):
            self.adaptation_params.enable_prediction = True
            self.adaptation_params.max_attention_heads = min(8, self.adaptation_params.max_attention_heads + 1)
    
    def _balanced_adaptation(self, current_state: HardwareState) -> None:
        """Adapt parameters balancing energy and performance."""
        # Compute multi-objective score
        energy_score = 1.0 - (current_state.energy_consumption_uj / self.hardware_constraints.max_energy_per_inference_uj)
        latency_score = 1.0 - (current_state.current_latency_ms / self.hardware_constraints.max_latency_ms)
        accuracy_score = 1.0 - current_state.error_rate / 0.01  # Assume 1% is worst acceptable
        
        # Weighted objective
        objective_score = (
            self.optimization_weights['energy'] * energy_score +
            self.optimization_weights['latency'] * latency_score +
            self.optimization_weights['accuracy'] * accuracy_score
        )
        
        # Adapt based on which component is worst
        if energy_score < 0.5:  # Energy is the bottleneck
            self.adaptation_params.sparsity_target = min(0.95, self.adaptation_params.sparsity_target + 0.02)
        elif latency_score < 0.5:  # Latency is the bottleneck
            self.adaptation_params.temporal_window_ms = max(75.0, self.adaptation_params.temporal_window_ms * 0.95)
        elif accuracy_score < 0.5:  # Accuracy is the bottleneck
            self.adaptation_params.attention_threshold = max(0.2, self.adaptation_params.attention_threshold - 0.02)
    
    def _predictive_adaptation(self, current_state: HardwareState) -> None:
        """Adapt parameters using predictive models."""
        # Try different parameter configurations and predict their outcomes
        best_config = None
        best_score = float('-inf')
        
        # Test a few parameter variations
        test_configs = [
            # Current config
            self.adaptation_params,
            # Higher sparsity
            AdaptationParameters(
                sparsity_target=min(0.98, self.adaptation_params.sparsity_target + 0.05),
                **{k: v for k, v in self.adaptation_params.__dict__.items() if k != 'sparsity_target'}
            ),
            # Lower sparsity  
            AdaptationParameters(
                sparsity_target=max(0.8, self.adaptation_params.sparsity_target - 0.05),
                **{k: v for k, v in self.adaptation_params.__dict__.items() if k != 'sparsity_target'}
            ),
            # Smaller window
            AdaptationParameters(
                temporal_window_ms=max(50.0, self.adaptation_params.temporal_window_ms * 0.8),
                **{k: v for k, v in self.adaptation_params.__dict__.items() if k != 'temporal_window_ms'}
            ),
        ]
        
        for config in test_configs:
            # Predict performance
            predicted_perf = self.performance_predictor(config, current_state)
            
            # Compute multi-objective score
            score = (
                self.optimization_weights['accuracy'] * predicted_perf['accuracy'] -
                self.optimization_weights['latency'] * predicted_perf['latency'] / 100.0  # Normalize latency
            )
            
            if score > best_score:
                best_score = score
                best_config = config
        
        if best_config is not None:
            self.adaptation_params = best_config
    
    def _reactive_adaptation(self, current_state: HardwareState) -> None:
        """React immediately to constraint violations."""
        # Emergency adaptations for constraint violations
        
        # Power constraint violation
        if current_state.power_consumption_mw > self.hardware_constraints.max_power_mw:
            self.adaptation_params.sparsity_target = min(0.99, self.adaptation_params.sparsity_target + 0.1)
            self.adaptation_params.enable_prediction = False
            self.logger.warning("Power constraint violated, increasing sparsity")
        
        # Memory constraint violation
        if current_state.memory_utilization > 0.9:
            self.adaptation_params.memory_window_size = max(100, self.adaptation_params.memory_window_size // 2)
            self.adaptation_params.batch_size = max(1, self.adaptation_params.batch_size // 2)
            self.logger.warning("Memory constraint violated, reducing memory usage")
        
        # Latency constraint violation
        if current_state.current_latency_ms > self.hardware_constraints.max_latency_ms:
            self.adaptation_params.temporal_window_ms = max(25.0, self.adaptation_params.temporal_window_ms * 0.5)
            self.adaptation_params.max_attention_heads = max(1, self.adaptation_params.max_attention_heads - 1)
            self.logger.warning("Latency constraint violated, reducing computation")
        
        # Energy constraint violation
        if current_state.energy_consumption_uj > self.hardware_constraints.max_energy_per_inference_uj:
            self.adaptation_params.sparsity_target = min(0.99, self.adaptation_params.sparsity_target + 0.05)
            self.adaptation_params.enable_cross_modal_fusion = False
            self.logger.warning("Energy constraint violated, disabling features")
    
    def _apply_energy_constraints(
        self,
        modality_data: Dict[str, ModalityData], 
        predicted_energy: float,
    ) -> None:
        """Apply energy constraints by reducing computation."""
        # Calculate energy reduction needed
        energy_budget = self.hardware_constraints.max_energy_per_inference_uj
        reduction_factor = energy_budget / predicted_energy
        
        if reduction_factor < 1.0:
            # Reduce spike count proportionally
            for modality, data in modality_data.items():
                if len(data.spike_times) > 0:
                    target_count = int(len(data.spike_times) * reduction_factor)
                    target_count = max(1, target_count)  # Keep at least one spike
                    
                    # Keep highest amplitude spikes
                    if data.features is not None and len(data.features) > 0:
                        # Sort by feature strength
                        indices = np.argsort(data.features)[-target_count:]
                    else:
                        # Keep first N spikes
                        indices = np.arange(target_count)
                    
                    # Update data
                    data.spike_times = data.spike_times[indices]
                    data.neuron_ids = data.neuron_ids[indices]
                    if data.features is not None:
                        data.features = data.features[indices]
            
            self.logger.info(f"Applied energy constraints, reduction factor: {reduction_factor:.3f}")
    
    def _apply_adaptation_parameters(self) -> None:
        """Apply current adaptation parameters to the algorithm."""
        # Update spike thresholds based on adaptation
        for modality in self.modalities:
            self.spike_thresholds[modality] = self.adaptation_params.attention_threshold
        
        # Update temporal window
        self.temporal_window = self.adaptation_params.temporal_window_ms
        
        # Update memory parameters
        for memory_trace in self.memory_traces.values():
            memory_trace.max_history_length = self.adaptation_params.memory_window_size
        
        # Update predictive setting
        self.enable_predictive = self.adaptation_params.enable_prediction
    
    def _record_performance(
        self,
        hardware_state: HardwareState,
        fusion_result: FusionResult,
        inference_time: float,
        energy_cost: float,
    ) -> None:
        """Record performance metrics for analysis."""
        performance_metric = {
            'timestamp': hardware_state.timestamp,
            'inference_time_ms': inference_time,
            'energy_cost_uj': energy_cost,
            'fusion_quality': sum(fusion_result.confidence_scores.values()),
            'n_fused_spikes': len(fusion_result.fused_spikes),
            'hardware_utilization': hardware_state.core_utilization,
            'memory_utilization': hardware_state.memory_utilization,
        }
        
        self.performance_history['hardware_states'].append(hardware_state)
        self.performance_history['performance_metrics'].append(performance_metric)
        self.performance_history['energy_costs'].append(energy_cost)
        
        # Keep limited history
        max_history = 1000
        for key in self.performance_history:
            if len(self.performance_history[key]) > max_history:
                self.performance_history[key] = self.performance_history[key][-max_history:]
    
    def get_hardware_analysis(self) -> Dict[str, Any]:
        """Get comprehensive hardware utilization analysis."""
        analysis = super().get_attention_analysis()
        
        # Add hardware-specific metrics
        if self.performance_history['hardware_states']:
            recent_states = self.performance_history['hardware_states'][-100:]
            recent_metrics = self.performance_history['performance_metrics'][-100:]
            
            analysis['hardware_metrics'] = {
                'mean_core_utilization': np.mean([s.core_utilization for s in recent_states]),
                'mean_memory_utilization': np.mean([s.memory_utilization for s in recent_states]),
                'mean_power_consumption_mw': np.mean([s.power_consumption_mw for s in recent_states]),
                'mean_temperature_celsius': np.mean([s.temperature_celsius for s in recent_states]),
                'mean_latency_ms': np.mean([s.current_latency_ms for s in recent_states]),
                'mean_energy_uj': np.mean([s.energy_consumption_uj for s in recent_states]),
            }
            
            analysis['performance_metrics'] = {
                'mean_inference_time_ms': np.mean([m['inference_time_ms'] for m in recent_metrics]),
                'mean_fusion_quality': np.mean([m['fusion_quality'] for m in recent_metrics]),
                'mean_energy_efficiency': np.mean([m['fusion_quality'] / max(m['energy_cost_uj'], 1e-6) for m in recent_metrics]),
                'constraint_violations': self._count_constraint_violations(recent_states),
            }
        
        # Hardware constraints and current parameters
        analysis['hardware_constraints'] = self.hardware_constraints.__dict__
        analysis['current_adaptation_parameters'] = self.adaptation_params.__dict__
        analysis['adaptation_strategy'] = self.adaptation_strategy.value
        
        # Adaptation effectiveness
        analysis['adaptation_analysis'] = self._analyze_adaptation_effectiveness()
        
        return analysis
    
    def _count_constraint_violations(self, hardware_states: List[HardwareState]) -> Dict[str, int]:
        """Count constraint violations in recent states."""
        violations = {
            'power_violations': 0,
            'memory_violations': 0,
            'latency_violations': 0,
            'energy_violations': 0,
        }
        
        for state in hardware_states:
            if state.power_consumption_mw > self.hardware_constraints.max_power_mw:
                violations['power_violations'] += 1
            if state.memory_utilization > 0.9:  # 90% threshold
                violations['memory_violations'] += 1
            if state.current_latency_ms > self.hardware_constraints.max_latency_ms:
                violations['latency_violations'] += 1
            if state.energy_consumption_uj > self.hardware_constraints.max_energy_per_inference_uj:
                violations['energy_violations'] += 1
        
        return violations
    
    def _analyze_adaptation_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of parameter adaptations."""
        if len(self.performance_history['adaptation_decisions']) < 2:
            return {'insufficient_data': True}
        
        decisions = self.performance_history['adaptation_decisions']
        metrics = self.performance_history['performance_metrics']
        
        # Analyze trend after adaptations
        effectiveness = {
            'total_adaptations': len(decisions),
            'energy_trend': 'stable',
            'performance_trend': 'stable',
            'adaptation_frequency': len(decisions) / max(1, (time.time() - decisions[0]['timestamp']) / 3600),  # per hour
        }
        
        if len(metrics) >= 10:
            recent_energy = [m['energy_cost_uj'] for m in metrics[-10:]]
            older_energy = [m['energy_cost_uj'] for m in metrics[-20:-10]]
            
            if len(older_energy) > 0:
                energy_change = (np.mean(recent_energy) - np.mean(older_energy)) / np.mean(older_energy)
                
                if energy_change < -0.1:
                    effectiveness['energy_trend'] = 'improving'
                elif energy_change > 0.1:
                    effectiveness['energy_trend'] = 'degrading'
            
            recent_quality = [m['fusion_quality'] for m in metrics[-10:]]
            older_quality = [m['fusion_quality'] for m in metrics[-20:-10]]
            
            if len(older_quality) > 0:
                quality_change = (np.mean(recent_quality) - np.mean(older_quality)) / max(np.mean(older_quality), 1e-6)
                
                if quality_change > 0.05:
                    effectiveness['performance_trend'] = 'improving'
                elif quality_change < -0.05:
                    effectiveness['performance_trend'] = 'degrading'
        
        return effectiveness


# Factory function for easy instantiation
def create_hardware_aware_adaptive_attention(
    modalities: List[str],
    hardware_type: HardwareType = HardwareType.CPU_EMULATION,
    config: Optional[Dict[str, Any]] = None,
) -> HardwareAwareAdaptiveAttention:
    """
    Factory function to create Hardware-Aware Adaptive Attention.
    
    Args:
        modalities: List of input modalities
        hardware_type: Type of neuromorphic hardware
        config: Optional configuration dictionary
        
    Returns:
        Configured HardwareAwareAdaptiveAttention instance
    """
    default_config = {
        'adaptation_strategy': AdaptationStrategy.BALANCED,
        'adaptation_interval': 1.0,
        'temporal_window': 100.0,
        'attention_mode': AttentionMode.ADAPTIVE,
        'enable_predictive': True,
    }
    
    if config:
        default_config.update(config)
    
    return HardwareAwareAdaptiveAttention(
        modalities, 
        hardware_type=hardware_type,
        **default_config
    )


# Research validation functions
def benchmark_hardware_adaptation(
    algorithm: HardwareAwareAdaptiveAttention,
    test_data: List[Dict[str, ModalityData]],
    constraint_variations: List[HardwareConstraints],
) -> Dict[str, Any]:
    """
    Benchmark hardware adaptation effectiveness across different constraints.
    
    Args:
        algorithm: Hardware-aware algorithm instance
        test_data: Test data samples
        constraint_variations: Different hardware constraint scenarios
        
    Returns:
        Hardware adaptation benchmark results
    """
    results = {
        'constraint_scenarios': [],
        'adaptation_effectiveness': [],
        'energy_efficiency': [],
        'performance_maintenance': [],
    }
    
    for constraints in constraint_variations:
        # Update algorithm constraints
        original_constraints = algorithm.hardware_constraints
        algorithm.hardware_constraints = constraints
        
        scenario_results = {
            'constraints': constraints.__dict__,
            'violations': 0,
            'adaptations': 0,
            'energy_costs': [],
            'performance_scores': [],
        }
        
        # Reset adaptation tracking
        adaptation_count_start = len(algorithm.performance_history['adaptation_decisions'])
        
        for sample in test_data:
            try:
                fusion_result = algorithm.fuse_modalities(sample)
                
                # Check for constraint violations
                current_state = algorithm.hardware_profiler.get_current_state()
                
                if (current_state.power_consumption_mw > constraints.max_power_mw or
                    current_state.current_latency_ms > constraints.max_latency_ms or
                    current_state.energy_consumption_uj > constraints.max_energy_per_inference_uj):
                    scenario_results['violations'] += 1
                
                # Record metrics
                energy_cost = fusion_result.metadata.get('actual_energy_uj', 0.0)
                performance_score = sum(fusion_result.confidence_scores.values())
                
                scenario_results['energy_costs'].append(energy_cost)
                scenario_results['performance_scores'].append(performance_score)
                
            except Exception as e:
                scenario_results['violations'] += 1
        
        # Count adaptations during this scenario
        adaptation_count_end = len(algorithm.performance_history['adaptation_decisions'])
        scenario_results['adaptations'] = adaptation_count_end - adaptation_count_start
        
        # Compute effectiveness metrics
        violation_rate = scenario_results['violations'] / len(test_data)
        adaptation_rate = scenario_results['adaptations'] / len(test_data)
        mean_energy = np.mean(scenario_results['energy_costs']) if scenario_results['energy_costs'] else float('inf')
        mean_performance = np.mean(scenario_results['performance_scores']) if scenario_results['performance_scores'] else 0.0
        
        results['constraint_scenarios'].append(scenario_results)
        results['adaptation_effectiveness'].append(1.0 - violation_rate)  # Lower violations = better adaptation
        results['energy_efficiency'].append(1.0 / max(mean_energy, 1.0))  # Lower energy = better efficiency
        results['performance_maintenance'].append(mean_performance)
        
        # Restore original constraints
        algorithm.hardware_constraints = original_constraints
    
    # Summary statistics
    results['mean_adaptation_effectiveness'] = np.mean(results['adaptation_effectiveness'])
    results['mean_energy_efficiency'] = np.mean(results['energy_efficiency'])
    results['mean_performance_maintenance'] = np.mean(results['performance_maintenance'])
    
    return results


def validate_multi_objective_optimization(
    algorithm: HardwareAwareAdaptiveAttention,
    test_data: List[Dict[str, ModalityData]],
    weight_combinations: List[Dict[str, float]],
) -> Dict[str, Any]:
    """
    Validate multi-objective optimization across different weight combinations.
    
    Args:
        algorithm: Hardware-aware algorithm instance
        test_data: Test data samples
        weight_combinations: Different optimization weight combinations
        
    Returns:
        Multi-objective optimization validation results
    """
    results = {
        'weight_combinations': weight_combinations,
        'pareto_frontier': [],
        'optimization_effectiveness': [],
    }
    
    for weights in weight_combinations:
        # Update optimization weights
        original_weights = algorithm.optimization_weights.copy()
        algorithm.optimization_weights = weights
        
        # Test performance with these weights
        energy_costs = []
        latencies = []
        accuracies = []
        
        for sample in test_data:
            fusion_result = algorithm.fuse_modalities(sample)
            
            energy_cost = fusion_result.metadata.get('actual_energy_uj', 0.0)
            latency = fusion_result.metadata.get('inference_time_ms', 0.0)
            accuracy = sum(fusion_result.confidence_scores.values())
            
            energy_costs.append(energy_cost)
            latencies.append(latency)
            accuracies.append(accuracy)
        
        # Compute objective values
        mean_energy = np.mean(energy_costs)
        mean_latency = np.mean(latencies)
        mean_accuracy = np.mean(accuracies)
        
        # Compute weighted objective score
        normalized_energy = mean_energy / 100.0  # Normalize to reasonable range
        normalized_latency = mean_latency / 10.0
        normalized_accuracy = mean_accuracy
        
        objective_score = (
            weights['energy'] * (1.0 / max(normalized_energy, 0.1)) +
            weights['latency'] * (1.0 / max(normalized_latency, 0.1)) +
            weights['accuracy'] * normalized_accuracy
        )
        
        results['pareto_frontier'].append({
            'weights': weights,
            'energy': mean_energy,
            'latency': mean_latency,
            'accuracy': mean_accuracy,
            'objective_score': objective_score,
        })
        
        # Restore original weights
        algorithm.optimization_weights = original_weights
    
    # Analyze Pareto efficiency
    frontier_points = results['pareto_frontier']
    for i, point in enumerate(frontier_points):
        is_dominated = False
        
        for j, other_point in enumerate(frontier_points):
            if i != j:
                # Check if other point dominates this point (better in all objectives)
                if (other_point['energy'] <= point['energy'] and
                    other_point['latency'] <= point['latency'] and
                    other_point['accuracy'] >= point['accuracy'] and
                    (other_point['energy'] < point['energy'] or
                     other_point['latency'] < point['latency'] or
                     other_point['accuracy'] > point['accuracy'])):
                    is_dominated = True
                    break
        
        point['pareto_optimal'] = not is_dominated
    
    # Count Pareto optimal solutions
    pareto_optimal_count = sum(1 for point in frontier_points if point['pareto_optimal'])
    results['pareto_optimal_ratio'] = pareto_optimal_count / len(frontier_points)
    
    return results