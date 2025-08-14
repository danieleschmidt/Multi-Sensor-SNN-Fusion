"""
Advanced Performance Optimizer for Neuromorphic Systems

Ultra-high performance optimization system specifically designed for
neuromorphic multi-modal fusion with hardware acceleration, memory
optimization, and adaptive performance tuning.
"""

import numpy as np
import torch
import threading
import time
import gc
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import psutil
import logging
from pathlib import Path

# Hardware acceleration imports
try:
    import torch.nn.functional as F
    import torch.cuda.amp as amp
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

# Intel MKL and OpenMP optimization
try:
    import mkl
    MKL_AVAILABLE = True
except ImportError:
    MKL_AVAILABLE = False

from ..algorithms.fusion import CrossModalFusion, ModalityData, FusionResult
from ..algorithms.temporal_spike_attention import TemporalSpikeAttention
from ..utils.robust_error_handling import robust_function


class OptimizationLevel(Enum):
    """Optimization levels for different deployment scenarios."""
    CONSERVATIVE = "conservative"    # Safe optimizations
    AGGRESSIVE = "aggressive"       # Maximum performance
    NEUROMORPHIC = "neuromorphic"   # Neuromorphic hardware optimized
    EDGE = "edge"                  # Edge device optimized


class PerformanceMetric(Enum):
    """Performance metrics to optimize."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    ENERGY_EFFICIENCY = "energy_efficiency"
    ACCURACY = "accuracy"


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    target_latency_ms: float = 10.0
    target_throughput_ops_per_sec: float = 100.0
    max_memory_mb: float = 1000.0
    enable_gpu_acceleration: bool = True
    enable_mixed_precision: bool = True
    enable_jit_compilation: bool = True
    enable_memory_pooling: bool = True
    enable_batch_processing: bool = True
    optimization_level: OptimizationLevel = OptimizationLevel.AGGRESSIVE
    target_hardware: str = "cpu"  # "cpu", "gpu", "loihi2", "akida"


@dataclass
class PerformanceProfile:
    """Performance profiling results."""
    latency_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    gpu_utilization: float
    cpu_utilization: float
    energy_estimate_mw: float
    accuracy_score: float
    timestamp: float = field(default_factory=time.time)


class MemoryOptimizer:
    """
    Advanced memory optimization for neuromorphic fusion systems.
    
    Features:
    - Memory pooling and reuse
    - Garbage collection optimization
    - Sparse data structure optimization
    - Memory-mapped file operations
    """
    
    def __init__(self, max_pool_size_mb: float = 500.0):
        """
        Initialize memory optimizer.
        
        Args:
            max_pool_size_mb: Maximum memory pool size in MB
        """
        self.max_pool_size_mb = max_pool_size_mb
        self.tensor_pools = {}  # Pre-allocated tensor pools
        self.memory_stats = {
            'allocations': 0,
            'deallocations': 0,
            'peak_usage_mb': 0.0,
            'current_usage_mb': 0.0,
            'pool_hits': 0,
            'pool_misses': 0,
        }
        
        self.logger = logging.getLogger(__name__)
        self._setup_memory_monitoring()
    
    def _setup_memory_monitoring(self):
        """Setup continuous memory monitoring."""
        def memory_monitor():
            while True:
                try:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    current_mb = memory_info.rss / (1024**2)
                    
                    self.memory_stats['current_usage_mb'] = current_mb
                    if current_mb > self.memory_stats['peak_usage_mb']:
                        self.memory_stats['peak_usage_mb'] = current_mb
                    
                    # Trigger GC if memory usage is high
                    if current_mb > self.max_pool_size_mb * 0.8:
                        self._optimize_memory_usage()
                    
                    time.sleep(1.0)  # Check every second
                    
                except Exception:
                    break
        
        monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
        monitor_thread.start()
    
    def get_optimized_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Get optimized tensor from pool or create new one.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            device: Device placement
            
        Returns:
            Optimized tensor
        """
        pool_key = (shape, dtype, device)
        
        if pool_key in self.tensor_pools and self.tensor_pools[pool_key]:
            # Reuse from pool
            tensor = self.tensor_pools[pool_key].pop()
            tensor.zero_()  # Clear previous data
            self.memory_stats['pool_hits'] += 1
            return tensor
        else:
            # Create new tensor
            tensor = torch.zeros(shape, dtype=dtype, device=device)
            self.memory_stats['allocations'] += 1
            self.memory_stats['pool_misses'] += 1
            return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse."""
        if tensor.numel() == 0:
            return
        
        pool_key = (tensor.shape, tensor.dtype, str(tensor.device))
        
        if pool_key not in self.tensor_pools:
            self.tensor_pools[pool_key] = deque(maxlen=10)  # Limit pool size
        
        # Only pool if we have space
        pool_size_mb = sum(
            len(pool) * torch.tensor(shape).prod().item() * 4 / (1024**2)  # Assume 4 bytes per element
            for (shape, _, _), pool in self.tensor_pools.items()
        )
        
        if pool_size_mb < self.max_pool_size_mb:
            self.tensor_pools[pool_key].append(tensor.detach())
        else:
            del tensor
            self.memory_stats['deallocations'] += 1
    
    def _optimize_memory_usage(self):
        """Optimize memory usage when high."""
        # Clear tensor pools
        for pool in self.tensor_pools.values():
            pool.clear()
        
        # Force garbage collection
        gc.collect()
        
        # CUDA memory cleanup if available
        if CUDA_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.logger.info("Memory optimization triggered")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics."""
        return self.memory_stats.copy()


class SpikeBatchProcessor:
    """
    High-performance batch processor for spike data.
    
    Features:
    - Automatic batching and vectorization
    - Memory-efficient sparse operations
    - Hardware-accelerated processing
    - Adaptive batch size optimization
    """
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        max_batch_size: int = 256,
        enable_gpu: bool = True,
    ):
        """
        Initialize batch processor.
        
        Args:
            initial_batch_size: Initial batch size
            max_batch_size: Maximum batch size
            enable_gpu: Enable GPU acceleration
        """
        self.batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.enable_gpu = enable_gpu and CUDA_AVAILABLE
        
        self.device = torch.device("cuda" if self.enable_gpu else "cpu")
        self.memory_optimizer = MemoryOptimizer()
        
        # Batch processing statistics
        self.processing_stats = {
            'batches_processed': 0,
            'total_samples': 0,
            'average_latency_ms': 0.0,
            'average_throughput': 0.0,
        }
        
        self.logger = logging.getLogger(__name__)
    
    def optimize_batch_size(self, target_latency_ms: float = 10.0):
        """
        Automatically optimize batch size for target latency.
        
        Args:
            target_latency_ms: Target latency in milliseconds
        """
        current_batch_size = self.batch_size
        best_batch_size = current_batch_size
        best_throughput = 0.0
        
        # Test different batch sizes
        test_sizes = [16, 32, 64, 128, 256]
        test_data = self._create_test_data()
        
        for test_size in test_sizes:
            if test_size > self.max_batch_size:
                continue
            
            self.batch_size = test_size
            
            # Measure performance
            start_time = time.time()
            n_iterations = 10
            
            for _ in range(n_iterations):
                self._process_batch_internal(test_data)
            
            total_time = time.time() - start_time
            latency_ms = (total_time / n_iterations) * 1000
            throughput = (test_size * n_iterations) / total_time
            
            if latency_ms <= target_latency_ms and throughput > best_throughput:
                best_batch_size = test_size
                best_throughput = throughput
        
        self.batch_size = best_batch_size
        self.logger.info(f"Optimized batch size to {best_batch_size} for {target_latency_ms}ms target")
    
    def process_spike_batch(
        self,
        spike_data_batch: List[Dict[str, ModalityData]],
    ) -> List[torch.Tensor]:
        """
        Process batch of spike data with optimizations.
        
        Args:
            spike_data_batch: Batch of spike data
            
        Returns:
            Batch of processed tensors
        """
        if not spike_data_batch:
            return []
        
        start_time = time.time()
        
        # Convert to optimized format
        batch_tensors = self._convert_to_batch_tensors(spike_data_batch)
        
        # Process on optimal device
        if self.enable_gpu:
            batch_tensors = [t.to(self.device) for t in batch_tensors]
        
        # Vectorized processing
        processed_tensors = self._process_vectorized(batch_tensors)
        
        # Update statistics
        processing_time = (time.time() - start_time) * 1000  # ms
        self.processing_stats['batches_processed'] += 1
        self.processing_stats['total_samples'] += len(spike_data_batch)
        
        # Update running averages
        alpha = 0.1  # EMA smoothing factor
        self.processing_stats['average_latency_ms'] = (
            (1 - alpha) * self.processing_stats['average_latency_ms'] + 
            alpha * processing_time
        )
        
        throughput = len(spike_data_batch) / (processing_time / 1000.0)
        self.processing_stats['average_throughput'] = (
            (1 - alpha) * self.processing_stats['average_throughput'] +
            alpha * throughput
        )
        
        return processed_tensors
    
    def _convert_to_batch_tensors(
        self,
        spike_data_batch: List[Dict[str, ModalityData]],
    ) -> List[torch.Tensor]:
        """Convert spike data batch to optimized tensor format."""
        if not spike_data_batch:
            return []
        
        # Determine common modalities
        all_modalities = set()
        for sample in spike_data_batch:
            all_modalities.update(sample.keys())
        
        batch_tensors = []
        
        for modality in sorted(all_modalities):
            # Find maximum dimensions for padding
            max_spikes = 0
            for sample in spike_data_batch:
                if modality in sample:
                    max_spikes = max(max_spikes, len(sample[modality].spike_times))
            
            if max_spikes == 0:
                continue
            
            # Create batch tensor
            batch_tensor = self.memory_optimizer.get_optimized_tensor(
                (len(spike_data_batch), max_spikes, 3),  # [batch, spikes, features]
                dtype=torch.float32,
                device=str(self.device)
            )
            
            # Fill batch tensor
            for batch_idx, sample in enumerate(spike_data_batch):
                if modality in sample:
                    data = sample[modality]
                    n_spikes = len(data.spike_times)
                    
                    if n_spikes > 0:
                        # Pack spike times, neuron IDs, and features
                        batch_tensor[batch_idx, :n_spikes, 0] = torch.from_numpy(data.spike_times)
                        batch_tensor[batch_idx, :n_spikes, 1] = torch.from_numpy(data.neuron_ids.astype(np.float32))
                        
                        if data.features is not None:
                            batch_tensor[batch_idx, :n_spikes, 2] = torch.from_numpy(data.features)
                        else:
                            batch_tensor[batch_idx, :n_spikes, 2] = 1.0  # Default amplitude
            
            batch_tensors.append(batch_tensor)
        
        return batch_tensors
    
    def _process_vectorized(self, batch_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply vectorized processing to batch tensors."""
        processed_tensors = []
        
        for tensor in batch_tensors:
            # Spike preprocessing (vectorized)
            spike_times = tensor[:, :, 0]
            neuron_ids = tensor[:, :, 1]
            amplitudes = tensor[:, :, 2]
            
            # Normalize spike times within each sample
            normalized_times = self._normalize_spike_times_vectorized(spike_times)
            
            # Apply temporal encoding
            temporal_features = self._temporal_encoding_vectorized(normalized_times, amplitudes)
            
            # Spatial encoding
            spatial_features = self._spatial_encoding_vectorized(neuron_ids, amplitudes)
            
            # Combine features
            combined_features = torch.cat([temporal_features, spatial_features], dim=-1)
            
            processed_tensors.append(combined_features)
        
        return processed_tensors
    
    def _normalize_spike_times_vectorized(self, spike_times: torch.Tensor) -> torch.Tensor:
        """Vectorized spike time normalization."""
        # Find min and max for each sample in batch
        batch_size = spike_times.shape[0]
        
        # Handle padding (zeros)
        mask = spike_times > 0
        
        min_times = torch.zeros(batch_size, device=spike_times.device)
        max_times = torch.zeros(batch_size, device=spike_times.device)
        
        for b in range(batch_size):
            valid_times = spike_times[b][mask[b]]
            if len(valid_times) > 0:
                min_times[b] = valid_times.min()
                max_times[b] = valid_times.max()
        
        # Normalize
        time_ranges = max_times - min_times
        time_ranges[time_ranges == 0] = 1.0  # Avoid division by zero
        
        normalized = (spike_times - min_times.unsqueeze(1)) / time_ranges.unsqueeze(1)
        normalized = normalized * mask.float()  # Keep padding as zeros
        
        return normalized
    
    def _temporal_encoding_vectorized(
        self,
        spike_times: torch.Tensor,
        amplitudes: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized temporal feature encoding."""
        batch_size, n_spikes = spike_times.shape
        
        # Temporal bins encoding
        n_bins = 32
        bin_edges = torch.linspace(0, 1, n_bins + 1, device=spike_times.device)
        
        # Create temporal histogram for each sample
        temporal_features = torch.zeros(batch_size, n_bins, device=spike_times.device)
        
        for b in range(batch_size):
            valid_mask = spike_times[b] > 0
            if valid_mask.any():
                valid_times = spike_times[b][valid_mask]
                valid_amps = amplitudes[b][valid_mask]
                
                # Weighted histogram
                bin_indices = torch.searchsorted(bin_edges[1:], valid_times)
                bin_indices = torch.clamp(bin_indices, 0, n_bins - 1)
                
                # Sum amplitudes in each bin
                for i, amp in zip(bin_indices, valid_amps):
                    temporal_features[b, i] += amp
        
        return temporal_features
    
    def _spatial_encoding_vectorized(
        self,
        neuron_ids: torch.Tensor,
        amplitudes: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized spatial feature encoding."""
        batch_size, n_spikes = neuron_ids.shape
        
        # Spatial topology encoding (assume 2D grid)
        grid_size = 16  # 16x16 spatial grid
        
        spatial_features = torch.zeros(batch_size, grid_size * grid_size, device=neuron_ids.device)
        
        for b in range(batch_size):
            valid_mask = neuron_ids[b] > 0
            if valid_mask.any():
                valid_ids = neuron_ids[b][valid_mask].long()
                valid_amps = amplitudes[b][valid_mask]
                
                # Map neuron IDs to spatial coordinates
                spatial_coords = valid_ids % (grid_size * grid_size)
                spatial_coords = torch.clamp(spatial_coords, 0, grid_size * grid_size - 1)
                
                # Sum amplitudes at each spatial location
                for coord, amp in zip(spatial_coords, valid_amps):
                    spatial_features[b, coord] += amp
        
        return spatial_features
    
    def _create_test_data(self) -> List[Dict[str, ModalityData]]:
        """Create test data for batch size optimization."""
        test_batch = []
        
        modalities = ['audio', 'vision', 'tactile']
        
        for _ in range(self.batch_size):
            sample = {}
            
            for modality in modalities:
                n_spikes = np.random.poisson(20)
                spike_times = np.sort(np.random.uniform(0, 100, n_spikes))
                neuron_ids = np.random.randint(0, 64, n_spikes)
                features = np.random.gamma(2.0, 0.5, n_spikes)
                
                sample[modality] = ModalityData(
                    modality_name=modality,
                    spike_times=spike_times,
                    neuron_ids=neuron_ids,
                    features=features,
                )
            
            test_batch.append(sample)
        
        return test_batch
    
    def _process_batch_internal(self, test_data: List[Dict[str, ModalityData]]):
        """Internal method for processing test batches."""
        batch_tensors = self._convert_to_batch_tensors(test_data)
        if self.enable_gpu:
            batch_tensors = [t.to(self.device) for t in batch_tensors]
        processed = self._process_vectorized(batch_tensors)
        
        # Return tensors to pool
        for tensor in batch_tensors + processed:
            self.memory_optimizer.return_tensor(tensor)


class NeuromorphicHardwareOptimizer:
    """
    Hardware-specific optimization for neuromorphic devices.
    
    Supports:
    - Intel Loihi 2 optimization
    - BrainChip Akida optimization
    - SpiNNaker optimization
    - General neuromorphic patterns
    """
    
    def __init__(self, target_hardware: str = "loihi2"):
        """
        Initialize hardware optimizer.
        
        Args:
            target_hardware: Target neuromorphic hardware
        """
        self.target_hardware = target_hardware.lower()
        self.hardware_specs = self._get_hardware_specs()
        self.optimization_cache = {}
        
        self.logger = logging.getLogger(__name__)
    
    def _get_hardware_specs(self) -> Dict[str, Any]:
        """Get hardware specifications for optimization."""
        specs = {
            "loihi2": {
                "max_neurons_per_core": 1024,
                "max_synapses_per_neuron": 64,
                "cores_per_chip": 128,
                "memory_per_core_kb": 128,
                "preferred_precision": "int8",
                "supports_plasticity": True,
                "energy_per_spike_pj": 23.6,
            },
            "akida": {
                "max_neurons_per_core": 256,
                "max_synapses_per_neuron": 32,
                "cores_per_chip": 16,
                "memory_per_core_kb": 64,
                "preferred_precision": "int4",
                "supports_plasticity": False,
                "energy_per_spike_pj": 45.2,
            },
            "spinnaker": {
                "max_neurons_per_core": 2048,
                "max_synapses_per_neuron": 1024,
                "cores_per_chip": 18,
                "memory_per_core_mb": 8,
                "preferred_precision": "float32",
                "supports_plasticity": True,
                "energy_per_spike_pj": 12.3,
            },
            "generic": {
                "max_neurons_per_core": 1000,
                "max_synapses_per_neuron": 100,
                "cores_per_chip": 16,
                "memory_per_core_kb": 256,
                "preferred_precision": "float32",
                "supports_plasticity": True,
                "energy_per_spike_pj": 50.0,
            }
        }
        
        return specs.get(self.target_hardware, specs["generic"])
    
    def optimize_model_for_hardware(
        self,
        model: CrossModalFusion,
        sample_data: Dict[str, ModalityData],
    ) -> Dict[str, Any]:
        """
        Optimize model for target neuromorphic hardware.
        
        Args:
            model: Fusion model to optimize
            sample_data: Sample data for profiling
            
        Returns:
            Optimization results and recommendations
        """
        optimization_key = f"{type(model).__name__}_{self.target_hardware}"
        
        if optimization_key in self.optimization_cache:
            return self.optimization_cache[optimization_key]
        
        start_time = time.time()
        
        # Analyze model complexity
        complexity_analysis = self._analyze_model_complexity(model, sample_data)
        
        # Generate hardware mapping strategy
        mapping_strategy = self._generate_mapping_strategy(complexity_analysis)
        
        # Optimize for memory constraints
        memory_optimization = self._optimize_memory_usage(complexity_analysis)
        
        # Generate quantization strategy
        quantization_strategy = self._generate_quantization_strategy()
        
        # Estimate performance
        performance_estimate = self._estimate_hardware_performance(
            complexity_analysis, mapping_strategy
        )
        
        optimization_result = {
            'target_hardware': self.target_hardware,
            'complexity_analysis': complexity_analysis,
            'mapping_strategy': mapping_strategy,
            'memory_optimization': memory_optimization,
            'quantization_strategy': quantization_strategy,
            'performance_estimate': performance_estimate,
            'optimization_time': time.time() - start_time,
            'recommendations': self._generate_hardware_recommendations(
                complexity_analysis, performance_estimate
            ),
        }
        
        self.optimization_cache[optimization_key] = optimization_result
        return optimization_result
    
    def _analyze_model_complexity(
        self,
        model: CrossModalFusion,
        sample_data: Dict[str, ModalityData],
    ) -> Dict[str, Any]:
        """Analyze model computational complexity."""
        analysis = {
            'n_modalities': len(getattr(model, 'modalities', [])),
            'total_neurons': 0,
            'total_synapses': 0,
            'memory_requirements_mb': 0.0,
            'computation_intensity': 0.0,
            'spike_density': {},
        }
        
        # Estimate from sample data
        total_spikes = 0
        for modality, data in sample_data.items():
            n_spikes = len(data.spike_times)
            unique_neurons = len(np.unique(data.neuron_ids)) if len(data.neuron_ids) > 0 else 0
            
            analysis['total_neurons'] += unique_neurons
            analysis['spike_density'][modality] = n_spikes / 100.0  # spikes per 100ms
            total_spikes += n_spikes
        
        # Estimate synapses (cross-modal connections)
        analysis['total_synapses'] = analysis['total_neurons'] * analysis['n_modalities'] * 10
        
        # Memory requirements
        neuron_memory = analysis['total_neurons'] * 64  # bytes per neuron
        synapse_memory = analysis['total_synapses'] * 8  # bytes per synapse
        analysis['memory_requirements_mb'] = (neuron_memory + synapse_memory) / (1024**2)
        
        # Computation intensity (operations per spike)
        analysis['computation_intensity'] = analysis['total_synapses'] / max(1, total_spikes)
        
        return analysis
    
    def _generate_mapping_strategy(self, complexity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy for mapping model to hardware cores."""
        specs = self.hardware_specs
        
        # Calculate required cores
        neurons_per_core = min(specs['max_neurons_per_core'], complexity_analysis['total_neurons'])
        required_cores = max(1, int(np.ceil(complexity_analysis['total_neurons'] / neurons_per_core)))
        
        # Check if model fits on single chip
        fits_on_chip = required_cores <= specs['cores_per_chip']
        
        strategy = {
            'neurons_per_core': neurons_per_core,
            'required_cores': required_cores,
            'fits_on_chip': fits_on_chip,
            'parallelization_factor': min(required_cores, specs['cores_per_chip']),
            'memory_per_core_mb': complexity_analysis['memory_requirements_mb'] / required_cores,
            'core_utilization': required_cores / specs['cores_per_chip'],
        }
        
        # Multi-chip strategy if needed
        if not fits_on_chip:
            strategy['chips_required'] = int(np.ceil(required_cores / specs['cores_per_chip']))
            strategy['inter_chip_communication'] = True
        else:
            strategy['chips_required'] = 1
            strategy['inter_chip_communication'] = False
        
        return strategy
    
    def _optimize_memory_usage(self, complexity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate memory optimization strategy."""
        specs = self.hardware_specs
        memory_per_core = specs.get('memory_per_core_kb', specs.get('memory_per_core_mb', 1) * 1024)
        
        # Calculate memory constraints
        available_memory = memory_per_core * specs['cores_per_chip']
        required_memory = complexity_analysis['memory_requirements_mb'] * 1024  # Convert to KB
        
        memory_optimization = {
            'available_memory_kb': available_memory,
            'required_memory_kb': required_memory,
            'memory_efficiency': available_memory / max(1, required_memory),
            'fits_in_memory': required_memory <= available_memory,
            'optimizations': [],
        }
        
        # Memory optimization strategies
        if required_memory > available_memory:
            memory_optimization['optimizations'].extend([
                'weight_pruning',
                'connection_sparsification',
                'temporal_compression',
            ])
        
        if complexity_analysis['computation_intensity'] > 100:
            memory_optimization['optimizations'].extend([
                'spike_buffering',
                'event_compression',
            ])
        
        return memory_optimization
    
    def _generate_quantization_strategy(self) -> Dict[str, Any]:
        """Generate quantization strategy for hardware."""
        specs = self.hardware_specs
        preferred_precision = specs.get('preferred_precision', 'float32')
        
        strategy = {
            'target_precision': preferred_precision,
            'weight_quantization': True,
            'activation_quantization': True,
            'quantization_aware_training': True,
            'calibration_required': preferred_precision in ['int8', 'int4'],
        }
        
        # Hardware-specific quantization
        if self.target_hardware == 'loihi2':
            strategy.update({
                'spike_encoding': 'temporal',
                'weight_bits': 8,
                'membrane_potential_bits': 16,
            })
        elif self.target_hardware == 'akida':
            strategy.update({
                'spike_encoding': 'rate',
                'weight_bits': 4,
                'activation_bits': 4,
            })
        elif self.target_hardware == 'spinnaker':
            strategy.update({
                'spike_encoding': 'temporal',
                'weight_bits': 32,  # Full precision
                'supports_dynamic_precision': True,
            })
        
        return strategy
    
    def _estimate_hardware_performance(
        self,
        complexity_analysis: Dict[str, Any],
        mapping_strategy: Dict[str, Any],
    ) -> Dict[str, float]:
        """Estimate performance on target hardware."""
        specs = self.hardware_specs
        
        # Latency estimation
        total_spikes = sum(complexity_analysis['spike_density'].values()) * 10  # 10 time windows
        processing_cycles = total_spikes * complexity_analysis['computation_intensity']
        
        # Assume 1MHz clock for neuromorphic chips
        latency_ms = processing_cycles / 1000.0  # Convert to ms
        
        # Parallelization benefit
        parallelization_factor = mapping_strategy['parallelization_factor']
        latency_ms = latency_ms / parallelization_factor
        
        # Memory access penalty
        if mapping_strategy['memory_per_core_mb'] > specs.get('memory_per_core_kb', 256) / 1024:
            latency_ms *= 1.5  # Memory bottleneck penalty
        
        # Inter-chip communication penalty
        if mapping_strategy.get('inter_chip_communication', False):
            latency_ms *= 2.0  # Communication overhead
        
        # Throughput estimation
        throughput_ops_per_sec = 1000.0 / max(0.1, latency_ms)  # Operations per second
        
        # Energy estimation
        energy_per_spike = specs['energy_per_spike_pj']
        total_energy_uj = (total_spikes * energy_per_spike) / 1000.0  # Convert to μJ
        
        # Memory access energy
        memory_accesses = total_spikes * 2  # Read and write
        memory_energy_uj = memory_accesses * 10 / 1000.0  # 10 pJ per access
        
        total_energy_uj += memory_energy_uj
        
        return {
            'estimated_latency_ms': latency_ms,
            'estimated_throughput_ops_per_sec': throughput_ops_per_sec,
            'estimated_energy_per_inference_uj': total_energy_uj,
            'memory_bandwidth_gbps': mapping_strategy['parallelization_factor'] * 0.1,
            'core_utilization_percent': mapping_strategy['core_utilization'] * 100,
        }
    
    def _generate_hardware_recommendations(
        self,
        complexity_analysis: Dict[str, Any],
        performance_estimate: Dict[str, float],
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Latency recommendations
        if performance_estimate['estimated_latency_ms'] > 10.0:
            recommendations.extend([
                "Consider model pruning to reduce computation",
                "Implement spike-based early termination",
                "Optimize cross-modal connection patterns",
            ])
        
        # Memory recommendations
        memory_req_mb = complexity_analysis['memory_requirements_mb']
        if memory_req_mb > 10.0:
            recommendations.extend([
                "Implement weight sharing across modalities",
                "Use sparse connectivity patterns",
                "Consider hierarchical processing",
            ])
        
        # Energy recommendations
        if performance_estimate['estimated_energy_per_inference_uj'] > 100.0:
            recommendations.extend([
                "Reduce spike generation rates",
                "Implement adaptive processing",
                "Use event-driven computation",
            ])
        
        # Hardware-specific recommendations
        if self.target_hardware == 'loihi2':
            recommendations.extend([
                "Leverage plasticity rules for adaptation",
                "Use compartmental neuron models",
                "Implement axonal delays",
            ])
        elif self.target_hardware == 'akida':
            recommendations.extend([
                "Use convolutional processing patterns",
                "Implement rate-based encoding",
                "Optimize for fixed-point arithmetic",
            ])
        
        return recommendations


class AdaptivePerformanceTuner:
    """
    Adaptive performance tuning system that continuously optimizes
    performance based on runtime characteristics and requirements.
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize adaptive performance tuner.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.performance_history = deque(maxlen=1000)
        self.tuning_parameters = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'memory_threshold': 0.8,
            'cpu_threshold': 0.7,
            'gpu_threshold': 0.8,
        }
        
        # Performance monitoring
        self.monitors = {
            'memory': MemoryOptimizer(),
            'batch_processor': SpikeBatchProcessor(),
        }
        
        self.logger = logging.getLogger(__name__)
        self.tuning_active = False
        self._start_adaptive_tuning()
    
    def _start_adaptive_tuning(self):
        """Start adaptive tuning thread."""
        def tuning_loop():
            while self.tuning_active:
                try:
                    self._adaptive_tuning_step()
                    time.sleep(5.0)  # Tune every 5 seconds
                except Exception as e:
                    self.logger.error(f"Adaptive tuning error: {e}")
        
        self.tuning_active = True
        tuning_thread = threading.Thread(target=tuning_loop, daemon=True)
        tuning_thread.start()
    
    def record_performance(self, profile: PerformanceProfile):
        """Record performance measurement for adaptive tuning."""
        self.performance_history.append(profile)
        
        # Trigger immediate tuning if performance is poor
        if (profile.latency_ms > self.config.target_latency_ms * 1.5 or
            profile.memory_usage_mb > self.config.max_memory_mb * 0.9):
            self._emergency_tuning()
    
    def _adaptive_tuning_step(self):
        """Perform one step of adaptive tuning."""
        if len(self.performance_history) < 10:
            return  # Need sufficient history
        
        recent_profiles = list(self.performance_history)[-10:]
        
        # Analyze trends
        latencies = [p.latency_ms for p in recent_profiles]
        throughputs = [p.throughput_ops_per_sec for p in recent_profiles]
        memory_usage = [p.memory_usage_mb for p in recent_profiles]
        
        # Tune based on performance trends
        self._tune_memory_management(memory_usage)
        self._tune_batch_processing(latencies, throughputs)
        self._tune_hardware_utilization(recent_profiles)
    
    def _tune_memory_management(self, memory_usage: List[float]):
        """Tune memory management parameters."""
        avg_memory = np.mean(memory_usage)
        memory_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
        
        # Adjust memory threshold based on usage
        if avg_memory > self.config.max_memory_mb * 0.8:
            self.tuning_parameters['memory_threshold'] = max(0.6, 
                self.tuning_parameters['memory_threshold'] - 0.05)
            self.monitors['memory']._optimize_memory_usage()
        elif avg_memory < self.config.max_memory_mb * 0.5:
            self.tuning_parameters['memory_threshold'] = min(0.9,
                self.tuning_parameters['memory_threshold'] + 0.05)
        
        # Handle memory growth trend
        if memory_trend > 1.0:  # Growing by > 1MB per measurement
            self.logger.warning("Memory usage growing, triggering optimization")
            self.monitors['memory']._optimize_memory_usage()
    
    def _tune_batch_processing(self, latencies: List[float], throughputs: List[float]):
        """Tune batch processing parameters."""
        avg_latency = np.mean(latencies)
        avg_throughput = np.mean(throughputs)
        
        batch_processor = self.monitors['batch_processor']
        current_batch_size = batch_processor.batch_size
        
        # Adjust batch size based on performance
        if avg_latency > self.config.target_latency_ms:
            # Latency too high, reduce batch size
            new_batch_size = max(16, int(current_batch_size * 0.8))
            batch_processor.batch_size = new_batch_size
            self.logger.info(f"Reduced batch size to {new_batch_size} for latency")
            
        elif (avg_latency < self.config.target_latency_ms * 0.5 and 
              avg_throughput < self.config.target_throughput_ops_per_sec):
            # Latency good but throughput low, increase batch size
            new_batch_size = min(batch_processor.max_batch_size, 
                                int(current_batch_size * 1.2))
            batch_processor.batch_size = new_batch_size
            self.logger.info(f"Increased batch size to {new_batch_size} for throughput")
    
    def _tune_hardware_utilization(self, profiles: List[PerformanceProfile]):
        """Tune hardware utilization parameters."""
        avg_cpu = np.mean([p.cpu_utilization for p in profiles])
        avg_gpu = np.mean([p.gpu_utilization for p in profiles])
        
        # Adjust processing parameters based on hardware utilization
        if avg_cpu > self.tuning_parameters['cpu_threshold']:
            # High CPU usage, reduce processing intensity
            self.tuning_parameters['cpu_threshold'] = min(0.9, 
                self.tuning_parameters['cpu_threshold'] + 0.05)
        
        if CUDA_AVAILABLE and avg_gpu < 0.5:
            # Low GPU usage, could increase GPU workload
            batch_processor = self.monitors['batch_processor']
            if not batch_processor.enable_gpu:
                batch_processor.enable_gpu = True
                batch_processor.device = torch.device("cuda")
                self.logger.info("Enabled GPU acceleration due to low utilization")
    
    def _emergency_tuning(self):
        """Emergency performance tuning for critical situations."""
        self.logger.warning("Emergency performance tuning triggered")
        
        # Aggressive memory cleanup
        self.monitors['memory']._optimize_memory_usage()
        
        # Reduce batch size immediately
        batch_processor = self.monitors['batch_processor']
        batch_processor.batch_size = max(8, batch_processor.batch_size // 2)
        
        # Force garbage collection
        gc.collect()
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()
    
    def get_tuning_status(self) -> Dict[str, Any]:
        """Get current tuning status and parameters."""
        return {
            'tuning_active': self.tuning_active,
            'parameters': self.tuning_parameters.copy(),
            'performance_history_length': len(self.performance_history),
            'memory_stats': self.monitors['memory'].get_memory_stats(),
            'batch_processing_stats': self.monitors['batch_processor'].processing_stats.copy(),
        }


# Factory function for creating optimized fusion models
@robust_function(critical_path=True)
def create_optimized_fusion_model(
    model_type: str,
    modalities: List[str],
    config: OptimizationConfig,
) -> Tuple[CrossModalFusion, Dict[str, Any]]:
    """
    Create optimized fusion model with performance enhancements.
    
    Args:
        model_type: Type of fusion model to create
        modalities: List of modalities
        config: Optimization configuration
        
    Returns:
        (optimized_model, optimization_info) tuple
    """
    start_time = time.time()
    
    # Import model classes
    if model_type.lower() == 'tsa':
        from ..algorithms.temporal_spike_attention import create_temporal_spike_attention
        model = create_temporal_spike_attention(modalities)
    else:
        from ..algorithms.fusion import AttentionMechanism
        model = AttentionMechanism(modalities)
    
    # Apply hardware-specific optimizations
    hardware_optimizer = NeuromorphicHardwareOptimizer(config.target_hardware)
    
    # Create sample data for optimization
    sample_data = {}
    for modality in modalities:
        n_spikes = np.random.poisson(30)
        sample_data[modality] = ModalityData(
            modality_name=modality,
            spike_times=np.sort(np.random.uniform(0, 100, n_spikes)),
            neuron_ids=np.random.randint(0, 32, n_spikes),
            features=np.random.gamma(2.0, 0.5, n_spikes),
        )
    
    # Optimize for target hardware
    optimization_info = hardware_optimizer.optimize_model_for_hardware(model, sample_data)
    
    # Apply optimizations if possible
    if hasattr(model, 'spike_thresholds') and config.optimization_level == OptimizationLevel.AGGRESSIVE:
        # Optimize spike thresholds
        for modality in modalities:
            if modality in model.spike_thresholds:
                model.spike_thresholds[modality] *= 0.9  # Reduce threshold for higher sensitivity
    
    # Setup performance tuner
    if config.optimization_level != OptimizationLevel.CONSERVATIVE:
        tuner = AdaptivePerformanceTuner(config)
        optimization_info['performance_tuner'] = tuner
    
    optimization_info.update({
        'model_type': model_type,
        'modalities': modalities,
        'optimization_time': time.time() - start_time,
        'config': config.__dict__,
    })
    
    return model, optimization_info


# Export key components
__all__ = [
    'OptimizationLevel',
    'PerformanceMetric',
    'OptimizationConfig',
    'PerformanceProfile',
    'MemoryOptimizer',
    'SpikeBatchProcessor',
    'NeuromorphicHardwareOptimizer',
    'AdaptivePerformanceTuner',
    'create_optimized_fusion_model',
]