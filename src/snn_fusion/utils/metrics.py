"""
Comprehensive Metrics for Multi-Modal Spiking Neural Networks

Implements evaluation metrics for spike trains, fusion quality, neuromorphic performance,
and task-specific measurements for real-time sensor fusion applications.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
import time


@dataclass
class SpikeStatistics:
    """Container for comprehensive spike statistics."""
    firing_rates: torch.Tensor
    cv_isi: torch.Tensor  # Coefficient of variation of inter-spike intervals
    synchrony_index: torch.Tensor
    burstiness: torch.Tensor
    spike_count: torch.Tensor
    active_neurons: torch.Tensor
    entropy: torch.Tensor
    

class SpikeMetrics:
    """
    Comprehensive spike train analysis and metrics computation.
    
    Provides methods for evaluating spike timing precision, population dynamics,
    and neuromorphic-specific performance measures.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize spike metrics calculator.
        
        Args:
            device: Computation device
        """
        self.device = device or torch.device('cpu')
        
    def compute_firing_rates(
        self, 
        spike_trains: torch.Tensor, 
        time_window: float = 1000.0,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Compute firing rates from spike trains.
        
        Args:
            spike_trains: Spike trains [batch, time, neurons]
            time_window: Time window for rate calculation (ms)
            dt: Time step (ms)
            
        Returns:
            firing_rates: Firing rates in Hz [batch, neurons]
        """
        spike_counts = spike_trains.sum(dim=1)  # Sum over time dimension
        time_duration = spike_trains.shape[1] * dt / 1000.0  # Convert to seconds
        firing_rates = spike_counts / time_duration
        
        return firing_rates
    
    def compute_cv_isi(self, spike_trains: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Compute coefficient of variation of inter-spike intervals.
        
        Args:
            spike_trains: Spike trains [batch, time, neurons]
            dt: Time step (ms)
            
        Returns:
            cv_isi: CV of ISI [batch, neurons]
        """
        batch_size, time_steps, n_neurons = spike_trains.shape
        cv_values = torch.zeros(batch_size, n_neurons, device=self.device)
        
        for b in range(batch_size):
            for n in range(n_neurons):
                spike_times = torch.where(spike_trains[b, :, n] > 0)[0].float() * dt
                
                if len(spike_times) > 2:
                    # Compute inter-spike intervals
                    isis = torch.diff(spike_times)
                    
                    if len(isis) > 0:
                        mean_isi = torch.mean(isis)
                        std_isi = torch.std(isis)
                        
                        if mean_isi > 0:
                            cv_values[b, n] = std_isi / mean_isi
                        
        return cv_values
    
    def compute_synchrony_index(
        self, 
        spike_trains: torch.Tensor, 
        window_size: int = 10
    ) -> torch.Tensor:
        """
        Compute population synchrony index using sliding window correlation.
        
        Args:
            spike_trains: Spike trains [batch, time, neurons]
            window_size: Window size for synchrony calculation
            
        Returns:
            synchrony: Population synchrony index [batch]
        """
        batch_size, time_steps, n_neurons = spike_trains.shape
        
        # Compute population activity (sum across neurons)
        pop_activity = spike_trains.sum(dim=2)  # [batch, time]
        
        synchrony_values = torch.zeros(batch_size, device=self.device)
        
        for b in range(batch_size):
            activity = pop_activity[b]
            
            if activity.std() > 0:
                # Compute autocorrelation at lag 1
                if time_steps > 1:
                    activity_t = activity[:-1]
                    activity_t1 = activity[1:]
                    
                    if len(activity_t) > 0 and len(activity_t1) > 0:
                        correlation = torch.corrcoef(torch.stack([activity_t, activity_t1]))
                        if correlation.numel() > 1:
                            synchrony_values[b] = correlation[0, 1]
                            
        return synchrony_values
    
    def compute_burstiness(self, spike_trains: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        Compute burstiness measure (deviation from Poisson process).
        
        Args:
            spike_trains: Spike trains [batch, time, neurons]
            dt: Time step (ms)
            
        Returns:
            burstiness: Burstiness index [batch, neurons]
        """
        batch_size, time_steps, n_neurons = spike_trains.shape
        burstiness = torch.zeros(batch_size, n_neurons, device=self.device)
        
        for b in range(batch_size):
            for n in range(n_neurons):
                spike_times = torch.where(spike_trains[b, :, n] > 0)[0].float() * dt
                
                if len(spike_times) > 2:
                    isis = torch.diff(spike_times)
                    
                    if len(isis) > 1:
                        mean_isi = torch.mean(isis)
                        std_isi = torch.std(isis)
                        
                        # Burstiness index: (CV^2 - 1) / (CV^2 + 1)
                        if mean_isi > 0:
                            cv = std_isi / mean_isi
                            burstiness[b, n] = (cv**2 - 1) / (cv**2 + 1)
                            
        return burstiness
    
    def compute_entropy(self, spike_trains: torch.Tensor, bin_size: int = 10) -> torch.Tensor:
        """
        Compute spike timing entropy.
        
        Args:
            spike_trains: Spike trains [batch, time, neurons]  
            bin_size: Bin size for histogram computation
            
        Returns:
            entropy: Spike timing entropy [batch, neurons]
        """
        batch_size, time_steps, n_neurons = spike_trains.shape
        entropy_values = torch.zeros(batch_size, n_neurons, device=self.device)
        
        n_bins = time_steps // bin_size
        
        for b in range(batch_size):
            for n in range(n_neurons):
                # Bin spike times
                spike_counts = torch.zeros(n_bins, device=self.device)
                
                for i in range(n_bins):
                    start_idx = i * bin_size
                    end_idx = min((i + 1) * bin_size, time_steps)
                    spike_counts[i] = spike_trains[b, start_idx:end_idx, n].sum()
                
                # Compute entropy
                total_spikes = spike_counts.sum()
                if total_spikes > 0:
                    probabilities = spike_counts / total_spikes
                    # Avoid log(0)
                    probabilities = probabilities[probabilities > 0]
                    if len(probabilities) > 0:
                        entropy_values[b, n] = -torch.sum(probabilities * torch.log(probabilities))
                        
        return entropy_values
    
    def compute_comprehensive_statistics(
        self, 
        spike_trains: torch.Tensor,
        dt: float = 1.0
    ) -> SpikeStatistics:
        """
        Compute comprehensive spike statistics.
        
        Args:
            spike_trains: Spike trains [batch, time, neurons]
            dt: Time step (ms)
            
        Returns:
            Comprehensive spike statistics
        """
        stats = SpikeStatistics(
            firing_rates=self.compute_firing_rates(spike_trains, dt=dt),
            cv_isi=self.compute_cv_isi(spike_trains, dt=dt),
            synchrony_index=self.compute_synchrony_index(spike_trains),
            burstiness=self.compute_burstiness(spike_trains, dt=dt),
            spike_count=spike_trains.sum(dim=1),
            active_neurons=(spike_trains.sum(dim=1) > 0).float().sum(dim=1),
            entropy=self.compute_entropy(spike_trains),
        )
        
        return stats
    
    def van_rossum_distance(
        self,
        spike_train_1: torch.Tensor,
        spike_train_2: torch.Tensor,
        tau: float = 10.0,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Compute Van Rossum distance between spike trains.
        
        Args:
            spike_train_1: First spike train [batch, time, neurons]
            spike_train_2: Second spike train [batch, time, neurons]
            tau: Filter time constant (ms)
            dt: Time step (ms)
            
        Returns:
            distances: Van Rossum distances [batch, neurons]
        """
        # Create exponential kernel
        time_steps = spike_train_1.shape[1]
        kernel_size = min(int(5 * tau / dt), time_steps)
        
        t = torch.arange(kernel_size, device=self.device, dtype=torch.float) * dt
        kernel = torch.exp(-t / tau)
        kernel = kernel / kernel.sum()  # Normalize
        
        batch_size, _, n_neurons = spike_train_1.shape
        distances = torch.zeros(batch_size, n_neurons, device=self.device)
        
        for b in range(batch_size):
            for n in range(n_neurons):
                # Convolve with exponential kernel
                s1_filtered = F.conv1d(
                    spike_train_1[b, :, n].unsqueeze(0).unsqueeze(0),
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=kernel_size//2
                ).squeeze()
                
                s2_filtered = F.conv1d(
                    spike_train_2[b, :, n].unsqueeze(0).unsqueeze(0),
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=kernel_size//2
                ).squeeze()
                
                # Compute squared difference
                diff = s1_filtered - s2_filtered
                distances[b, n] = torch.sum(diff**2) * dt / tau
                
        return distances
    
    def victor_purpura_distance(
        self,
        spike_train_1: torch.Tensor,
        spike_train_2: torch.Tensor,
        cost_parameter: float = 1.0,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Compute Victor-Purpura distance between spike trains.
        
        Args:
            spike_train_1: First spike train [batch, time, neurons]
            spike_train_2: Second spike train [batch, time, neurons]
            cost_parameter: Cost parameter for spike timing
            dt: Time step (ms)
            
        Returns:
            distances: Victor-Purpura distances [batch, neurons]
        """
        batch_size, time_steps, n_neurons = spike_train_1.shape
        distances = torch.zeros(batch_size, n_neurons, device=self.device)
        
        for b in range(batch_size):
            for n in range(n_neurons):
                # Extract spike times
                times_1 = torch.where(spike_train_1[b, :, n] > 0)[0].float() * dt
                times_2 = torch.where(spike_train_2[b, :, n] > 0)[0].float() * dt
                
                # Compute VP distance using dynamic programming
                distances[b, n] = self._vp_distance_dp(times_1, times_2, cost_parameter)
                
        return distances
    
    def _vp_distance_dp(
        self,
        times_1: torch.Tensor,
        times_2: torch.Tensor,
        cost_parameter: float
    ) -> torch.Tensor:
        """Dynamic programming implementation of Victor-Purpura distance."""
        n1, n2 = len(times_1), len(times_2)
        
        # Initialize DP table
        dp = torch.zeros(n1 + 1, n2 + 1, device=self.device)
        
        # Base cases
        dp[0, :] = torch.arange(n2 + 1, device=self.device)  # Insert all spikes from train 2
        dp[:, 0] = torch.arange(n1 + 1, device=self.device)  # Delete all spikes from train 1
        
        # Fill DP table
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                # Cost of matching spikes
                time_diff = abs(times_1[i-1] - times_2[j-1])
                match_cost = min(cost_parameter * time_diff, 2.0)  # Cap at cost of delete+insert
                
                # Three operations: match, delete, insert
                match = dp[i-1, j-1] + match_cost
                delete = dp[i-1, j] + 1.0
                insert = dp[i, j-1] + 1.0
                
                dp[i, j] = min(match, delete, insert)
        
        return dp[n1, n2]


class FusionMetrics:
    """
    Metrics for evaluating multi-modal fusion quality and performance.
    
    Provides methods to assess fusion effectiveness, modality contributions,
    and cross-modal information integration.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize fusion metrics calculator."""
        self.device = device or torch.device('cpu')
        
    def compute_modality_importance(
        self,
        predictions: Dict[str, torch.Tensor],
        fusion_prediction: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute relative importance of each modality for fusion.
        
        Args:
            predictions: Single-modality predictions {modality: predictions}
            fusion_prediction: Fused prediction
            targets: Ground truth targets
            
        Returns:
            modality_importance: Importance scores for each modality
        """
        fusion_accuracy = self._compute_accuracy(fusion_prediction, targets)
        importance_scores = {}
        
        for modality, pred in predictions.items():
            # Leave-one-out: compute performance without this modality
            other_modalities = {k: v for k, v in predictions.items() if k != modality}
            
            if len(other_modalities) > 0:
                # Simple average fusion of remaining modalities
                remaining_pred = torch.stack(list(other_modalities.values())).mean(dim=0)
                remaining_accuracy = self._compute_accuracy(remaining_pred, targets)
                
                # Importance = performance drop when removing this modality
                importance_scores[modality] = fusion_accuracy - remaining_accuracy
            else:
                importance_scores[modality] = fusion_accuracy
                
        # Normalize importance scores
        total_importance = sum(abs(score) for score in importance_scores.values())
        if total_importance > 0:
            importance_scores = {k: v / total_importance for k, v in importance_scores.items()}
            
        return importance_scores
    
    def compute_cross_modal_correlation(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute cross-modal feature correlations.
        
        Args:
            features: Modality features {modality: features [batch, feature_dim]}
            
        Returns:
            correlations: Pairwise correlations between modalities
        """
        correlations = {}
        modality_names = list(features.keys())
        
        for i, mod1 in enumerate(modality_names):
            for j, mod2 in enumerate(modality_names[i+1:], i+1):
                # Flatten features for correlation computation
                feat1 = features[mod1].flatten()
                feat2 = features[mod2].flatten()
                
                # Compute Pearson correlation
                if len(feat1) > 1 and len(feat2) > 1:
                    correlation = torch.corrcoef(torch.stack([feat1, feat2]))[0, 1]
                    correlations[(mod1, mod2)] = correlation.item()
                else:
                    correlations[(mod1, mod2)] = 0.0
                    
        return correlations
    
    def compute_fusion_efficiency(
        self,
        single_modal_predictions: Dict[str, torch.Tensor],
        fusion_prediction: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Compute fusion efficiency compared to best single modality.
        
        Args:
            single_modal_predictions: Single-modality predictions
            fusion_prediction: Fused prediction
            targets: Ground truth targets
            
        Returns:
            efficiency: Fusion efficiency (>1 means fusion helps)
        """
        # Get best single-modality performance
        best_single_acc = 0.0
        for pred in single_modal_predictions.values():
            acc = self._compute_accuracy(pred, targets)
            best_single_acc = max(best_single_acc, acc)
        
        # Get fusion performance
        fusion_acc = self._compute_accuracy(fusion_prediction, targets)
        
        # Efficiency = fusion_performance / best_single_performance
        efficiency = fusion_acc / max(best_single_acc, 1e-6)
        
        return efficiency
    
    def compute_complementarity_score(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> float:
        """
        Compute how complementary different modalities are.
        
        Args:
            predictions: Modality predictions
            targets: Ground truth targets
            
        Returns:
            complementarity: Score indicating modality complementarity
        """
        modalities = list(predictions.keys())
        n_modalities = len(modalities)
        
        if n_modalities < 2:
            return 0.0
        
        # Compute pairwise disagreement on correct samples
        total_complementarity = 0.0
        n_pairs = 0
        
        for i in range(n_modalities):
            for j in range(i + 1, n_modalities):
                pred1 = predictions[modalities[i]]
                pred2 = predictions[modalities[j]]
                
                # Get predicted classes
                class1 = torch.argmax(pred1, dim=-1)
                class2 = torch.argmax(pred2, dim=-1)
                true_class = targets
                
                # Where both are wrong, check if they disagree (complementary errors)
                both_wrong = (class1 != true_class) & (class2 != true_class)
                if torch.sum(both_wrong) > 0:
                    disagree_when_wrong = (class1 != class2) & both_wrong
                    complementarity = torch.sum(disagree_when_wrong).float() / torch.sum(both_wrong).float()
                    total_complementarity += complementarity.item()
                
                n_pairs += 1
        
        return total_complementarity / n_pairs if n_pairs > 0 else 0.0
    
    def analyze_modality_dropout(
        self,
        model: torch.nn.Module,
        test_loader,
        modalities: List[str],
        num_samples: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance degradation when modalities are dropped.
        
        Args:
            model: Multi-modal model
            test_loader: Test data loader
            modalities: List of modality names
            num_samples: Number of test samples to use
            
        Returns:
            dropout_analysis: Performance with different modality combinations
        """
        results = {}
        model.eval()
        
        with torch.no_grad():
            sample_count = 0
            total_correct = {combo: 0 for combo in self._get_modality_combinations(modalities)}
            total_samples = 0
            
            for batch in test_loader:
                if sample_count >= num_samples:
                    break
                
                inputs = batch['inputs'] if isinstance(batch, dict) else batch[0]
                targets = batch['targets'] if isinstance(batch, dict) else batch[1]
                
                batch_size = targets.shape[0]
                
                # Test all modality combinations
                for combo in self._get_modality_combinations(modalities):
                    # Create masked input (simulate dropout)
                    masked_inputs = self._mask_modalities(inputs, combo, modalities)
                    
                    # Forward pass
                    outputs = model(masked_inputs)
                    predictions = torch.argmax(outputs, dim=-1)
                    
                    # Count correct predictions
                    correct = (predictions == targets).sum().item()
                    total_correct[combo] += correct
                
                total_samples += batch_size
                sample_count += batch_size
            
            # Convert to accuracies
            for combo in total_correct:
                accuracy = total_correct[combo] / total_samples
                combo_str = '+'.join(sorted(combo)) if combo else 'none'
                results[combo_str] = {
                    'accuracy': accuracy,
                    'num_modalities': len(combo),
                    'modalities': list(combo)
                }
        
        return results
    
    def _compute_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute classification accuracy."""
        pred_classes = torch.argmax(predictions, dim=-1)
        correct = (pred_classes == targets).sum().item()
        total = targets.numel()
        return correct / total if total > 0 else 0.0
    
    def _get_modality_combinations(self, modalities: List[str]) -> List[Tuple[str, ...]]:
        """Generate all possible modality combinations."""
        from itertools import combinations
        
        all_combinations = []
        for r in range(len(modalities) + 1):  # Include empty set
            for combo in combinations(modalities, r):
                all_combinations.append(combo)
        
        return all_combinations
    
    def _mask_modalities(
        self,
        inputs: Dict[str, torch.Tensor],
        keep_modalities: Tuple[str, ...],
        all_modalities: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Mask out modalities not in keep_modalities."""
        masked = inputs.copy()
        
        for modality in all_modalities:
            if modality not in keep_modalities and modality in masked:
                # Zero out the modality
                masked[modality] = torch.zeros_like(masked[modality])
        
        return masked


class PerformanceMetrics:
    """
    Real-time performance and efficiency metrics for neuromorphic systems.
    
    Measures latency, throughput, energy efficiency, and resource utilization
    for deployment on neuromorphic hardware.
    """
    
    def __init__(self):
        """Initialize performance metrics tracker."""
        self.reset_counters()
        
    def reset_counters(self):
        """Reset all performance counters."""
        self.inference_times = []
        self.throughput_samples = []
        self.memory_usage = []
        self.energy_estimates = []
        self.spike_counts = []
        
    def measure_inference_latency(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Measure inference latency statistics.
        
        Args:
            model: Model to benchmark
            sample_input: Sample input tensor
            num_runs: Number of timing runs
            warmup_runs: Number of warmup runs
            
        Returns:
            latency_stats: Latency statistics (mean, std, min, max)
        """
        model.eval()
        latencies = []
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(sample_input)
                
        # Timing runs
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(sample_input)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
        }
    
    def estimate_energy_consumption(
        self,
        spike_counts: Dict[str, int],
        layer_sizes: Dict[str, int],
        hardware_platform: str = 'loihi2'
    ) -> Dict[str, float]:
        """
        Estimate energy consumption based on spike counts and hardware platform.
        
        Args:
            spike_counts: Spike counts per layer
            layer_sizes: Number of neurons per layer
            hardware_platform: Target hardware platform
            
        Returns:
            energy_estimates: Energy consumption estimates
        """
        # Energy coefficients (estimates based on literature)
        energy_per_spike = {
            'loihi2': 23e-12,  # 23 pJ per spike
            'akida': 50e-12,   # 50 pJ per spike
            'spinnaker': 100e-12,  # 100 pJ per spike
            'gpu': 1e-9,       # 1 nJ per spike (much higher)
            'cpu': 10e-9,      # 10 nJ per spike (highest)
        }
        
        spike_energy = energy_per_spike.get(hardware_platform, 100e-12)
        
        total_spikes = sum(spike_counts.values())
        total_energy = total_spikes * spike_energy
        
        # Add static power consumption (estimated)
        static_power = {
            'loihi2': 100e-3,    # 100 mW
            'akida': 50e-3,      # 50 mW
            'spinnaker': 1.0,    # 1 W
            'gpu': 250.0,        # 250 W
            'cpu': 100.0,        # 100 W
        }
        
        inference_time = 0.01  # Assume 10ms inference time
        static_energy = static_power.get(hardware_platform, 100e-3) * inference_time
        
        return {
            'total_energy_j': total_energy + static_energy,
            'spike_energy_j': total_energy,
            'static_energy_j': static_energy,
            'energy_per_spike_j': spike_energy,
            'total_spikes': total_spikes,
            'energy_efficiency_ops_per_j': total_spikes / max(total_energy, 1e-12)
        }
    
    def compute_throughput(
        self,
        model: torch.nn.Module,
        data_loader,
        max_batches: int = 50
    ) -> Dict[str, float]:
        """
        Compute model throughput (samples per second).
        
        Args:
            model: Model to benchmark
            data_loader: Data loader for throughput testing
            max_batches: Maximum number of batches to process
            
        Returns:
            throughput_stats: Throughput statistics
        """
        model.eval()
        total_samples = 0
        total_time = 0.0
        batch_times = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch['inputs']
                batch_size = inputs.shape[0] if hasattr(inputs, 'shape') else len(inputs)
                
                start_time = time.perf_counter()
                _ = model(inputs)
                end_time = time.perf_counter()
                
                batch_time = end_time - start_time
                batch_times.append(batch_time)
                total_time += batch_time
                total_samples += batch_size
        
        throughput = total_samples / total_time if total_time > 0 else 0.0
        
        return {
            'throughput_samples_per_sec': throughput,
            'total_samples': total_samples,
            'total_time_sec': total_time,
            'mean_batch_time_sec': np.mean(batch_times),
            'std_batch_time_sec': np.std(batch_times),
        }
    
    def analyze_sparsity(self, spike_trains: torch.Tensor) -> Dict[str, float]:
        """
        Analyze spike sparsity for neuromorphic efficiency.
        
        Args:
            spike_trains: Spike trains [batch, time, neurons]
            
        Returns:
            sparsity_stats: Sparsity analysis
        """
        total_elements = spike_trains.numel()
        total_spikes = torch.sum(spike_trains).item()
        
        sparsity = 1.0 - (total_spikes / total_elements)
        
        # Spatial sparsity (across neurons)
        spatial_activity = torch.mean(spike_trains, dim=(0, 1))  # [neurons]
        spatial_sparsity = torch.sum(spatial_activity == 0).item() / len(spatial_activity)
        
        # Temporal sparsity (across time)
        temporal_activity = torch.mean(spike_trains, dim=(0, 2))  # [time]
        temporal_sparsity = torch.sum(temporal_activity == 0).item() / len(temporal_activity)
        
        return {
            'overall_sparsity': sparsity,
            'spatial_sparsity': spatial_sparsity,
            'temporal_sparsity': temporal_sparsity,
            'spike_rate_hz': total_spikes / (spike_trains.shape[1] * 0.001),  # Assuming 1ms timesteps
            'active_neurons_percent': (spatial_activity > 0).float().mean().item() * 100,
        }


def create_comprehensive_metrics(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Create a comprehensive metrics suite for neuromorphic evaluation.
    
    Args:
        device: Computation device
        
    Returns:
        metrics_suite: Dictionary containing all metric calculators
    """
    return {
        'spike_metrics': SpikeMetrics(device),
        'fusion_metrics': FusionMetrics(device),
        'performance_metrics': PerformanceMetrics(),
    }


# Utility functions for quick metric computation
def quick_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Quick accuracy computation."""
    pred_classes = torch.argmax(predictions, dim=-1)
    return (pred_classes == targets).float().mean().item()


def quick_firing_rate(spike_trains: torch.Tensor, dt: float = 1.0) -> float:
    """Quick firing rate computation."""
    total_spikes = torch.sum(spike_trains).item()
    total_time = spike_trains.shape[1] * dt / 1000.0  # Convert to seconds
    total_neurons = spike_trains.shape[0] * spike_trains.shape[2]
    return total_spikes / (total_time * total_neurons)


def quick_sparsity(spike_trains: torch.Tensor) -> float:
    """Quick sparsity computation."""
    return 1.0 - (torch.sum(spike_trains).item() / spike_trains.numel())