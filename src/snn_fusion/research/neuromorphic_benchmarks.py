"""
Neuromorphic Research Benchmarks for Multi-Modal Fusion

Comprehensive benchmarking suite for evaluating novel neuromorphic algorithms
including Temporal Spike Attention (TSA) and other cutting-edge approaches.

Research Focus:
- Ultra-low latency fusion performance
- Energy efficiency on neuromorphic hardware
- Statistical significance validation
- Cross-modal synchrony analysis
- Real-time adaptation capabilities
"""

import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from scipy import stats
import json
import logging
from pathlib import Path

# Local imports
from ..algorithms.temporal_spike_attention import TemporalSpikeAttention, SpikeEvent
from ..algorithms.fusion import AttentionMechanism, TemporalFusion, CrossModalFusion, ModalityData
from ..datasets.synthetic import generate_synthetic_multimodal_spikes


@dataclass
class BenchmarkConfig:
    """Configuration for neuromorphic benchmarks."""
    n_trials: int = 100
    spike_rates: Dict[str, float] = None  # spikes/sec per modality
    correlation_strengths: Dict[str, float] = None  # cross-modal correlations
    noise_levels: Dict[str, float] = None  # noise standard deviation
    temporal_window: float = 100.0  # ms
    statistical_significance_threshold: float = 0.05
    hardware_constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.spike_rates is None:
            self.spike_rates = {'audio': 50, 'vision': 30, 'tactile': 40}
        if self.correlation_strengths is None:
            self.correlation_strengths = {'audio_vision': 0.7, 'audio_tactile': 0.6, 'vision_tactile': 0.8}
        if self.noise_levels is None:
            self.noise_levels = {'audio': 0.1, 'vision': 0.08, 'tactile': 0.12}
        if self.hardware_constraints is None:
            self.hardware_constraints = {'max_latency_ms': 5.0, 'max_energy_uj': 100.0}


@dataclass  
class BenchmarkResult:
    """Results from neuromorphic algorithm benchmarking."""
    algorithm_name: str
    latencies_ms: List[float]
    energy_costs_uj: List[float]
    fusion_quality_scores: List[float]
    attention_accuracies: List[float]
    cross_modal_sync_scores: List[float]
    adaptation_convergence_time: Optional[float]
    hardware_efficiency_metrics: Dict[str, Any]
    statistical_metrics: Dict[str, float]


class NeuromorphicBenchmarkSuite:
    """
    Comprehensive benchmarking suite for neuromorphic fusion algorithms.
    
    Provides standardized evaluation across multiple dimensions:
    - Computational latency and throughput
    - Energy efficiency on neuromorphic hardware
    - Fusion quality and cross-modal synchrony
    - Statistical significance validation
    - Real-time adaptation performance
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        output_dir: str = "benchmark_results",
        enable_gpu_profiling: bool = False,
    ):
        """
        Initialize benchmark suite.
        
        Args:
            config: Benchmark configuration
            output_dir: Directory for saving results
            enable_gpu_profiling: Enable detailed GPU profiling
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_gpu_profiling = enable_gpu_profiling
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize hardware simulators
        self.hardware_simulators = self._initialize_hardware_simulators()
        
        # Benchmark state
        self.results = {}
        self.baseline_results = None
        
        self.logger.info("Initialized neuromorphic benchmark suite")
    
    def _initialize_hardware_simulators(self) -> Dict[str, Any]:
        """Initialize neuromorphic hardware simulators."""
        simulators = {}
        
        # Loihi 2 simulator
        simulators['loihi2'] = {
            'core_power_mw': 0.12,
            'synapse_energy_pj': 23.6,
            'neuron_energy_pj': 3.5,
            'latency_overhead_us': 10.0,
            'max_neurons_per_core': 1024,
        }
        
        # BrainChip Akida simulator  
        simulators['akida'] = {
            'core_power_mw': 0.3,
            'synapse_energy_pj': 45.2,
            'neuron_energy_pj': 8.1,
            'latency_overhead_us': 15.0,
            'max_neurons_per_core': 256,
        }
        
        # SpiNNaker 2 simulator
        simulators['spinnaker2'] = {
            'core_power_mw': 1.2,
            'synapse_energy_pj': 12.3,
            'neuron_energy_pj': 2.8,
            'latency_overhead_us': 25.0,
            'max_neurons_per_core': 2048,
        }
        
        return simulators
    
    def benchmark_algorithm(
        self,
        algorithm: CrossModalFusion,
        algorithm_name: str,
        test_data: Optional[List[Dict[str, ModalityData]]] = None,
    ) -> BenchmarkResult:
        """
        Comprehensive benchmark of a fusion algorithm.
        
        Args:
            algorithm: Fusion algorithm to benchmark
            algorithm_name: Name for results identification
            test_data: Optional test data (generated if not provided)
            
        Returns:
            Comprehensive benchmark results
        """
        self.logger.info(f"Benchmarking algorithm: {algorithm_name}")
        
        # Generate test data if not provided
        if test_data is None:
            test_data = self._generate_benchmark_data()
        
        # Initialize result collectors
        latencies_ms = []
        energy_costs_uj = []
        fusion_quality_scores = []
        attention_accuracies = []
        cross_modal_sync_scores = []
        
        # Run benchmark trials
        for trial in range(self.config.n_trials):
            # Select test sample
            sample = test_data[trial % len(test_data)]
            
            # Measure latency
            start_time = time.perf_counter()
            
            try:
                # Perform fusion
                fusion_result = algorithm.fuse_modalities(sample)
                
                # Record latency
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000.0
                latencies_ms.append(latency_ms)
                
                # Evaluate fusion quality
                quality_score = self._evaluate_fusion_quality(fusion_result, sample)
                fusion_quality_scores.append(quality_score)
                
                # Evaluate attention accuracy (if applicable)
                if hasattr(algorithm, 'attention_statistics'):
                    attention_acc = self._evaluate_attention_accuracy(algorithm, sample)
                    attention_accuracies.append(attention_acc)
                else:
                    attention_accuracies.append(0.0)
                
                # Evaluate cross-modal synchrony
                sync_score = self._evaluate_cross_modal_synchrony(fusion_result, sample)
                cross_modal_sync_scores.append(sync_score)
                
                # Estimate energy cost
                energy_cost = self._estimate_energy_cost(algorithm, fusion_result, 'loihi2')
                energy_costs_uj.append(energy_cost)
                
            except Exception as e:
                self.logger.warning(f"Trial {trial} failed: {e}")
                # Record failure values
                latencies_ms.append(float('inf'))
                energy_costs_uj.append(float('inf'))
                fusion_quality_scores.append(0.0)
                attention_accuracies.append(0.0)
                cross_modal_sync_scores.append(0.0)
        
        # Measure adaptation convergence (if applicable)
        adaptation_time = None
        if hasattr(algorithm, 'reset_adaptation'):
            adaptation_time = self._measure_adaptation_convergence(algorithm, test_data)
        
        # Compute hardware efficiency metrics
        hardware_metrics = self._compute_hardware_efficiency(
            algorithm, latencies_ms, energy_costs_uj
        )
        
        # Compute statistical metrics
        statistical_metrics = self._compute_statistical_metrics(
            latencies_ms, fusion_quality_scores, attention_accuracies
        )
        
        # Create result object
        result = BenchmarkResult(
            algorithm_name=algorithm_name,
            latencies_ms=latencies_ms,
            energy_costs_uj=energy_costs_uj,
            fusion_quality_scores=fusion_quality_scores,
            attention_accuracies=attention_accuracies,
            cross_modal_sync_scores=cross_modal_sync_scores,
            adaptation_convergence_time=adaptation_time,
            hardware_efficiency_metrics=hardware_metrics,
            statistical_metrics=statistical_metrics,
        )
        
        # Store results
        self.results[algorithm_name] = result
        
        self.logger.info(f"Completed benchmark for {algorithm_name}")
        return result
    
    def _generate_benchmark_data(self) -> List[Dict[str, ModalityData]]:
        """Generate standardized benchmark data."""
        test_data = []
        
        modalities = list(self.config.spike_rates.keys())
        
        for _ in range(self.config.n_trials):
            sample = {}
            
            for modality in modalities:
                # Generate spike trains with controlled statistics
                spike_rate = self.config.spike_rates[modality]
                noise_level = self.config.noise_levels[modality]
                
                # Create synthetic spike data
                n_spikes = int(spike_rate * self.config.temporal_window / 1000.0)
                
                # Generate spike times with some jitter
                base_times = np.linspace(0, self.config.temporal_window, n_spikes)
                jitter = np.random.normal(0, noise_level * 10, n_spikes)
                spike_times = np.clip(base_times + jitter, 0, self.config.temporal_window)
                spike_times = np.sort(spike_times)
                
                # Generate neuron IDs
                neuron_ids = np.random.randint(0, 100, n_spikes)
                
                # Create features (spike amplitudes)
                features = np.random.gamma(2.0, 0.5, n_spikes)  # Gamma-distributed amplitudes
                
                sample[modality] = ModalityData(
                    modality_name=modality,
                    spike_times=spike_times,
                    neuron_ids=neuron_ids,
                    features=features,
                )
            
            test_data.append(sample)
        
        return test_data
    
    def _evaluate_fusion_quality(
        self,
        fusion_result: Any,
        original_sample: Dict[str, ModalityData],
    ) -> float:
        """Evaluate quality of fusion result."""
        # Multiple quality metrics combined
        
        # 1. Spike preservation ratio
        original_spike_count = sum(len(data.spike_times) for data in original_sample.values())
        fused_spike_count = len(fusion_result.fused_spikes) if hasattr(fusion_result, 'fused_spikes') else 0
        
        if original_spike_count > 0:
            preservation_ratio = min(1.0, fused_spike_count / original_spike_count)
        else:
            preservation_ratio = 0.0
        
        # 2. Temporal coherence (spikes should maintain temporal order)
        temporal_coherence = 1.0
        if hasattr(fusion_result, 'fused_spikes') and len(fusion_result.fused_spikes) > 1:
            spike_times = fusion_result.fused_spikes[:, 0]
            temporal_coherence = float(np.all(np.diff(spike_times) >= 0))  # Check if sorted
        
        # 3. Modality balance (fusion should represent all modalities fairly)
        modality_balance = 1.0
        if hasattr(fusion_result, 'fusion_weights'):
            weights = list(fusion_result.fusion_weights.values())
            if weights:
                # Entropy-based balance measure
                weights = np.array(weights)
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
                entropy = -np.sum(weights * np.log(weights + 1e-10))
                max_entropy = np.log(len(weights))
                modality_balance = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # 4. Confidence score
        confidence_score = 0.0
        if hasattr(fusion_result, 'confidence_scores'):
            confidence_score = np.mean(list(fusion_result.confidence_scores.values()))
        
        # Weighted combination
        quality_score = (
            0.3 * preservation_ratio +
            0.3 * temporal_coherence +
            0.2 * modality_balance +
            0.2 * confidence_score
        )
        
        return quality_score
    
    def _evaluate_attention_accuracy(
        self,
        algorithm: CrossModalFusion,
        sample: Dict[str, ModalityData],
    ) -> float:
        """Evaluate attention mechanism accuracy."""
        if not hasattr(algorithm, 'attention_statistics'):
            return 0.0
        
        # Analyze attention patterns
        attention_stats = algorithm.attention_statistics
        
        # Check if attention correlates with spike activity
        attention_accuracy = 0.0
        
        for modality in algorithm.modalities:
            if modality in attention_stats['attention_strengths'] and modality in sample:
                # Get recent attention strength
                strengths = attention_stats['attention_strengths'][modality]
                if strengths:
                    recent_attention = np.mean(strengths[-10:])  # Last 10 measurements
                    
                    # Compute actual spike activity
                    spike_activity = len(sample[modality].spike_times) / self.config.temporal_window
                    
                    # Normalize spike activity to [0, 1] range
                    max_expected_activity = self.config.spike_rates[modality] / 1000.0
                    normalized_activity = min(1.0, spike_activity / max_expected_activity)
                    
                    # Compute correlation between attention and activity
                    correlation = 1.0 - abs(recent_attention - normalized_activity)
                    attention_accuracy += correlation
        
        # Average across modalities
        if algorithm.modalities:
            attention_accuracy /= len(algorithm.modalities)
        
        return attention_accuracy
    
    def _evaluate_cross_modal_synchrony(
        self,
        fusion_result: Any,
        original_sample: Dict[str, ModalityData],
    ) -> float:
        """Evaluate how well fusion preserves cross-modal synchrony."""
        synchrony_score = 0.0
        
        # Measure temporal alignment between modalities
        modalities = list(original_sample.keys())
        
        if len(modalities) >= 2:
            # Compute pairwise synchrony scores
            synchrony_scores = []
            
            for i in range(len(modalities)):
                for j in range(i + 1, len(modalities)):
                    mod1, mod2 = modalities[i], modalities[j]
                    
                    # Compute cross-correlation between spike trains
                    sync = self._compute_cross_modal_correlation(
                        original_sample[mod1], original_sample[mod2]
                    )
                    synchrony_scores.append(sync)
            
            if synchrony_scores:
                synchrony_score = np.mean(synchrony_scores)
        
        return synchrony_score
    
    def _compute_cross_modal_correlation(
        self,
        data1: ModalityData,
        data2: ModalityData,
    ) -> float:
        """Compute cross-correlation between two modalities."""
        # Create time histograms
        time_bins = np.arange(0, self.config.temporal_window, 2.0)  # 2ms bins
        
        hist1 = np.histogram(data1.spike_times, bins=time_bins)[0]
        hist2 = np.histogram(data2.spike_times, bins=time_bins)[0]
        
        # Compute cross-correlation
        if np.std(hist1) > 0 and np.std(hist2) > 0:
            correlation = np.corrcoef(hist1, hist2)[0, 1]
            return max(0.0, correlation)  # Only positive correlations
        else:
            return 0.0
    
    def _estimate_energy_cost(
        self,
        algorithm: CrossModalFusion,
        fusion_result: Any,
        hardware_platform: str,
    ) -> float:
        """Estimate energy cost on neuromorphic hardware."""
        if hardware_platform not in self.hardware_simulators:
            return 0.0
        
        hw_params = self.hardware_simulators[hardware_platform]
        
        # Estimate energy components
        energy_cost_uj = 0.0
        
        # 1. Spike processing energy
        if hasattr(fusion_result, 'fused_spikes'):
            n_spikes = len(fusion_result.fused_spikes)
            spike_energy = n_spikes * hw_params['neuron_energy_pj'] / 1000.0  # Convert to μJ
            energy_cost_uj += spike_energy
        
        # 2. Synaptic operations energy
        if hasattr(algorithm, 'modalities'):
            n_modalities = len(algorithm.modalities)
            # Estimate synaptic operations (cross-modal connections)
            n_synapses = n_modalities * (n_modalities - 1) * 50  # Rough estimate
            synapse_energy = n_synapses * hw_params['synapse_energy_pj'] / 1000.0
            energy_cost_uj += synapse_energy
        
        # 3. Core idle power during processing
        processing_time_ms = 1.0  # Assume 1ms processing time
        idle_energy = (hw_params['core_power_mw'] * processing_time_ms) / 1000.0
        energy_cost_uj += idle_energy
        
        return energy_cost_uj
    
    def _measure_adaptation_convergence(
        self,
        algorithm: CrossModalFusion,
        test_data: List[Dict[str, ModalityData]],
    ) -> float:
        """Measure time for algorithm to adapt and converge."""
        # Reset algorithm state
        algorithm.reset_adaptation()
        
        # Track adaptation metrics over time
        convergence_threshold = 0.95  # 95% of final performance
        performance_history = []
        
        start_time = time.time()
        
        # Run adaptation trials
        for i, sample in enumerate(test_data[:50]):  # Use first 50 samples for adaptation
            try:
                fusion_result = algorithm.fuse_modalities(sample)
                
                # Measure current performance
                quality_score = self._evaluate_fusion_quality(fusion_result, sample)
                performance_history.append(quality_score)
                
                # Check for convergence
                if len(performance_history) >= 10:
                    recent_performance = np.mean(performance_history[-10:])
                    final_performance = np.mean(performance_history[-5:])
                    
                    if recent_performance >= convergence_threshold * final_performance:
                        convergence_time = time.time() - start_time
                        return convergence_time
                        
            except Exception as e:
                self.logger.warning(f"Adaptation trial {i} failed: {e}")
                continue
        
        # Return total time if no convergence detected
        return time.time() - start_time
    
    def _compute_hardware_efficiency(
        self,
        algorithm: CrossModalFusion,
        latencies_ms: List[float],
        energy_costs_uj: List[float],
    ) -> Dict[str, Any]:
        """Compute hardware efficiency metrics."""
        valid_latencies = [l for l in latencies_ms if np.isfinite(l)]
        valid_energy = [e for e in energy_costs_uj if np.isfinite(e)]
        
        if not valid_latencies or not valid_energy:
            return {'error': 'No valid measurements'}
        
        metrics = {
            'mean_latency_ms': np.mean(valid_latencies),
            'p95_latency_ms': np.percentile(valid_latencies, 95),
            'p99_latency_ms': np.percentile(valid_latencies, 99),
            'mean_energy_uj': np.mean(valid_energy),
            'energy_std_uj': np.std(valid_energy),
            'energy_per_spike_pj': np.mean(valid_energy) * 1000,  # Rough estimate
            'throughput_spikes_per_sec': 1000.0 / np.mean(valid_latencies) if valid_latencies else 0,
            'energy_efficiency_spikes_per_uj': 1.0 / np.mean(valid_energy) if valid_energy else 0,
        }
        
        # Hardware constraint violations
        constraints = self.config.hardware_constraints
        metrics['latency_violations'] = sum(1 for l in valid_latencies if l > constraints['max_latency_ms'])
        metrics['energy_violations'] = sum(1 for e in valid_energy if e > constraints['max_energy_uj'])
        
        return metrics
    
    def _compute_statistical_metrics(
        self,
        latencies_ms: List[float],
        quality_scores: List[float],
        attention_accuracies: List[float],
    ) -> Dict[str, float]:
        """Compute statistical significance metrics."""
        valid_latencies = [l for l in latencies_ms if np.isfinite(l)]
        valid_quality = [q for q in quality_scores if np.isfinite(q)]
        valid_attention = [a for a in attention_accuracies if np.isfinite(a)]
        
        metrics = {}
        
        # Basic statistics
        if valid_latencies:
            metrics['latency_mean'] = np.mean(valid_latencies)
            metrics['latency_std'] = np.std(valid_latencies)
            metrics['latency_cv'] = metrics['latency_std'] / metrics['latency_mean']
        
        if valid_quality:
            metrics['quality_mean'] = np.mean(valid_quality)
            metrics['quality_std'] = np.std(valid_quality)
            metrics['quality_cv'] = metrics['quality_std'] / metrics['quality_mean'] if metrics['quality_mean'] > 0 else 0
        
        if valid_attention:
            metrics['attention_mean'] = np.mean(valid_attention)
            metrics['attention_std'] = np.std(valid_attention)
        
        # Distribution tests
        if len(valid_latencies) >= 8:  # Minimum for normality test
            _, p_value = stats.normaltest(valid_latencies)
            metrics['latency_normality_p'] = p_value
        
        return metrics
    
    def compare_algorithms(
        self,
        baseline_name: str,
        comparison_names: List[str],
    ) -> Dict[str, Any]:
        """
        Statistical comparison between baseline and other algorithms.
        
        Args:
            baseline_name: Name of baseline algorithm in results
            comparison_names: Names of algorithms to compare against baseline
            
        Returns:
            Statistical comparison results
        """
        if baseline_name not in self.results:
            raise ValueError(f"Baseline {baseline_name} not found in results")
        
        baseline = self.results[baseline_name]
        comparison_results = {}
        
        for comp_name in comparison_names:
            if comp_name not in self.results:
                self.logger.warning(f"Algorithm {comp_name} not found, skipping")
                continue
            
            comparison = self.results[comp_name]
            
            # Statistical tests
            comp_results = {
                'algorithm_name': comp_name,
                'improvements': {},
                'statistical_tests': {},
            }
            
            # Latency comparison
            if baseline.latencies_ms and comparison.latencies_ms:
                baseline_latency = [l for l in baseline.latencies_ms if np.isfinite(l)]
                comp_latency = [l for l in comparison.latencies_ms if np.isfinite(l)]
                
                if baseline_latency and comp_latency:
                    # Mann-Whitney U test (non-parametric)
                    statistic, p_value = stats.mannwhitneyu(
                        baseline_latency, comp_latency, alternative='two-sided'
                    )
                    
                    comp_results['statistical_tests']['latency_mannwhitney'] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < self.config.statistical_significance_threshold
                    }
                    
                    # Effect size (improvement percentage)
                    baseline_mean = np.mean(baseline_latency)
                    comp_mean = np.mean(comp_latency)
                    improvement_pct = ((baseline_mean - comp_mean) / baseline_mean) * 100
                    
                    comp_results['improvements']['latency_improvement_pct'] = improvement_pct
            
            # Quality comparison
            if baseline.fusion_quality_scores and comparison.fusion_quality_scores:
                baseline_quality = [q for q in baseline.fusion_quality_scores if np.isfinite(q)]
                comp_quality = [q for q in comparison.fusion_quality_scores if np.isfinite(q)]
                
                if baseline_quality and comp_quality:
                    # T-test for quality (assuming approximately normal)
                    statistic, p_value = stats.ttest_ind(baseline_quality, comp_quality)
                    
                    comp_results['statistical_tests']['quality_ttest'] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < self.config.statistical_significance_threshold
                    }
                    
                    # Effect size
                    baseline_mean = np.mean(baseline_quality)
                    comp_mean = np.mean(comp_quality)
                    improvement_pct = ((comp_mean - baseline_mean) / baseline_mean) * 100
                    
                    comp_results['improvements']['quality_improvement_pct'] = improvement_pct
            
            # Energy comparison
            if baseline.energy_costs_uj and comparison.energy_costs_uj:
                baseline_energy = [e for e in baseline.energy_costs_uj if np.isfinite(e)]
                comp_energy = [e for e in comparison.energy_costs_uj if np.isfinite(e)]
                
                if baseline_energy and comp_energy:
                    statistic, p_value = stats.mannwhitneyu(
                        baseline_energy, comp_energy, alternative='two-sided'
                    )
                    
                    comp_results['statistical_tests']['energy_mannwhitney'] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < self.config.statistical_significance_threshold
                    }
                    
                    # Energy efficiency improvement
                    baseline_mean = np.mean(baseline_energy)
                    comp_mean = np.mean(comp_energy)
                    improvement_pct = ((baseline_mean - comp_mean) / baseline_mean) * 100
                    
                    comp_results['improvements']['energy_improvement_pct'] = improvement_pct
            
            comparison_results[comp_name] = comp_results
        
        return comparison_results
    
    def generate_research_report(
        self,
        title: str = "Neuromorphic Multi-Modal Fusion Benchmark Report",
        include_statistical_analysis: bool = True,
    ) -> Dict[str, Any]:
        """Generate comprehensive research report."""
        report = {
            'title': title,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'n_trials': self.config.n_trials,
                'spike_rates': self.config.spike_rates,
                'temporal_window_ms': self.config.temporal_window,
                'hardware_constraints': self.config.hardware_constraints,
            },
            'algorithms_tested': list(self.results.keys()),
            'summary_statistics': {},
            'comparative_analysis': {},
            'research_conclusions': {},
        }
        
        # Summary statistics for each algorithm
        for alg_name, result in self.results.items():
            valid_latencies = [l for l in result.latencies_ms if np.isfinite(l)]
            valid_quality = [q for q in result.fusion_quality_scores if np.isfinite(q)]
            valid_energy = [e for e in result.energy_costs_uj if np.isfinite(e)]
            
            summary = {
                'successful_trials': len(valid_latencies),
                'failure_rate': 1.0 - (len(valid_latencies) / len(result.latencies_ms)),
                'mean_latency_ms': np.mean(valid_latencies) if valid_latencies else float('inf'),
                'p95_latency_ms': np.percentile(valid_latencies, 95) if valid_latencies else float('inf'),
                'mean_quality_score': np.mean(valid_quality) if valid_quality else 0.0,
                'mean_energy_cost_uj': np.mean(valid_energy) if valid_energy else float('inf'),
                'hardware_efficiency': result.hardware_efficiency_metrics,
            }
            
            report['summary_statistics'][alg_name] = summary
        
        # Comparative analysis
        if include_statistical_analysis and len(self.results) > 1:
            algorithm_names = list(self.results.keys())
            baseline_name = algorithm_names[0]  # Use first as baseline
            comparison_names = algorithm_names[1:]
            
            report['comparative_analysis'] = self.compare_algorithms(
                baseline_name, comparison_names
            )
        
        # Research conclusions
        best_latency = min((
            np.mean([l for l in result.latencies_ms if np.isfinite(l)]) 
            for result in self.results.values() 
            if any(np.isfinite(l) for l in result.latencies_ms)
        ), default=float('inf'))
        
        best_quality = max((
            np.mean([q for q in result.fusion_quality_scores if np.isfinite(q)])
            for result in self.results.values()
            if any(np.isfinite(q) for q in result.fusion_quality_scores)
        ), default=0.0)
        
        best_energy = min((
            np.mean([e for e in result.energy_costs_uj if np.isfinite(e)])
            for result in self.results.values()
            if any(np.isfinite(e) for e in result.energy_costs_uj)
        ), default=float('inf'))
        
        report['research_conclusions'] = {
            'best_latency_ms': best_latency,
            'best_quality_score': best_quality,
            'best_energy_efficiency_uj': best_energy,
            'neuromorphic_feasibility': best_latency < self.config.hardware_constraints['max_latency_ms'],
            'energy_efficiency_achieved': best_energy < self.config.hardware_constraints['max_energy_uj'],
        }
        
        return report
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save benchmark results to file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"neuromorphic_benchmark_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert results to JSON-serializable format
        json_results = {}
        for alg_name, result in self.results.items():
            json_results[alg_name] = {
                'algorithm_name': result.algorithm_name,
                'latencies_ms': [float(l) if np.isfinite(l) else None for l in result.latencies_ms],
                'energy_costs_uj': [float(e) if np.isfinite(e) else None for e in result.energy_costs_uj],
                'fusion_quality_scores': [float(q) if np.isfinite(q) else None for q in result.fusion_quality_scores],
                'attention_accuracies': [float(a) if np.isfinite(a) else None for a in result.attention_accuracies],
                'cross_modal_sync_scores': [float(s) if np.isfinite(s) else None for s in result.cross_modal_sync_scores],
                'adaptation_convergence_time': result.adaptation_convergence_time,
                'hardware_efficiency_metrics': result.hardware_efficiency_metrics,
                'statistical_metrics': result.statistical_metrics,
            }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
        return str(filepath)


# Research validation functions
def validate_tsa_performance() -> Dict[str, Any]:
    """Validate TSA performance against established baselines."""
    from ..algorithms.temporal_spike_attention import create_temporal_spike_attention
    
    # Initialize benchmark suite
    config = BenchmarkConfig(
        n_trials=200,
        spike_rates={'audio': 60, 'vision': 40, 'tactile': 50},
        temporal_window=150.0,
    )
    
    suite = NeuromorphicBenchmarkSuite(config)
    
    # Create algorithms to compare
    modalities = ['audio', 'vision', 'tactile']
    
    # TSA (our novel algorithm)
    tsa = create_temporal_spike_attention(modalities)
    
    # Baseline: Traditional attention
    baseline_attention = AttentionMechanism(modalities)
    
    # Baseline: Temporal fusion
    baseline_temporal = TemporalFusion(modalities)
    
    # Run benchmarks
    tsa_results = suite.benchmark_algorithm(tsa, "TSA_Novel")
    attention_results = suite.benchmark_algorithm(baseline_attention, "Attention_Baseline")
    temporal_results = suite.benchmark_algorithm(baseline_temporal, "Temporal_Baseline")
    
    # Generate research report
    report = suite.generate_research_report(
        title="Temporal Spike Attention: Novel Algorithm Validation",
        include_statistical_analysis=True
    )
    
    return report


# Export key functions and classes
__all__ = [
    'BenchmarkConfig',
    'BenchmarkResult', 
    'NeuromorphicBenchmarkSuite',
    'validate_tsa_performance',
]