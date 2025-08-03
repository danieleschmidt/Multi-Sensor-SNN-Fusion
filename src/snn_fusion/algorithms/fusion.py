"""
Cross-Modal Fusion Algorithms

Implements fusion mechanisms for combining information from multiple
sensory modalities in neuromorphic computing systems.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FusionStrategy(Enum):
    """Types of fusion strategies."""
    EARLY = "early"
    LATE = "late"
    INTERMEDIATE = "intermediate"
    ATTENTION = "attention"
    TEMPORAL = "temporal"
    SPATIOTEMPORAL = "spatiotemporal"


@dataclass
class ModalityData:
    """Data structure for modality-specific information."""
    modality_name: str
    spike_times: np.ndarray
    neuron_ids: np.ndarray
    features: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    temporal_info: Optional[Dict[str, Any]] = None


@dataclass
class FusionResult:
    """Result of cross-modal fusion."""
    fused_spikes: np.ndarray
    fusion_weights: Dict[str, float]
    attention_map: Optional[np.ndarray]
    temporal_alignment: Optional[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


class CrossModalFusion(ABC):
    """
    Abstract base class for cross-modal fusion mechanisms.
    
    Provides common interface for combining information from
    multiple sensory modalities in neuromorphic systems.
    """
    
    def __init__(
        self,
        modalities: List[str],
        fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION,
        temporal_window: float = 50.0,  # ms
    ):
        """
        Initialize cross-modal fusion.
        
        Args:
            modalities: List of modality names
            fusion_strategy: Strategy for fusion
            temporal_window: Temporal window for fusion in ms
        """
        self.modalities = modalities
        self.fusion_strategy = fusion_strategy
        self.temporal_window = temporal_window
        self.logger = logging.getLogger(__name__)
        
        # Fusion state
        self.modality_weights = {mod: 1.0 for mod in modalities}
        self.fusion_history = []
        
        self.logger.info(f"Initialized cross-modal fusion for {modalities}")
    
    @abstractmethod
    def fuse_modalities(
        self,
        modality_data: Dict[str, ModalityData],
    ) -> FusionResult:
        """
        Perform cross-modal fusion.
        
        Args:
            modality_data: Dictionary of modality data
            
        Returns:
            Fusion result
        """
        pass
    
    def set_modality_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for different modalities."""
        for modality in self.modalities:
            if modality in weights:
                self.modality_weights[modality] = weights[modality]
        
        self.logger.info(f"Updated modality weights: {self.modality_weights}")
    
    def get_fusion_history(self, n_recent: int = 10) -> List[Dict[str, Any]]:
        """Get recent fusion history."""
        return self.fusion_history[-n_recent:]


class AttentionMechanism(CrossModalFusion):
    """
    Attention-based cross-modal fusion.
    
    Implements attention mechanisms to dynamically weight different
    modalities based on their relevance and reliability.
    """
    
    def __init__(
        self,
        modalities: List[str],
        temporal_window: float = 50.0,
        attention_type: str = "cross_modal",
        decay_factor: float = 0.9,
    ):
        """
        Initialize attention mechanism.
        
        Args:
            modalities: List of modality names
            temporal_window: Temporal window in ms
            attention_type: Type of attention ('self', 'cross_modal', 'temporal')
            decay_factor: Temporal decay factor
        """
        super().__init__(modalities, FusionStrategy.ATTENTION, temporal_window)
        
        self.attention_type = attention_type
        self.decay_factor = decay_factor
        
        # Attention parameters
        self.attention_weights = np.ones((len(modalities), len(modalities)))
        self.temporal_attention = np.ones(100)  # Attention over time
        
        # Learning parameters
        self.learning_rate = 0.01
        self.attention_history = []
    
    def fuse_modalities(
        self,
        modality_data: Dict[str, ModalityData],
    ) -> FusionResult:
        """
        Perform attention-based fusion.
        
        Args:
            modality_data: Dictionary of modality data
            
        Returns:
            Fusion result with attention
        """
        try:
            # Extract features from each modality
            modality_features = self._extract_modality_features(modality_data)
            
            # Compute attention weights
            attention_map = self._compute_attention_weights(modality_features)
            
            # Apply attention-weighted fusion
            fused_result = self._apply_attention_fusion(
                modality_data, modality_features, attention_map
            )
            
            # Update attention history
            self._update_attention_history(attention_map, modality_features)
            
            return fused_result
            
        except Exception as e:
            self.logger.error(f"Failed to perform attention fusion: {e}")
            raise
    
    def _extract_modality_features(
        self,
        modality_data: Dict[str, ModalityData],
    ) -> Dict[str, np.ndarray]:
        """Extract features from each modality."""
        features = {}
        
        for modality, data in modality_data.items():
            if data.features is not None:
                features[modality] = data.features
            else:
                # Extract basic spike-based features
                features[modality] = self._compute_spike_features(data)
        
        return features
    
    def _compute_spike_features(self, data: ModalityData) -> np.ndarray:
        """Compute basic features from spike data."""
        # Create time bins
        time_bins = np.arange(0, self.temporal_window, 1.0)  # 1ms bins
        
        # Compute spike rate in each bin
        spike_rates = []
        for i in range(len(time_bins) - 1):
            bin_start = time_bins[i]
            bin_end = time_bins[i + 1]
            
            # Count spikes in this time bin
            bin_spikes = np.sum(
                (data.spike_times >= bin_start) & (data.spike_times < bin_end)
            )
            spike_rates.append(bin_spikes)
        
        # Additional features
        features = np.array([
            np.mean(spike_rates),  # Mean firing rate
            np.std(spike_rates),   # Firing rate variability
            len(data.spike_times), # Total spike count
            np.max(spike_rates) if spike_rates else 0,  # Peak firing rate
        ])
        
        return features
    
    def _compute_attention_weights(
        self,
        modality_features: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Compute attention weights between modalities."""
        n_modalities = len(self.modalities)
        attention_scores = np.zeros((n_modalities, n_modalities))
        
        # Get ordered feature arrays
        feature_arrays = []
        for modality in self.modalities:
            if modality in modality_features:
                feature_arrays.append(modality_features[modality])
            else:
                feature_arrays.append(np.zeros(4))  # Default feature size
        
        # Compute pairwise attention
        for i, features_i in enumerate(feature_arrays):
            for j, features_j in enumerate(feature_arrays):
                if i != j:
                    # Compute similarity/relevance
                    similarity = self._compute_feature_similarity(features_i, features_j)
                    attention_scores[i, j] = similarity
                else:
                    attention_scores[i, j] = 1.0  # Self-attention
        
        # Apply softmax to get attention weights
        attention_weights = self._softmax_2d(attention_scores)
        
        return attention_weights
    
    def _compute_feature_similarity(
        self,
        features_i: np.ndarray,
        features_j: np.ndarray,
    ) -> float:
        """Compute similarity between feature vectors."""
        # Normalize features
        norm_i = features_i / (np.linalg.norm(features_i) + 1e-8)
        norm_j = features_j / (np.linalg.norm(features_j) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(norm_i, norm_j)
        
        return max(0, similarity)  # ReLU activation
    
    def _softmax_2d(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to 2D array along last dimension."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _apply_attention_fusion(
        self,
        modality_data: Dict[str, ModalityData],
        modality_features: Dict[str, np.ndarray],
        attention_map: np.ndarray,
    ) -> FusionResult:
        """Apply attention-weighted fusion."""
        # Combine spike trains with attention weighting
        all_spike_times = []
        all_neuron_ids = []
        
        # Calculate confidence scores
        confidence_scores = {}
        
        for i, modality in enumerate(self.modalities):
            if modality not in modality_data:
                confidence_scores[modality] = 0.0
                continue
            
            data = modality_data[modality]
            
            # Get attention weight for this modality
            attention_weight = np.mean(attention_map[i, :])
            
            # Apply attention to spike selection
            n_spikes = len(data.spike_times)
            if n_spikes > 0:
                # Randomly select spikes based on attention weight
                keep_prob = min(1.0, attention_weight * 2)  # Scale factor
                keep_mask = np.random.random(n_spikes) < keep_prob
                
                selected_spikes = data.spike_times[keep_mask]
                selected_neurons = data.neuron_ids[keep_mask]
                
                all_spike_times.extend(selected_spikes)
                all_neuron_ids.extend(selected_neurons)
                
                confidence_scores[modality] = attention_weight
            else:
                confidence_scores[modality] = 0.0
        
        # Create fused spike array
        if all_spike_times:
            # Sort by time
            sort_indices = np.argsort(all_spike_times)
            fused_spikes = np.column_stack([
                np.array(all_spike_times)[sort_indices],
                np.array(all_neuron_ids)[sort_indices]
            ])
        else:
            fused_spikes = np.empty((0, 2))
        
        # Calculate fusion weights
        total_attention = np.sum([confidence_scores[mod] for mod in self.modalities])
        fusion_weights = {}
        if total_attention > 0:
            for modality in self.modalities:
                fusion_weights[modality] = confidence_scores[modality] / total_attention
        else:
            for modality in self.modalities:
                fusion_weights[modality] = 1.0 / len(self.modalities)
        
        return FusionResult(
            fused_spikes=fused_spikes,
            fusion_weights=fusion_weights,
            attention_map=attention_map,
            temporal_alignment=None,
            confidence_scores=confidence_scores,
            metadata={
                'fusion_type': 'attention',
                'attention_type': self.attention_type,
                'n_fused_spikes': len(fused_spikes),
            }
        )
    
    def _update_attention_history(
        self,
        attention_map: np.ndarray,
        modality_features: Dict[str, np.ndarray],
    ) -> None:
        """Update attention learning history."""
        self.attention_history.append({
            'timestamp': np.time.time(),
            'attention_map': attention_map.copy(),
            'modality_features': modality_features.copy(),
        })
        
        # Keep limited history
        if len(self.attention_history) > 100:
            self.attention_history.pop(0)


class TemporalFusion(CrossModalFusion):
    """
    Temporal-based cross-modal fusion.
    
    Aligns and fuses modalities based on temporal synchrony
    and causal relationships between spike patterns.
    """
    
    def __init__(
        self,
        modalities: List[str],
        temporal_window: float = 50.0,
        synchrony_threshold: float = 5.0,  # ms
        causality_window: float = 20.0,   # ms
    ):
        """
        Initialize temporal fusion.
        
        Args:
            modalities: List of modality names
            temporal_window: Temporal window in ms
            synchrony_threshold: Threshold for synchrony detection
            causality_window: Window for causality analysis
        """
        super().__init__(modalities, FusionStrategy.TEMPORAL, temporal_window)
        
        self.synchrony_threshold = synchrony_threshold
        self.causality_window = causality_window
        
        # Temporal alignment parameters
        self.temporal_offsets = {mod: 0.0 for mod in modalities}
        self.synchrony_history = []
    
    def fuse_modalities(
        self,
        modality_data: Dict[str, ModalityData],
    ) -> FusionResult:
        """
        Perform temporal-based fusion.
        
        Args:
            modality_data: Dictionary of modality data
            
        Returns:
            Fusion result with temporal alignment
        """
        try:
            # Detect temporal synchrony
            synchrony_analysis = self._analyze_temporal_synchrony(modality_data)
            
            # Align modalities temporally
            aligned_data = self._align_modalities(modality_data, synchrony_analysis)
            
            # Perform temporal fusion
            fused_result = self._perform_temporal_fusion(aligned_data, synchrony_analysis)
            
            return fused_result
            
        except Exception as e:
            self.logger.error(f"Failed to perform temporal fusion: {e}")
            raise
    
    def _analyze_temporal_synchrony(
        self,
        modality_data: Dict[str, ModalityData],
    ) -> Dict[str, Any]:
        """Analyze temporal synchrony between modalities."""
        synchrony_info = {
            'cross_correlations': {},
            'lag_analysis': {},
            'synchrony_events': [],
            'causality_matrix': np.zeros((len(self.modalities), len(self.modalities))),
        }
        
        modality_list = list(modality_data.keys())
        
        # Compute cross-correlations
        for i, mod_i in enumerate(modality_list):
            for j, mod_j in enumerate(modality_list):
                if i != j:
                    correlation, lags = self._compute_cross_correlation(
                        modality_data[mod_i], modality_data[mod_j]
                    )
                    
                    synchrony_info['cross_correlations'][f"{mod_i}_{mod_j}"] = correlation
                    synchrony_info['lag_analysis'][f"{mod_i}_{mod_j}"] = lags
                    
                    # Analyze causality
                    causality_score = self._analyze_causality(
                        modality_data[mod_i], modality_data[mod_j]
                    )
                    synchrony_info['causality_matrix'][i, j] = causality_score
        
        # Detect synchrony events
        synchrony_info['synchrony_events'] = self._detect_synchrony_events(modality_data)
        
        return synchrony_info
    
    def _compute_cross_correlation(
        self,
        data_i: ModalityData,
        data_j: ModalityData,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cross-correlation between two modalities."""
        # Create time histograms
        time_bins = np.arange(0, self.temporal_window, 1.0)
        
        hist_i = np.histogram(data_i.spike_times, bins=time_bins)[0]
        hist_j = np.histogram(data_j.spike_times, bins=time_bins)[0]
        
        # Compute cross-correlation
        correlation = np.correlate(hist_i, hist_j, mode='full')
        
        # Create lag vector
        n_bins = len(time_bins) - 1
        lags = np.arange(-n_bins + 1, n_bins)
        
        return correlation, lags
    
    def _analyze_causality(
        self,
        data_i: ModalityData,
        data_j: ModalityData,
    ) -> float:
        """Analyze causal relationship between modalities."""
        # Simple causality analysis based on temporal precedence
        causality_score = 0.0
        
        for spike_i in data_i.spike_times:
            # Find spikes in data_j that occur within causality window
            future_spikes = data_j.spike_times[
                (data_j.spike_times > spike_i) &
                (data_j.spike_times <= spike_i + self.causality_window)
            ]
            
            # Score based on number of future spikes
            causality_score += len(future_spikes)
        
        # Normalize by total spikes
        if len(data_i.spike_times) > 0:
            causality_score /= len(data_i.spike_times)
        
        return causality_score
    
    def _detect_synchrony_events(
        self,
        modality_data: Dict[str, ModalityData],
    ) -> List[Dict[str, Any]]:
        """Detect synchronous events across modalities."""
        synchrony_events = []
        
        # Create combined spike times
        all_spikes = []
        for modality, data in modality_data.items():
            for spike_time in data.spike_times:
                all_spikes.append((spike_time, modality))
        
        # Sort by time
        all_spikes.sort(key=lambda x: x[0])
        
        # Detect synchrony events
        i = 0
        while i < len(all_spikes):
            event_start = all_spikes[i][0]
            event_end = event_start + self.synchrony_threshold
            
            # Find all spikes within synchrony window
            event_spikes = []
            modalities_involved = set()
            
            j = i
            while j < len(all_spikes) and all_spikes[j][0] <= event_end:
                event_spikes.append(all_spikes[j])
                modalities_involved.add(all_spikes[j][1])
                j += 1
            
            # Record event if multiple modalities involved
            if len(modalities_involved) > 1:
                synchrony_events.append({
                    'start_time': event_start,
                    'end_time': event_end,
                    'modalities': list(modalities_involved),
                    'spike_count': len(event_spikes),
                    'synchrony_strength': len(modalities_involved) / len(self.modalities),
                })
            
            i = j
        
        return synchrony_events
    
    def _align_modalities(
        self,
        modality_data: Dict[str, ModalityData],
        synchrony_analysis: Dict[str, Any],
    ) -> Dict[str, ModalityData]:
        """Align modalities based on synchrony analysis."""
        aligned_data = {}
        
        # Use lag analysis to determine temporal offsets
        lag_analysis = synchrony_analysis['lag_analysis']
        
        # Calculate optimal offsets (simplified approach)
        for modality in self.modalities:
            if modality in modality_data:
                # Find average lag to other modalities
                total_lag = 0.0
                lag_count = 0
                
                for key, lags in lag_analysis.items():
                    if modality in key:
                        correlations = synchrony_analysis['cross_correlations'][key]
                        max_corr_idx = np.argmax(correlations)
                        optimal_lag = lags[max_corr_idx]
                        
                        total_lag += optimal_lag
                        lag_count += 1
                
                if lag_count > 0:
                    average_lag = total_lag / lag_count
                    self.temporal_offsets[modality] = average_lag
                
                # Apply temporal offset
                data = modality_data[modality]
                aligned_spike_times = data.spike_times + self.temporal_offsets[modality]
                
                aligned_data[modality] = ModalityData(
                    modality_name=data.modality_name,
                    spike_times=aligned_spike_times,
                    neuron_ids=data.neuron_ids,
                    features=data.features,
                    attention_weights=data.attention_weights,
                    temporal_info=data.temporal_info,
                )
        
        return aligned_data
    
    def _perform_temporal_fusion(
        self,
        aligned_data: Dict[str, ModalityData],
        synchrony_analysis: Dict[str, Any],
    ) -> FusionResult:
        """Perform temporal fusion of aligned modalities."""
        # Combine aligned spike trains
        all_spike_times = []
        all_neuron_ids = []
        
        neuron_offset = 0
        fusion_weights = {}
        confidence_scores = {}
        
        for modality in self.modalities:
            if modality in aligned_data:
                data = aligned_data[modality]
                
                # Add spikes with neuron offset
                all_spike_times.extend(data.spike_times)
                all_neuron_ids.extend(data.neuron_ids + neuron_offset)
                
                # Calculate fusion weight based on synchrony
                synchrony_events = synchrony_analysis['synchrony_events']
                modality_synchrony = sum(
                    1 for event in synchrony_events
                    if modality in event['modalities']
                )
                
                total_events = len(synchrony_events)
                sync_ratio = modality_synchrony / max(1, total_events)
                
                fusion_weights[modality] = sync_ratio
                confidence_scores[modality] = sync_ratio
                
                neuron_offset += len(np.unique(data.neuron_ids))
            else:
                fusion_weights[modality] = 0.0
                confidence_scores[modality] = 0.0
        
        # Normalize fusion weights
        total_weight = sum(fusion_weights.values())
        if total_weight > 0:
            fusion_weights = {k: v / total_weight for k, v in fusion_weights.items()}
        
        # Create fused spike array
        if all_spike_times:
            sort_indices = np.argsort(all_spike_times)
            fused_spikes = np.column_stack([
                np.array(all_spike_times)[sort_indices],
                np.array(all_neuron_ids)[sort_indices]
            ])
        else:
            fused_spikes = np.empty((0, 2))
        
        return FusionResult(
            fused_spikes=fused_spikes,
            fusion_weights=fusion_weights,
            attention_map=None,
            temporal_alignment=synchrony_analysis,
            confidence_scores=confidence_scores,
            metadata={
                'fusion_type': 'temporal',
                'temporal_offsets': self.temporal_offsets,
                'synchrony_events': len(synchrony_analysis['synchrony_events']),
            }
        )


class SpatioTemporalFusion(CrossModalFusion):
    """
    Spatio-temporal fusion for neuromorphic networks.
    
    Combines spatial and temporal information across modalities
    for comprehensive multi-dimensional fusion.
    """
    
    def __init__(
        self,
        modalities: List[str],
        spatial_dimensions: Dict[str, Tuple[int, int]],
        temporal_window: float = 50.0,
        spatial_kernel_size: int = 3,
    ):
        """
        Initialize spatio-temporal fusion.
        
        Args:
            modalities: List of modality names
            spatial_dimensions: Spatial dimensions for each modality
            temporal_window: Temporal window in ms
            spatial_kernel_size: Size of spatial convolution kernel
        """
        super().__init__(modalities, FusionStrategy.SPATIOTEMPORAL, temporal_window)
        
        self.spatial_dimensions = spatial_dimensions
        self.spatial_kernel_size = spatial_kernel_size
        
        # Spatio-temporal state
        self.spatial_maps = {}
        self.temporal_dynamics = {}
    
    def fuse_modalities(
        self,
        modality_data: Dict[str, ModalityData],
    ) -> FusionResult:
        """
        Perform spatio-temporal fusion.
        
        Args:
            modality_data: Dictionary of modality data
            
        Returns:
            Fusion result with spatio-temporal processing
        """
        try:
            # Create spatial representations
            spatial_maps = self._create_spatial_representations(modality_data)
            
            # Analyze temporal dynamics
            temporal_dynamics = self._analyze_temporal_dynamics(modality_data)
            
            # Perform spatio-temporal convolution
            fused_result = self._spatiotemporal_convolution(
                spatial_maps, temporal_dynamics, modality_data
            )
            
            return fused_result
            
        except Exception as e:
            self.logger.error(f"Failed to perform spatio-temporal fusion: {e}")
            raise
    
    def _create_spatial_representations(
        self,
        modality_data: Dict[str, ModalityData],
    ) -> Dict[str, np.ndarray]:
        """Create spatial spike maps for each modality."""
        spatial_maps = {}
        
        for modality, data in modality_data.items():
            if modality in self.spatial_dimensions:
                height, width = self.spatial_dimensions[modality]
                
                # Create 2D spatial map
                spatial_map = np.zeros((height, width))
                
                # Map neuron IDs to spatial coordinates
                for neuron_id in data.neuron_ids:
                    y = neuron_id // width
                    x = neuron_id % width
                    
                    if 0 <= y < height and 0 <= x < width:
                        # Count spikes at this location
                        spike_count = np.sum(data.neuron_ids == neuron_id)
                        spatial_map[y, x] = spike_count
                
                spatial_maps[modality] = spatial_map
            else:
                # Create 1D representation as fallback
                max_neuron = np.max(data.neuron_ids) if len(data.neuron_ids) > 0 else 0
                spatial_map = np.zeros(max_neuron + 1)
                
                for neuron_id in data.neuron_ids:
                    spatial_map[neuron_id] += 1
                
                spatial_maps[modality] = spatial_map.reshape(-1, 1)
        
        return spatial_maps
    
    def _analyze_temporal_dynamics(
        self,
        modality_data: Dict[str, ModalityData],
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze temporal dynamics of each modality."""
        temporal_dynamics = {}
        
        for modality, data in modality_data.items():
            dynamics = {
                'spike_rate_over_time': self._compute_temporal_spike_rate(data),
                'burst_analysis': self._analyze_burst_patterns(data),
                'temporal_correlation': self._compute_temporal_autocorrelation(data),
            }
            
            temporal_dynamics[modality] = dynamics
        
        return temporal_dynamics
    
    def _compute_temporal_spike_rate(self, data: ModalityData) -> np.ndarray:
        """Compute spike rate over time."""
        time_bins = np.arange(0, self.temporal_window, 1.0)  # 1ms bins
        spike_rates = np.histogram(data.spike_times, bins=time_bins)[0]
        
        return spike_rates
    
    def _analyze_burst_patterns(self, data: ModalityData) -> Dict[str, Any]:
        """Analyze burst patterns in spike data."""
        if len(data.spike_times) < 2:
            return {'burst_count': 0, 'average_burst_length': 0, 'burst_intervals': []}
        
        # Find inter-spike intervals
        isi = np.diff(np.sort(data.spike_times))
        
        # Detect bursts (ISI < 5ms threshold)
        burst_threshold = 5.0  # ms
        burst_indices = np.where(isi < burst_threshold)[0]
        
        # Analyze burst structure
        bursts = []
        if len(burst_indices) > 0:
            current_burst = [burst_indices[0]]
            
            for i in range(1, len(burst_indices)):
                if burst_indices[i] == burst_indices[i-1] + 1:
                    current_burst.append(burst_indices[i])
                else:
                    if len(current_burst) > 1:
                        bursts.append(current_burst)
                    current_burst = [burst_indices[i]]
            
            if len(current_burst) > 1:
                bursts.append(current_burst)
        
        burst_analysis = {
            'burst_count': len(bursts),
            'average_burst_length': np.mean([len(burst) for burst in bursts]) if bursts else 0,
            'burst_intervals': [len(burst) for burst in bursts],
        }
        
        return burst_analysis
    
    def _compute_temporal_autocorrelation(self, data: ModalityData) -> np.ndarray:
        """Compute temporal autocorrelation of spike train."""
        spike_rates = self._compute_temporal_spike_rate(data)
        
        if len(spike_rates) > 1:
            # Compute autocorrelation
            autocorr = np.correlate(spike_rates, spike_rates, mode='full')
            # Normalize
            autocorr = autocorr / np.max(autocorr)
            
            # Return central portion
            center = len(autocorr) // 2
            window_size = min(50, center)
            return autocorr[center-window_size:center+window_size+1]
        else:
            return np.array([1.0])
    
    def _spatiotemporal_convolution(
        self,
        spatial_maps: Dict[str, np.ndarray],
        temporal_dynamics: Dict[str, Dict[str, Any]],
        modality_data: Dict[str, ModalityData],
    ) -> FusionResult:
        """Perform spatio-temporal convolution fusion."""
        # Create unified spatial representation
        unified_spatial = self._create_unified_spatial_map(spatial_maps)
        
        # Apply spatial filtering
        filtered_spatial = self._apply_spatial_filter(unified_spatial)
        
        # Combine with temporal information
        fusion_weights = self._compute_spatiotemporal_weights(
            spatial_maps, temporal_dynamics
        )
        
        # Generate fused spike train
        fused_spikes = self._generate_fused_spikes(
            modality_data, fusion_weights, filtered_spatial
        )
        
        confidence_scores = {}
        for modality in self.modalities:
            if modality in fusion_weights:
                confidence_scores[modality] = fusion_weights[modality]
            else:
                confidence_scores[modality] = 0.0
        
        return FusionResult(
            fused_spikes=fused_spikes,
            fusion_weights=fusion_weights,
            attention_map=filtered_spatial,
            temporal_alignment=temporal_dynamics,
            confidence_scores=confidence_scores,
            metadata={
                'fusion_type': 'spatiotemporal',
                'spatial_kernel_size': self.spatial_kernel_size,
                'unified_spatial_shape': unified_spatial.shape,
            }
        )
    
    def _create_unified_spatial_map(
        self,
        spatial_maps: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Create unified spatial representation."""
        if not spatial_maps:
            return np.array([[0]])
        
        # Find maximum dimensions
        max_height = 0
        max_width = 0
        
        for spatial_map in spatial_maps.values():
            if spatial_map.ndim == 2:
                h, w = spatial_map.shape
                max_height = max(max_height, h)
                max_width = max(max_width, w)
            else:
                max_height = max(max_height, len(spatial_map))
                max_width = max(max_width, 1)
        
        # Create unified map
        unified_map = np.zeros((max_height, max_width))
        
        for modality, spatial_map in spatial_maps.items():
            if spatial_map.ndim == 2:
                h, w = spatial_map.shape
                unified_map[:h, :w] += spatial_map
            else:
                h = len(spatial_map)
                unified_map[:h, 0] += spatial_map.flatten()
        
        return unified_map
    
    def _apply_spatial_filter(self, spatial_map: np.ndarray) -> np.ndarray:
        """Apply spatial filtering (simplified convolution)."""
        # Simple Gaussian-like kernel
        kernel_size = self.spatial_kernel_size
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        # Apply convolution (simplified)
        filtered = np.zeros_like(spatial_map)
        pad = kernel_size // 2
        
        for i in range(pad, spatial_map.shape[0] - pad):
            for j in range(pad, spatial_map.shape[1] - pad):
                region = spatial_map[i-pad:i+pad+1, j-pad:j+pad+1]
                if region.shape == kernel.shape:
                    filtered[i, j] = np.sum(region * kernel)
                else:
                    filtered[i, j] = spatial_map[i, j]
        
        return filtered
    
    def _compute_spatiotemporal_weights(
        self,
        spatial_maps: Dict[str, np.ndarray],
        temporal_dynamics: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute fusion weights based on spatio-temporal features."""
        weights = {}
        
        for modality in self.modalities:
            weight = 0.0
            
            if modality in spatial_maps:
                # Spatial contribution
                spatial_activity = np.sum(spatial_maps[modality])
                weight += spatial_activity * 0.5
            
            if modality in temporal_dynamics:
                # Temporal contribution
                dynamics = temporal_dynamics[modality]
                temporal_activity = np.sum(dynamics['spike_rate_over_time'])
                burst_activity = dynamics['burst_analysis']['burst_count']
                
                weight += temporal_activity * 0.3
                weight += burst_activity * 0.2
            
            weights[modality] = weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _generate_fused_spikes(
        self,
        modality_data: Dict[str, ModalityData],
        fusion_weights: Dict[str, float],
        spatial_attention: np.ndarray,
    ) -> np.ndarray:
        """Generate fused spike train."""
        all_spike_times = []
        all_neuron_ids = []
        
        neuron_offset = 0
        
        for modality in self.modalities:
            if modality in modality_data and modality in fusion_weights:
                data = modality_data[modality]
                weight = fusion_weights[modality]
                
                # Select spikes based on fusion weight
                n_spikes = len(data.spike_times)
                if n_spikes > 0:
                    keep_prob = min(1.0, weight * 2)
                    keep_mask = np.random.random(n_spikes) < keep_prob
                    
                    selected_spikes = data.spike_times[keep_mask]
                    selected_neurons = data.neuron_ids[keep_mask] + neuron_offset
                    
                    all_spike_times.extend(selected_spikes)
                    all_neuron_ids.extend(selected_neurons)
                
                neuron_offset += len(np.unique(data.neuron_ids))
        
        # Create fused spike array
        if all_spike_times:
            sort_indices = np.argsort(all_spike_times)
            fused_spikes = np.column_stack([
                np.array(all_spike_times)[sort_indices],
                np.array(all_neuron_ids)[sort_indices]
            ])
        else:
            fused_spikes = np.empty((0, 2))
        
        return fused_spikes