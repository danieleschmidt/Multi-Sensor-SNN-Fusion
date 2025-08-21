"""
Neuromorphic Security Framework

Implements security measures specifically designed for neuromorphic computing systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hashlib
import hmac
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum


class ThreatLevel(Enum):
    """Security threat levels."""
    BENIGN = "BENIGN"
    SUSPICIOUS = "SUSPICIOUS"
    MALICIOUS = "MALICIOUS"
    CRITICAL = "CRITICAL"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    description: str
    source_info: Dict[str, Any]
    metadata: Dict[str, Any]


class SpikeTrainAnalyzer:
    """Analyzes spike trains for security threats and anomalies."""
    
    def __init__(
        self,
        anomaly_threshold: float = 0.85,
        temporal_window: int = 100,
        frequency_analysis: bool = True
    ):
        self.anomaly_threshold = anomaly_threshold
        self.temporal_window = temporal_window
        self.frequency_analysis = frequency_analysis
        
        self.baseline_stats = {}
        self.pattern_history = []
        
    def analyze_spike_patterns(self, spikes: torch.Tensor) -> Dict[str, float]:
        """Analyze spike train patterns for anomalies."""
        
        if not isinstance(spikes, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        if spikes.dim() == 3:
            batch_size = spikes.size(0)
            results = []
            for b in range(batch_size):
                result = self._analyze_single_spike_train(spikes[b])
                results.append(result)
            return self._aggregate_batch_results(results)
        else:
            return self._analyze_single_spike_train(spikes)
    
    def _analyze_single_spike_train(self, spikes: torch.Tensor) -> Dict[str, float]:
        """Analyze a single spike train."""
        neurons, time_steps = spikes.shape
        
        total_spikes = spikes.sum().item()
        spike_rate = total_spikes / (neurons * time_steps)
        
        temporal_stats = self._analyze_temporal_patterns(spikes)
        spatial_stats = self._analyze_spatial_patterns(spikes)
        
        all_stats = {
            'total_spikes': total_spikes,
            'spike_rate': spike_rate,
            **temporal_stats,
            **spatial_stats,
        }
        
        anomaly_score = self._calculate_anomaly_score(all_stats)
        all_stats['anomaly_score'] = anomaly_score
        
        return all_stats
    
    def _analyze_temporal_patterns(self, spikes: torch.Tensor) -> Dict[str, float]:
        """Analyze temporal patterns in spike trains."""
        time_steps = spikes.size(1)
        
        spike_times = []
        for neuron in range(spikes.size(0)):
            neuron_spikes = torch.nonzero(spikes[neuron]).flatten()
            if len(neuron_spikes) > 1:
                isis = torch.diff(neuron_spikes.float()).cpu().numpy()
                spike_times.extend(isis)
        
        if len(spike_times) > 0:
            spike_times = np.array(spike_times)
            isi_mean = np.mean(spike_times)
            isi_std = np.std(spike_times)
            isi_cv = isi_std / isi_mean if isi_mean > 0 else 0
        else:
            isi_mean = isi_std = isi_cv = 0
        
        temporal_activity = spikes.sum(dim=0).float()
        if len(temporal_activity) > 1:
            temp_corr = torch.corrcoef(torch.stack([
                temporal_activity[:-1], 
                temporal_activity[1:]
            ]))[0, 1].item()
        else:
            temp_corr = 0
        
        burst_threshold = 3
        bursts = (temporal_activity > burst_threshold).sum().item()
        burst_ratio = bursts / time_steps
        
        return {
            'isi_mean': isi_mean,
            'isi_std': isi_std,
            'isi_cv': isi_cv,
            'temporal_correlation': temp_corr,
            'burst_ratio': burst_ratio
        }
    
    def _analyze_spatial_patterns(self, spikes: torch.Tensor) -> Dict[str, float]:
        """Analyze spatial patterns in spike trains."""
        neurons = spikes.size(0)
        
        neuron_rates = spikes.mean(dim=1)
        rate_mean = neuron_rates.mean().item()
        rate_std = neuron_rates.std().item()
        rate_cv = rate_std / rate_mean if rate_mean > 0 else 0
        
        if neurons > 1:
            spike_corr_matrix = torch.corrcoef(spikes)
            mask = ~torch.eye(neurons, dtype=bool)
            off_diag_corr = spike_corr_matrix[mask]
            spatial_corr_mean = off_diag_corr.mean().item()
            spatial_corr_std = off_diag_corr.std().item()
        else:
            spatial_corr_mean = spatial_corr_std = 0
        
        active_neurons = (neuron_rates > 0).sum().item()
        active_ratio = active_neurons / neurons
        
        return {
            'rate_mean': rate_mean,
            'rate_std': rate_std,
            'rate_cv': rate_cv,
            'spatial_correlation_mean': spatial_corr_mean,
            'spatial_correlation_std': spatial_corr_std,
            'active_neuron_ratio': active_ratio
        }
    
    def _calculate_anomaly_score(self, stats: Dict[str, float]) -> float:
        """Calculate overall anomaly score based on statistics."""
        anomaly_indicators = []
        
        if stats.get('spike_rate', 0) > 0.8 or stats.get('spike_rate', 0) < 0.001:
            anomaly_indicators.append(0.3)
        
        if abs(stats.get('temporal_correlation', 0)) > 0.9:
            anomaly_indicators.append(0.2)
        
        if abs(stats.get('spatial_correlation_mean', 0)) > 0.8:
            anomaly_indicators.append(0.2)
        
        if stats.get('burst_ratio', 0) > 0.5:
            anomaly_indicators.append(0.15)
        
        return sum(anomaly_indicators)
    
    def _aggregate_batch_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate results across batch."""
        if not results:
            return {}
        
        aggregated = {}
        for key in results[0].keys():
            values = [r[key] for r in results]
            aggregated[key] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated


class AdversarialDetector:
    """Detects adversarial attacks on neuromorphic systems."""
    
    def __init__(
        self,
        detection_threshold: float = 0.7,
        ensemble_size: int = 5,
        confidence_threshold: float = 0.9
    ):
        self.detection_threshold = detection_threshold
        self.ensemble_size = ensemble_size
        self.confidence_threshold = confidence_threshold
        
        self.spike_analyzer = SpikeTrainAnalyzer()
        self.detection_history = []
        
    def detect_adversarial_input(
        self,
        input_data: torch.Tensor,
        model: nn.Module,
        original_prediction: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Detect adversarial inputs using multiple detection methods."""
        
        detection_results = {
            'is_adversarial': False,
            'confidence': 0.0,
            'threat_level': ThreatLevel.BENIGN,
            'detection_methods': {},
            'metadata': {}
        }
        
        try:
            methods = [
                self._statistical_detection,
                self._prediction_consistency_check,
                self._spike_pattern_analysis
            ]
            
            method_results = []
            for method in methods:
                try:
                    result = method(input_data, model, original_prediction)
                    method_results.append(result)
                    detection_results['detection_methods'][method.__name__] = result
                except Exception as e:
                    logging.warning(f"Detection method {method.__name__} failed: {e}")
                    continue
            
            if method_results:
                detection_results = self._make_ensemble_decision(
                    method_results, 
                    detection_results
                )
            
            self._log_detection_event(detection_results, input_data)
            
        except Exception as e:
            logging.error(f"Adversarial detection failed: {e}")
            detection_results['error'] = str(e)
            detection_results['threat_level'] = ThreatLevel.SUSPICIOUS
        
        return detection_results
    
    def _statistical_detection(
        self, 
        input_data: torch.Tensor, 
        model: nn.Module, 
        original_prediction: Optional[torch.Tensor]
    ) -> Dict[str, Any]:
        """Statistical anomaly detection."""
        
        mean_val = input_data.mean().item()
        std_val = input_data.std().item()
        min_val = input_data.min().item()
        max_val = input_data.max().item()
        
        anomaly_score = 0.0
        
        if min_val < -10 or max_val > 10:
            anomaly_score += 0.3
        
        if std_val > 5.0 or std_val < 0.001:
            anomaly_score += 0.2
        
        if torch.isnan(input_data).any() or torch.isinf(input_data).any():
            anomaly_score += 0.5
        
        return {
            'anomaly_score': anomaly_score,
            'is_anomalous': anomaly_score > self.detection_threshold,
            'statistics': {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val
            }
        }
    
    def _prediction_consistency_check(
        self, 
        input_data: torch.Tensor, 
        model: nn.Module, 
        original_prediction: Optional[torch.Tensor]
    ) -> Dict[str, Any]:
        """Check prediction consistency with noise injection."""
        
        if original_prediction is None:
            with torch.no_grad():
                original_prediction = model(input_data)
        
        consistency_scores = []
        
        for _ in range(5):
            noise = torch.randn_like(input_data) * 0.01
            noisy_input = input_data + noise
            
            with torch.no_grad():
                noisy_prediction = model(noisy_input)
            
            if original_prediction.dim() > 1:
                consistency = F.cosine_similarity(
                    original_prediction.flatten(),
                    noisy_prediction.flatten(),
                    dim=0
                ).item()
            else:
                consistency = 1.0 - torch.abs(original_prediction - noisy_prediction).mean().item()
            
            consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores)
        anomaly_score = 1.0 - avg_consistency
        
        return {
            'anomaly_score': anomaly_score,
            'is_anomalous': anomaly_score > self.detection_threshold,
            'consistency_score': avg_consistency,
            'consistency_std': np.std(consistency_scores)
        }
    
    def _spike_pattern_analysis(
        self, 
        input_data: torch.Tensor, 
        model: nn.Module, 
        original_prediction: Optional[torch.Tensor]
    ) -> Dict[str, Any]:
        """Analyze spike patterns for adversarial characteristics."""
        
        if input_data.max() > 1.0 or input_data.min() < 0.0:
            normalized = torch.sigmoid(input_data)
            spikes = (normalized > 0.5).float()
        else:
            spikes = input_data
        
        pattern_analysis = self.spike_analyzer.analyze_spike_patterns(spikes)
        anomaly_score = pattern_analysis.get('anomaly_score', 0.0)
        
        return {
            'anomaly_score': anomaly_score,
            'is_anomalous': anomaly_score > self.detection_threshold,
            'pattern_analysis': pattern_analysis
        }
    
    def _make_ensemble_decision(
        self, 
        method_results: List[Dict[str, Any]], 
        detection_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make ensemble decision based on multiple detection methods."""
        
        anomaly_scores = [r.get('anomaly_score', 0.0) for r in method_results]
        is_anomalous_votes = [r.get('is_anomalous', False) for r in method_results]
        
        avg_anomaly_score = np.mean(anomaly_scores)
        anomaly_vote_ratio = sum(is_anomalous_votes) / len(is_anomalous_votes)
        
        is_adversarial = (
            avg_anomaly_score > self.detection_threshold or 
            anomaly_vote_ratio > 0.5
        )
        
        if avg_anomaly_score > 0.9:
            threat_level = ThreatLevel.CRITICAL
        elif avg_anomaly_score > 0.7:
            threat_level = ThreatLevel.MALICIOUS
        elif avg_anomaly_score > 0.3:
            threat_level = ThreatLevel.SUSPICIOUS
        else:
            threat_level = ThreatLevel.BENIGN
        
        detection_results.update({
            'is_adversarial': is_adversarial,
            'confidence': avg_anomaly_score,
            'threat_level': threat_level,
            'ensemble_score': avg_anomaly_score,
            'vote_ratio': anomaly_vote_ratio
        })
        
        return detection_results
    
    def _log_detection_event(self, detection_results: Dict[str, Any], input_data: torch.Tensor):
        """Log security detection event."""
        
        event = SecurityEvent(
            timestamp=time.time(),
            event_type="adversarial_detection",
            threat_level=detection_results['threat_level'],
            description=f"Adversarial detection: {detection_results['is_adversarial']}",
            source_info={
                'input_shape': list(input_data.shape),
                'ensemble_score': detection_results.get('ensemble_score', 0.0)
            },
            metadata=detection_results
        )
        
        self.detection_history.append(event)
        
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]


class SecurityValidator:
    """Security validation for neuromorphic systems."""
    
    @staticmethod
    def validate_input_safety(data: torch.Tensor, max_memory_gb: float = 4.0) -> bool:
        """Validate input data for memory safety."""
        memory_bytes = data.numel() * data.element_size()
        memory_gb = memory_bytes / (1024 ** 3)
        
        if memory_gb > max_memory_gb:
            raise ValueError(f"Input data too large: {memory_gb:.2f}GB > {max_memory_gb}GB")
        
        return True
    
    @staticmethod
    def sanitize_file_path(path: str) -> str:
        """Sanitize file paths to prevent directory traversal."""
        import os
        path = os.path.normpath(path)
        if '..' in path or path.startswith('/'):
            raise ValueError(f"Unsafe file path: {path}")
        return path