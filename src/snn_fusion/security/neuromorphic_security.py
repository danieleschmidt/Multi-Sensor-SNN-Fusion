"""
Neuromorphic Security Framework

Advanced security measures specifically designed for neuromorphic computing
systems, including spike train validation, temporal integrity checking,
and hardware-specific security controls.
"""

import hashlib
import hmac
import time
import secrets
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
from pathlib import Path
import json

from ..algorithms.fusion import ModalityData
from ..utils.robust_error_handling import robust_function, ErrorCategory, ErrorSeverity


class SecurityThreat(Enum):
    """Types of security threats in neuromorphic systems."""
    SPIKE_INJECTION = "spike_injection"
    TEMPORAL_MANIPULATION = "temporal_manipulation"
    DATA_CORRUPTION = "data_corruption"
    REPLAY_ATTACK = "replay_attack"
    MODEL_EXTRACTION = "model_extraction"
    HARDWARE_TAMPERING = "hardware_tampering"
    SIDE_CHANNEL = "side_channel"
    ADVERSARIAL_SPIKES = "adversarial_spikes"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    threat_type: SecurityThreat
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    source_modality: Optional[str] = None
    detection_method: Optional[str] = None
    mitigation_applied: bool = False
    additional_data: Dict[str, Any] = field(default_factory=dict)


class SpikeTrainValidator:
    """
    Validates spike trains for security threats and data integrity.
    
    Detects:
    - Abnormal spike patterns
    - Temporal anomalies
    - Statistical deviations
    - Adversarial spike injections
    """
    
    def __init__(
        self,
        baseline_stats: Optional[Dict[str, Any]] = None,
        anomaly_threshold: float = 3.0,  # Standard deviations
    ):
        """
        Initialize spike train validator.
        
        Args:
            baseline_stats: Baseline statistics for normal operation
            anomaly_threshold: Threshold for anomaly detection (in std devs)
        """
        self.baseline_stats = baseline_stats or {}
        self.anomaly_threshold = anomaly_threshold
        self.logger = logging.getLogger(__name__)
        
        # Security monitoring
        self.validation_history = []
        self.threat_counters = {threat.value: 0 for threat in SecurityThreat}
    
    def validate_spike_statistics(
        self,
        modality_data: ModalityData,
        modality_name: str,
    ) -> Tuple[bool, List[SecurityEvent]]:
        """
        Validate spike train statistics for anomalies.
        
        Args:
            modality_data: Spike data to validate
            modality_name: Name of the modality
            
        Returns:
            (is_valid, security_events) tuple
        """
        events = []
        is_valid = True
        
        if len(modality_data.spike_times) == 0:
            return True, events  # Empty data is considered valid
        
        # Check spike rate
        duration = np.max(modality_data.spike_times) - np.min(modality_data.spike_times)
        if duration > 0:
            spike_rate = len(modality_data.spike_times) / (duration / 1000.0)  # spikes/sec
            
            if modality_name in self.baseline_stats:
                expected_rate = self.baseline_stats[modality_name].get('mean_spike_rate', 50)
                rate_std = self.baseline_stats[modality_name].get('spike_rate_std', 10)
                
                # Check for abnormal spike rates
                if abs(spike_rate - expected_rate) > self.anomaly_threshold * rate_std:
                    severity = 'high' if spike_rate > expected_rate * 2 else 'medium'
                    
                    events.append(SecurityEvent(
                        timestamp=time.time(),
                        threat_type=SecurityThreat.SPIKE_INJECTION if spike_rate > expected_rate else SecurityThreat.DATA_CORRUPTION,
                        severity=severity,
                        description=f"Abnormal spike rate: {spike_rate:.2f} (expected: {expected_rate:.2f})",
                        source_modality=modality_name,
                        detection_method="statistical_validation",
                        additional_data={'spike_rate': spike_rate, 'expected_rate': expected_rate}
                    ))
                    
                    if severity == 'high':
                        is_valid = False
        
        # Check inter-spike intervals for regularity attacks
        if len(modality_data.spike_times) > 1:
            isi = np.diff(modality_data.spike_times)
            isi_cv = np.std(isi) / np.mean(isi) if np.mean(isi) > 0 else 0
            
            # Extremely regular patterns might indicate artificial injection
            if isi_cv < 0.1 and len(isi) > 10:
                events.append(SecurityEvent(
                    timestamp=time.time(),
                    threat_type=SecurityThreat.SPIKE_INJECTION,
                    severity='medium',
                    description=f"Suspiciously regular spike pattern (CV: {isi_cv:.3f})",
                    source_modality=modality_name,
                    detection_method="temporal_regularity_check",
                    additional_data={'isi_cv': isi_cv}
                ))
        
        # Check for temporal clustering (burst attacks)
        if len(modality_data.spike_times) > 5:
            # Look for abnormal clustering
            window_size = 10.0  # 10ms windows
            time_windows = np.arange(
                np.min(modality_data.spike_times),
                np.max(modality_data.spike_times) + window_size,
                window_size
            )
            
            spike_counts = np.histogram(modality_data.spike_times, bins=time_windows)[0]
            
            if len(spike_counts) > 0:
                # Check for abnormal bursts
                max_burst = np.max(spike_counts)
                mean_count = np.mean(spike_counts)
                
                if max_burst > mean_count + self.anomaly_threshold * np.std(spike_counts):
                    events.append(SecurityEvent(
                        timestamp=time.time(),
                        threat_type=SecurityThreat.ADVERSARIAL_SPIKES,
                        severity='medium',
                        description=f"Abnormal temporal clustering detected (max: {max_burst}, mean: {mean_count:.2f})",
                        source_modality=modality_name,
                        detection_method="burst_detection",
                        additional_data={'max_burst': int(max_burst), 'mean_count': mean_count}
                    ))
        
        return is_valid, events
    
    def validate_neuron_distribution(
        self,
        modality_data: ModalityData,
        modality_name: str,
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Validate neuron ID distribution for tampering."""
        events = []
        is_valid = True
        
        if len(modality_data.neuron_ids) == 0:
            return True, events
        
        # Check for suspicious neuron ID patterns
        unique_neurons = np.unique(modality_data.neuron_ids)
        total_spikes = len(modality_data.neuron_ids)
        
        # Check for overly concentrated activity
        neuron_counts = np.bincount(modality_data.neuron_ids)
        max_activity = np.max(neuron_counts) if len(neuron_counts) > 0 else 0
        
        concentration_ratio = max_activity / total_spikes if total_spikes > 0 else 0
        
        if concentration_ratio > 0.8:  # More than 80% of spikes from single neuron
            events.append(SecurityEvent(
                timestamp=time.time(),
                threat_type=SecurityThreat.SPIKE_INJECTION,
                severity='high',
                description=f"Excessive spike concentration in single neuron ({concentration_ratio:.2%})",
                source_modality=modality_name,
                detection_method="neuron_concentration_check",
                additional_data={'concentration_ratio': concentration_ratio}
            ))
            is_valid = False
        
        # Check for sequential neuron ID patterns (might indicate synthetic data)
        if len(unique_neurons) > 5:
            sequential_count = 0
            for i in range(len(unique_neurons) - 1):
                if unique_neurons[i+1] - unique_neurons[i] == 1:
                    sequential_count += 1
            
            sequential_ratio = sequential_count / len(unique_neurons)
            
            if sequential_ratio > 0.9:  # >90% sequential
                events.append(SecurityEvent(
                    timestamp=time.time(),
                    threat_type=SecurityThreat.DATA_CORRUPTION,
                    severity='medium',
                    description=f"Suspicious sequential neuron pattern ({sequential_ratio:.2%})",
                    source_modality=modality_name,
                    detection_method="sequential_pattern_check",
                    additional_data={'sequential_ratio': sequential_ratio}
                ))
        
        return is_valid, events
    
    def validate_features(
        self,
        modality_data: ModalityData,
        modality_name: str,
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Validate spike features for adversarial manipulation."""
        events = []
        is_valid = True
        
        if modality_data.features is None or len(modality_data.features) == 0:
            return True, events
        
        features = modality_data.features
        
        # Check for out-of-range values
        if np.any(features < 0) or np.any(features > 10):  # Reasonable range
            events.append(SecurityEvent(
                timestamp=time.time(),
                threat_type=SecurityThreat.DATA_CORRUPTION,
                severity='high',
                description="Features outside expected range detected",
                source_modality=modality_name,
                detection_method="feature_range_check",
                additional_data={
                    'min_feature': float(np.min(features)),
                    'max_feature': float(np.max(features))
                }
            ))
            is_valid = False
        
        # Check for abnormal feature distributions
        if len(features) > 10:
            # Check for uniform distribution (might indicate synthetic data)
            hist, _ = np.histogram(features, bins=10)
            hist_std = np.std(hist)
            hist_mean = np.mean(hist)
            
            if hist_std < 0.1 * hist_mean and hist_mean > 0:  # Too uniform
                events.append(SecurityEvent(
                    timestamp=time.time(),
                    threat_type=SecurityThreat.ADVERSARIAL_SPIKES,
                    severity='medium',
                    description="Suspiciously uniform feature distribution",
                    source_modality=modality_name,
                    detection_method="feature_distribution_check",
                    additional_data={'hist_uniformity': hist_std / hist_mean}
                ))
        
        return is_valid, events


class TemporalIntegrityChecker:
    """
    Ensures temporal integrity of spike data to prevent replay attacks
    and temporal manipulation.
    """
    
    def __init__(
        self,
        max_timestamp_drift: float = 1000.0,  # ms
        replay_detection_window: float = 5000.0,  # ms
    ):
        """
        Initialize temporal integrity checker.
        
        Args:
            max_timestamp_drift: Maximum allowed timestamp drift
            replay_detection_window: Window for replay detection
        """
        self.max_timestamp_drift = max_timestamp_drift
        self.replay_detection_window = replay_detection_window
        self.recent_signatures = {}  # Hash signatures of recent data
        self.logger = logging.getLogger(__name__)
    
    def check_temporal_ordering(
        self,
        modality_data: ModalityData,
        modality_name: str,
        current_time: float,
    ) -> Tuple[bool, List[SecurityEvent]]:
        """
        Check temporal ordering and detect timing attacks.
        
        Args:
            modality_data: Spike data to check
            modality_name: Name of the modality
            current_time: Current system time in ms
            
        Returns:
            (is_valid, security_events) tuple
        """
        events = []
        is_valid = True
        
        if len(modality_data.spike_times) == 0:
            return True, events
        
        # Check if spikes are properly ordered
        if not np.all(np.diff(modality_data.spike_times) >= 0):
            events.append(SecurityEvent(
                timestamp=time.time(),
                threat_type=SecurityThreat.TEMPORAL_MANIPULATION,
                severity='high',
                description="Spike times are not properly ordered",
                source_modality=modality_name,
                detection_method="temporal_ordering_check",
            ))
            is_valid = False
        
        # Check for future timestamps (clock skew attack)
        max_spike_time = np.max(modality_data.spike_times)
        if max_spike_time > current_time + self.max_timestamp_drift:
            events.append(SecurityEvent(
                timestamp=time.time(),
                threat_type=SecurityThreat.TEMPORAL_MANIPULATION,
                severity='high',
                description=f"Future timestamps detected (max: {max_spike_time}, current: {current_time})",
                source_modality=modality_name,
                detection_method="future_timestamp_check",
                additional_data={'max_spike_time': max_spike_time, 'current_time': current_time}
            ))
            is_valid = False
        
        # Check for overly old timestamps (replay attack)
        min_spike_time = np.min(modality_data.spike_times)
        if current_time - min_spike_time > self.replay_detection_window:
            events.append(SecurityEvent(
                timestamp=time.time(),
                threat_type=SecurityThreat.REPLAY_ATTACK,
                severity='medium',
                description=f"Old spike data detected (age: {current_time - min_spike_time:.2f}ms)",
                source_modality=modality_name,
                detection_method="replay_detection",
                additional_data={'data_age_ms': current_time - min_spike_time}
            ))
        
        return is_valid, events
    
    def check_replay_attack(
        self,
        modality_data: ModalityData,
        modality_name: str,
    ) -> Tuple[bool, List[SecurityEvent]]:
        """Check for replay attacks using data signatures."""
        events = []
        is_valid = True
        
        # Create signature of the spike data
        data_signature = self._create_data_signature(modality_data)
        
        current_time = time.time()
        
        # Check against recent signatures
        if modality_name in self.recent_signatures:
            for timestamp, signature in list(self.recent_signatures[modality_name]):
                # Remove old signatures
                if current_time - timestamp > self.replay_detection_window / 1000.0:
                    self.recent_signatures[modality_name].remove((timestamp, signature))
                    continue
                
                # Check for exact matches (replay attack)
                if signature == data_signature:
                    events.append(SecurityEvent(
                        timestamp=current_time,
                        threat_type=SecurityThreat.REPLAY_ATTACK,
                        severity='high',
                        description="Exact data replay detected",
                        source_modality=modality_name,
                        detection_method="signature_replay_detection",
                        additional_data={'signature': signature[:16]}  # First 16 chars
                    ))
                    is_valid = False
        else:
            self.recent_signatures[modality_name] = []
        
        # Store current signature
        self.recent_signatures[modality_name].append((current_time, data_signature))
        
        # Limit signature history
        if len(self.recent_signatures[modality_name]) > 100:
            self.recent_signatures[modality_name] = self.recent_signatures[modality_name][-100:]
        
        return is_valid, events
    
    def _create_data_signature(self, modality_data: ModalityData) -> str:
        """Create hash signature of spike data."""
        # Create a hash of the essential data
        hasher = hashlib.sha256()
        
        # Include spike times (rounded to avoid floating point precision issues)
        spike_times_rounded = np.round(modality_data.spike_times, decimals=3)
        hasher.update(spike_times_rounded.tobytes())
        
        # Include neuron IDs
        hasher.update(modality_data.neuron_ids.tobytes())
        
        # Include features if available
        if modality_data.features is not None:
            features_rounded = np.round(modality_data.features, decimals=3)
            hasher.update(features_rounded.tobytes())
        
        return hasher.hexdigest()


class NeuromorphicSecurityManager:
    """
    Comprehensive security manager for neuromorphic systems.
    
    Integrates multiple security mechanisms:
    - Spike train validation
    - Temporal integrity checking
    - Access control and authentication
    - Security event monitoring and response
    """
    
    def __init__(
        self,
        security_config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
    ):
        """
        Initialize neuromorphic security manager.
        
        Args:
            security_config: Security configuration parameters
            enable_logging: Enable security event logging
        """
        self.config = security_config or self._default_security_config()
        self.enable_logging = enable_logging
        
        # Initialize components
        self.spike_validator = SpikeTrainValidator(
            baseline_stats=self.config.get('baseline_stats'),
            anomaly_threshold=self.config.get('anomaly_threshold', 3.0),
        )
        
        self.temporal_checker = TemporalIntegrityChecker(
            max_timestamp_drift=self.config.get('max_timestamp_drift', 1000.0),
            replay_detection_window=self.config.get('replay_detection_window', 5000.0),
        )
        
        # Security event tracking
        self.security_events: List[SecurityEvent] = []
        self.threat_counts = {threat.value: 0 for threat in SecurityThreat}
        self.blocked_requests = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if enable_logging:
            self._setup_security_logging()
        
        self.logger.info("NeuromorphicSecurityManager initialized")
    
    def _default_security_config(self) -> Dict[str, Any]:
        """Default security configuration."""
        return {
            'anomaly_threshold': 3.0,
            'max_timestamp_drift': 1000.0,
            'replay_detection_window': 5000.0,
            'max_security_events': 1000,
            'alert_threshold': 10,  # Alert after 10 security events
            'block_threshold': 50,  # Block after 50 security events
        }
    
    def _setup_security_logging(self) -> None:
        """Setup security-specific logging."""
        security_handler = logging.FileHandler('neuromorphic_security.log')
        security_formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        security_handler.setFormatter(security_formatter)
        self.logger.addHandler(security_handler)
    
    @robust_function(critical_path=True)
    def validate_modality_data(
        self,
        modality_data: Dict[str, ModalityData],
        current_time: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive validation of multi-modal spike data.
        
        Args:
            modality_data: Dictionary of modality spike data
            current_time: Current system time (ms)
            
        Returns:
            (is_valid, validation_report) tuple
        """
        if current_time is None:
            current_time = time.time() * 1000.0  # Convert to ms
        
        validation_report = {
            'timestamp': time.time(),
            'overall_valid': True,
            'modality_results': {},
            'security_events': [],
            'threat_summary': {},
        }
        
        all_events = []
        
        # Validate each modality
        for modality_name, data in modality_data.items():
            modality_valid = True
            modality_events = []
            
            # Spike statistics validation
            stats_valid, stats_events = self.spike_validator.validate_spike_statistics(data, modality_name)
            modality_valid = modality_valid and stats_valid
            modality_events.extend(stats_events)
            
            # Neuron distribution validation
            neuron_valid, neuron_events = self.spike_validator.validate_neuron_distribution(data, modality_name)
            modality_valid = modality_valid and neuron_valid
            modality_events.extend(neuron_events)
            
            # Feature validation
            feature_valid, feature_events = self.spike_validator.validate_features(data, modality_name)
            modality_valid = modality_valid and feature_valid
            modality_events.extend(feature_events)
            
            # Temporal integrity checks
            temporal_valid, temporal_events = self.temporal_checker.check_temporal_ordering(
                data, modality_name, current_time
            )
            modality_valid = modality_valid and temporal_valid
            modality_events.extend(temporal_events)
            
            # Replay attack detection
            replay_valid, replay_events = self.temporal_checker.check_replay_attack(data, modality_name)
            modality_valid = modality_valid and replay_valid
            modality_events.extend(replay_events)
            
            validation_report['modality_results'][modality_name] = {
                'valid': modality_valid,
                'events': [event.__dict__ for event in modality_events],
                'threat_counts': {},
            }
            
            # Update overall validity
            if not modality_valid:
                validation_report['overall_valid'] = False
            
            all_events.extend(modality_events)
        
        # Process security events
        for event in all_events:
            self.security_events.append(event)
            self.threat_counts[event.threat_type.value] += 1
            
            # Log security events
            if self.enable_logging:
                self.logger.warning(
                    f"Security event: {event.threat_type.value} in {event.source_modality} - {event.description}"
                )
        
        # Limit security event history
        max_events = self.config.get('max_security_events', 1000)
        if len(self.security_events) > max_events:
            self.security_events = self.security_events[-max_events:]
        
        validation_report['security_events'] = [event.__dict__ for event in all_events]
        validation_report['threat_summary'] = self.threat_counts.copy()
        
        # Check if we should block due to too many threats
        total_recent_threats = len([
            event for event in self.security_events
            if time.time() - event.timestamp < 300  # Last 5 minutes
        ])
        
        block_threshold = self.config.get('block_threshold', 50)
        if total_recent_threats > block_threshold:
            validation_report['overall_valid'] = False
            validation_report['blocked'] = True
            self.blocked_requests += 1
            
            self.logger.critical(f"Blocking request due to excessive security threats: {total_recent_threats}")
        
        return validation_report['overall_valid'], validation_report
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        recent_events = [
            event for event in self.security_events
            if time.time() - event.timestamp < 3600  # Last hour
        ]
        
        status = {
            'timestamp': time.time(),
            'total_events': len(self.security_events),
            'recent_events': len(recent_events),
            'blocked_requests': self.blocked_requests,
            'threat_counts': self.threat_counts.copy(),
            'recent_threat_counts': {},
            'security_level': self._assess_security_level(),
            'recommendations': self._generate_security_recommendations(),
        }
        
        # Recent threat counts
        for event in recent_events:
            threat_type = event.threat_type.value
            if threat_type not in status['recent_threat_counts']:
                status['recent_threat_counts'][threat_type] = 0
            status['recent_threat_counts'][threat_type] += 1
        
        return status
    
    def _assess_security_level(self) -> str:
        """Assess current security level."""
        recent_events = [
            event for event in self.security_events
            if time.time() - event.timestamp < 1800  # Last 30 minutes
        ]
        
        if len(recent_events) == 0:
            return 'green'
        elif len(recent_events) < 5:
            return 'yellow'
        elif len(recent_events) < 20:
            return 'orange'
        else:
            return 'red'
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current threats."""
        recommendations = []
        
        recent_events = [
            event for event in self.security_events
            if time.time() - event.timestamp < 3600  # Last hour
        ]
        
        # Count recent threat types
        recent_threats = {}
        for event in recent_events:
            threat_type = event.threat_type.value
            recent_threats[threat_type] = recent_threats.get(threat_type, 0) + 1
        
        # Generate specific recommendations
        if recent_threats.get('spike_injection', 0) > 5:
            recommendations.append("Consider implementing stricter spike rate validation")
        
        if recent_threats.get('replay_attack', 0) > 3:
            recommendations.append("Reduce replay detection window for enhanced security")
        
        if recent_threats.get('temporal_manipulation', 0) > 3:
            recommendations.append("Review temporal synchronization mechanisms")
        
        if recent_threats.get('adversarial_spikes', 0) > 5:
            recommendations.append("Implement advanced adversarial spike detection")
        
        if len(recent_events) > 20:
            recommendations.append("Consider implementing rate limiting")
        
        if not recommendations:
            recommendations.append("Security status is normal")
        
        return recommendations
    
    def export_security_report(self, filepath: str) -> None:
        """Export comprehensive security report."""
        report = {
            'timestamp': time.time(),
            'security_status': self.get_security_status(),
            'configuration': self.config,
            'recent_events': [
                event.__dict__ for event in self.security_events[-100:]  # Last 100 events
            ],
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Security report exported to {filepath}")


# Decorator for securing neuromorphic functions
def neuromorphic_secure(
    security_manager: Optional[NeuromorphicSecurityManager] = None,
    block_on_failure: bool = True,
) -> Callable:
    """
    Decorator for securing neuromorphic functions with automatic validation.
    
    Args:
        security_manager: Custom security manager (optional)
        block_on_failure: Block execution if validation fails
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use provided security manager or create default
            sec_manager = security_manager or NeuromorphicSecurityManager()
            
            # Try to extract modality data from arguments
            modality_data = None
            
            if args and isinstance(args[0], dict):
                # Assume first argument is modality data
                potential_data = args[0]
                if all(isinstance(v, ModalityData) for v in potential_data.values()):
                    modality_data = potential_data
            
            # Validate if we have modality data
            if modality_data:
                is_valid, validation_report = sec_manager.validate_modality_data(modality_data)
                
                if not is_valid and block_on_failure:
                    raise SecurityError(
                        f"Security validation failed for {func.__name__}: {validation_report['threat_summary']}"
                    )
                
                # Add validation report to kwargs for function access
                kwargs['_security_validation'] = validation_report
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class SecurityError(Exception):
    """Exception raised for security violations."""
    pass


# Export key components
__all__ = [
    'SecurityThreat',
    'SecurityEvent', 
    'SpikeTrainValidator',
    'TemporalIntegrityChecker',
    'NeuromorphicSecurityManager',
    'neuromorphic_secure',
    'SecurityError',
]