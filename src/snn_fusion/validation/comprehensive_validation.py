"""
Comprehensive Validation Framework

Advanced validation system for neuromorphic multi-modal fusion, including
data validation, model validation, result validation, and end-to-end
system validation with automated testing capabilities.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import json
from pathlib import Path
import unittest

from ..algorithms.fusion import ModalityData, FusionResult, CrossModalFusion
from ..algorithms.temporal_spike_attention import TemporalSpikeAttention
from ..utils.robust_error_handling import robust_function, SecurityError
from ..security.neuromorphic_security import NeuromorphicSecurityManager


class ValidationLevel(Enum):
    """Validation levels for different stages."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    COMPREHENSIVE = "comprehensive"
    PRODUCTION = "production"


class ValidationResult(Enum):
    """Validation result status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class ValidationCheck:
    """Individual validation check."""
    name: str
    description: str
    level: ValidationLevel
    result: ValidationResult
    message: str
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    timestamp: float
    validation_level: ValidationLevel
    overall_result: ValidationResult
    checks: List[ValidationCheck]
    summary: Dict[str, int]
    recommendations: List[str]
    execution_time: float


class DataValidator:
    """
    Validates multi-modal spike data for correctness and consistency.
    
    Checks:
    - Data format and structure
    - Temporal consistency
    - Statistical properties
    - Cross-modal relationships
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE):
        """
        Initialize data validator.
        
        Args:
            validation_level: Level of validation to perform
        """
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
    
    def validate_modality_data(
        self,
        modality_data: Dict[str, ModalityData],
        expected_modalities: Optional[List[str]] = None,
    ) -> List[ValidationCheck]:
        """
        Validate multi-modal spike data.
        
        Args:
            modality_data: Dictionary of modality data to validate
            expected_modalities: Expected modalities (optional)
            
        Returns:
            List of validation checks
        """
        checks = []
        
        # Basic structure validation
        checks.extend(self._validate_data_structure(modality_data, expected_modalities))
        
        # Individual modality validation
        for modality, data in modality_data.items():
            checks.extend(self._validate_single_modality(modality, data))
        
        # Cross-modal validation (if comprehensive)
        if self.validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.PRODUCTION]:
            checks.extend(self._validate_cross_modal_consistency(modality_data))
        
        return checks
    
    def _validate_data_structure(
        self,
        modality_data: Dict[str, ModalityData],
        expected_modalities: Optional[List[str]],
    ) -> List[ValidationCheck]:
        """Validate basic data structure."""
        checks = []
        start_time = time.time()
        
        # Check if data is provided
        if not modality_data:
            checks.append(ValidationCheck(
                name="data_presence",
                description="Check if modality data is provided",
                level=ValidationLevel.BASIC,
                result=ValidationResult.FAIL,
                message="No modality data provided",
                execution_time=time.time() - start_time,
            ))
            return checks
        
        checks.append(ValidationCheck(
            name="data_presence",
            description="Check if modality data is provided",
            level=ValidationLevel.BASIC,
            result=ValidationResult.PASS,
            message=f"Found {len(modality_data)} modalities",
            execution_time=time.time() - start_time,
        ))
        
        # Check expected modalities
        if expected_modalities:
            start_time = time.time()
            missing_modalities = set(expected_modalities) - set(modality_data.keys())
            extra_modalities = set(modality_data.keys()) - set(expected_modalities)
            
            if missing_modalities:
                checks.append(ValidationCheck(
                    name="expected_modalities",
                    description="Check if all expected modalities are present",
                    level=ValidationLevel.BASIC,
                    result=ValidationResult.FAIL,
                    message=f"Missing modalities: {missing_modalities}",
                    execution_time=time.time() - start_time,
                    metadata={"missing": list(missing_modalities)},
                ))
            elif extra_modalities:
                checks.append(ValidationCheck(
                    name="expected_modalities",
                    description="Check if all expected modalities are present",
                    level=ValidationLevel.BASIC,
                    result=ValidationResult.WARNING,
                    message=f"Unexpected modalities: {extra_modalities}",
                    execution_time=time.time() - start_time,
                    metadata={"extra": list(extra_modalities)},
                ))
            else:
                checks.append(ValidationCheck(
                    name="expected_modalities",
                    description="Check if all expected modalities are present",
                    level=ValidationLevel.BASIC,
                    result=ValidationResult.PASS,
                    message="All expected modalities present",
                    execution_time=time.time() - start_time,
                ))
        
        return checks
    
    def _validate_single_modality(
        self,
        modality: str,
        data: ModalityData,
    ) -> List[ValidationCheck]:
        """Validate single modality data."""
        checks = []
        
        # Check data types and formats
        checks.append(self._check_data_types(modality, data))
        
        # Check data ranges and validity
        checks.append(self._check_data_ranges(modality, data))
        
        # Check temporal consistency
        checks.append(self._check_temporal_consistency(modality, data))
        
        # Statistical validation (if intermediate or higher)
        if self.validation_level in [ValidationLevel.INTERMEDIATE, ValidationLevel.COMPREHENSIVE, ValidationLevel.PRODUCTION]:
            checks.append(self._check_statistical_properties(modality, data))
        
        # Advanced validation (if comprehensive or production)
        if self.validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.PRODUCTION]:
            checks.extend(self._check_advanced_properties(modality, data))
        
        return checks
    
    def _check_data_types(self, modality: str, data: ModalityData) -> ValidationCheck:
        """Check data types and formats."""
        start_time = time.time()
        
        issues = []
        
        # Check spike times
        if not isinstance(data.spike_times, np.ndarray):
            issues.append("spike_times must be numpy array")
        elif data.spike_times.dtype not in [np.float32, np.float64]:
            issues.append("spike_times must be float type")
        
        # Check neuron IDs
        if not isinstance(data.neuron_ids, np.ndarray):
            issues.append("neuron_ids must be numpy array")
        elif data.neuron_ids.dtype not in [np.int32, np.int64]:
            issues.append("neuron_ids must be integer type")
        
        # Check features (if present)
        if data.features is not None:
            if not isinstance(data.features, np.ndarray):
                issues.append("features must be numpy array")
            elif data.features.dtype not in [np.float32, np.float64]:
                issues.append("features must be float type")
        
        # Check array lengths
        if len(data.spike_times) != len(data.neuron_ids):
            issues.append("spike_times and neuron_ids must have same length")
        
        if data.features is not None and len(data.features) != len(data.spike_times):
            issues.append("features must have same length as spike_times")
        
        if issues:
            result = ValidationResult.FAIL
            message = f"Data type issues: {'; '.join(issues)}"
        else:
            result = ValidationResult.PASS
            message = "Data types are valid"
        
        return ValidationCheck(
            name=f"{modality}_data_types",
            description="Check data types and formats",
            level=ValidationLevel.BASIC,
            result=result,
            message=message,
            execution_time=time.time() - start_time,
            metadata={"issues": issues},
        )
    
    def _check_data_ranges(self, modality: str, data: ModalityData) -> ValidationCheck:
        """Check data ranges and validity."""
        start_time = time.time()
        
        issues = []
        
        if len(data.spike_times) > 0:
            # Check spike times for negative values
            if np.any(data.spike_times < 0):
                issues.append("negative spike times found")
            
            # Check for NaN or infinite values
            if np.any(~np.isfinite(data.spike_times)):
                issues.append("non-finite spike times found")
            
            # Check neuron IDs for negative values
            if np.any(data.neuron_ids < 0):
                issues.append("negative neuron IDs found")
            
            # Check features (if present)
            if data.features is not None:
                if np.any(~np.isfinite(data.features)):
                    issues.append("non-finite features found")
                if np.any(data.features < 0):
                    issues.append("negative features found (may be valid depending on encoding)")
        
        if issues:
            result = ValidationResult.FAIL
            message = f"Data range issues: {'; '.join(issues)}"
        else:
            result = ValidationResult.PASS
            message = "Data ranges are valid"
        
        return ValidationCheck(
            name=f"{modality}_data_ranges",
            description="Check data ranges and validity",
            level=ValidationLevel.BASIC,
            result=result,
            message=message,
            execution_time=time.time() - start_time,
            metadata={"issues": issues},
        )
    
    def _check_temporal_consistency(self, modality: str, data: ModalityData) -> ValidationCheck:
        """Check temporal consistency of spike data."""
        start_time = time.time()
        
        if len(data.spike_times) <= 1:
            return ValidationCheck(
                name=f"{modality}_temporal_consistency",
                description="Check temporal ordering of spikes",
                level=ValidationLevel.BASIC,
                result=ValidationResult.SKIP,
                message="Insufficient data for temporal check",
                execution_time=time.time() - start_time,
            )
        
        # Check if spike times are sorted
        is_sorted = np.all(np.diff(data.spike_times) >= 0)
        
        if not is_sorted:
            result = ValidationResult.FAIL
            message = "Spike times are not temporally ordered"
        else:
            result = ValidationResult.PASS
            message = "Spike times are properly ordered"
        
        # Additional temporal statistics
        time_span = np.max(data.spike_times) - np.min(data.spike_times)
        mean_isi = np.mean(np.diff(data.spike_times)) if len(data.spike_times) > 1 else 0
        
        return ValidationCheck(
            name=f"{modality}_temporal_consistency",
            description="Check temporal ordering of spikes",
            level=ValidationLevel.BASIC,
            result=result,
            message=message,
            execution_time=time.time() - start_time,
            metadata={
                "time_span": float(time_span),
                "mean_isi": float(mean_isi),
                "n_spikes": len(data.spike_times),
            },
        )
    
    def _check_statistical_properties(self, modality: str, data: ModalityData) -> ValidationCheck:
        """Check statistical properties of spike data."""
        start_time = time.time()
        
        if len(data.spike_times) < 10:
            return ValidationCheck(
                name=f"{modality}_statistical_properties",
                description="Check statistical properties",
                level=ValidationLevel.INTERMEDIATE,
                result=ValidationResult.SKIP,
                message="Insufficient data for statistical analysis",
                execution_time=time.time() - start_time,
            )
        
        warnings = []
        stats = {}
        
        # Spike rate analysis
        time_span = np.max(data.spike_times) - np.min(data.spike_times)
        if time_span > 0:
            spike_rate = len(data.spike_times) / (time_span / 1000.0)  # spikes/sec
            stats['spike_rate'] = spike_rate
            
            # Check for reasonable spike rates
            if spike_rate > 1000:  # Very high spike rate
                warnings.append(f"very high spike rate: {spike_rate:.1f} spikes/sec")
            elif spike_rate < 0.1:  # Very low spike rate
                warnings.append(f"very low spike rate: {spike_rate:.1f} spikes/sec")
        
        # Inter-spike interval analysis
        if len(data.spike_times) > 1:
            isi = np.diff(data.spike_times)
            stats['mean_isi'] = float(np.mean(isi))
            stats['std_isi'] = float(np.std(isi))
            stats['cv_isi'] = float(np.std(isi) / np.mean(isi)) if np.mean(isi) > 0 else 0
            
            # Check for suspicious regularity
            if stats['cv_isi'] < 0.05:  # Very regular
                warnings.append(f"suspiciously regular ISI (CV: {stats['cv_isi']:.3f})")
        
        # Neuron distribution analysis
        unique_neurons = len(np.unique(data.neuron_ids))
        stats['unique_neurons'] = unique_neurons
        stats['spikes_per_neuron'] = len(data.spike_times) / unique_neurons if unique_neurons > 0 else 0
        
        # Check for concentration in few neurons
        if unique_neurons > 0:
            neuron_counts = np.bincount(data.neuron_ids)
            max_neuron_activity = np.max(neuron_counts)
            concentration_ratio = max_neuron_activity / len(data.spike_times)
            stats['max_neuron_concentration'] = float(concentration_ratio)
            
            if concentration_ratio > 0.8:
                warnings.append(f"high neuron concentration: {concentration_ratio:.2%}")
        
        if warnings:
            result = ValidationResult.WARNING
            message = f"Statistical warnings: {'; '.join(warnings)}"
        else:
            result = ValidationResult.PASS
            message = "Statistical properties are normal"
        
        return ValidationCheck(
            name=f"{modality}_statistical_properties",
            description="Check statistical properties",
            level=ValidationLevel.INTERMEDIATE,
            result=result,
            message=message,
            execution_time=time.time() - start_time,
            metadata={"statistics": stats, "warnings": warnings},
        )
    
    def _check_advanced_properties(self, modality: str, data: ModalityData) -> List[ValidationCheck]:
        """Check advanced properties of spike data."""
        checks = []
        
        # Burst detection
        checks.append(self._check_burst_patterns(modality, data))
        
        # Temporal clustering
        checks.append(self._check_temporal_clustering(modality, data))
        
        # Feature distribution (if features available)
        if data.features is not None:
            checks.append(self._check_feature_distribution(modality, data))
        
        return checks
    
    def _check_burst_patterns(self, modality: str, data: ModalityData) -> ValidationCheck:
        """Check for burst patterns in spike data."""
        start_time = time.time()
        
        if len(data.spike_times) < 20:
            return ValidationCheck(
                name=f"{modality}_burst_patterns",
                description="Check for burst patterns",
                level=ValidationLevel.COMPREHENSIVE,
                result=ValidationResult.SKIP,
                message="Insufficient data for burst analysis",
                execution_time=time.time() - start_time,
            )
        
        # Simple burst detection based on ISI
        isi = np.diff(data.spike_times)
        burst_threshold = 5.0  # ms
        
        # Find potential bursts
        in_burst = isi < burst_threshold
        burst_starts = np.where(np.diff(np.concatenate(([False], in_burst))))[0]
        burst_ends = np.where(np.diff(np.concatenate((in_burst, [False]))))[0]
        
        n_bursts = len(burst_starts)
        burst_ratio = np.sum(in_burst) / len(isi) if len(isi) > 0 else 0
        
        warnings = []
        if burst_ratio > 0.8:
            warnings.append(f"excessive burst activity: {burst_ratio:.2%}")
        elif burst_ratio < 0.01 and n_bursts == 0:
            warnings.append("no burst activity detected (may be normal)")
        
        if warnings:
            result = ValidationResult.WARNING
            message = f"Burst analysis: {'; '.join(warnings)}"
        else:
            result = ValidationResult.PASS
            message = f"Normal burst patterns ({n_bursts} bursts, {burst_ratio:.2%} ratio)"
        
        return ValidationCheck(
            name=f"{modality}_burst_patterns",
            description="Check for burst patterns",
            level=ValidationLevel.COMPREHENSIVE,
            result=result,
            message=message,
            execution_time=time.time() - start_time,
            metadata={
                "n_bursts": n_bursts,
                "burst_ratio": float(burst_ratio),
                "burst_threshold_ms": burst_threshold,
            },
        )
    
    def _check_temporal_clustering(self, modality: str, data: ModalityData) -> ValidationCheck:
        """Check for temporal clustering anomalies."""
        start_time = time.time()
        
        if len(data.spike_times) < 50:
            return ValidationCheck(
                name=f"{modality}_temporal_clustering",
                description="Check temporal clustering",
                level=ValidationLevel.COMPREHENSIVE,
                result=ValidationResult.SKIP,
                message="Insufficient data for clustering analysis",
                execution_time=time.time() - start_time,
            )
        
        # Bin spikes into time windows
        time_window = 10.0  # 10ms windows
        time_range = np.max(data.spike_times) - np.min(data.spike_times)
        n_bins = max(10, int(time_range / time_window))
        
        spike_counts, _ = np.histogram(data.spike_times, bins=n_bins)
        
        # Analyze clustering
        mean_count = np.mean(spike_counts)
        std_count = np.std(spike_counts)
        max_count = np.max(spike_counts)
        
        clustering_ratio = (max_count - mean_count) / std_count if std_count > 0 else 0
        
        warnings = []
        if clustering_ratio > 5.0:
            warnings.append(f"high temporal clustering detected (ratio: {clustering_ratio:.1f})")
        elif std_count < mean_count * 0.1 and mean_count > 0:
            warnings.append("suspiciously uniform temporal distribution")
        
        if warnings:
            result = ValidationResult.WARNING
            message = f"Clustering analysis: {'; '.join(warnings)}"
        else:
            result = ValidationResult.PASS
            message = "Normal temporal clustering"
        
        return ValidationCheck(
            name=f"{modality}_temporal_clustering",
            description="Check temporal clustering",
            level=ValidationLevel.COMPREHENSIVE,
            result=result,
            message=message,
            execution_time=time.time() - start_time,
            metadata={
                "clustering_ratio": float(clustering_ratio),
                "mean_count": float(mean_count),
                "std_count": float(std_count),
                "max_count": int(max_count),
            },
        )
    
    def _check_feature_distribution(self, modality: str, data: ModalityData) -> ValidationCheck:
        """Check feature distribution if features are available."""
        start_time = time.time()
        
        if data.features is None or len(data.features) < 10:
            return ValidationCheck(
                name=f"{modality}_feature_distribution",
                description="Check feature distribution",
                level=ValidationLevel.COMPREHENSIVE,
                result=ValidationResult.SKIP,
                message="No features available or insufficient data",
                execution_time=time.time() - start_time,
            )
        
        features = data.features
        
        # Basic feature statistics
        mean_feature = np.mean(features)
        std_feature = np.std(features)
        min_feature = np.min(features)
        max_feature = np.max(features)
        
        warnings = []
        
        # Check for suspicious distributions
        if std_feature < mean_feature * 0.01 and mean_feature > 0:
            warnings.append("features have very low variance")
        
        if min_feature == max_feature:
            warnings.append("all features have identical values")
        
        # Check for outliers (simple method)
        if std_feature > 0:
            z_scores = np.abs((features - mean_feature) / std_feature)
            n_outliers = np.sum(z_scores > 4.0)  # More than 4 standard deviations
            outlier_ratio = n_outliers / len(features)
            
            if outlier_ratio > 0.05:  # More than 5% outliers
                warnings.append(f"high outlier ratio: {outlier_ratio:.2%}")
        
        if warnings:
            result = ValidationResult.WARNING
            message = f"Feature distribution warnings: {'; '.join(warnings)}"
        else:
            result = ValidationResult.PASS
            message = "Feature distribution is normal"
        
        return ValidationCheck(
            name=f"{modality}_feature_distribution",
            description="Check feature distribution",
            level=ValidationLevel.COMPREHENSIVE,
            result=result,
            message=message,
            execution_time=time.time() - start_time,
            metadata={
                "mean": float(mean_feature),
                "std": float(std_feature),
                "min": float(min_feature),
                "max": float(max_feature),
                "n_features": len(features),
            },
        )
    
    def _validate_cross_modal_consistency(
        self,
        modality_data: Dict[str, ModalityData],
    ) -> List[ValidationCheck]:
        """Validate consistency across modalities."""
        checks = []
        
        if len(modality_data) < 2:
            return checks  # Need at least 2 modalities
        
        # Check temporal alignment
        checks.append(self._check_temporal_alignment(modality_data))
        
        # Check for suspicious correlations
        checks.append(self._check_cross_modal_correlations(modality_data))
        
        return checks
    
    def _check_temporal_alignment(self, modality_data: Dict[str, ModalityData]) -> ValidationCheck:
        """Check temporal alignment across modalities."""
        start_time = time.time()
        
        modalities = list(modality_data.keys())
        time_ranges = {}
        
        # Calculate time ranges for each modality
        for modality, data in modality_data.items():
            if len(data.spike_times) > 0:
                time_ranges[modality] = {
                    'min': np.min(data.spike_times),
                    'max': np.max(data.spike_times),
                    'span': np.max(data.spike_times) - np.min(data.spike_times),
                }
        
        if len(time_ranges) < 2:
            return ValidationCheck(
                name="cross_modal_temporal_alignment",
                description="Check temporal alignment across modalities",
                level=ValidationLevel.COMPREHENSIVE,
                result=ValidationResult.SKIP,
                message="Insufficient modalities with data",
                execution_time=time.time() - start_time,
            )
        
        # Check for significant temporal misalignment
        all_mins = [tr['min'] for tr in time_ranges.values()]
        all_maxs = [tr['max'] for tr in time_ranges.values()]
        
        min_gap = max(all_mins) - min(all_mins)  # Gap between earliest and latest start
        max_gap = max(all_maxs) - min(all_maxs)  # Gap between earliest and latest end
        
        warnings = []
        if min_gap > 1000:  # More than 1 second offset
            warnings.append(f"large start time gap: {min_gap:.1f}ms")
        if max_gap > 1000:  # More than 1 second difference in end times
            warnings.append(f"large end time gap: {max_gap:.1f}ms")
        
        # Check for drastically different time spans
        spans = [tr['span'] for tr in time_ranges.values()]
        min_span, max_span = min(spans), max(spans)
        
        if max_span > 0 and (min_span / max_span) < 0.5:
            warnings.append(f"large span difference: {min_span:.1f}ms to {max_span:.1f}ms")
        
        if warnings:
            result = ValidationResult.WARNING
            message = f"Temporal alignment issues: {'; '.join(warnings)}"
        else:
            result = ValidationResult.PASS
            message = "Good temporal alignment across modalities"
        
        return ValidationCheck(
            name="cross_modal_temporal_alignment",
            description="Check temporal alignment across modalities",
            level=ValidationLevel.COMPREHENSIVE,
            result=result,
            message=message,
            execution_time=time.time() - start_time,
            metadata={
                "time_ranges": time_ranges,
                "min_gap": float(min_gap),
                "max_gap": float(max_gap),
            },
        )
    
    def _check_cross_modal_correlations(self, modality_data: Dict[str, ModalityData]) -> ValidationCheck:
        """Check for suspicious cross-modal correlations."""
        start_time = time.time()
        
        modalities = list(modality_data.keys())
        
        if len(modalities) < 2:
            return ValidationCheck(
                name="cross_modal_correlations",
                description="Check cross-modal correlations",
                level=ValidationLevel.COMPREHENSIVE,
                result=ValidationResult.SKIP,
                message="Need at least 2 modalities",
                execution_time=time.time() - start_time,
            )
        
        # Create binned spike rates for correlation analysis
        time_window = 20.0  # 20ms windows
        correlations = {}
        
        # Find common time range
        all_times = np.concatenate([data.spike_times for data in modality_data.values()])
        if len(all_times) == 0:
            return ValidationCheck(
                name="cross_modal_correlations",
                description="Check cross-modal correlations",
                level=ValidationLevel.COMPREHENSIVE,
                result=ValidationResult.SKIP,
                message="No spike data available",
                execution_time=time.time() - start_time,
            )
        
        min_time, max_time = np.min(all_times), np.max(all_times)
        n_bins = max(10, int((max_time - min_time) / time_window))
        
        # Create spike rate vectors
        spike_rates = {}
        for modality, data in modality_data.items():
            if len(data.spike_times) > 0:
                counts, _ = np.histogram(data.spike_times, bins=n_bins, range=(min_time, max_time))
                spike_rates[modality] = counts
        
        # Calculate pairwise correlations
        warnings = []
        high_correlations = []
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i+1:], i+1):
                if mod1 in spike_rates and mod2 in spike_rates:
                    rates1, rates2 = spike_rates[mod1], spike_rates[mod2]
                    
                    if np.std(rates1) > 0 and np.std(rates2) > 0:
                        correlation = np.corrcoef(rates1, rates2)[0, 1]
                        
                        if not np.isnan(correlation):
                            correlations[f"{mod1}_{mod2}"] = float(correlation)
                            
                            # Check for suspiciously high correlations
                            if abs(correlation) > 0.95:
                                high_correlations.append(f"{mod1}-{mod2}: {correlation:.3f}")
        
        if high_correlations:
            warnings.append(f"very high correlations: {', '.join(high_correlations)}")
        
        if warnings:
            result = ValidationResult.WARNING
            message = f"Correlation warnings: {'; '.join(warnings)}"
        else:
            result = ValidationResult.PASS
            message = "Normal cross-modal correlations"
        
        return ValidationCheck(
            name="cross_modal_correlations",
            description="Check cross-modal correlations",
            level=ValidationLevel.COMPREHENSIVE,
            result=result,
            message=message,
            execution_time=time.time() - start_time,
            metadata={
                "correlations": correlations,
                "n_bins": n_bins,
                "time_window_ms": time_window,
            },
        )


class ModelValidator:
    """
    Validates neuromorphic fusion models for correctness and performance.
    
    Checks:
    - Model architecture
    - Parameter validation
    - Input/output compatibility
    - Performance characteristics
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
    
    def validate_fusion_model(
        self,
        model: CrossModalFusion,
        test_data: Optional[Dict[str, ModalityData]] = None,
    ) -> List[ValidationCheck]:
        """
        Validate fusion model.
        
        Args:
            model: Fusion model to validate
            test_data: Optional test data for functionality testing
            
        Returns:
            List of validation checks
        """
        checks = []
        
        # Basic model validation
        checks.extend(self._validate_model_structure(model))
        
        # Functionality testing (if test data provided)
        if test_data is not None:
            checks.extend(self._validate_model_functionality(model, test_data))
        
        # Performance validation (if comprehensive)
        if self.validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.PRODUCTION]:
            checks.extend(self._validate_model_performance(model, test_data))
        
        return checks
    
    def _validate_model_structure(self, model: CrossModalFusion) -> List[ValidationCheck]:
        """Validate basic model structure."""
        checks = []
        start_time = time.time()
        
        # Check if model has required attributes
        required_attrs = ['modalities', 'fuse_modalities']
        missing_attrs = []
        
        for attr in required_attrs:
            if not hasattr(model, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            checks.append(ValidationCheck(
                name="model_structure",
                description="Check model structure and required attributes",
                level=ValidationLevel.BASIC,
                result=ValidationResult.FAIL,
                message=f"Missing required attributes: {missing_attrs}",
                execution_time=time.time() - start_time,
                metadata={"missing_attributes": missing_attrs},
            ))
        else:
            checks.append(ValidationCheck(
                name="model_structure",
                description="Check model structure and required attributes",
                level=ValidationLevel.BASIC,
                result=ValidationResult.PASS,
                message="Model structure is valid",
                execution_time=time.time() - start_time,
            ))
        
        # Check modalities configuration
        start_time = time.time()
        if hasattr(model, 'modalities'):
            if not model.modalities:
                result = ValidationResult.FAIL
                message = "No modalities configured"
            elif not all(isinstance(mod, str) for mod in model.modalities):
                result = ValidationResult.FAIL
                message = "Invalid modality types"
            else:
                result = ValidationResult.PASS
                message = f"Valid modalities: {model.modalities}"
        else:
            result = ValidationResult.SKIP
            message = "No modalities attribute found"
        
        checks.append(ValidationCheck(
            name="modalities_configuration",
            description="Check modalities configuration",
            level=ValidationLevel.BASIC,
            result=result,
            message=message,
            execution_time=time.time() - start_time,
        ))
        
        return checks
    
    def _validate_model_functionality(
        self,
        model: CrossModalFusion,
        test_data: Dict[str, ModalityData],
    ) -> List[ValidationCheck]:
        """Validate model functionality with test data."""
        checks = []
        
        # Test basic fusion operation
        start_time = time.time()
        try:
            result = model.fuse_modalities(test_data)
            
            # Check if result has expected structure
            if hasattr(result, 'fused_spikes'):
                checks.append(ValidationCheck(
                    name="basic_fusion_functionality",
                    description="Test basic fusion functionality",
                    level=ValidationLevel.INTERMEDIATE,
                    result=ValidationResult.PASS,
                    message="Basic fusion operation successful",
                    execution_time=time.time() - start_time,
                ))
            else:
                checks.append(ValidationCheck(
                    name="basic_fusion_functionality",
                    description="Test basic fusion functionality",
                    level=ValidationLevel.INTERMEDIATE,
                    result=ValidationResult.FAIL,
                    message="Fusion result missing expected structure",
                    execution_time=time.time() - start_time,
                ))
                
        except Exception as e:
            checks.append(ValidationCheck(
                name="basic_fusion_functionality",
                description="Test basic fusion functionality",
                level=ValidationLevel.INTERMEDIATE,
                result=ValidationResult.FAIL,
                message=f"Fusion operation failed: {str(e)}",
                execution_time=time.time() - start_time,
                metadata={"error": str(e)},
            ))
        
        return checks
    
    def _validate_model_performance(
        self,
        model: CrossModalFusion,
        test_data: Optional[Dict[str, ModalityData]],
    ) -> List[ValidationCheck]:
        """Validate model performance characteristics."""
        checks = []
        
        if test_data is None:
            return checks
        
        # Performance timing test
        start_time = time.time()
        latencies = []
        
        # Run multiple iterations for timing
        n_iterations = 10
        for _ in range(n_iterations):
            iter_start = time.time()
            try:
                model.fuse_modalities(test_data)
                latencies.append((time.time() - iter_start) * 1000)  # Convert to ms
            except Exception as e:
                latencies.append(float('inf'))  # Mark as failed
        
        valid_latencies = [l for l in latencies if np.isfinite(l)]
        
        if valid_latencies:
            mean_latency = np.mean(valid_latencies)
            p95_latency = np.percentile(valid_latencies, 95)
            
            # Check performance thresholds
            if mean_latency > 100:  # 100ms threshold
                result = ValidationResult.WARNING
                message = f"High mean latency: {mean_latency:.2f}ms"
            elif mean_latency > 500:  # 500ms critical threshold
                result = ValidationResult.FAIL
                message = f"Excessive mean latency: {mean_latency:.2f}ms"
            else:
                result = ValidationResult.PASS
                message = f"Good performance: {mean_latency:.2f}ms mean latency"
        else:
            result = ValidationResult.FAIL
            message = "All performance tests failed"
            mean_latency = float('inf')
            p95_latency = float('inf')
        
        checks.append(ValidationCheck(
            name="model_performance",
            description="Test model performance characteristics",
            level=ValidationLevel.COMPREHENSIVE,
            result=result,
            message=message,
            execution_time=time.time() - start_time,
            metadata={
                "mean_latency_ms": float(mean_latency),
                "p95_latency_ms": float(p95_latency),
                "n_iterations": n_iterations,
                "success_rate": len(valid_latencies) / n_iterations,
            },
        ))
        
        return checks


class SystemValidator:
    """
    End-to-end system validation including security and integration testing.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE):
        self.validation_level = validation_level
        self.data_validator = DataValidator(validation_level)
        self.model_validator = ModelValidator(validation_level)
        self.logger = logging.getLogger(__name__)
    
    def validate_complete_system(
        self,
        model: CrossModalFusion,
        test_data: Dict[str, ModalityData],
        security_manager: Optional[NeuromorphicSecurityManager] = None,
    ) -> ValidationReport:
        """
        Perform complete system validation.
        
        Args:
            model: Fusion model to validate
            test_data: Test data for validation
            security_manager: Optional security manager for security testing
            
        Returns:
            Comprehensive validation report
        """
        start_time = time.time()
        all_checks = []
        
        # Data validation
        data_checks = self.data_validator.validate_modality_data(test_data)
        all_checks.extend(data_checks)
        
        # Model validation
        model_checks = self.model_validator.validate_fusion_model(model, test_data)
        all_checks.extend(model_checks)
        
        # Security validation (if security manager provided)
        if security_manager is not None:
            security_checks = self._validate_security_integration(
                model, test_data, security_manager
            )
            all_checks.extend(security_checks)
        
        # Integration testing
        integration_checks = self._validate_integration(model, test_data)
        all_checks.extend(integration_checks)
        
        # Generate overall result and summary
        overall_result = self._determine_overall_result(all_checks)
        summary = self._generate_summary(all_checks)
        recommendations = self._generate_recommendations(all_checks)
        
        report = ValidationReport(
            timestamp=time.time(),
            validation_level=self.validation_level,
            overall_result=overall_result,
            checks=all_checks,
            summary=summary,
            recommendations=recommendations,
            execution_time=time.time() - start_time,
        )
        
        return report
    
    def _validate_security_integration(
        self,
        model: CrossModalFusion,
        test_data: Dict[str, ModalityData],
        security_manager: NeuromorphicSecurityManager,
    ) -> List[ValidationCheck]:
        """Validate security integration."""
        checks = []
        
        # Test security validation
        start_time = time.time()
        try:
            is_valid, validation_report = security_manager.validate_modality_data(test_data)
            
            if is_valid:
                result = ValidationResult.PASS
                message = "Security validation passed"
            else:
                # Check if this is expected (test data might have intentional issues)
                n_events = len(validation_report.get('security_events', []))
                if n_events > 0:
                    result = ValidationResult.WARNING
                    message = f"Security events detected: {n_events}"
                else:
                    result = ValidationResult.FAIL
                    message = "Security validation failed"
            
            checks.append(ValidationCheck(
                name="security_integration",
                description="Test security manager integration",
                level=ValidationLevel.PRODUCTION,
                result=result,
                message=message,
                execution_time=time.time() - start_time,
                metadata={"validation_report": validation_report},
            ))
            
        except Exception as e:
            checks.append(ValidationCheck(
                name="security_integration",
                description="Test security manager integration",
                level=ValidationLevel.PRODUCTION,
                result=ValidationResult.FAIL,
                message=f"Security integration failed: {str(e)}",
                execution_time=time.time() - start_time,
                metadata={"error": str(e)},
            ))
        
        return checks
    
    def _validate_integration(
        self,
        model: CrossModalFusion,
        test_data: Dict[str, ModalityData],
    ) -> List[ValidationCheck]:
        """Validate end-to-end integration."""
        checks = []
        
        # Test complete pipeline
        start_time = time.time()
        try:
            # Run fusion
            fusion_result = model.fuse_modalities(test_data)
            
            # Validate fusion result structure
            required_fields = ['fused_spikes', 'fusion_weights', 'confidence_scores']
            missing_fields = []
            
            for field in required_fields:
                if not hasattr(fusion_result, field):
                    missing_fields.append(field)
            
            if missing_fields:
                result = ValidationResult.FAIL
                message = f"Missing fusion result fields: {missing_fields}"
            else:
                # Check result validity
                if len(fusion_result.fused_spikes) == 0:
                    result = ValidationResult.WARNING
                    message = "No spikes in fusion result"
                else:
                    result = ValidationResult.PASS
                    message = f"Complete integration successful ({len(fusion_result.fused_spikes)} fused spikes)"
            
            checks.append(ValidationCheck(
                name="end_to_end_integration",
                description="Test complete integration pipeline",
                level=ValidationLevel.COMPREHENSIVE,
                result=result,
                message=message,
                execution_time=time.time() - start_time,
                metadata={
                    "n_fused_spikes": len(fusion_result.fused_spikes) if hasattr(fusion_result, 'fused_spikes') else 0,
                    "fusion_weights": getattr(fusion_result, 'fusion_weights', {}),
                },
            ))
            
        except Exception as e:
            checks.append(ValidationCheck(
                name="end_to_end_integration",
                description="Test complete integration pipeline",
                level=ValidationLevel.COMPREHENSIVE,
                result=ValidationResult.FAIL,
                message=f"Integration test failed: {str(e)}",
                execution_time=time.time() - start_time,
                metadata={"error": str(e)},
            ))
        
        return checks
    
    def _determine_overall_result(self, checks: List[ValidationCheck]) -> ValidationResult:
        """Determine overall validation result."""
        if not checks:
            return ValidationResult.SKIP
        
        results = [check.result for check in checks]
        
        if ValidationResult.FAIL in results:
            return ValidationResult.FAIL
        elif ValidationResult.WARNING in results:
            return ValidationResult.WARNING
        elif ValidationResult.PASS in results:
            return ValidationResult.PASS
        else:
            return ValidationResult.SKIP
    
    def _generate_summary(self, checks: List[ValidationCheck]) -> Dict[str, int]:
        """Generate summary statistics."""
        summary = {
            'total_checks': len(checks),
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'skipped': 0,
        }
        
        for check in checks:
            if check.result == ValidationResult.PASS:
                summary['passed'] += 1
            elif check.result == ValidationResult.FAIL:
                summary['failed'] += 1
            elif check.result == ValidationResult.WARNING:
                summary['warnings'] += 1
            else:
                summary['skipped'] += 1
        
        return summary
    
    def _generate_recommendations(self, checks: List[ValidationCheck]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Collect failed and warning checks
        issues = [check for check in checks if check.result in [ValidationResult.FAIL, ValidationResult.WARNING]]
        
        # Generate specific recommendations
        for check in issues:
            if 'data_types' in check.name:
                recommendations.append("Fix data type issues to ensure proper processing")
            elif 'temporal_consistency' in check.name:
                recommendations.append("Verify spike data temporal ordering")
            elif 'statistical_properties' in check.name:
                recommendations.append("Review data collection and preprocessing pipeline")
            elif 'performance' in check.name:
                recommendations.append("Optimize model performance for production deployment")
            elif 'security' in check.name:
                recommendations.append("Address security vulnerabilities before deployment")
            elif 'integration' in check.name:
                recommendations.append("Fix integration issues for reliable operation")
        
        # General recommendations
        if not recommendations:
            recommendations.append("All validations passed - system ready for deployment")
        
        return list(set(recommendations))  # Remove duplicates


# Automated testing framework
class NeuromorphicTestSuite(unittest.TestCase):
    """Automated test suite for neuromorphic fusion systems."""
    
    def setUp(self):
        """Set up test environment."""
        self.validator = SystemValidator(ValidationLevel.COMPREHENSIVE)
        self.test_data = self._create_test_data()
    
    def _create_test_data(self) -> Dict[str, ModalityData]:
        """Create synthetic test data."""
        test_data = {}
        
        modalities = ['audio', 'vision', 'tactile']
        
        for modality in modalities:
            # Create synthetic spike data
            n_spikes = np.random.poisson(50)
            spike_times = np.sort(np.random.uniform(0, 1000, n_spikes))  # 1 second window
            neuron_ids = np.random.randint(0, 32, n_spikes)
            features = np.random.gamma(2.0, 0.5, n_spikes)
            
            test_data[modality] = ModalityData(
                modality_name=modality,
                spike_times=spike_times,
                neuron_ids=neuron_ids,
                features=features,
            )
        
        return test_data
    
    def test_data_validation(self):
        """Test data validation functionality."""
        checks = self.validator.data_validator.validate_modality_data(self.test_data)
        
        # Should have at least basic checks
        self.assertGreater(len(checks), 0)
        
        # Should not have any critical failures with synthetic data
        critical_failures = [c for c in checks if c.result == ValidationResult.FAIL]
        self.assertEqual(len(critical_failures), 0, f"Critical validation failures: {critical_failures}")
    
    def test_temporal_spike_attention_validation(self):
        """Test TSA model validation."""
        from ..algorithms.temporal_spike_attention import create_temporal_spike_attention
        
        # Create TSA model
        modalities = list(self.test_data.keys())
        tsa_model = create_temporal_spike_attention(modalities)
        
        # Validate model
        checks = self.validator.model_validator.validate_fusion_model(tsa_model, self.test_data)
        
        # Should pass basic validation
        failed_checks = [c for c in checks if c.result == ValidationResult.FAIL]
        self.assertEqual(len(failed_checks), 0, f"Model validation failures: {failed_checks}")


# Export key components
__all__ = [
    'ValidationLevel',
    'ValidationResult',
    'ValidationCheck',
    'ValidationReport',
    'DataValidator',
    'ModelValidator',
    'SystemValidator',
    'NeuromorphicTestSuite',
]