"""
Metrics Repository Implementation

Provides data access operations for performance metrics tracking
and spike data analysis in neuromorphic training.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass

from .base import BaseRepository


@dataclass
class PerformanceMetric:
    """Performance Metric model class."""
    id: Optional[int] = None
    training_run_id: Optional[int] = None
    hardware_profile_id: Optional[int] = None
    metric_name: str = ""
    metric_value: float = 0.0
    metric_unit: Optional[str] = None
    measurement_context: Optional[str] = None
    recorded_at: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MetricsRepository(BaseRepository[PerformanceMetric]):
    """Repository for performance metrics data access operations."""
    
    def _get_table_name(self) -> str:
        return "performance_metrics"
    
    def _to_dict(self, metric: PerformanceMetric) -> Dict[str, Any]:
        """Convert metric object to dictionary for database storage."""
        return {
            'id': metric.id,
            'training_run_id': metric.training_run_id,
            'hardware_profile_id': metric.hardware_profile_id,
            'metric_name': metric.metric_name,
            'metric_value': metric.metric_value,
            'metric_unit': metric.metric_unit,
            'measurement_context': metric.measurement_context,
            'recorded_at': metric.recorded_at,
            'metadata_json': self._serialize_json_field(metric.metadata)
        }
    
    def _from_dict(self, data: Dict[str, Any]) -> PerformanceMetric:
        """Convert dictionary from database to metric object."""
        return PerformanceMetric(
            id=data.get('id'),
            training_run_id=data.get('training_run_id'),
            hardware_profile_id=data.get('hardware_profile_id'),
            metric_name=data.get('metric_name', ''),
            metric_value=data.get('metric_value', 0.0),
            metric_unit=data.get('metric_unit'),
            measurement_context=data.get('measurement_context'),
            recorded_at=data.get('recorded_at'),
            metadata=self._deserialize_json_field(data.get('metadata_json')) or {}
        )