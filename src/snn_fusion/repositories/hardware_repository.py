"""
Hardware Repository Implementation

Provides data access operations for neuromorphic hardware deployment
profiles and performance benchmarking.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass

from .base import BaseRepository


@dataclass
class HardwareProfile:
    """Hardware Profile model class."""
    id: Optional[int] = None
    name: str = ""
    hardware_type: str = ""
    model_id: Optional[int] = None
    deployment_config: Dict[str, Any] = None
    inference_latency_ms: Optional[float] = None
    power_consumption_mw: Optional[float] = None
    accuracy_deployed: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    optimization_applied: bool = False
    deployed_at: Optional[str] = None
    benchmark_results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.deployment_config is None:
            self.deployment_config = {}
        if self.benchmark_results is None:
            self.benchmark_results = {}


class HardwareRepository(BaseRepository[HardwareProfile]):
    """Repository for hardware profile data access operations."""
    
    def _get_table_name(self) -> str:
        return "hardware_profiles"
    
    def _to_dict(self, profile: HardwareProfile) -> Dict[str, Any]:
        """Convert hardware profile object to dictionary for database storage."""
        return {
            'id': profile.id,
            'name': profile.name,
            'hardware_type': profile.hardware_type,
            'model_id': profile.model_id,
            'deployment_config_json': self._serialize_json_field(profile.deployment_config),
            'inference_latency_ms': profile.inference_latency_ms,
            'power_consumption_mw': profile.power_consumption_mw,
            'accuracy_deployed': profile.accuracy_deployed,
            'memory_usage_mb': profile.memory_usage_mb,
            'throughput_samples_per_sec': profile.throughput_samples_per_sec,
            'optimization_applied': profile.optimization_applied,
            'deployed_at': profile.deployed_at,
            'benchmark_results_json': self._serialize_json_field(profile.benchmark_results)
        }
    
    def _from_dict(self, data: Dict[str, Any]) -> HardwareProfile:
        """Convert dictionary from database to hardware profile object."""
        return HardwareProfile(
            id=data.get('id'),
            name=data.get('name', ''),
            hardware_type=data.get('hardware_type', ''),
            model_id=data.get('model_id'),
            deployment_config=self._deserialize_json_field(data.get('deployment_config_json')) or {},
            inference_latency_ms=data.get('inference_latency_ms'),
            power_consumption_mw=data.get('power_consumption_mw'),
            accuracy_deployed=data.get('accuracy_deployed'),
            memory_usage_mb=data.get('memory_usage_mb'),
            throughput_samples_per_sec=data.get('throughput_samples_per_sec'),
            optimization_applied=bool(data.get('optimization_applied', False)),
            deployed_at=data.get('deployed_at'),
            benchmark_results=self._deserialize_json_field(data.get('benchmark_results_json')) or {}
        )