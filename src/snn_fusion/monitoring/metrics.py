"""
Metrics Collection System

Implements comprehensive metrics collection for neuromorphic computing
experiments, system performance, and hardware utilization tracking.
"""

import os
import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
from collections import defaultdict, deque
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class NeuromorphicMetrics:
    """Neuromorphic-specific metrics."""
    spike_rate: float = 0.0
    membrane_potential_mean: float = 0.0
    membrane_potential_std: float = 0.0
    synaptic_activity: float = 0.0
    adaptation_rate: float = 0.0
    reservoir_sparsity: float = 0.0
    temporal_correlation: float = 0.0
    plasticity_changes: float = 0.0
    energy_consumption: float = 0.0
    latency_ms: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0
    network_sent_mb_s: float = 0.0
    network_recv_mb_s: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_temperature: float = 0.0
    load_average_1m: float = 0.0
    load_average_5m: float = 0.0
    load_average_15m: float = 0.0


class MetricsCollector:
    """
    Central metrics collection and storage system.
    
    Collects, aggregates, and stores metrics from neuromorphic experiments,
    system performance monitoring, and hardware utilization tracking.
    """
    
    def __init__(
        self,
        max_points: int = 10000,
        aggregation_window: int = 60,
        storage_backend: Optional[str] = None,
    ):
        """
        Initialize metrics collector.
        
        Args:
            max_points: Maximum points to store per metric
            aggregation_window: Aggregation window in seconds
            storage_backend: Storage backend ('memory', 'disk', 'database')
        """
        self.max_points = max_points
        self.aggregation_window = aggregation_window
        self.storage_backend = storage_backend or 'memory'
        self.logger = logging.getLogger(__name__)
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Collection state
        self.collection_active = False
        self.collection_thread: Optional[threading.Thread] = None
        self.collection_interval = 1.0  # seconds
        
        # Callbacks for real-time processing
        self.callbacks: List[Callable[[str, MetricPoint], None]] = []
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self.last_disk_io = None
        self.last_network_io = None
        
        self.logger.info(f"Initialized metrics collector with {storage_backend} backend")
    
    def start_collection(self, interval: float = 1.0) -> None:
        """
        Start automatic metrics collection.
        
        Args:
            interval: Collection interval in seconds
        """
        if self.collection_active:
            self.logger.warning("Metrics collection already active")
            return
        
        self.collection_interval = interval
        self.collection_active = True
        
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()
        
        self.logger.info(f"Started metrics collection with {interval}s interval")
    
    def stop_collection(self) -> None:
        """Stop automatic metrics collection."""
        if not self.collection_active:
            return
        
        self.collection_active = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        self.logger.info("Stopped metrics collection")
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a single metric point.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for grouping/filtering
            metadata: Optional metadata
        """
        try:
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {},
                metadata=metadata
            )
            
            with self._lock:
                self.metrics[name].append(point)
                
                # Update metadata
                if metadata:
                    self.metric_metadata[name] = metadata
            
            # Call registered callbacks
            for callback in self.callbacks:
                try:
                    callback(name, point)
                except Exception as e:
                    self.logger.error(f"Metric callback error: {e}")
            
            # Auto-aggregate if window reached
            self._maybe_aggregate(name)
            
        except Exception as e:
            self.logger.error(f"Failed to record metric {name}: {e}")
    
    def record_neuromorphic_metrics(
        self,
        metrics: NeuromorphicMetrics,
        experiment_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        """
        Record neuromorphic-specific metrics.
        
        Args:
            metrics: Neuromorphic metrics data
            experiment_id: Optional experiment identifier
            model_id: Optional model identifier
        """
        tags = {}
        if experiment_id:
            tags['experiment_id'] = experiment_id
        if model_id:
            tags['model_id'] = model_id
        
        # Record each metric individually
        for field, value in asdict(metrics).items():
            if value is not None:
                self.record_metric(
                    name=f"neuromorphic.{field}",
                    value=float(value),
                    tags=tags,
                    metadata={'type': 'neuromorphic', 'category': field}
                )
    
    def record_system_metrics(self, metrics: SystemMetrics) -> None:
        """
        Record system performance metrics.
        
        Args:
            metrics: System metrics data
        """
        tags = {'type': 'system'}
        
        # Record each metric individually
        for field, value in asdict(metrics).items():
            if value is not None:
                self.record_metric(
                    name=f"system.{field}",
                    value=float(value),
                    tags=tags,
                    metadata={'type': 'system', 'category': field}
                )
    
    def get_metric_history(
        self,
        name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[MetricPoint]:
        """
        Get metric history within time range.
        
        Args:
            name: Metric name
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            
        Returns:
            List of metric points
        """
        with self._lock:
            points = list(self.metrics.get(name, []))
        
        # Filter by time range if specified
        if start_time or end_time:
            filtered_points = []
            for point in points:
                if start_time and point.timestamp < start_time:
                    continue
                if end_time and point.timestamp > end_time:
                    continue
                filtered_points.append(point)
            points = filtered_points
        
        return points
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """
        Get latest value for metric.
        
        Args:
            name: Metric name
            
        Returns:
            Latest metric value or None
        """
        with self._lock:
            points = self.metrics.get(name)
            if points:
                return points[-1].value
        return None
    
    def get_aggregated_metrics(
        self,
        name: str,
        aggregation: str = 'mean',
    ) -> Optional[float]:
        """
        Get aggregated metric value.
        
        Args:
            name: Metric name
            aggregation: Aggregation type (mean, min, max, sum, count)
            
        Returns:
            Aggregated value or None
        """
        with self._lock:
            return self.aggregated_metrics.get(name, {}).get(aggregation)
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """
        Get comprehensive metric summary.
        
        Args:
            name: Metric name
            
        Returns:
            Metric summary statistics
        """
        points = self.get_metric_history(name)
        if not points:
            return {}
        
        values = [p.value for p in points]
        
        return {
            'name': name,
            'count': len(values),
            'latest': values[-1] if values else None,
            'mean': np.mean(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'std': np.std(values),
            'first_timestamp': points[0].timestamp,
            'last_timestamp': points[-1].timestamp,
            'duration_seconds': points[-1].timestamp - points[0].timestamp,
            'metadata': self.metric_metadata.get(name, {}),
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary for all collected metrics."""
        with self._lock:
            metric_names = list(self.metrics.keys())
        
        return {
            name: self.get_metric_summary(name)
            for name in metric_names
        }
    
    def add_callback(self, callback: Callable[[str, MetricPoint], None]) -> None:
        """
        Add callback for real-time metric processing.
        
        Args:
            callback: Function to call with (metric_name, metric_point)
        """
        self.callbacks.append(callback)
        self.logger.info("Added metrics callback")
    
    def remove_callback(self, callback: Callable[[str, MetricPoint], None]) -> None:
        """Remove metrics callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            self.logger.info("Removed metrics callback")
    
    def export_metrics(
        self,
        format: str = 'json',
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        Export metrics data.
        
        Args:
            format: Export format ('json', 'csv')
            filename: Optional filename to save to
            
        Returns:
            Exported data as string or None if saved to file
        """
        try:
            if format == 'json':
                data = {}
                for name in self.metrics.keys():
                    points = self.get_metric_history(name)
                    data[name] = [
                        {
                            'timestamp': p.timestamp,
                            'value': p.value,
                            'tags': p.tags,
                            'metadata': p.metadata,
                        }
                        for p in points
                    ]
                
                json_data = json.dumps(data, indent=2)
                
                if filename:
                    with open(filename, 'w') as f:
                        f.write(json_data)
                    self.logger.info(f"Exported metrics to {filename}")
                    return None
                else:
                    return json_data
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return None
    
    def clear_metrics(self, metric_name: Optional[str] = None) -> None:
        """
        Clear stored metrics.
        
        Args:
            metric_name: Specific metric to clear, or None for all
        """
        with self._lock:
            if metric_name:
                self.metrics.pop(metric_name, None)
                self.aggregated_metrics.pop(metric_name, None)
                self.metric_metadata.pop(metric_name, None)
                self.logger.info(f"Cleared metric: {metric_name}")
            else:
                self.metrics.clear()
                self.aggregated_metrics.clear()
                self.metric_metadata.clear()
                self.logger.info("Cleared all metrics")
    
    def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        self.logger.info("Started metrics collection loop")
        
        while self.collection_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                if system_metrics:
                    self.record_system_metrics(system_metrics)
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                time.sleep(self.collection_interval)
        
        self.logger.info("Stopped metrics collection loop")
    
    def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect current system metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Disk usage and I/O
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Calculate disk I/O rates
            disk_read_rate = 0.0
            disk_write_rate = 0.0
            if self.last_disk_io and disk_io:
                time_delta = self.collection_interval
                read_delta = disk_io.read_bytes - self.last_disk_io.read_bytes
                write_delta = disk_io.write_bytes - self.last_disk_io.write_bytes
                disk_read_rate = (read_delta / time_delta) / (1024 * 1024)  # MB/s
                disk_write_rate = (write_delta / time_delta) / (1024 * 1024)  # MB/s
            self.last_disk_io = disk_io
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_sent_rate = 0.0
            network_recv_rate = 0.0
            if self.last_network_io and network_io:
                time_delta = self.collection_interval
                sent_delta = network_io.bytes_sent - self.last_network_io.bytes_sent
                recv_delta = network_io.bytes_recv - self.last_network_io.bytes_recv
                network_sent_rate = (sent_delta / time_delta) / (1024 * 1024)  # MB/s
                network_recv_rate = (recv_delta / time_delta) / (1024 * 1024)  # MB/s
            self.last_network_io = network_io
            
            # Load average
            load_avg = os.getloadavg()
            
            # GPU metrics (if available)
            gpu_util = 0.0
            gpu_memory = 0.0
            gpu_temp = 0.0
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    # PyTorch GPU metrics
                    gpu_memory = torch.cuda.memory_used() / torch.cuda.max_memory_reserved() * 100
                except:
                    pass
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk_usage.used / disk_usage.total * 100,
                disk_read_mb_s=disk_read_rate,
                disk_write_mb_s=disk_write_rate,
                network_sent_mb_s=network_sent_rate,
                network_recv_mb_s=network_recv_rate,
                gpu_utilization=gpu_util,
                gpu_memory_percent=gpu_memory,
                gpu_temperature=gpu_temp,
                load_average_1m=load_avg[0],
                load_average_5m=load_avg[1],
                load_average_15m=load_avg[2],
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return None
    
    def _maybe_aggregate(self, metric_name: str) -> None:
        """Aggregate metrics if aggregation window is reached."""
        try:
            points = list(self.metrics[metric_name])
            if len(points) < 2:
                return
            
            # Check if enough time has passed for aggregation
            latest_time = points[-1].timestamp
            window_start = latest_time - self.aggregation_window
            
            # Get points in current window
            window_points = [p for p in points if p.timestamp >= window_start]
            if len(window_points) < 2:
                return
            
            # Calculate aggregations
            values = [p.value for p in window_points]
            
            aggregations = {
                'mean': np.mean(values),
                'min': np.min(values),
                'max': np.max(values),
                'sum': np.sum(values),
                'count': len(values),
                'std': np.std(values),
                'last': values[-1],
            }
            
            with self._lock:
                self.aggregated_metrics[metric_name] = aggregations
                
        except Exception as e:
            self.logger.error(f"Failed to aggregate metric {metric_name}: {e}")


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_global_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def record_metric(name: str, value: float, **kwargs) -> None:
    """Convenience function to record metric using global collector."""
    get_global_collector().record_metric(name, value, **kwargs)


def start_system_monitoring(interval: float = 1.0) -> None:
    """Start system metrics monitoring using global collector."""
    get_global_collector().start_collection(interval)


def stop_system_monitoring() -> None:
    """Stop system metrics monitoring."""
    get_global_collector().stop_collection()