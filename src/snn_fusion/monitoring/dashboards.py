"""
Monitoring Dashboards and Real-time Visualization

Implements web-based dashboards and real-time monitoring interfaces
for neuromorphic computing experiments and system metrics.
"""

import json
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import asdict
import logging
from collections import defaultdict, deque

try:
    from flask import Flask, render_template, jsonify, request, Response
    from flask_socketio import SocketIO, emit, join_room, leave_room
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from .metrics import MetricsCollector, get_global_collector
from .monitoring import SystemMonitor, ExperimentMonitor


class MetricsDashboard:
    """
    Web-based metrics dashboard for neuromorphic computing monitoring.
    
    Provides interactive visualization of system metrics, experiment progress,
    and neuromorphic-specific performance indicators.
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        system_monitor: Optional[SystemMonitor] = None,
        experiment_monitor: Optional[ExperimentMonitor] = None,
        port: int = 8080,
        debug: bool = False,
    ):
        """
        Initialize metrics dashboard.
        
        Args:
            metrics_collector: Metrics collector instance
            system_monitor: System monitor instance  
            experiment_monitor: Experiment monitor instance
            port: Dashboard port
            debug: Debug mode
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask and Flask-SocketIO required for dashboard")
        
        self.metrics_collector = metrics_collector or get_global_collector()
        self.system_monitor = system_monitor
        self.experiment_monitor = experiment_monitor
        self.port = port
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Flask app setup
        self.app = Flask(__name__, template_folder='templates')
        self.app.config['SECRET_KEY'] = 'neuromorphic-dashboard-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Dashboard state
        self.connected_clients: List[str] = []
        self.update_thread: Optional[threading.Thread] = None
        self.dashboard_active = False
        
        # Data caching for performance
        self.cache_timeout = 5.0  # seconds
        self.cached_data: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        
        self._setup_routes()
        self._setup_socketio_events()
        
        self.logger.info(f"Initialized metrics dashboard on port {port}")
    
    def start_dashboard(self, host: str = '0.0.0.0') -> None:
        """
        Start the dashboard server.
        
        Args:
            host: Host address to bind to
        """
        try:
            self.dashboard_active = True
            
            # Start background update thread
            self.update_thread = threading.Thread(
                target=self._update_loop,
                daemon=True
            )
            self.update_thread.start()
            
            self.logger.info(f"Starting dashboard server on {host}:{self.port}")
            self.socketio.run(
                self.app,
                host=host,
                port=self.port,
                debug=self.debug,
                use_reloader=False
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard: {e}")
            raise
    
    def stop_dashboard(self) -> None:
        """Stop the dashboard server."""
        self.dashboard_active = False
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)
        
        self.logger.info("Stopped dashboard server")
    
    def _setup_routes(self) -> None:
        """Setup Flask routes for dashboard."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/metrics/summary')
        def metrics_summary():
            """Get metrics summary."""
            try:
                summary = self._get_cached_data('metrics_summary', self._collect_metrics_summary)
                return jsonify(summary)
            except Exception as e:
                self.logger.error(f"Failed to get metrics summary: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/system/health')
        def system_health():
            """Get system health status."""
            try:
                if self.system_monitor:
                    health = self.system_monitor.get_system_health()
                    return jsonify(health)
                else:
                    return jsonify({'status': 'unknown', 'message': 'System monitor not available'})
            except Exception as e:
                self.logger.error(f"Failed to get system health: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/experiments/active')
        def active_experiments():
            """Get active experiments."""
            try:
                if self.experiment_monitor:
                    experiments = self.experiment_monitor.get_active_experiments()
                    # Convert to JSON-serializable format
                    serializable_experiments = {}
                    for exp_id, status in experiments.items():
                        serializable_experiments[exp_id] = asdict(status)
                    return jsonify(serializable_experiments)
                else:
                    return jsonify({})
            except Exception as e:
                self.logger.error(f"Failed to get active experiments: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/metrics/history/<metric_name>')
        def metric_history(metric_name):
            """Get metric history."""
            try:
                # Get time range from query parameters
                hours = request.args.get('hours', 1, type=int)
                start_time = time.time() - (hours * 3600)
                
                history = self.metrics_collector.get_metric_history(
                    metric_name, start_time=start_time
                )
                
                # Convert to chart-friendly format
                data = [
                    {
                        'timestamp': point.timestamp * 1000,  # JavaScript timestamp
                        'value': point.value,
                        'tags': point.tags
                    }
                    for point in history
                ]
                
                return jsonify(data)
                
            except Exception as e:
                self.logger.error(f"Failed to get metric history: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/hardware/utilization')
        def hardware_utilization():
            """Get hardware utilization metrics."""
            try:
                utilization = self._get_cached_data('hardware_utilization', self._collect_hardware_utilization)
                return jsonify(utilization)
            except Exception as e:
                self.logger.error(f"Failed to get hardware utilization: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_socketio_events(self) -> None:
        """Setup SocketIO events for real-time updates."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            client_id = request.sid
            self.connected_clients.append(client_id)
            self.logger.info(f"Client connected: {client_id}")
            
            # Send initial data
            emit('initial_data', self._get_initial_dashboard_data())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            client_id = request.sid
            if client_id in self.connected_clients:
                self.connected_clients.remove(client_id)
            self.logger.info(f"Client disconnected: {client_id}")
        
        @self.socketio.on('subscribe_metric')
        def handle_subscribe_metric(data):
            """Handle metric subscription."""
            metric_name = data.get('metric_name')
            if metric_name:
                join_room(f"metric_{metric_name}")
                self.logger.info(f"Client subscribed to metric: {metric_name}")
        
        @self.socketio.on('unsubscribe_metric')
        def handle_unsubscribe_metric(data):
            """Handle metric unsubscription."""
            metric_name = data.get('metric_name')
            if metric_name:
                leave_room(f"metric_{metric_name}")
                self.logger.info(f"Client unsubscribed from metric: {metric_name}")
    
    def _update_loop(self) -> None:
        """Background thread for pushing real-time updates."""
        self.logger.info("Started dashboard update loop")
        
        while self.dashboard_active:
            try:
                if self.connected_clients:
                    # Emit real-time updates
                    self._emit_system_updates()
                    self._emit_experiment_updates()
                    self._emit_metric_updates()
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                time.sleep(5.0)
        
        self.logger.info("Stopped dashboard update loop")
    
    def _emit_system_updates(self) -> None:
        """Emit system health updates."""
        try:
            if self.system_monitor:
                health = self.system_monitor.get_system_health()
                self.socketio.emit('system_health_update', health)
        except Exception as e:
            self.logger.error(f"Failed to emit system updates: {e}")
    
    def _emit_experiment_updates(self) -> None:
        """Emit experiment status updates."""
        try:
            if self.experiment_monitor:
                experiments = self.experiment_monitor.get_active_experiments()
                serializable_experiments = {}
                for exp_id, status in experiments.items():
                    serializable_experiments[exp_id] = asdict(status)
                
                self.socketio.emit('experiments_update', serializable_experiments)
        except Exception as e:
            self.logger.error(f"Failed to emit experiment updates: {e}")
    
    def _emit_metric_updates(self) -> None:
        """Emit latest metric values."""
        try:
            # Get latest values for key metrics
            key_metrics = [
                'system.cpu_percent',
                'system.memory_percent',
                'system.gpu_memory_percent',
                'neuromorphic.spike_rate',
                'neuromorphic.accuracy',
            ]
            
            updates = {}
            for metric_name in key_metrics:
                value = self.metrics_collector.get_latest_value(metric_name)
                if value is not None:
                    updates[metric_name] = {
                        'value': value,
                        'timestamp': time.time() * 1000
                    }
            
            if updates:
                self.socketio.emit('metrics_update', updates)
                
        except Exception as e:
            self.logger.error(f"Failed to emit metric updates: {e}")
    
    def _get_initial_dashboard_data(self) -> Dict[str, Any]:
        """Get initial data for dashboard."""
        try:
            data = {
                'metrics_summary': self._collect_metrics_summary(),
                'hardware_utilization': self._collect_hardware_utilization(),
                'timestamp': time.time() * 1000
            }
            
            if self.system_monitor:
                data['system_health'] = self.system_monitor.get_system_health()
            
            if self.experiment_monitor:
                experiments = self.experiment_monitor.get_active_experiments()
                data['active_experiments'] = {
                    exp_id: asdict(status) for exp_id, status in experiments.items()
                }
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to get initial dashboard data: {e}")
            return {'error': str(e)}
    
    def _get_cached_data(self, cache_key: str, data_function: Callable) -> Any:
        """Get cached data or refresh if expired."""
        current_time = time.time()
        
        if (cache_key in self.cache_timestamps and 
            current_time - self.cache_timestamps[cache_key] < self.cache_timeout and
            cache_key in self.cached_data):
            return self.cached_data[cache_key]
        
        # Refresh cache
        try:
            data = data_function()
            self.cached_data[cache_key] = data
            self.cache_timestamps[cache_key] = current_time
            return data
        except Exception as e:
            self.logger.error(f"Failed to refresh cache for {cache_key}: {e}")
            return self.cached_data.get(cache_key, {})
    
    def _collect_metrics_summary(self) -> Dict[str, Any]:
        """Collect metrics summary."""
        try:
            summary = self.metrics_collector.get_all_metrics_summary()
            
            # Categorize metrics
            categorized = {
                'system': {},
                'neuromorphic': {},
                'experiment': {},
                'performance': {}
            }
            
            for metric_name, metric_summary in summary.items():
                if metric_name.startswith('system.'):
                    categorized['system'][metric_name] = metric_summary
                elif metric_name.startswith('neuromorphic.'):
                    categorized['neuromorphic'][metric_name] = metric_summary
                elif metric_name.startswith('experiment.'):
                    categorized['experiment'][metric_name] = metric_summary
                elif metric_name.startswith('performance.'):
                    categorized['performance'][metric_name] = metric_summary
            
            return categorized
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics summary: {e}")
            return {}
    
    def _collect_hardware_utilization(self) -> Dict[str, Any]:
        """Collect hardware utilization data."""
        try:
            # Get system metrics
            cpu_util = self.metrics_collector.get_latest_value('system.cpu_percent') or 0
            memory_util = self.metrics_collector.get_latest_value('system.memory_percent') or 0
            gpu_util = self.metrics_collector.get_latest_value('system.gpu_memory_percent') or 0
            disk_util = self.metrics_collector.get_latest_value('system.disk_usage_percent') or 0
            
            return {
                'cpu': cpu_util,
                'memory': memory_util,
                'gpu': gpu_util,
                'disk': disk_util,
                'timestamp': time.time() * 1000
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect hardware utilization: {e}")
            return {}


class RealtimeMonitor:
    """
    Real-time monitoring interface for neuromorphic experiments.
    
    Provides streaming data and live updates for active experiments
    and system monitoring without full dashboard overhead.
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        update_interval: float = 1.0,
    ):
        """
        Initialize real-time monitor.
        
        Args:
            metrics_collector: Metrics collector instance
            update_interval: Update interval in seconds
        """
        self.metrics_collector = metrics_collector or get_global_collector()
        self.update_interval = update_interval
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Subscribers
        self.metric_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.system_subscribers: List[Callable] = []
        self.experiment_subscribers: List[Callable] = []
        
        # Data buffers
        self.metric_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        self.logger.info("Initialized real-time monitor")
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.active:
            self.logger.warning("Real-time monitoring already active")
            return
        
        self.active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"Started real-time monitoring with {self.update_interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if not self.active:
            return
        
        self.active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Stopped real-time monitoring")
    
    def subscribe_metric(self, metric_name: str, callback: Callable[[str, float, float], None]) -> None:
        """
        Subscribe to metric updates.
        
        Args:
            metric_name: Metric to subscribe to
            callback: Function called with (metric_name, value, timestamp)
        """
        self.metric_subscribers[metric_name].append(callback)
        self.logger.info(f"Added subscriber for metric: {metric_name}")
    
    def unsubscribe_metric(self, metric_name: str, callback: Callable) -> None:
        """Unsubscribe from metric updates."""
        if callback in self.metric_subscribers[metric_name]:
            self.metric_subscribers[metric_name].remove(callback)
            self.logger.info(f"Removed subscriber for metric: {metric_name}")
    
    def subscribe_system_updates(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to system health updates.
        
        Args:
            callback: Function called with system health data
        """
        self.system_subscribers.append(callback)
        self.logger.info("Added system health subscriber")
    
    def subscribe_experiment_updates(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to experiment updates.
        
        Args:
            callback: Function called with experiment data
        """
        self.experiment_subscribers.append(callback)
        self.logger.info("Added experiment subscriber")
    
    def get_metric_buffer(self, metric_name: str, max_points: int = 100) -> List[Dict[str, Any]]:
        """
        Get buffered metric data.
        
        Args:
            metric_name: Metric name
            max_points: Maximum points to return
            
        Returns:
            List of metric points
        """
        buffer = self.metric_buffers[metric_name]
        points = list(buffer)[-max_points:]
        
        return [
            {
                'timestamp': point['timestamp'],
                'value': point['value']
            }
            for point in points
        ]
    
    def get_live_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of live metrics."""
        try:
            summary = {}
            current_time = time.time()
            
            # Get latest values for key metrics
            key_metrics = [
                'system.cpu_percent',
                'system.memory_percent', 
                'system.gpu_memory_percent',
                'neuromorphic.spike_rate',
                'neuromorphic.accuracy',
                'neuromorphic.latency_ms',
            ]
            
            for metric_name in key_metrics:
                value = self.metrics_collector.get_latest_value(metric_name)
                if value is not None:
                    summary[metric_name] = {
                        'value': value,
                        'timestamp': current_time
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get live metrics summary: {e}")
            return {}
    
    def _monitoring_loop(self) -> None:
        """Main real-time monitoring loop."""
        self.logger.info("Started real-time monitoring loop")
        
        while self.active:
            try:
                current_time = time.time()
                
                # Update metric buffers and notify subscribers
                self._update_metric_subscribers(current_time)
                
                # Update system subscribers
                self._update_system_subscribers()
                
                # Update experiment subscribers
                self._update_experiment_subscribers()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in real-time monitoring loop: {e}")
                time.sleep(self.update_interval)
        
        self.logger.info("Stopped real-time monitoring loop")
    
    def _update_metric_subscribers(self, current_time: float) -> None:
        """Update metric subscribers with latest values."""
        try:
            for metric_name, subscribers in self.metric_subscribers.items():
                if not subscribers:
                    continue
                
                value = self.metrics_collector.get_latest_value(metric_name)
                if value is not None:
                    # Add to buffer
                    self.metric_buffers[metric_name].append({
                        'timestamp': current_time,
                        'value': value
                    })
                    
                    # Notify subscribers
                    for subscriber in subscribers:
                        try:
                            subscriber(metric_name, value, current_time)
                        except Exception as e:
                            self.logger.error(f"Metric subscriber error: {e}")
                            
        except Exception as e:
            self.logger.error(f"Failed to update metric subscribers: {e}")
    
    def _update_system_subscribers(self) -> None:
        """Update system health subscribers."""
        try:
            if not self.system_subscribers:
                return
            
            # Get basic system info from metrics
            system_data = {
                'cpu_percent': self.metrics_collector.get_latest_value('system.cpu_percent'),
                'memory_percent': self.metrics_collector.get_latest_value('system.memory_percent'),
                'disk_usage_percent': self.metrics_collector.get_latest_value('system.disk_usage_percent'),
                'timestamp': time.time()
            }
            
            for subscriber in self.system_subscribers:
                try:
                    subscriber(system_data)
                except Exception as e:
                    self.logger.error(f"System subscriber error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to update system subscribers: {e}")
    
    def _update_experiment_subscribers(self) -> None:
        """Update experiment subscribers."""
        try:
            if not self.experiment_subscribers:
                return
            
            # Get experiment metrics
            experiment_data = {
                'timestamp': time.time(),
                'active_count': 0,  # Would need experiment monitor integration
                'metrics': {}
            }
            
            # Add any experiment-related metrics
            for metric_name in ['neuromorphic.accuracy', 'neuromorphic.loss']:
                value = self.metrics_collector.get_latest_value(metric_name)
                if value is not None:
                    experiment_data['metrics'][metric_name] = value
            
            for subscriber in self.experiment_subscribers:
                try:
                    subscriber(experiment_data)
                except Exception as e:
                    self.logger.error(f"Experiment subscriber error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to update experiment subscribers: {e}")