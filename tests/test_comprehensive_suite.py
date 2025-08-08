"""
Comprehensive Test Suite for SNN-Fusion Framework

Complete test coverage for all major components including unit tests,
integration tests, performance tests, and security tests.
"""

import pytest
import numpy as np
import torch
import tempfile
import time
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import sqlite3
from datetime import datetime, timedelta

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from snn_fusion.datasets.transforms import (
    TemporalJitter, ModalityDropout, SpikeMasking, SpikeNoise,
    create_training_transforms
)
from snn_fusion.utils.error_handling import (
    ErrorHandler, ErrorCategory, ErrorSeverity, SNNFusionError,
    handle_errors, ErrorContext
)
from snn_fusion.utils.security_enhanced import (
    InputSanitizer, SecurityLevel, ThreatType, SecureConfig
)
from snn_fusion.monitoring.comprehensive_monitoring import (
    MetricsCollector, HealthMonitor, AlertManager, AlertSeverity,
    ComprehensiveMonitor
)
from snn_fusion.optimization.performance import (
    PerformanceProfiler, MemoryOptimizer, ComputeOptimizer,
    CacheManager, PerformanceOptimizer, OptimizationLevel
)
from snn_fusion.scaling.concurrent_processing import (
    ConcurrentProcessor, ProcessingConfig, ProcessingMode,
    ProcessingTask, TaskPriority, SpikeProcessingPipeline
)
from snn_fusion.scaling.advanced_scaling import (
    WorkerNode, NodeStatus, LoadBalancer, LoadBalancingStrategy,
    AutoScaler, ScalingConfig, ScalingDirection
)


class TestDatasetTransforms:
    """Test suite for dataset transforms."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample multi-modal data."""
        return {
            'audio': torch.randn(100, 2, 64),
            'events': torch.randn(100, 128, 128, 2),
            'imu': torch.randn(100, 6),
            'label': torch.tensor(3)
        }
    
    def test_temporal_jitter(self, sample_data):
        """Test temporal jittering transform."""
        transform = TemporalJitter(max_jitter=5, probability=1.0)
        result = transform(sample_data)
        
        assert isinstance(result, dict)
        assert 'audio' in result
        assert result['audio'].shape == sample_data['audio'].shape
        assert result['label'] == sample_data['label']
    
    def test_modality_dropout(self, sample_data):
        """Test modality dropout transform."""
        transform = ModalityDropout(dropout_probability=1.0, min_modalities=1)
        result = transform(sample_data)
        
        assert isinstance(result, dict)
        # At least one modality should be zeroed out
        zero_modalities = sum(
            1 for key in ['audio', 'events', 'imu']
            if torch.all(result[key] == 0)
        )
        assert zero_modalities > 0
    
    def test_spike_masking(self, sample_data):
        """Test spike masking transform."""
        transform = SpikeMasking(mask_probability=0.5, probability=1.0)
        result = transform(sample_data)
        
        assert isinstance(result, dict)
        assert result['audio'].shape == sample_data['audio'].shape
    
    def test_spike_noise(self, sample_data):
        """Test spike noise transform."""
        # Convert to binary spikes for noise testing
        binary_data = {
            'audio': (torch.randn(100, 2, 64) > 0).float(),
            'events': (torch.randn(100, 128, 128, 2) > 0).float(),
            'label': torch.tensor(3)
        }
        
        transform = SpikeNoise(noise_probability=0.1, probability=1.0)
        result = transform(binary_data)
        
        assert isinstance(result, dict)
        assert result['audio'].shape == binary_data['audio'].shape
    
    def test_create_training_transforms(self, sample_data):
        """Test training transform pipeline."""
        transforms = create_training_transforms()
        result = transforms(sample_data)
        
        assert isinstance(result, dict)
        assert all(key in result for key in sample_data.keys())


class TestErrorHandling:
    """Test suite for error handling system."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ErrorHandler(
                log_errors=True,
                save_error_reports=True,
                error_report_dir=temp_dir,
                max_recovery_attempts=2
            )
    
    def test_snn_fusion_error(self):
        """Test custom SNN-Fusion exception."""
        error = SNNFusionError(
            "Test error",
            category=ErrorCategory.DATA_ERROR,
            severity=ErrorSeverity.HIGH,
            context={'test': True}
        )
        
        assert str(error) == "Test error"
        assert error.category == ErrorCategory.DATA_ERROR
        assert error.severity == ErrorSeverity.HIGH
        assert error.context == {'test': True}
    
    def test_error_categorization(self, error_handler):
        """Test automatic error categorization."""
        data_error = ValueError("Invalid tensor shape")
        report = error_handler.handle_error(data_error)
        
        assert report.category == ErrorCategory.DATA_ERROR
        assert "shape" in report.message.lower()
    
    def test_error_recovery(self, error_handler):
        """Test error recovery mechanisms."""
        # Test data error recovery
        data_error = ValueError("missing data")
        report = error_handler.handle_error(data_error)
        
        assert report.recovery_attempted
        assert report.recovery_successful  # Should succeed with fallback
    
    def test_error_statistics(self, error_handler):
        """Test error statistics collection."""
        # Generate some errors
        for i in range(5):
            error_handler.handle_error(ValueError(f"Test error {i}"))
        
        stats = error_handler.get_error_statistics()
        assert stats['total_errors'] == 5
        assert stats['recovery_stats']['attempted'] > 0
    
    def test_handle_errors_decorator(self, error_handler):
        """Test error handling decorator."""
        @handle_errors(error_handler=error_handler, return_on_error="fallback")
        def failing_function():
            raise ValueError("Decorator test error")
        
        result = failing_function()
        assert result == "fallback"
    
    def test_error_context_manager(self, error_handler):
        """Test error context manager."""
        with ErrorContext(error_handler=error_handler, reraise=False):
            raise RuntimeError("Context manager test")
        
        # Should not raise exception due to reraise=False
        stats = error_handler.get_error_statistics()
        assert stats['total_errors'] >= 1


class TestSecurityEnhanced:
    """Test suite for enhanced security features."""
    
    @pytest.fixture
    def sanitizer(self):
        """Create input sanitizer for testing."""
        return InputSanitizer(SecurityLevel.STRICT)
    
    def test_string_sanitization(self, sanitizer):
        """Test string input sanitization."""
        clean_string = sanitizer.sanitize_string("Hello World", max_length=50)
        assert clean_string == "Hello World"
        
        # Test malicious input blocking
        with pytest.raises(ValueError):
            sanitizer.sanitize_string("<script>alert('xss')</script>")
        
        with pytest.raises(ValueError):
            sanitizer.sanitize_string("'; DROP TABLE users; --")
    
    def test_filename_sanitization(self, sanitizer):
        """Test filename sanitization."""
        safe_filename = sanitizer.sanitize_filename("document.pdf")
        assert safe_filename == "document.pdf"
        
        # Test path traversal prevention
        with pytest.raises(ValueError):
            sanitizer.sanitize_filename("../../../secret.txt")
        
        # Test dangerous characters removal
        safe_name = sanitizer.sanitize_filename("file<>with*bad?chars.txt")
        assert "<" not in safe_name and ">" not in safe_name
    
    def test_path_validation(self, sanitizer):
        """Test file path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            allowed_dirs = [temp_dir]
            test_file = Path(temp_dir) / "test.txt"
            
            # Valid path
            validated_path = sanitizer.validate_file_path(test_file, allowed_dirs)
            assert isinstance(validated_path, Path)
            
            # Invalid path (outside allowed directory)
            with pytest.raises(ValueError):
                sanitizer.validate_file_path("/etc/passwd", allowed_dirs)
    
    def test_numeric_validation(self, sanitizer):
        """Test numeric input validation."""
        # Valid numeric input
        result = sanitizer.sanitize_numeric_input("42", min_value=0, max_value=100)
        assert result == 42
        
        # Invalid range
        with pytest.raises(ValueError):
            sanitizer.sanitize_numeric_input("150", min_value=0, max_value=100)
        
        # Injection attempt
        with pytest.raises(ValueError):
            sanitizer.sanitize_numeric_input("42; rm -rf /", min_value=0, max_value=100)
    
    def test_json_validation(self, sanitizer):
        """Test JSON input validation."""
        valid_json = {"name": "test", "value": 42}
        result = sanitizer.validate_json_input(valid_json)
        assert isinstance(result, dict)
        assert "name" in result
        
        # Test malicious JSON
        malicious_json = {"name": "<script>alert('xss')</script>", "value": 42}
        result = sanitizer.validate_json_input(malicious_json)
        assert "<script>" not in str(result)
    
    def test_security_violations_tracking(self, sanitizer):
        """Test security violations tracking."""
        # Trigger some violations
        try:
            sanitizer.sanitize_string("<script>alert('test')</script>")
        except ValueError:
            pass
        
        try:
            sanitizer.sanitize_filename("../../../etc/passwd")
        except ValueError:
            pass
        
        summary = sanitizer.get_violation_summary()
        assert summary['total_violations'] >= 2
        assert 'injection' in summary['violations_by_type'] or 'path_traversal' in summary['violations_by_type']
    
    def test_secure_config(self):
        """Test secure configuration management."""
        config = SecureConfig()
        
        test_config = {
            "api_key": "secret-key-123",
            "database_url": "postgresql://user:pass@localhost/db",
            "debug": True
        }
        
        validated = config.validate_config(test_config)
        assert "api_key" in validated
        assert "database_url" in validated
        # Sensitive values should be encrypted/modified
        assert validated["api_key"] != "secret-key-123"


class TestMonitoringSystem:
    """Test suite for monitoring and alerting."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector for testing."""
        return MetricsCollector(retention_hours=1)
    
    @pytest.fixture
    def health_monitor(self):
        """Create health monitor for testing."""
        monitor = HealthMonitor(check_interval=1)
        yield monitor
        monitor.stop_monitoring()
    
    @pytest.fixture
    def alert_manager(self):
        """Create alert manager for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_file:
            manager = AlertManager(storage_file=temp_file.name)
            yield manager
            # Cleanup
            Path(temp_file.name).unlink(missing_ok=True)
    
    def test_metrics_collection(self, metrics_collector):
        """Test metrics collection and retrieval."""
        from snn_fusion.monitoring.comprehensive_monitoring import MetricType
        
        # Record some metrics
        metrics_collector.record_metric("test_counter", 10, MetricType.COUNTER)
        metrics_collector.set_gauge("test_gauge", 75.5)
        
        # Test retrieval
        counter_metrics = metrics_collector.get_metrics("test_counter")
        assert len(counter_metrics) == 1
        assert counter_metrics[0].value == 10
        
        gauge_value = metrics_collector.get_latest_value("test_gauge")
        assert gauge_value == 75.5
    
    def test_aggregated_metrics(self, metrics_collector):
        """Test metrics aggregation."""
        from snn_fusion.monitoring.comprehensive_monitoring import MetricType
        
        # Record multiple values
        for i in range(10):
            metrics_collector.record_metric("test_metric", i, MetricType.GAUGE)
        
        aggregated = metrics_collector.get_aggregated_metrics(60)
        assert "test_metric" in aggregated
        assert aggregated["test_metric"]["count"] == 10
        assert aggregated["test_metric"]["avg"] == 4.5
        assert aggregated["test_metric"]["min"] == 0
        assert aggregated["test_metric"]["max"] == 9
    
    def test_health_monitoring(self, health_monitor):
        """Test health check system."""
        # Start monitoring
        health_monitor.start_monitoring()
        time.sleep(2)  # Let it run a health check cycle
        
        # Get health status
        overall_health = health_monitor.get_overall_health()
        assert "status" in overall_health
        assert "checks" in overall_health
        assert len(overall_health["checks"]) > 0
    
    def test_alert_management(self, alert_manager):
        """Test alert creation and management."""
        # Trigger an alert
        alert_id = alert_manager.trigger_alert(
            name="test_alert",
            message="This is a test alert",
            severity=AlertSeverity.WARNING
        )
        
        assert alert_id is not None
        
        # Check active alerts
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].name == "test_alert"
        
        # Resolve alert
        resolved = alert_manager.resolve_alert(alert_id)
        assert resolved
        
        # Check that alert is no longer active
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 0
    
    def test_comprehensive_monitoring(self):
        """Test integrated monitoring system."""
        config = {
            'health_check_interval': 1,
            'metrics_retention_hours': 1
        }
        
        monitor = ComprehensiveMonitor(config)
        
        # Record some metrics
        monitor.metrics.set_gauge('test.cpu_usage', 75.0)
        monitor.metrics.increment_counter('test.requests')
        
        # Get dashboard
        dashboard = monitor.get_monitoring_dashboard()
        assert 'system_health' in dashboard
        assert 'metrics_summary' in dashboard
        assert 'alert_summary' in dashboard
        
        monitor.stop()


class TestPerformanceOptimization:
    """Test suite for performance optimization."""
    
    @pytest.fixture
    def profiler(self):
        """Create performance profiler."""
        return PerformanceProfiler(enable_detailed_profiling=True)
    
    @pytest.fixture
    def memory_optimizer(self):
        """Create memory optimizer."""
        return MemoryOptimizer(OptimizationLevel.BASIC)
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager."""
        return CacheManager(max_memory_mb=10.0)  # Small cache for testing
    
    def test_performance_profiling(self, profiler):
        """Test performance profiling."""
        profile_id = profiler.start_profiling("test_operation")
        
        # Simulate some work
        time.sleep(0.1)
        
        metrics = profiler.end_profiling(profile_id)
        assert metrics.execution_time >= 0.1
        assert metrics.operations_per_second > 0
    
    def test_memory_optimization(self, memory_optimizer):
        """Test memory optimization features."""
        # Test array allocation from pool
        array1 = memory_optimizer.allocate_array((100, 64), np.float32)
        assert array1.shape == (100, 64)
        assert array1.dtype == np.float32
        
        # Return to pool and allocate again
        memory_optimizer.deallocate_array(array1)
        array2 = memory_optimizer.allocate_array((100, 64), np.float32)
        
        # Should get the same array from pool (or a new one)
        assert array2.shape == (100, 64)
    
    def test_cache_operations(self, cache_manager):
        """Test caching functionality."""
        # Test put and get
        cache_manager.put("test_key", "test_value")
        result = cache_manager.get("test_key")
        assert result == "test_value"
        
        # Test cache miss
        missing = cache_manager.get("nonexistent_key")
        assert missing is None
        
        # Test cache stats
        stats = cache_manager.get_stats()
        assert stats['hit_rate'] >= 0.0
        assert stats['cached_items'] >= 1
    
    def test_performance_optimizer_integration(self):
        """Test integrated performance optimizer."""
        optimizer = PerformanceOptimizer(
            optimization_level=OptimizationLevel.BASIC,
            max_memory_mb=50.0,
            max_workers=2
        )
        
        # Test spike encoding optimization
        test_data = np.random.randn(100, 64)
        
        def dummy_encoder(data):
            return data * 2
        
        result = optimizer.optimize_spike_encoding(test_data, dummy_encoder)
        assert result.shape == test_data.shape
        
        # Test caching (second call should be faster)
        result2 = optimizer.optimize_spike_encoding(test_data, dummy_encoder)
        assert np.array_equal(result, result2)
        
        # Get optimization report
        report = optimizer.get_optimization_report()
        assert 'optimization_level' in report
        assert 'cache_stats' in report
        
        optimizer.cleanup_resources()


class TestConcurrentProcessing:
    """Test suite for concurrent processing."""
    
    @pytest.fixture
    def processor_config(self):
        """Create processing configuration."""
        return ProcessingConfig(
            mode=ProcessingMode.THREADED,
            max_threads=4,
            max_processes=2,
            batch_size=8
        )
    
    @pytest.fixture
    def processor(self, processor_config):
        """Create concurrent processor."""
        processor = ConcurrentProcessor(processor_config)
        yield processor
        processor.shutdown()
    
    def test_batch_processing(self, processor):
        """Test asynchronous batch processing."""
        test_data = [np.random.randn(10, 5) for _ in range(20)]
        
        def simple_processor(data):
            time.sleep(0.01)  # Simulate processing
            return data * 2
        
        # Run async batch processing
        async def run_test():
            results = await processor.process_batch_async(
                test_data[:10], simple_processor, batch_size=4
            )
            return results
        
        results = asyncio.run(run_test())
        assert len(results) == 10
        assert all(isinstance(r, np.ndarray) for r in results)
    
    def test_task_pipeline(self, processor):
        """Test task pipeline execution."""
        def task_a(x):
            return x * 2
        
        def task_b(y, dep_a):
            return y + dep_a
        
        def task_c(dep_a, dep_b):
            return dep_a * dep_b
        
        tasks = [
            ProcessingTask("task_a", task_a, args=(5,), priority=TaskPriority.HIGH),
            ProcessingTask("task_b", task_b, args=(3,), dependencies=["task_a"]),
            ProcessingTask("task_c", task_c, dependencies=["task_a", "task_b"])
        ]
        
        results = processor.execute_pipeline(tasks)
        assert "task_a" in results
        assert "task_b" in results
        assert "task_c" in results
        assert results["task_a"] == 10  # 5 * 2
        assert results["task_b"] == 13  # 3 + 10
        assert results["task_c"] == 130  # 10 * 13
    
    def test_multimodal_processing(self, processor):
        """Test multi-modal data processing."""
        modal_data = {
            'audio': [np.random.randn(50, 32) for _ in range(5)],
            'events': [np.random.randn(50, 64) for _ in range(5)],
            'imu': [np.random.randn(50, 6) for _ in range(5)]
        }
        
        processors = {
            'audio': lambda x: np.mean(x, axis=0),
            'events': lambda x: np.max(x, axis=0),
            'imu': lambda x: np.std(x, axis=0)
        }
        
        results = processor.process_multi_modal_batch(modal_data, processors)
        assert len(results) == 3
        assert 'audio' in results
        assert 'events' in results
        assert 'imu' in results
    
    def test_spike_processing_pipeline(self, processor):
        """Test spike processing pipeline."""
        pipeline = SpikeProcessingPipeline(processor)
        
        # Test spike encoding
        input_data = [np.random.randn(20, 16) for _ in range(5)]
        encoder_config = {'rate_factor': 50.0, 'time_steps': 25}
        
        async def run_encoding():
            return await pipeline.encode_spike_trains(input_data, encoder_config)
        
        encoded = asyncio.run(run_encoding())
        assert len(encoded) == 5
        assert all(spike_train.shape[-1] == 25 for spike_train in encoded)  # time_steps


class TestAdvancedScaling:
    """Test suite for advanced scaling and load balancing."""
    
    @pytest.fixture
    def load_balancer(self):
        """Create load balancer for testing."""
        return LoadBalancer(LoadBalancingStrategy.LEAST_RESPONSE_TIME)
    
    @pytest.fixture
    def scaling_config(self):
        """Create scaling configuration."""
        return ScalingConfig(
            min_nodes=2,
            max_nodes=10,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            cooldown_period=1  # Short for testing
        )
    
    def test_worker_node(self):
        """Test worker node functionality."""
        node = WorkerNode(
            node_id="test-node",
            host="localhost",
            port=8080,
            capacity=100,
            weight=1.0
        )
        
        assert node.load_percentage == 0.0
        assert node.health_score > 0
        
        # Simulate load
        node.current_load = 50
        assert node.load_percentage == 50.0
        
        # Add response times
        node.response_times.extend([0.1, 0.2, 0.3])
        assert node.average_response_time == 0.2
    
    def test_load_balancer_strategies(self, load_balancer):
        """Test different load balancing strategies."""
        # Add test nodes
        nodes = [
            WorkerNode(f"node-{i}", "localhost", 8000+i, 100, 1.0)
            for i in range(3)
        ]
        
        for node in nodes:
            load_balancer.add_node(node)
        
        # Test node selection
        selected_node = load_balancer.get_node("test-request")
        assert selected_node is not None
        assert selected_node.node_id.startswith("node-")
        
        # Test load increment and metrics update
        load_balancer.increment_node_load(selected_node.node_id)
        load_balancer.update_node_metrics(
            selected_node.node_id, 
            response_time=0.5, 
            success=True,
            cpu_usage=60.0,
            memory_usage=50.0
        )
        
        # Get cluster stats
        stats = load_balancer.get_cluster_stats()
        assert stats['total_nodes'] == 3
        assert stats['healthy_nodes'] == 3
        assert stats['total_capacity'] == 300
    
    def test_auto_scaler(self, load_balancer, scaling_config):
        """Test auto-scaling functionality."""
        auto_scaler = AutoScaler(scaling_config, load_balancer)
        
        # Add initial nodes
        for i in range(scaling_config.min_nodes):
            node = WorkerNode(f"initial-{i}", "localhost", 8000+i, 100, 1.0)
            load_balancer.add_node(node)
        
        # Test scaling decision with high load
        from snn_fusion.scaling.advanced_scaling import ScalingMetrics
        high_load_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            total_requests_per_second=100.0,
            average_response_time=2.0,
            error_rate=0.5,
            cpu_usage=90.0,
            memory_usage=85.0,
            active_connections=200,
            queue_depth=50
        )
        
        decision = auto_scaler._make_scaling_decision(high_load_metrics)
        assert decision['action'] in [ScalingDirection.UP, ScalingDirection.STABLE]
        
        # Test scaling report
        report = auto_scaler.get_scaling_report()
        assert 'cluster_status' in report
        assert 'scaling_config' in report
        assert 'predictions' in report


class TestIntegrationScenarios:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_processing(self):
        """Test complete processing pipeline."""
        # Create components
        config = ProcessingConfig(mode=ProcessingMode.THREADED, max_threads=2)
        processor = ConcurrentProcessor(config)
        
        try:
            # Create test data pipeline
            input_data = [np.random.randn(50, 32) for _ in range(10)]
            
            def processing_pipeline(data):
                # Simulate multi-step processing
                step1 = data * 2  # Encoding
                time.sleep(0.01)  # Simulate computation
                step2 = np.mean(step1, axis=1, keepdims=True)  # Feature extraction
                return step2
            
            # Process data
            async def run_pipeline():
                return await processor.process_batch_async(
                    input_data, processing_pipeline, batch_size=4
                )
            
            results = asyncio.run(run_pipeline())
            assert len(results) == 10
            assert all(r.shape[1] == 1 for r in results)  # Feature dimension
            
        finally:
            processor.shutdown()
    
    def test_monitoring_with_scaling(self):
        """Test monitoring system with auto-scaling."""
        # Create monitoring system
        monitor = ComprehensiveMonitor({'health_check_interval': 1})
        
        # Create load balancer and scaler
        lb = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        config = ScalingConfig(min_nodes=1, max_nodes=5, cooldown_period=1)
        scaler = AutoScaler(config, lb)
        
        try:
            # Add initial node
            node = WorkerNode("test-node", "localhost", 8080, 100, 1.0)
            lb.add_node(node)
            
            # Record some metrics
            monitor.metrics.set_gauge('system.cpu_usage', 85.0)
            monitor.metrics.set_gauge('system.memory_usage', 80.0)
            
            # Get integrated dashboard
            dashboard = monitor.get_monitoring_dashboard()
            scaling_report = scaler.get_scaling_report()
            
            assert 'system_health' in dashboard
            assert 'cluster_status' in scaling_report
            
        finally:
            monitor.stop()
            scaler.stop_monitoring()
    
    def test_security_with_error_handling(self):
        """Test security system with error handling."""
        sanitizer = InputSanitizer(SecurityLevel.STRICT)
        error_handler = ErrorHandler(log_errors=True, save_error_reports=False)
        
        # Test security violation handling
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "rm -rf /"
        ]
        
        violations = 0
        for malicious_input in malicious_inputs:
            try:
                sanitizer.sanitize_string(malicious_input)
            except ValueError as e:
                error_report = error_handler.handle_error(e, 
                    context={'input': malicious_input[:20] + '...'})
                assert error_report.category == ErrorCategory.VALIDATION_ERROR
                violations += 1
        
        assert violations == len(malicious_inputs)
        
        # Check security summary
        security_summary = sanitizer.get_violation_summary()
        assert security_summary['total_violations'] >= violations


# Performance and Load Tests
class TestPerformanceLoad:
    """Performance and load testing."""
    
    def test_concurrent_processing_performance(self):
        """Test performance under concurrent load."""
        config = ProcessingConfig(
            mode=ProcessingMode.HYBRID,
            max_threads=8,
            batch_size=32
        )
        
        processor = ConcurrentProcessor(config)
        
        try:
            # Generate large dataset
            large_dataset = [np.random.randn(100, 64) for _ in range(500)]
            
            def cpu_intensive_task(data):
                # Simulate CPU-intensive processing
                return np.fft.fft2(data).real
            
            start_time = time.time()
            
            # Process in batches
            async def run_load_test():
                results = await processor.process_batch_async(
                    large_dataset[:100], cpu_intensive_task, batch_size=20
                )
                return results
            
            results = asyncio.run(run_load_test())
            processing_time = time.time() - start_time
            
            assert len(results) == 100
            assert processing_time < 30.0  # Should complete within 30 seconds
            
            # Check performance stats
            stats = processor.get_performance_stats()
            assert stats['success_rate'] > 90.0  # At least 90% success rate
            
        finally:
            processor.shutdown()
    
    def test_memory_optimization_under_load(self):
        """Test memory optimization under high memory pressure."""
        optimizer = MemoryOptimizer(OptimizationLevel.AGGRESSIVE)
        
        # Simulate high memory usage
        large_arrays = []
        for i in range(100):
            array = optimizer.allocate_array((1000, 1000), np.float32)
            large_arrays.append(array)
        
        # Check memory stats
        memory_stats = optimizer.get_memory_stats()
        assert memory_stats['rss_mb'] > 0
        
        # Deallocate arrays
        for array in large_arrays:
            optimizer.deallocate_array(array)
        
        # Force garbage collection
        optimizer.optimize_garbage_collection()
        
        # Memory should be reclaimed
        final_stats = optimizer.get_memory_stats()
        assert 'pool_sizes' in final_stats


# Run all tests
if __name__ == "__main__":
    print("Running Comprehensive Test Suite...")
    
    # Run pytest with coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--maxfail=10",
        "--durations=10"
    ])
    
    print("✓ Comprehensive test suite completed!")