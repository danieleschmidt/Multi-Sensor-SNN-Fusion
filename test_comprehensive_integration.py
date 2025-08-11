#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for SNN Fusion Framework

Tests all three generations (Simple, Robust, Optimized) with realistic
scenarios and validates production readiness across all components.
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import json
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test framework
class TestResults:
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        self.performance_metrics = {}
        
    def add_result(self, test_name: str, passed: bool, error: str = None, metrics: Dict = None):
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            print(f"✓ {test_name}")
        else:
            self.tests_failed += 1
            self.failures.append(f"{test_name}: {error}")
            print(f"✗ {test_name}: {error}")
        
        if metrics:
            self.performance_metrics[test_name] = metrics
    
    def summary(self):
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        return {
            'total_tests': self.tests_run,
            'passed': self.tests_passed,
            'failed': self.tests_failed,
            'success_rate': f"{success_rate:.1f}%",
            'failures': self.failures,
            'performance_metrics': self.performance_metrics
        }


def test_generation_1_basic_functionality(results: TestResults):
    """Test Generation 1: Make it Work (Simple)"""
    print("\n=== GENERATION 1 TESTS: BASIC FUNCTIONALITY ===")
    
    # Test 1: Configuration System
    try:
        from snn_fusion.utils.config import create_default_config, validate_config
        config = create_default_config()
        validate_config(config)
        results.add_result("G1.1 Configuration System", True)
    except Exception as e:
        results.add_result("G1.1 Configuration System", False, str(e))
    
    # Test 2: Logging System
    try:
        from snn_fusion.utils.logging import setup_logging, get_logger
        logger = setup_logging(log_level="INFO", enable_console=False)
        module_logger = get_logger("test_module")
        module_logger.info("Test log message")
        results.add_result("G1.2 Logging System", True)
    except Exception as e:
        results.add_result("G1.2 Logging System", False, str(e))
    
    # Test 3: Spike Encoding
    try:
        import numpy as np
        from snn_fusion.algorithms.encoding import RateEncoder, TemporalEncoder
        
        # Rate encoding test
        rate_encoder = RateEncoder(n_neurons=32, duration=100.0, max_rate=50.0)
        test_data = np.random.randn(100)
        spike_data = rate_encoder.encode(test_data)
        
        # Temporal encoding test
        temporal_encoder = TemporalEncoder(n_neurons=32, duration=100.0, encoding_window=50.0)
        spike_data_temp = temporal_encoder.encode(test_data)
        
        results.add_result("G1.3 Spike Encoding", True, 
                         metrics={'rate_spikes': len(spike_data.spike_times),
                                'temporal_spikes': len(spike_data_temp.spike_times)})
    except Exception as e:
        results.add_result("G1.3 Spike Encoding", False, str(e))
    
    # Test 4: Basic Model Components (without torch)
    try:
        # Test model structure exists
        from snn_fusion.models import lsm, multimodal_lsm, hierarchical_fusion
        from snn_fusion.models import neurons, attention, readouts
        
        # Verify classes exist
        assert hasattr(lsm, 'LiquidStateMachine')
        assert hasattr(neurons, 'AdaptiveLIF')
        assert hasattr(attention, 'CrossModalAttention')
        assert hasattr(readouts, 'LinearReadout')
        
        results.add_result("G1.4 Model Components Structure", True)
    except Exception as e:
        results.add_result("G1.4 Model Components Structure", False, str(e))
    
    # Test 5: Training Infrastructure
    try:
        from snn_fusion.training import trainer, losses, plasticity
        
        # Verify training classes exist
        assert hasattr(trainer, 'SNNTrainer')
        assert hasattr(losses, 'TemporalCrossEntropyLoss')
        assert hasattr(plasticity, 'STDPLearner')
        
        results.add_result("G1.5 Training Infrastructure", True)
    except Exception as e:
        results.add_result("G1.5 Training Infrastructure", False, str(e))
    
    # Test 6: Data Pipeline
    try:
        from snn_fusion.datasets import maven_dataset, loaders
        
        # Verify data classes exist
        assert hasattr(maven_dataset, 'MAVENDataset')
        assert hasattr(loaders, 'MultiModalCollate')
        
        results.add_result("G1.6 Data Pipeline Structure", True)
    except Exception as e:
        results.add_result("G1.6 Data Pipeline Structure", False, str(e))


def test_generation_2_robustness(results: TestResults):
    """Test Generation 2: Make it Robust (Reliable)"""
    print("\n=== GENERATION 2 TESTS: ROBUSTNESS & RELIABILITY ===")
    
    # Test 1: Error Handling System
    try:
        from snn_fusion.utils.error_handling import (
            ErrorHandler, SNNFusionException, ModelError, DataError
        )
        
        # Test error handler
        error_handler = ErrorHandler()
        
        # Test custom exceptions
        try:
            raise ModelError("Test model error", severity="HIGH")
        except ModelError as e:
            error_context = error_handler.handle_error(e)
            assert error_context.error_type == "ModelError"
        
        results.add_result("G2.1 Error Handling System", True)
    except Exception as e:
        results.add_result("G2.1 Error Handling System", False, str(e))
    
    # Test 2: Security Framework
    try:
        from snn_fusion.utils.security_enhanced import (
            InputSanitizer, SecureConfig, SecurityLevel
        )
        
        # Test input sanitization
        sanitizer = InputSanitizer(SecurityLevel.STRICT)
        
        # Test safe input
        safe_input = sanitizer.sanitize_string("Hello World", max_length=100)
        assert safe_input == "Hello World"
        
        # Test dangerous input
        try:
            sanitizer.sanitize_string("<script>alert('xss')</script>")
            results.add_result("G2.2 Security Framework", False, "Dangerous input not blocked")
        except ValueError:
            # Expected behavior
            pass
        
        # Test secure config
        config = SecureConfig()
        test_config = {"api_key": "secret-123", "debug": True}
        validated_config = config.validate_config(test_config)
        
        results.add_result("G2.2 Security Framework", True)
    except Exception as e:
        results.add_result("G2.2 Security Framework", False, str(e))
    
    # Test 3: Health Monitoring System
    try:
        from snn_fusion.utils.health_monitoring import (
            HealthMonitor, HealthStatus, MetricType, HealthMetric
        )
        
        # Test health monitor initialization
        monitor = HealthMonitor(monitoring_interval=1.0, enable_alerts=False)
        
        # Test snapshot taking
        snapshot = monitor.take_snapshot()
        assert snapshot.overall_status in [s for s in HealthStatus]
        assert isinstance(snapshot.metrics, list)
        
        # Test custom metrics
        def custom_metric():
            return 42.0
        
        monitor.register_custom_metric("test_metric", custom_metric)
        
        results.add_result("G2.3 Health Monitoring System", True, 
                         metrics={'metrics_count': len(snapshot.metrics)})
    except Exception as e:
        results.add_result("G2.3 Health Monitoring System", False, str(e))
    
    # Test 4: Comprehensive Metrics
    try:
        import numpy as np
        from snn_fusion.utils.metrics import (
            SpikeMetrics, FusionMetrics, PerformanceMetrics, 
            create_comprehensive_metrics
        )
        
        # Test spike metrics
        spike_metrics = SpikeMetrics()
        
        # Create dummy spike trains
        spike_trains = np.random.randint(0, 2, (4, 100, 64)).astype(float)
        
        # Test firing rate computation
        firing_rates = spike_metrics.compute_firing_rates(spike_trains)
        assert firing_rates.shape == (4, 64)
        
        # Test fusion metrics
        fusion_metrics = FusionMetrics()
        
        single_modal_preds = {
            "audio": np.random.randn(10, 5),
            "vision": np.random.randn(10, 5),
            "tactile": np.random.randn(10, 5)
        }
        fusion_pred = np.random.randn(10, 5)
        targets = np.random.randint(0, 5, 10)
        
        importance = fusion_metrics.compute_modality_importance(
            single_modal_preds, fusion_pred, targets
        )
        assert isinstance(importance, dict)
        assert len(importance) == 3
        
        results.add_result("G2.4 Comprehensive Metrics", True,
                         metrics={'spike_trains_shape': spike_trains.shape,
                                'modality_importance': importance})
    except Exception as e:
        results.add_result("G2.4 Comprehensive Metrics", False, str(e))
    
    # Test 5: Validation and Data Integrity
    try:
        from snn_fusion.utils.security_enhanced import InputSanitizer, SecurityLevel
        
        sanitizer = InputSanitizer(SecurityLevel.PARANOID)
        
        # Test filename validation
        safe_filename = sanitizer.sanitize_filename("test_file.txt")
        assert safe_filename == "test_file.txt"
        
        # Test numeric validation
        validated_num = sanitizer.sanitize_numeric_input("42", min_value=0, max_value=100)
        assert validated_num == 42
        
        # Test JSON validation
        test_json = {"name": "Test", "value": 123}
        validated_json = sanitizer.validate_json_input(test_json)
        assert "name" in validated_json
        
        results.add_result("G2.5 Data Validation", True)
    except Exception as e:
        results.add_result("G2.5 Data Validation", False, str(e))


def test_generation_3_optimization(results: TestResults):
    """Test Generation 3: Make it Scale (Optimized)"""
    print("\n=== GENERATION 3 TESTS: SCALABILITY & OPTIMIZATION ===")
    
    # Test 1: Performance Optimizer
    try:
        from snn_fusion.optimization.performance_optimizer import (
            PerformanceOptimizer, OptimizationLevel, MemoryStrategy
        )
        
        # Test optimizer initialization
        optimizer = PerformanceOptimizer(
            optimization_level=OptimizationLevel.BALANCED,
            memory_strategy=MemoryStrategy.DEFAULT
        )
        
        # Test hardware detection
        assert optimizer.hardware_info['cpu_count'] > 0
        
        # Test function profiling
        def test_function():
            time.sleep(0.01)  # 10ms delay
            return "test_result"
        
        result, profile = optimizer.profile_function(test_function, operation_name="test_op")
        assert result == "test_result"
        assert profile.operation_name == "test_op"
        assert profile.execution_time > 0.009  # Should be around 10ms
        
        # Test batch processing optimization
        def batch_processor(batch):
            return [item * 2 for item in batch]
        
        data = list(range(100))
        processed = optimizer.optimize_batch_processing(batch_processor, data, batch_size=10)
        assert len(processed) == 100
        assert processed[0] == 0
        assert processed[10] == 20
        
        results.add_result("G3.1 Performance Optimizer", True,
                         metrics={'cpu_count': optimizer.hardware_info['cpu_count'],
                                'execution_time': profile.execution_time})
    except Exception as e:
        results.add_result("G3.1 Performance Optimizer", False, str(e))
    
    # Test 2: Memory Optimization
    try:
        from snn_fusion.optimization.performance_optimizer import PerformanceOptimizer
        
        optimizer = PerformanceOptimizer(memory_strategy=MemoryStrategy.LOW_MEMORY)
        
        # Test memory usage monitoring
        initial_memory = optimizer._get_memory_usage()
        
        # Test memory optimization
        optimization_result = optimizer.optimize_memory_usage(target_reduction_percent=5.0)
        
        assert 'initial_memory_mb' in optimization_result
        assert 'techniques_applied' in optimization_result
        assert len(optimization_result['techniques_applied']) > 0
        
        results.add_result("G3.2 Memory Optimization", True,
                         metrics=optimization_result)
    except Exception as e:
        results.add_result("G3.2 Memory Optimization", False, str(e))
    
    # Test 3: Distributed Deployment System
    try:
        from snn_fusion.deployment.distributed_deployment import (
            DistributedDeploymentManager, DeploymentConfig, DeploymentStrategy,
            LoadBalancingStrategy, create_deployment_config, create_node_info
        )
        
        # Test deployment config creation
        config = create_deployment_config(
            deployment_id="test_deployment",
            strategy=DeploymentStrategy.MULTI_NODE,
            min_nodes=1,
            max_nodes=3
        )
        
        # Test deployment manager initialization
        manager = DistributedDeploymentManager(config)
        
        # Test node registration
        node_info = create_node_info(
            hostname="test-node-1",
            ip_address="127.0.0.1",
            port=8080,
            capabilities={"cpu_cores": 4, "memory_gb": 8}
        )
        
        success = manager.register_node(node_info)
        assert success
        
        # Test deployment status
        status = manager.get_deployment_status()
        assert status['deployment_id'] == "test_deployment"
        assert status['active_nodes'] == 1
        
        # Test load balancer
        selected_node = manager.route_request({"task_id": "test_task"})
        assert selected_node is not None
        assert selected_node.node_id == node_info.node_id
        
        results.add_result("G3.3 Distributed Deployment", True,
                         metrics={'active_nodes': status['active_nodes']})
    except Exception as e:
        results.add_result("G3.3 Distributed Deployment", False, str(e))
    
    # Test 4: Load Balancing and Auto-Scaling
    try:
        from snn_fusion.deployment.distributed_deployment import (
            LoadBalancer, AutoScaler, LoadBalancingStrategy
        )
        
        # Test load balancer
        load_balancer = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        # Add test nodes
        node1 = create_node_info("node1", "127.0.0.1", 8081)
        node2 = create_node_info("node2", "127.0.0.1", 8082)
        
        load_balancer.add_node(node1)
        load_balancer.add_node(node2)
        
        # Test node selection
        selected1 = load_balancer.select_node()
        selected2 = load_balancer.select_node()
        
        assert selected1.node_id != selected2.node_id  # Round robin should alternate
        
        # Test load balancer status
        status = load_balancer.get_status()
        assert status['total_nodes'] == 2
        assert status['strategy'] == 'round_robin'
        
        results.add_result("G3.4 Load Balancing", True,
                         metrics={'nodes_managed': status['total_nodes']})
    except Exception as e:
        results.add_result("G3.4 Load Balancing", False, str(e))
    
    # Test 5: Service Discovery and Registry
    try:
        from snn_fusion.deployment.distributed_deployment import ServiceRegistry
        
        # Test service registry
        registry = ServiceRegistry()
        registry.initialize()
        
        # Test node registration in service registry
        node_info = create_node_info("service-node", "192.168.1.100", 9090, region="us-east")
        registry.register_node(node_info)
        
        # Test node discovery
        all_nodes = registry.discover_nodes()
        assert len(all_nodes) == 1
        assert all_nodes[0]['hostname'] == "service-node"
        
        # Test region-based discovery
        us_nodes = registry.discover_nodes(region="us-east")
        assert len(us_nodes) == 1
        
        eu_nodes = registry.discover_nodes(region="eu-west")
        assert len(eu_nodes) == 0
        
        # Test service registry status
        status = registry.get_status()
        assert status['total_services'] == 1
        
        results.add_result("G3.5 Service Discovery", True,
                         metrics={'registered_services': status['total_services']})
    except Exception as e:
        results.add_result("G3.5 Service Discovery", False, str(e))


def test_integration_scenarios(results: TestResults):
    """Test realistic integration scenarios"""
    print("\n=== INTEGRATION SCENARIO TESTS ===")
    
    # Test 1: End-to-End Data Processing Pipeline
    try:
        from snn_fusion.utils.config import create_default_config
        from snn_fusion.utils.logging import setup_logging
        from snn_fusion.algorithms.encoding import RateEncoder
        from snn_fusion.utils.metrics import SpikeMetrics
        
        # Setup configuration and logging
        config = create_default_config()
        logger = setup_logging(log_level="WARNING", enable_console=False)
        
        # Create synthetic data
        import numpy as np
        synthetic_data = np.random.randn(1000)
        
        # Encode data
        encoder = RateEncoder(n_neurons=64, duration=100.0, max_rate=50.0)
        spike_data = encoder.encode(synthetic_data)
        
        # Create spike train tensor for metrics
        spike_trains = np.random.randint(0, 2, (1, 100, 64)).astype(float)
        
        # Compute metrics
        spike_metrics = SpikeMetrics()
        firing_rates = spike_metrics.compute_firing_rates(spike_trains)
        
        # Validate pipeline
        assert len(spike_data.spike_times) > 0
        assert firing_rates.shape[0] == 1
        assert firing_rates.shape[1] == 64
        
        results.add_result("I1 End-to-End Data Pipeline", True,
                         metrics={'spikes_generated': len(spike_data.spike_times),
                                'neurons': 64})
    except Exception as e:
        results.add_result("I1 End-to-End Data Pipeline", False, str(e))
    
    # Test 2: Security + Error Handling Integration
    try:
        from snn_fusion.utils.security_enhanced import InputSanitizer, SecurityLevel
        from snn_fusion.utils.error_handling import ErrorHandler
        
        # Setup components
        sanitizer = InputSanitizer(SecurityLevel.STRICT)
        error_handler = ErrorHandler()
        
        # Test malicious input handling with error recovery
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "rm -rf /"
        ]
        
        blocked_count = 0
        for malicious_input in malicious_inputs:
            try:
                sanitizer.sanitize_string(malicious_input)
            except ValueError as e:
                # Log security error
                error_handler.handle_error(e, {'input': malicious_input[:20]})
                blocked_count += 1
        
        # All malicious inputs should be blocked
        assert blocked_count == len(malicious_inputs)
        
        # Check error statistics
        stats = error_handler.get_error_statistics()
        assert stats['total_errors'] == len(malicious_inputs)
        
        results.add_result("I2 Security + Error Handling", True,
                         metrics={'blocked_inputs': blocked_count,
                                'total_errors': stats['total_errors']})
    except Exception as e:
        results.add_result("I2 Security + Error Handling", False, str(e))
    
    # Test 3: Performance Monitoring + Health Check Integration
    try:
        from snn_fusion.optimization.performance_optimizer import PerformanceOptimizer
        from snn_fusion.utils.health_monitoring import HealthMonitor
        
        # Setup performance optimizer
        optimizer = PerformanceOptimizer(enable_profiling=True)
        
        # Setup health monitor
        monitor = HealthMonitor(monitoring_interval=0.1, enable_alerts=False)
        
        # Profile a computational task
        def cpu_intensive_task():
            result = 0
            for i in range(100000):
                result += i * i
            return result
        
        # Profile the task
        result, profile = optimizer.profile_function(cpu_intensive_task, operation_name="cpu_task")
        
        # Take health snapshot
        snapshot = monitor.take_snapshot()
        
        # Validate integration
        assert result > 0
        assert profile.execution_time > 0
        assert len(snapshot.metrics) > 0
        
        # Get optimization summary
        opt_summary = optimizer.get_optimization_summary()
        
        results.add_result("I3 Performance + Health Monitoring", True,
                         metrics={'execution_time': profile.execution_time,
                                'health_metrics': len(snapshot.metrics),
                                'cpu_cores': opt_summary['hardware_info']['cpu_count']})
    except Exception as e:
        results.add_result("I3 Performance + Health Monitoring", False, str(e))
    
    # Test 4: Distributed System + Load Balancing Integration
    try:
        from snn_fusion.deployment.distributed_deployment import (
            DistributedDeploymentManager, create_deployment_config, create_node_info
        )
        
        # Create deployment configuration
        config = create_deployment_config("integration_test", min_nodes=2, max_nodes=5)
        
        # Create deployment manager
        manager = DistributedDeploymentManager(config)
        
        # Register multiple nodes
        nodes = []
        for i in range(3):
            node = create_node_info(f"node-{i}", "127.0.0.1", 8080 + i, 
                                  capabilities={"cpu": 4, "memory": 8},
                                  region="us-east")
            nodes.append(node)
            success = manager.register_node(node)
            assert success
        
        # Test request routing
        requests_routed = 0
        node_usage = {}
        
        for i in range(10):  # Route 10 requests
            request_data = {"request_id": i, "data_size": 1024}
            selected_node = manager.route_request(request_data)
            
            if selected_node:
                requests_routed += 1
                node_usage[selected_node.node_id] = node_usage.get(selected_node.node_id, 0) + 1
        
        # Check that requests were distributed
        assert requests_routed == 10
        assert len(node_usage) > 1  # Multiple nodes should have received requests
        
        # Get deployment status
        status = manager.get_deployment_status()
        assert status['active_nodes'] == 3
        
        results.add_result("I4 Distributed System Integration", True,
                         metrics={'active_nodes': status['active_nodes'],
                                'requests_routed': requests_routed,
                                'node_distribution': len(node_usage)})
    except Exception as e:
        results.add_result("I4 Distributed System Integration", False, str(e))


def test_performance_benchmarks(results: TestResults):
    """Run performance benchmarks"""
    print("\n=== PERFORMANCE BENCHMARK TESTS ===")
    
    # Test 1: Encoding Performance Benchmark
    try:
        import numpy as np
        import time
        from snn_fusion.algorithms.encoding import RateEncoder, TemporalEncoder, CochlearEncoder
        
        # Test data sizes
        data_sizes = [100, 1000, 5000]
        encoding_results = {}
        
        for size in data_sizes:
            test_data = np.random.randn(size)
            
            # Rate encoding benchmark
            rate_encoder = RateEncoder(n_neurons=64, duration=100.0)
            start_time = time.time()
            rate_result = rate_encoder.encode(test_data)
            rate_time = time.time() - start_time
            
            # Temporal encoding benchmark  
            temporal_encoder = TemporalEncoder(n_neurons=64, duration=100.0)
            start_time = time.time()
            temporal_result = temporal_encoder.encode(test_data)
            temporal_time = time.time() - start_time
            
            encoding_results[f'size_{size}'] = {
                'rate_encoding_time': rate_time,
                'temporal_encoding_time': temporal_time,
                'rate_spikes': len(rate_result.spike_times),
                'temporal_spikes': len(temporal_result.spike_times)
            }
        
        # Validate performance (should complete within reasonable time)
        max_time = max(result['rate_encoding_time'] for result in encoding_results.values())
        assert max_time < 5.0  # Should complete in under 5 seconds
        
        results.add_result("P1 Encoding Performance", True, metrics=encoding_results)
    except Exception as e:
        results.add_result("P1 Encoding Performance", False, str(e))
    
    # Test 2: Memory Usage Benchmark
    try:
        from snn_fusion.optimization.performance_optimizer import PerformanceOptimizer
        import gc
        
        optimizer = PerformanceOptimizer()
        
        # Measure baseline memory
        gc.collect()
        baseline_memory = optimizer._get_memory_usage()
        
        # Allocate some memory (simulate workload)
        large_arrays = []
        for i in range(10):
            arr = np.random.randn(1000, 1000)  # ~8MB each
            large_arrays.append(arr)
        
        # Measure peak memory
        peak_memory = optimizer._get_memory_usage()
        memory_increase = peak_memory - baseline_memory
        
        # Test memory optimization
        optimization_result = optimizer.optimize_memory_usage(target_reduction_percent=10.0)
        
        # Clean up
        del large_arrays
        final_memory = optimizer._get_memory_usage()
        
        # Validate memory management
        assert memory_increase > 50  # Should have increased by at least 50MB
        assert 'memory_saved_mb' in optimization_result
        
        memory_metrics = {
            'baseline_memory_mb': baseline_memory,
            'peak_memory_mb': peak_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': memory_increase,
            'optimization_result': optimization_result
        }
        
        results.add_result("P2 Memory Usage Benchmark", True, metrics=memory_metrics)
    except Exception as e:
        results.add_result("P2 Memory Usage Benchmark", False, str(e))
    
    # Test 3: Batch Processing Performance
    try:
        from snn_fusion.optimization.performance_optimizer import PerformanceOptimizer
        
        optimizer = PerformanceOptimizer()
        
        # Test different batch sizes
        data = list(range(1000))
        
        def simple_processor(item):
            return item * item
        
        def batch_processor(batch):
            return [simple_processor(item) for item in batch]
        
        batch_results = {}
        
        for batch_size in [10, 50, 100]:
            start_time = time.time()
            results_batch = optimizer.optimize_batch_processing(
                batch_processor, data, batch_size=batch_size, parallel=False
            )
            processing_time = time.time() - start_time
            
            batch_results[f'batch_size_{batch_size}'] = {
                'processing_time': processing_time,
                'throughput': len(data) / processing_time if processing_time > 0 else 0,
                'results_length': len(results_batch)
            }
            
            # Validate results
            assert len(results_batch) == len(data)
            assert results_batch[10] == 100  # 10 * 10 = 100
        
        results.add_result("P3 Batch Processing Performance", True, metrics=batch_results)
    except Exception as e:
        results.add_result("P3 Batch Processing Performance", False, str(e))


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🧠 SNN FUSION FRAMEWORK - COMPREHENSIVE INTEGRATION TEST SUITE")
    print("=" * 70)
    print("Testing all three generations with realistic scenarios...")
    
    # Initialize test results
    results = TestResults()
    
    # Run test suites
    test_generation_1_basic_functionality(results)
    test_generation_2_robustness(results)
    test_generation_3_optimization(results)
    test_integration_scenarios(results)
    test_performance_benchmarks(results)
    
    # Generate comprehensive report
    print("\n" + "=" * 70)
    print("🏆 COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    
    summary = results.summary()
    
    print(f"Total Tests Run: {summary['total_tests']}")
    print(f"Tests Passed: {summary['passed']}")
    print(f"Tests Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']}")
    
    if summary['failed'] > 0:
        print(f"\n❌ FAILURES ({summary['failed']}):")
        for failure in summary['failures']:
            print(f"  • {failure}")
    
    # Performance metrics summary
    if summary['performance_metrics']:
        print(f"\n📊 PERFORMANCE METRICS:")
        for test_name, metrics in summary['performance_metrics'].items():
            print(f"  • {test_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    print(f"    - {metric_name}: {value:.3f}")
                else:
                    print(f"    - {metric_name}: {value}")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    if summary['passed'] == summary['total_tests']:
        print("🎉 EXCELLENT! All tests passed - Framework is production-ready")
        grade = "A+"
    elif summary['failed'] <= 2:
        print("✅ VERY GOOD! Minor issues detected - Framework is nearly production-ready")
        grade = "A"
    elif summary['failed'] <= 5:
        print("⚠️ GOOD! Some issues detected - Framework needs minor fixes")
        grade = "B+"
    else:
        print("❌ NEEDS WORK! Multiple issues detected - Framework needs significant fixes")
        grade = "C"
    
    print(f"Framework Grade: {grade}")
    print(f"Production Readiness: {float(summary['success_rate'].rstrip('%')):.1f}%")
    
    # Save detailed report
    report_file = "comprehensive_test_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'summary': summary,
            'grade': grade,
            'production_readiness_percent': float(summary['success_rate'].rstrip('%'))
        }, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return summary['failed'] == 0


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)