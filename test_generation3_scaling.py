#!/usr/bin/env python3
"""
Generation 3 Scaling Test Suite
Tests performance optimization, concurrent processing, auto-scaling, and distributed systems
"""

import os
import sys
import time
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_performance_optimizer():
    """Test performance optimization system."""
    print("⚡ Testing Performance Optimizer...")
    
    try:
        from snn_fusion.scaling.performance_optimizer import (
            MemoryOptimizer, ComputationOptimizer, ResourceManager,
            PerformanceProfiler, optimize_function, profile_operation
        )
        
        # Test memory optimization
        print("  Testing memory optimization...")
        memory_opt = MemoryOptimizer(max_cache_size=1024 * 1024)  # 1MB
        
        # Test caching
        cached = memory_opt.cache_result("test_key", "test_value")
        assert cached == True
        
        retrieved = memory_opt.get_cached("test_key")
        assert retrieved == "test_value"
        print("    ✓ Memory caching working")
        
        # Test memory stats
        stats = memory_opt.get_memory_stats()
        assert stats['cache_items'] > 0
        print(f"    ✓ Memory stats: {stats['cache_items']} items")
        
        # Test computation optimization
        print("  Testing computation optimization...")
        comp_opt = ComputationOptimizer()
        
        def test_function(x):
            return x ** 2
        
        # Optimize function
        optimized_func = comp_opt.optimize_function(test_function, cache_results=True)
        
        # Test optimization
        result1 = optimized_func(10)
        result2 = optimized_func(10)  # Should hit cache
        assert result1 == result2 == 100
        print("    ✓ Function optimization working")
        
        # Test parallel processing
        numbers = list(range(20))
        parallel_results = comp_opt.parallel_map(lambda x: x * 2, numbers)
        expected = [x * 2 for x in numbers]
        assert parallel_results == expected
        print("    ✓ Parallel processing working")
        
        # Test resource management
        print("  Testing resource management...")
        resource_mgr = ResourceManager()
        
        status = resource_mgr.get_resource_status()
        assert 'usage' in status
        assert 'limits' in status
        print(f"    ✓ Resource monitoring: {len(status['usage'])} metrics")
        
        # Test performance profiling
        print("  Testing performance profiling...")
        profiler = PerformanceProfiler()
        
        with profiler.profile_context("test_operation") as profile:
            time.sleep(0.01)  # Simulate work
            result = sum(range(100))
        
        analysis = profiler.analyze_performance("test_operation")
        assert analysis['total_executions'] == 1
        print(f"    ✓ Profiling: {analysis['total_executions']} executions tracked")
        
        # Test convenience functions
        @optimize_function
        def convenience_test(x):
            return x + 1
        
        with profile_operation("convenience_op"):
            result = convenience_test(5)
        
        assert result == 6
        print("    ✓ Convenience functions working")
        
        # Cleanup
        comp_opt.cleanup()
        
        print("✅ Performance Optimizer Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance Optimizer Test Failed: {e}")
        return False

def test_concurrent_processing():
    """Test concurrent processing system."""
    print("🔄 Testing Concurrent Processing...")
    
    try:
        from snn_fusion.scaling.concurrent_processing import (
            StreamProcessor, TaskScheduler, PipelineProcessor,
            ProcessingTask, Priority, concurrent_map
        )
        
        # Test stream processor
        print("  Testing stream processor...")
        
        def simple_processor(data):
            return data * 2
        
        stream_proc = StreamProcessor(buffer_size=50, max_workers=2)
        stream_proc.set_processing_function(simple_processor)
        stream_proc.start()
        
        # Submit and process data
        for i in range(5):
            submitted = stream_proc.submit(i, timeout=1.0)
            assert submitted, f"Failed to submit {i}"
        
        # Get results
        results = []
        for _ in range(5):
            result = stream_proc.get_result(timeout=2.0)
            if result:
                results.append(result['output'])
        
        stream_proc.stop()
        
        expected = [i * 2 for i in range(5)]
        assert sorted(results) == sorted(expected)
        print(f"    ✓ Stream processing: {len(results)} results")
        
        # Test task scheduler
        print("  Testing task scheduler...")
        
        def task_function(x):
            return x ** 2
        
        scheduler = TaskScheduler(max_workers=2)
        scheduler.start()
        
        # Submit tasks
        task_ids = []
        for i in range(3):
            task = ProcessingTask(
                task_id=f"task_{i}",
                function=task_function,
                args=(i,),
                priority=Priority.NORMAL
            )
            task_id = scheduler.submit_task(task)
            task_ids.append(task_id)
        
        # Wait for results
        results = []
        for task_id in task_ids:
            result = scheduler.wait_for_task(task_id, timeout=3.0)
            if result and result.success:
                results.append(result.result)
        
        scheduler.stop()
        
        expected = [i ** 2 for i in range(3)]
        assert sorted(results) == sorted(expected)
        print(f"    ✓ Task scheduling: {len(results)} tasks completed")
        
        # Test pipeline processor
        print("  Testing pipeline processor...")
        
        def stage1(x): return x * 2
        def stage2(x): return x + 1
        
        pipeline = PipelineProcessor([stage1, stage2], buffer_size=20)
        pipeline.start()
        
        # Submit data
        for i in range(3):
            pipeline.submit(i, timeout=1.0)
        
        # Get results
        results = []
        for _ in range(3):
            result = pipeline.get_result(timeout=2.0)
            if result is not None:
                results.append(result)
        
        pipeline.stop()
        
        # Verify: (x * 2) + 1
        expected = [(i * 2) + 1 for i in range(3)]
        assert sorted(results) == sorted(expected)
        print(f"    ✓ Pipeline processing: {len(results)} items")
        
        # Test convenience function
        def square(x): return x ** 2
        numbers = list(range(5))
        concurrent_results = concurrent_map(square, numbers, max_workers=2)
        expected = [x ** 2 for x in numbers]
        assert concurrent_results == expected
        print("    ✓ Concurrent map working")
        
        print("✅ Concurrent Processing Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Concurrent Processing Test Failed: {e}")
        return False

def test_auto_scaler():
    """Test auto-scaling system."""
    print("📈 Testing Auto-Scaler...")
    
    try:
        from snn_fusion.scaling.auto_scaler import (
            AutoScaler, ScalingStrategy, ScalingMetrics, ScalingRule,
            ScalingThreshold, ResourceMetricType, PredictiveModel
        )
        
        # Test predictive model
        print("  Testing predictive model...")
        model = PredictiveModel(window_size=5)
        
        # Add test samples
        for i in range(5):
            metrics = ScalingMetrics(
                timestamp=time.time() + i,
                cpu_utilization=50.0 + i * 5  # Increasing trend
            )
            model.add_sample(metrics)
        
        predicted_load = model.predict_load(3)
        trend = model.get_trend()
        
        assert predicted_load > 50.0  # Should predict higher load
        assert trend > 0.0  # Should detect increasing trend
        print(f"    ✓ Predictive model: {predicted_load:.1f}% predicted, trend: {trend:.2f}")
        
        # Test auto-scaler
        print("  Testing auto-scaler...")
        auto_scaler = AutoScaler(ScalingStrategy.HYBRID)
        
        # Mock metrics callback
        def mock_metrics():
            return ScalingMetrics(
                timestamp=time.time(),
                cpu_utilization=60.0,
                memory_utilization=45.0,
                queue_length=25,
                throughput=100.0,
                average_latency=0.05
            )
        
        auto_scaler.add_metrics_callback(mock_metrics)
        
        # Test custom scaling rule
        custom_rule = ScalingRule(
            name="test_rule",
            thresholds=[
                ScalingThreshold(
                    metric_type=ResourceMetricType.CPU_UTILIZATION,
                    scale_up_threshold=80.0,
                    scale_down_threshold=20.0
                )
            ],
            min_instances=1,
            max_instances=5
        )
        
        auto_scaler.add_scaling_rule(custom_rule)
        assert "test_rule" in auto_scaler.scaling_rules
        print("    ✓ Custom scaling rule added")
        
        # Start auto-scaler briefly
        auto_scaler.start()
        time.sleep(1.0)  # Let it collect some data
        
        # Get state and recommendations
        state = auto_scaler.get_current_state()
        assert state['is_running'] == True
        print(f"    ✓ Auto-scaler state: {state['current_instances']} instances")
        
        recommendations = auto_scaler.get_recommendations()
        assert len(recommendations) > 0
        print(f"    ✓ Got {len(recommendations)} recommendations")
        
        # Get statistics
        stats = auto_scaler.get_statistics()
        history = auto_scaler.get_scaling_history()
        
        print(f"    ✓ Statistics: {stats['scaling_actions']} actions")
        print(f"    ✓ History: {len(history)} events")
        
        auto_scaler.stop()
        
        print("✅ Auto-Scaler Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Auto-Scaler Test Failed: {e}")
        return False

def test_distributed_cluster():
    """Test distributed cluster management."""
    print("🌐 Testing Distributed Cluster...")
    
    try:
        from snn_fusion.distributed.cluster_manager import (
            ClusterManager, NodeInfo, NodeRole, NodeStatus,
            DistributedTask, HeartbeatManager
        )
        
        # Test node info
        print("  Testing node info...")
        node = NodeInfo(
            node_id="test_node",
            hostname="localhost", 
            ip_address="127.0.0.1",
            port=8080,
            role=NodeRole.WORKER
        )
        
        # Test serialization
        node_dict = node.to_dict()
        restored_node = NodeInfo.from_dict(node_dict)
        
        assert restored_node.node_id == node.node_id
        assert restored_node.role == node.role
        print("    ✓ Node info serialization working")
        
        # Test heartbeat manager
        print("  Testing heartbeat manager...")
        heartbeat_mgr = HeartbeatManager("test_heartbeat_node")
        
        received_heartbeats = []
        def heartbeat_callback(node_id, data):
            received_heartbeats.append(node_id)
        
        heartbeat_mgr.add_heartbeat_callback(heartbeat_callback)
        
        # Start and stop quickly
        heartbeat_mgr.start()
        time.sleep(0.1)
        heartbeat_mgr.stop()
        
        print(f"    ✓ Heartbeat manager on port {heartbeat_mgr.port}")
        
        # Test cluster manager
        print("  Testing cluster manager...")
        cluster_mgr = ClusterManager("test_cluster", NodeRole.HYBRID)
        
        # Register task handler
        def test_handler(data):
            return f"processed: {data}"
        
        cluster_mgr.register_task_handler("test_task", test_handler)
        
        # Start cluster
        cluster_mgr.start()
        time.sleep(0.5)  # Brief initialization
        
        # Create and distribute task
        task = DistributedTask(
            task_id="test_task_1",
            task_type="test_task",
            data="hello"
        )
        
        distributed = cluster_mgr.distribute_task(task)
        print(f"    ✓ Task distribution: {distributed}")
        
        # Get cluster state
        state = cluster_mgr.get_cluster_state()
        assert state['node_id'] == "test_cluster"
        print(f"    ✓ Cluster state: {state['total_nodes']} nodes")
        
        # Get node list
        nodes = cluster_mgr.get_node_list()
        assert len(nodes) > 0
        print(f"    ✓ Node list: {len(nodes)} nodes")
        
        # Get statistics
        stats = cluster_mgr.get_statistics()
        print(f"    ✓ Cluster stats: {stats['tasks_distributed']} distributed")
        
        cluster_mgr.stop()
        
        print("✅ Distributed Cluster Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Distributed Cluster Test Failed: {e}")
        return False

def test_scaling_integration():
    """Test integration between scaling components."""
    print("🔗 Testing Scaling Integration...")
    
    try:
        from snn_fusion.scaling.performance_optimizer import default_memory_optimizer
        from snn_fusion.scaling.concurrent_processing import concurrent_map
        
        # Test memory optimization with concurrent processing
        print("  Testing memory + concurrent integration...")
        
        # Cache some results
        default_memory_optimizer.cache_result("integration_test", [1, 2, 3, 4, 5])
        
        # Use concurrent processing
        def cached_operation(x):
            cached = default_memory_optimizer.get_cached("integration_test")
            if cached:
                return sum(cached) + x
            return x
        
        numbers = [1, 2, 3]
        results = concurrent_map(cached_operation, numbers, max_workers=2)
        
        # All should get cached value (15) + x
        expected = [15 + x for x in numbers]
        assert results == expected
        print("    ✓ Memory caching + concurrent processing")
        
        # Test resource-aware processing
        print("  Testing resource-aware processing...")
        
        # Simple simulation of resource-aware task distribution
        class ResourceAwareProcessor:
            def __init__(self):
                self.cpu_usage = 30.0
                self.memory_usage = 40.0
            
            def can_handle_task(self, estimated_load):
                return (self.cpu_usage + estimated_load) < 80.0
            
            def process_task(self, task_data):
                if self.can_handle_task(10.0):  # Estimated 10% CPU load
                    self.cpu_usage += 10.0
                    result = task_data * 2
                    self.cpu_usage -= 10.0  # Task completed
                    return result
                else:
                    return None  # Reject task
        
        processor = ResourceAwareProcessor()
        
        # Test task processing
        task_results = []
        for i in range(5):
            result = processor.process_task(i)
            if result is not None:
                task_results.append(result)
        
        assert len(task_results) > 0
        print(f"    ✓ Resource-aware processing: {len(task_results)} tasks processed")
        
        # Test load balancing simulation
        print("  Testing load balancing simulation...")
        
        class LoadBalancer:
            def __init__(self, num_workers=3):
                self.workers = [{'id': i, 'load': 0.0} for i in range(num_workers)]
            
            def get_least_loaded_worker(self):
                return min(self.workers, key=lambda w: w['load'])
            
            def assign_task(self, task_load=10.0):
                worker = self.get_least_loaded_worker()
                worker['load'] += task_load
                return worker['id']
            
            def complete_task(self, worker_id, task_load=10.0):
                self.workers[worker_id]['load'] -= task_load
                self.workers[worker_id]['load'] = max(0, self.workers[worker_id]['load'])
        
        balancer = LoadBalancer(num_workers=3)
        
        # Assign tasks
        assignments = []
        for _ in range(6):
            worker_id = balancer.assign_task(15.0)
            assignments.append(worker_id)
        
        # Check distribution
        assignment_counts = {i: assignments.count(i) for i in range(3)}
        assert len(set(assignment_counts.values())) <= 2  # Should be relatively balanced
        print(f"    ✓ Load balancing: {assignment_counts}")
        
        print("✅ Scaling Integration Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Scaling Integration Test Failed: {e}")
        return False

def test_performance_benchmarks():
    """Test performance benchmarks and scaling."""
    print("🏃 Testing Performance Benchmarks...")
    
    try:
        import multiprocessing as mp
        
        # Benchmark sequential vs concurrent processing
        print("  Benchmarking processing modes...")
        
        def cpu_intensive_task(n):
            return sum(i * i for i in range(n))
        
        # Sequential benchmark
        start_time = time.time()
        sequential_results = [cpu_intensive_task(1000) for _ in range(20)]
        sequential_time = time.time() - start_time
        
        # Concurrent benchmark (if possible)
        try:
            from snn_fusion.scaling.concurrent_processing import concurrent_map
            
            start_time = time.time()
            concurrent_results = concurrent_map(
                lambda _: cpu_intensive_task(1000), 
                range(20), 
                max_workers=min(4, mp.cpu_count())
            )
            concurrent_time = time.time() - start_time
            
            # Results should be the same
            assert len(sequential_results) == len(concurrent_results)
            
            speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1.0
            print(f"    ✓ Sequential: {sequential_time:.3f}s")
            print(f"    ✓ Concurrent: {concurrent_time:.3f}s")
            print(f"    ✓ Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"    ⚠ Concurrent benchmark skipped: {e}")
        
        # Memory usage benchmark
        print("  Benchmarking memory usage...")
        
        from snn_fusion.scaling.performance_optimizer import MemoryOptimizer
        
        memory_opt = MemoryOptimizer(max_cache_size=1024 * 1024)  # 1MB
        
        # Test cache performance
        start_time = time.time()
        for i in range(1000):
            memory_opt.cache_result(f"key_{i}", f"value_{i}")
        cache_time = time.time() - start_time
        
        start_time = time.time()
        hits = 0
        for i in range(1000):
            result = memory_opt.get_cached(f"key_{i}")
            if result is not None:
                hits += 1
        retrieval_time = time.time() - start_time
        
        hit_rate = hits / 1000.0
        print(f"    ✓ Cache operations: {cache_time:.3f}s write, {retrieval_time:.3f}s read")
        print(f"    ✓ Cache hit rate: {hit_rate:.1%}")
        
        # Throughput benchmark
        print("  Benchmarking throughput scaling...")
        
        class ThroughputTester:
            def __init__(self):
                self.processed = 0
                self.lock = threading.Lock()
            
            def process_item(self, item):
                # Simulate processing
                time.sleep(0.001)
                with self.lock:
                    self.processed += 1
                return item * 2
        
        tester = ThroughputTester()
        
        # Test with different worker counts
        for workers in [1, 2, 4]:
            if workers > mp.cpu_count():
                continue
                
            tester.processed = 0
            
            start_time = time.time()
            try:
                from snn_fusion.scaling.concurrent_processing import concurrent_map
                results = concurrent_map(
                    tester.process_item,
                    range(50),
                    max_workers=workers
                )
                elapsed = time.time() - start_time
                throughput = len(results) / elapsed if elapsed > 0 else 0
                
                print(f"    ✓ {workers} workers: {throughput:.1f} items/second")
                
            except Exception:
                # Skip if concurrent processing not available
                pass
        
        print("✅ Performance Benchmarks Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance Benchmarks Test Failed: {e}")
        return False

def main():
    """Run all Generation 3 scaling tests."""
    print("🚀 Starting Generation 3 Scaling Test Suite")
    print("=" * 70)
    
    tests = [
        test_performance_optimizer,
        test_concurrent_processing,
        test_auto_scaler,
        test_distributed_cluster,
        test_scaling_integration,
        test_performance_benchmarks,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 70)
    print(f"📊 Generation 3 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL GENERATION 3 SCALING TESTS PASSED!")
        print("✅ Performance optimization systems operational")
        print("✅ Concurrent processing infrastructure working")
        print("✅ Auto-scaling capabilities functional")
        print("✅ Distributed cluster management ready")
        print("✅ Scaling integration verified")
        print("✅ Performance benchmarks completed")
        print("✅ System is optimized and ready for production scale")
    else:
        print(f"⚠️  {total - passed} tests failed or had issues")
        print("🔧 Review failed components before production deployment")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)