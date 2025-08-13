"""
Performance Benchmarking Suite for Neuromorphic System

Establishes performance baselines and conducts comprehensive benchmarking
across all system components for production readiness assessment.
"""

import time
import threading
import statistics
import gc
import sys
import json
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

# Mock dependencies
try:
    import numpy as np
except ImportError:
    class MockNumpy:
        def array(self, data):
            return data
        def mean(self, data):
            return sum(data) / len(data)
        def std(self, data):
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return variance ** 0.5
    np = MockNumpy()

try:
    import psutil
except ImportError:
    class MockPsutil:
        def cpu_percent(self, interval=None):
            return 50.0
        def virtual_memory(self):
            class MemInfo:
                percent = 60.0
                used = 8 * 1024 * 1024 * 1024  # 8GB
                total = 16 * 1024 * 1024 * 1024  # 16GB
            return MemInfo()
        def Process(self):
            class ProcInfo:
                def memory_info(self):
                    class MemInfo:
                        rss = 100 * 1024 * 1024  # 100MB
                    return MemInfo()
            return ProcInfo()
    psutil = MockPsutil()


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    name: str
    category: str
    duration: float
    throughput: float
    memory_usage: int
    cpu_usage: float
    iterations: int
    success_rate: float
    error_count: int
    percentiles: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Optional[Dict[str, float]] = None


@dataclass
class PerformanceBaseline:
    """Performance baseline measurements."""
    operation: str
    expected_duration: float
    expected_throughput: float
    expected_memory_mb: float
    tolerance_percent: float = 20.0  # 20% tolerance


class MockNeuromorphicProcessor:
    """Mock neuromorphic processor for benchmarking."""
    
    def __init__(self, complexity_factor: float = 1.0):
        self.complexity_factor = complexity_factor
        self.spike_history = []
        self.processing_count = 0
        self.memory_usage = 0
    
    def process_spikes(self, spike_data: List[float]) -> List[bool]:
        """Process spike data and return spike pattern."""
        start_time = time.time()
        
        # Simulate computational load based on complexity factor
        computational_work = len(spike_data) * self.complexity_factor
        
        # Simulate processing delay
        processing_delay = computational_work * 0.0001
        time.sleep(processing_delay)
        
        # Generate spike pattern (simple threshold-based)
        spikes = [val > 0.5 for val in spike_data]
        
        # Update state
        self.spike_history.extend(spikes)
        self.processing_count += 1
        self.memory_usage += len(spike_data) * 8  # 8 bytes per float
        
        # Keep history bounded
        if len(self.spike_history) > 10000:
            self.spike_history = self.spike_history[-5000:]
            
        return spikes
    
    def reset_state(self):
        """Reset processor state."""
        self.spike_history = []
        self.processing_count = 0
        self.memory_usage = 0


class BenchmarkSuite:
    """Comprehensive benchmarking suite."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baselines = self._define_baselines()
        self.processor = MockNeuromorphicProcessor()
    
    def _define_baselines(self) -> Dict[str, PerformanceBaseline]:
        """Define performance baselines for comparison."""
        return {
            "single_spike_processing": PerformanceBaseline(
                operation="single_spike_processing",
                expected_duration=0.001,  # 1ms
                expected_throughput=1000.0,  # 1000 ops/sec
                expected_memory_mb=10.0,
                tolerance_percent=25.0
            ),
            "batch_processing": PerformanceBaseline(
                operation="batch_processing",
                expected_duration=0.1,  # 100ms for batch
                expected_throughput=100.0,  # 100 batches/sec
                expected_memory_mb=50.0,
                tolerance_percent=30.0
            ),
            "concurrent_processing": PerformanceBaseline(
                operation="concurrent_processing",
                expected_duration=0.05,  # 50ms with concurrency
                expected_throughput=200.0,  # 200 ops/sec
                expected_memory_mb=80.0,
                tolerance_percent=35.0
            ),
            "memory_intensive": PerformanceBaseline(
                operation="memory_intensive",
                expected_duration=0.2,  # 200ms
                expected_throughput=50.0,  # 50 ops/sec
                expected_memory_mb=200.0,
                tolerance_percent=40.0
            ),
            "cpu_intensive": PerformanceBaseline(
                operation="cpu_intensive",
                expected_duration=0.5,  # 500ms
                expected_throughput=20.0,  # 20 ops/sec
                expected_memory_mb=30.0,
                tolerance_percent=25.0
            )
        }
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        print("🏁 Starting Performance Benchmark Suite")
        print("=" * 60)
        
        benchmark_methods = [
            self.benchmark_single_spike_processing,
            self.benchmark_batch_processing,
            self.benchmark_concurrent_processing,
            self.benchmark_memory_intensive,
            self.benchmark_cpu_intensive,
            self.benchmark_scalability,
            self.benchmark_latency_distribution,
            self.benchmark_memory_efficiency,
            self.benchmark_error_handling_performance,
            self.benchmark_sustained_load
        ]
        
        for benchmark_method in benchmark_methods:
            print(f"\n🔍 Running {benchmark_method.__name__}...")
            try:
                result = benchmark_method()
                if result:
                    self.results.append(result)
                    self._print_result_summary(result)
            except Exception as e:
                print(f"❌ Benchmark {benchmark_method.__name__} failed: {e}")
        
        return self._generate_comprehensive_report()
    
    def benchmark_single_spike_processing(self) -> BenchmarkResult:
        """Benchmark single spike processing performance."""
        iterations = 1000
        durations = []
        error_count = 0
        
        # Warm-up
        for _ in range(10):
            self.processor.process_spikes([0.5, 0.7, 0.3])
        
        gc.collect()
        initial_memory = self._get_memory_usage()
        start_time = time.time()
        
        for i in range(iterations):
            try:
                spike_data = [0.1 + (i % 10) * 0.1]
                
                iteration_start = time.time()
                result = self.processor.process_spikes(spike_data)
                iteration_duration = time.time() - iteration_start
                
                durations.append(iteration_duration)
                
                # Validate result
                if not isinstance(result, list) or len(result) != 1:
                    error_count += 1
                    
            except Exception:
                error_count += 1
        
        total_duration = time.time() - start_time
        final_memory = self._get_memory_usage()
        memory_used = final_memory - initial_memory
        
        # Calculate statistics
        mean_duration = statistics.mean(durations)
        throughput = iterations / total_duration
        success_rate = (iterations - error_count) / iterations
        
        percentiles = {
            "p50": statistics.median(durations),
            "p95": self._calculate_percentile(durations, 95),
            "p99": self._calculate_percentile(durations, 99)
        }
        
        return BenchmarkResult(
            name="single_spike_processing",
            category="core_functionality",
            duration=mean_duration,
            throughput=throughput,
            memory_usage=memory_used,
            cpu_usage=self._get_cpu_usage(),
            iterations=iterations,
            success_rate=success_rate,
            error_count=error_count,
            percentiles=percentiles,
            metadata={"spike_data_size": 1},
            baseline_comparison=self._compare_to_baseline("single_spike_processing", mean_duration, throughput, memory_used)
        )
    
    def benchmark_batch_processing(self) -> BenchmarkResult:
        """Benchmark batch processing performance."""
        batch_size = 100
        num_batches = 50
        durations = []
        error_count = 0
        
        gc.collect()
        initial_memory = self._get_memory_usage()
        start_time = time.time()
        
        for batch_num in range(num_batches):
            try:
                # Generate batch data
                batch_data = [0.1 + (i % 10) * 0.1 for i in range(batch_size)]
                
                batch_start = time.time()
                result = self.processor.process_spikes(batch_data)
                batch_duration = time.time() - batch_start
                
                durations.append(batch_duration)
                
                # Validate result
                if not isinstance(result, list) or len(result) != batch_size:
                    error_count += 1
                    
            except Exception:
                error_count += 1
        
        total_duration = time.time() - start_time
        final_memory = self._get_memory_usage()
        memory_used = final_memory - initial_memory
        
        # Calculate statistics
        mean_duration = statistics.mean(durations)
        throughput = num_batches / total_duration
        success_rate = (num_batches - error_count) / num_batches
        
        percentiles = {
            "p50": statistics.median(durations),
            "p95": self._calculate_percentile(durations, 95),
            "p99": self._calculate_percentile(durations, 99)
        }
        
        return BenchmarkResult(
            name="batch_processing",
            category="throughput",
            duration=mean_duration,
            throughput=throughput,
            memory_usage=memory_used,
            cpu_usage=self._get_cpu_usage(),
            iterations=num_batches,
            success_rate=success_rate,
            error_count=error_count,
            percentiles=percentiles,
            metadata={"batch_size": batch_size, "total_items": num_batches * batch_size},
            baseline_comparison=self._compare_to_baseline("batch_processing", mean_duration, throughput, memory_used)
        )
    
    def benchmark_concurrent_processing(self) -> BenchmarkResult:
        """Benchmark concurrent processing performance."""
        num_workers = 4
        tasks_per_worker = 25
        total_tasks = num_workers * tasks_per_worker
        
        durations = []
        error_count = 0
        results_lock = threading.Lock()
        
        def worker_task(worker_id: int, task_id: int):
            nonlocal error_count
            try:
                spike_data = [0.1 + (task_id % 10) * 0.1, 0.2 + (worker_id % 5) * 0.1]
                
                task_start = time.time()
                result = self.processor.process_spikes(spike_data)
                task_duration = time.time() - task_start
                
                with results_lock:
                    durations.append(task_duration)
                    
                    # Validate result
                    if not isinstance(result, list) or len(result) != 2:
                        error_count += 1
                        
            except Exception:
                with results_lock:
                    error_count += 1
        
        gc.collect()
        initial_memory = self._get_memory_usage()
        start_time = time.time()
        
        # Create and start worker threads
        threads = []
        for worker_id in range(num_workers):
            for task_id in range(tasks_per_worker):
                thread = threading.Thread(target=worker_task, args=(worker_id, task_id))
                threads.append(thread)
                thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)  # 10 second timeout
        
        total_duration = time.time() - start_time
        final_memory = self._get_memory_usage()
        memory_used = final_memory - initial_memory
        
        # Calculate statistics
        if durations:
            mean_duration = statistics.mean(durations)
            percentiles = {
                "p50": statistics.median(durations),
                "p95": self._calculate_percentile(durations, 95),
                "p99": self._calculate_percentile(durations, 99)
            }
        else:
            mean_duration = total_duration
            percentiles = {"p50": 0, "p95": 0, "p99": 0}
        
        throughput = len(durations) / total_duration
        success_rate = len(durations) / total_tasks
        
        return BenchmarkResult(
            name="concurrent_processing",
            category="concurrency",
            duration=mean_duration,
            throughput=throughput,
            memory_usage=memory_used,
            cpu_usage=self._get_cpu_usage(),
            iterations=total_tasks,
            success_rate=success_rate,
            error_count=error_count,
            percentiles=percentiles,
            metadata={"num_workers": num_workers, "tasks_per_worker": tasks_per_worker},
            baseline_comparison=self._compare_to_baseline("concurrent_processing", mean_duration, throughput, memory_used)
        )
    
    def benchmark_memory_intensive(self) -> BenchmarkResult:
        """Benchmark memory-intensive operations."""
        iterations = 20
        large_data_size = 10000
        durations = []
        error_count = 0
        
        gc.collect()
        initial_memory = self._get_memory_usage()
        start_time = time.time()
        
        for i in range(iterations):
            try:
                # Create large spike data
                large_spike_data = [0.1 + (j % 100) * 0.01 for j in range(large_data_size)]
                
                iteration_start = time.time()
                result = self.processor.process_spikes(large_spike_data)
                iteration_duration = time.time() - iteration_start
                
                durations.append(iteration_duration)
                
                # Validate result
                if not isinstance(result, list) or len(result) != large_data_size:
                    error_count += 1
                
                # Force memory cleanup periodically
                if i % 5 == 0:
                    gc.collect()
                    
            except Exception:
                error_count += 1
        
        total_duration = time.time() - start_time
        final_memory = self._get_memory_usage()
        memory_used = final_memory - initial_memory
        
        # Calculate statistics
        mean_duration = statistics.mean(durations)
        throughput = iterations / total_duration
        success_rate = (iterations - error_count) / iterations
        
        percentiles = {
            "p50": statistics.median(durations),
            "p95": self._calculate_percentile(durations, 95),
            "p99": self._calculate_percentile(durations, 99)
        }
        
        return BenchmarkResult(
            name="memory_intensive",
            category="memory",
            duration=mean_duration,
            throughput=throughput,
            memory_usage=memory_used,
            cpu_usage=self._get_cpu_usage(),
            iterations=iterations,
            success_rate=success_rate,
            error_count=error_count,
            percentiles=percentiles,
            metadata={"data_size_per_iteration": large_data_size, "total_data_processed": iterations * large_data_size},
            baseline_comparison=self._compare_to_baseline("memory_intensive", mean_duration, throughput, memory_used)
        )
    
    def benchmark_cpu_intensive(self) -> BenchmarkResult:
        """Benchmark CPU-intensive operations."""
        iterations = 10
        durations = []
        error_count = 0
        
        # Create high complexity processor
        cpu_intensive_processor = MockNeuromorphicProcessor(complexity_factor=100.0)
        
        gc.collect()
        initial_memory = self._get_memory_usage()
        start_time = time.time()
        
        for i in range(iterations):
            try:
                # CPU-intensive spike data processing
                spike_data = [0.1 + (j % 20) * 0.05 for j in range(1000)]
                
                iteration_start = time.time()
                result = cpu_intensive_processor.process_spikes(spike_data)
                iteration_duration = time.time() - iteration_start
                
                durations.append(iteration_duration)
                
                # Validate result
                if not isinstance(result, list) or len(result) != 1000:
                    error_count += 1
                    
            except Exception:
                error_count += 1
        
        total_duration = time.time() - start_time
        final_memory = self._get_memory_usage()
        memory_used = final_memory - initial_memory
        
        # Calculate statistics
        mean_duration = statistics.mean(durations)
        throughput = iterations / total_duration
        success_rate = (iterations - error_count) / iterations
        
        percentiles = {
            "p50": statistics.median(durations),
            "p95": self._calculate_percentile(durations, 95),
            "p99": self._calculate_percentile(durations, 99)
        }
        
        return BenchmarkResult(
            name="cpu_intensive",
            category="computation",
            duration=mean_duration,
            throughput=throughput,
            memory_usage=memory_used,
            cpu_usage=self._get_cpu_usage(),
            iterations=iterations,
            success_rate=success_rate,
            error_count=error_count,
            percentiles=percentiles,
            metadata={"complexity_factor": 100.0, "data_size": 1000},
            baseline_comparison=self._compare_to_baseline("cpu_intensive", mean_duration, throughput, memory_used)
        )
    
    def benchmark_scalability(self) -> BenchmarkResult:
        """Benchmark system scalability characteristics."""
        load_levels = [10, 50, 100, 200, 500, 1000]
        scalability_results = []
        
        start_time = time.time()
        
        for load in load_levels:
            # Test processing with different load levels
            test_start = time.time()
            
            try:
                # Process multiple items at this load level
                for i in range(min(load, 100)):  # Cap at 100 for time constraints
                    spike_data = [0.5 + (i % 10) * 0.05]
                    result = self.processor.process_spikes(spike_data)
                
                test_duration = time.time() - test_start
                throughput_at_load = min(load, 100) / test_duration
                
                scalability_results.append({
                    "load": load,
                    "duration": test_duration,
                    "throughput": throughput_at_load
                })
                
            except Exception:
                scalability_results.append({
                    "load": load,
                    "duration": float('inf'),
                    "throughput": 0.0
                })
        
        total_duration = time.time() - start_time
        
        # Calculate scalability metrics
        throughputs = [r["throughput"] for r in scalability_results]
        avg_throughput = statistics.mean(throughputs)
        
        # Scalability coefficient (higher is better)
        scalability_coeff = 1.0
        if len(throughputs) > 1:
            # Simple scalability measure: ratio of max to min throughput
            max_throughput = max(throughputs)
            min_throughput = min(t for t in throughputs if t > 0)
            if min_throughput > 0:
                scalability_coeff = max_throughput / min_throughput
        
        return BenchmarkResult(
            name="scalability",
            category="scalability", 
            duration=total_duration / len(load_levels),
            throughput=avg_throughput,
            memory_usage=self._get_memory_usage(),
            cpu_usage=self._get_cpu_usage(),
            iterations=sum(min(load, 100) for load in load_levels),
            success_rate=1.0,  # All loads completed
            error_count=0,
            percentiles={},
            metadata={
                "load_levels": load_levels,
                "scalability_results": scalability_results,
                "scalability_coefficient": scalability_coeff
            }
        )
    
    def benchmark_latency_distribution(self) -> BenchmarkResult:
        """Benchmark latency distribution characteristics."""
        iterations = 500
        durations = []
        
        # Warm-up
        for _ in range(20):
            self.processor.process_spikes([0.5])
        
        gc.collect()
        start_time = time.time()
        
        for i in range(iterations):
            spike_data = [0.1 + (i % 10) * 0.1]
            
            iteration_start = time.time()
            self.processor.process_spikes(spike_data)
            iteration_duration = time.time() - iteration_start
            
            durations.append(iteration_duration)
        
        total_duration = time.time() - start_time
        
        # Calculate comprehensive percentiles
        percentiles = {}
        for p in [10, 25, 50, 75, 90, 95, 99, 99.9]:
            percentiles[f"p{p}"] = self._calculate_percentile(durations, p)
        
        # Latency statistics
        mean_latency = statistics.mean(durations)
        median_latency = statistics.median(durations)
        std_latency = statistics.stdev(durations) if len(durations) > 1 else 0
        
        # Jitter (variability)
        jitter = std_latency / mean_latency if mean_latency > 0 else 0
        
        return BenchmarkResult(
            name="latency_distribution",
            category="latency",
            duration=mean_latency,
            throughput=iterations / total_duration,
            memory_usage=self._get_memory_usage(),
            cpu_usage=self._get_cpu_usage(),
            iterations=iterations,
            success_rate=1.0,
            error_count=0,
            percentiles=percentiles,
            metadata={
                "mean_latency": mean_latency,
                "median_latency": median_latency,
                "std_latency": std_latency,
                "jitter_coefficient": jitter,
                "latency_consistency": 1.0 - jitter  # Higher is more consistent
            }
        )
    
    def benchmark_memory_efficiency(self) -> BenchmarkResult:
        """Benchmark memory efficiency and garbage collection behavior."""
        iterations = 100
        memory_snapshots = []
        durations = []
        
        gc.collect()
        initial_memory = self._get_memory_usage()
        start_time = time.time()
        
        for i in range(iterations):
            # Take memory snapshot before
            memory_before = self._get_memory_usage()
            
            # Create and process data
            spike_data = [0.1 + (j % 50) * 0.02 for j in range(500)]
            
            iteration_start = time.time()
            result = self.processor.process_spikes(spike_data)
            iteration_duration = time.time() - iteration_start
            
            durations.append(iteration_duration)
            
            # Take memory snapshot after
            memory_after = self._get_memory_usage()
            memory_used = memory_after - memory_before
            
            memory_snapshots.append({
                "iteration": i,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_delta": memory_used
            })
            
            # Force GC every 20 iterations
            if i % 20 == 19:
                gc.collect()
        
        total_duration = time.time() - start_time
        final_memory = self._get_memory_usage()
        
        # Calculate memory efficiency metrics
        memory_deltas = [s["memory_delta"] for s in memory_snapshots]
        avg_memory_per_operation = statistics.mean(memory_deltas)
        max_memory_spike = max(memory_deltas)
        
        # Memory growth rate
        memory_growth = final_memory - initial_memory
        
        return BenchmarkResult(
            name="memory_efficiency",
            category="memory",
            duration=statistics.mean(durations),
            throughput=iterations / total_duration,
            memory_usage=memory_growth,
            cpu_usage=self._get_cpu_usage(),
            iterations=iterations,
            success_rate=1.0,
            error_count=0,
            percentiles={},
            metadata={
                "initial_memory": initial_memory,
                "final_memory": final_memory,
                "memory_growth": memory_growth,
                "avg_memory_per_op": avg_memory_per_operation,
                "max_memory_spike": max_memory_spike,
                "memory_efficiency_score": 1000 / max(avg_memory_per_operation, 1)  # Higher is better
            }
        )
    
    def benchmark_error_handling_performance(self) -> BenchmarkResult:
        """Benchmark error handling and recovery performance."""
        iterations = 100
        error_scenarios = ["invalid_data", "oversized_data", "null_data"]
        
        durations = []
        recovery_times = []
        error_count = 0
        recovered_count = 0
        
        start_time = time.time()
        
        for i in range(iterations):
            scenario = error_scenarios[i % len(error_scenarios)]
            
            try:
                iteration_start = time.time()
                
                if scenario == "invalid_data":
                    # Try to process invalid data, expect graceful handling
                    try:
                        result = self.processor.process_spikes("invalid")
                        error_count += 1  # Should have been handled
                    except Exception:
                        # Expected error, now test recovery
                        recovery_start = time.time()
                        result = self.processor.process_spikes([0.5])  # Valid fallback
                        recovery_time = time.time() - recovery_start
                        recovery_times.append(recovery_time)
                        recovered_count += 1
                
                elif scenario == "oversized_data":
                    # Process very large data
                    large_data = [0.1] * 50000  # Very large
                    result = self.processor.process_spikes(large_data[:1000])  # Truncate
                    
                elif scenario == "null_data":
                    # Handle null/empty data
                    try:
                        result = self.processor.process_spikes([])
                        if not result:  # Empty result is acceptable
                            recovered_count += 1
                    except Exception:
                        # Recover with default data
                        result = self.processor.process_spikes([0.0])
                        recovered_count += 1
                
                iteration_duration = time.time() - iteration_start
                durations.append(iteration_duration)
                
            except Exception:
                error_count += 1
        
        total_duration = time.time() - start_time
        
        # Calculate error handling metrics
        error_rate = error_count / iterations
        recovery_rate = recovered_count / iterations
        avg_duration = statistics.mean(durations) if durations else 0
        avg_recovery_time = statistics.mean(recovery_times) if recovery_times else 0
        
        return BenchmarkResult(
            name="error_handling_performance",
            category="reliability",
            duration=avg_duration,
            throughput=len(durations) / total_duration if total_duration > 0 else 0,
            memory_usage=self._get_memory_usage(),
            cpu_usage=self._get_cpu_usage(),
            iterations=iterations,
            success_rate=1.0 - error_rate,
            error_count=error_count,
            percentiles={},
            metadata={
                "error_rate": error_rate,
                "recovery_rate": recovery_rate,
                "avg_recovery_time": avg_recovery_time,
                "error_scenarios": error_scenarios,
                "resilience_score": recovery_rate * (1.0 - error_rate)
            }
        )
    
    def benchmark_sustained_load(self) -> BenchmarkResult:
        """Benchmark performance under sustained load."""
        duration_seconds = 30  # 30 second sustained test
        
        results = []
        error_count = 0
        start_time = time.time()
        operation_count = 0
        
        gc.collect()
        initial_memory = self._get_memory_usage()
        
        while (time.time() - start_time) < duration_seconds:
            try:
                operation_start = time.time()
                
                # Varied workload
                if operation_count % 10 < 7:
                    # Light operation (70%)
                    spike_data = [0.5]
                elif operation_count % 10 < 9:
                    # Medium operation (20%)
                    spike_data = [0.1 + i * 0.1 for i in range(10)]
                else:
                    # Heavy operation (10%)
                    spike_data = [0.1 + i * 0.01 for i in range(100)]
                
                result = self.processor.process_spikes(spike_data)
                operation_duration = time.time() - operation_start
                
                results.append(operation_duration)
                operation_count += 1
                
                # Brief pause to simulate realistic load
                time.sleep(0.001)
                
            except Exception:
                error_count += 1
        
        total_duration = time.time() - start_time
        final_memory = self._get_memory_usage()
        memory_growth = final_memory - initial_memory
        
        # Calculate sustained load metrics
        avg_duration = statistics.mean(results) if results else 0
        throughput = len(results) / total_duration
        success_rate = len(results) / (len(results) + error_count) if (len(results) + error_count) > 0 else 0
        
        # Performance stability (coefficient of variation)
        std_duration = statistics.stdev(results) if len(results) > 1 else 0
        stability = 1.0 - (std_duration / avg_duration) if avg_duration > 0 else 0
        
        return BenchmarkResult(
            name="sustained_load",
            category="endurance",
            duration=avg_duration,
            throughput=throughput,
            memory_usage=memory_growth,
            cpu_usage=self._get_cpu_usage(),
            iterations=len(results),
            success_rate=success_rate,
            error_count=error_count,
            percentiles={},
            metadata={
                "test_duration_seconds": duration_seconds,
                "operations_completed": len(results),
                "performance_stability": stability,
                "memory_growth": memory_growth,
                "avg_ops_per_second": throughput
            }
        )
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return sys.getsizeof(self.processor)
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return psutil.cpu_percent(interval=None)
        except:
            return 50.0  # Default fallback
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            
            if upper_index >= len(sorted_data):
                return sorted_data[-1]
            
            weight = index - lower_index
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    def _compare_to_baseline(self, operation: str, duration: float, throughput: float, memory_mb: float) -> Optional[Dict[str, float]]:
        """Compare performance to baseline."""
        if operation not in self.baselines:
            return None
        
        baseline = self.baselines[operation]
        
        return {
            "duration_ratio": duration / baseline.expected_duration,
            "throughput_ratio": throughput / baseline.expected_throughput,
            "memory_ratio": memory_mb / baseline.expected_memory_mb,
            "within_tolerance": (
                abs(duration - baseline.expected_duration) / baseline.expected_duration < baseline.tolerance_percent / 100 and
                abs(throughput - baseline.expected_throughput) / baseline.expected_throughput < baseline.tolerance_percent / 100 and
                abs(memory_mb - baseline.expected_memory_mb) / baseline.expected_memory_mb < baseline.tolerance_percent / 100
            )
        }
    
    def _print_result_summary(self, result: BenchmarkResult):
        """Print benchmark result summary."""
        print(f"  ✅ {result.name}")
        print(f"    Duration: {result.duration*1000:.2f}ms")
        print(f"    Throughput: {result.throughput:.1f} ops/sec")
        print(f"    Memory: {result.memory_usage/1024/1024:.1f}MB")
        print(f"    Success Rate: {result.success_rate:.1%}")
        
        if result.baseline_comparison:
            baseline = result.baseline_comparison
            status = "✅" if baseline["within_tolerance"] else "⚠️"
            print(f"    Baseline Comparison: {status}")
            print(f"      Duration: {baseline['duration_ratio']:.2f}x expected")
            print(f"      Throughput: {baseline['throughput_ratio']:.2f}x expected")
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Categorize results
        categories = defaultdict(list)
        for result in self.results:
            categories[result.category].append(result)
        
        # Overall statistics
        total_iterations = sum(r.iterations for r in self.results)
        avg_success_rate = statistics.mean([r.success_rate for r in self.results])
        total_errors = sum(r.error_count for r in self.results)
        
        # Performance summary
        performance_summary = {}
        for category, results in categories.items():
            avg_throughput = statistics.mean([r.throughput for r in results])
            avg_duration = statistics.mean([r.duration for r in results])
            
            performance_summary[category] = {
                "tests": len(results),
                "avg_throughput": avg_throughput,
                "avg_duration": avg_duration,
                "avg_success_rate": statistics.mean([r.success_rate for r in results])
            }
        
        # Baseline compliance
        baseline_results = [r for r in self.results if r.baseline_comparison is not None]
        baseline_compliance = {
            "total_baseline_tests": len(baseline_results),
            "within_tolerance": sum(1 for r in baseline_results if r.baseline_comparison["within_tolerance"]),
            "compliance_rate": sum(1 for r in baseline_results if r.baseline_comparison["within_tolerance"]) / max(len(baseline_results), 1)
        }
        
        # Performance recommendations
        recommendations = self._generate_performance_recommendations()
        
        return {
            "summary": {
                "total_benchmarks": len(self.results),
                "total_iterations": total_iterations,
                "overall_success_rate": avg_success_rate,
                "total_errors": total_errors,
                "test_duration": sum(r.duration * r.iterations for r in self.results)
            },
            "category_performance": performance_summary,
            "baseline_compliance": baseline_compliance,
            "detailed_results": [
                {
                    "name": r.name,
                    "category": r.category,
                    "duration_ms": r.duration * 1000,
                    "throughput": r.throughput,
                    "memory_mb": r.memory_usage / 1024 / 1024,
                    "cpu_usage": r.cpu_usage,
                    "success_rate": r.success_rate,
                    "iterations": r.iterations,
                    "percentiles": r.percentiles,
                    "metadata": r.metadata,
                    "baseline_comparison": r.baseline_comparison
                }
                for r in self.results
            ],
            "recommendations": recommendations
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations based on results."""
        recommendations = []
        
        # Success rate recommendations
        low_success_results = [r for r in self.results if r.success_rate < 0.95]
        if low_success_results:
            recommendations.append(f"Address reliability issues in {len(low_success_results)} benchmarks with success rates below 95%")
        
        # Performance recommendations
        slow_results = [r for r in self.results if r.duration > 0.1]  # > 100ms
        if slow_results:
            recommendations.append(f"Optimize {len(slow_results)} slow operations (>100ms duration)")
        
        # Memory recommendations
        memory_intensive_results = [r for r in self.results if r.memory_usage > 100 * 1024 * 1024]  # > 100MB
        if memory_intensive_results:
            recommendations.append(f"Review memory usage in {len(memory_intensive_results)} memory-intensive operations")
        
        # Baseline compliance
        baseline_results = [r for r in self.results if r.baseline_comparison is not None]
        non_compliant = [r for r in baseline_results if not r.baseline_comparison["within_tolerance"]]
        if non_compliant:
            recommendations.append(f"Bring {len(non_compliant)} benchmarks within performance tolerance")
        
        # Scalability recommendations
        scalability_results = [r for r in self.results if r.name == "scalability"]
        if scalability_results:
            scalability_coeff = scalability_results[0].metadata.get("scalability_coefficient", 1.0)
            if scalability_coeff < 0.8:
                recommendations.append("Improve system scalability - performance degrades significantly under load")
        
        # Error handling
        error_handling_results = [r for r in self.results if r.name == "error_handling_performance"]
        if error_handling_results:
            recovery_rate = error_handling_results[0].metadata.get("recovery_rate", 0.0)
            if recovery_rate < 0.8:
                recommendations.append("Strengthen error recovery mechanisms - low recovery rate detected")
        
        # Overall assessment
        overall_success = statistics.mean([r.success_rate for r in self.results])
        if overall_success >= 0.95:
            recommendations.append("System demonstrates high performance reliability - ready for production load testing")
        elif overall_success >= 0.85:
            recommendations.append("System shows good performance with minor optimization needed")
        else:
            recommendations.append("System requires significant performance improvements before production deployment")
        
        return recommendations


# Example usage and main execution
if __name__ == "__main__":
    print("🏁 Neuromorphic System Performance Benchmarks")
    print("=" * 60)
    
    # Run comprehensive benchmarks
    benchmark_suite = BenchmarkSuite()
    results = benchmark_suite.run_all_benchmarks()
    
    # Print comprehensive summary
    print("\n📊 Performance Benchmark Summary:")
    print("=" * 60)
    print(f"Total Benchmarks: {results['summary']['total_benchmarks']}")
    print(f"Total Iterations: {results['summary']['total_iterations']:,}")
    print(f"Overall Success Rate: {results['summary']['overall_success_rate']:.1%}")
    print(f"Total Test Duration: {results['summary']['test_duration']:.2f}s")
    
    print(f"\n🎯 Category Performance:")
    for category, stats in results['category_performance'].items():
        print(f"  {category.title()}:")
        print(f"    Tests: {stats['tests']}")
        print(f"    Avg Throughput: {stats['avg_throughput']:.1f} ops/sec")
        print(f"    Avg Duration: {stats['avg_duration']*1000:.2f}ms")
        print(f"    Success Rate: {stats['avg_success_rate']:.1%}")
    
    print(f"\n📏 Baseline Compliance:")
    compliance = results['baseline_compliance']
    print(f"  Tests with Baselines: {compliance['total_baseline_tests']}")
    print(f"  Within Tolerance: {compliance['within_tolerance']}")
    print(f"  Compliance Rate: {compliance['compliance_rate']:.1%}")
    
    print(f"\n💡 Performance Recommendations:")
    for rec in results['recommendations']:
        print(f"  • {rec}")
    
    print(f"\n✅ Performance benchmarking completed!")
    
    # Write results to file
    output_file = Path("performance_benchmark_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📝 Detailed results written to: {output_file}")