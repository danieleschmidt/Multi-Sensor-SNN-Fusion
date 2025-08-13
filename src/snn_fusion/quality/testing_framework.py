"""
Comprehensive Testing Framework for SNN-Fusion

Implements automated testing with unit tests, integration tests,
performance benchmarks, and quality assurance metrics.
"""

import time
import sys
import traceback
import inspect
import logging
from typing import Dict, List, Optional, Any, Callable, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import hashlib
import threading
from collections import defaultdict, deque
import warnings


class TestResult(Enum):
    """Test execution results."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class TestCategory(Enum):
    """Categories of tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    FUNCTIONAL = "functional"
    SYSTEM = "system"


@dataclass
class TestCase:
    """Represents a test case."""
    name: str
    test_function: Callable
    category: TestCategory = TestCategory.UNIT
    description: str = ""
    tags: List[str] = field(default_factory=list)
    timeout: float = 30.0
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    expected_duration: float = 1.0
    criticality: str = "medium"  # low, medium, high, critical
    
    def __post_init__(self):
        if not self.description:
            self.description = self.test_function.__doc__ or f"Test for {self.name}"


@dataclass
class TestExecution:
    """Records test execution details."""
    test_name: str
    result: TestResult
    duration: float
    start_time: float
    end_time: float
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """Collection of related test cases."""
    name: str
    description: str = ""
    test_cases: List[TestCase] = field(default_factory=list)
    setup_suite: Optional[Callable] = None
    teardown_suite: Optional[Callable] = None
    parallel: bool = False


class TestDiscovery:
    """
    Automatically discovers test functions and classes.
    """
    
    def __init__(self, test_patterns: List[str] = None):
        self.test_patterns = test_patterns or ["test_*", "*_test"]
        self.discovered_tests = []
    
    def discover_from_module(self, module: Any) -> List[TestCase]:
        """Discover tests from a Python module."""
        tests = []
        
        for name, obj in inspect.getmembers(module):
            if self._is_test_function(name, obj):
                test_case = self._create_test_case(name, obj)
                tests.append(test_case)
            elif self._is_test_class(name, obj):
                class_tests = self._discover_from_test_class(obj)
                tests.extend(class_tests)
        
        return tests
    
    def _is_test_function(self, name: str, obj: Any) -> bool:
        """Check if object is a test function."""
        return (
            callable(obj) and
            any(pattern.replace('*', '') in name for pattern in self.test_patterns) and
            not name.startswith('_')
        )
    
    def _is_test_class(self, name: str, obj: Any) -> bool:
        """Check if object is a test class."""
        return (
            inspect.isclass(obj) and
            any(pattern.replace('*', '') in name for pattern in self.test_patterns)
        )
    
    def _create_test_case(self, name: str, func: Callable) -> TestCase:
        """Create TestCase from function."""
        # Extract metadata from function annotations or docstring
        category = TestCategory.UNIT
        tags = []
        timeout = 30.0
        
        # Parse docstring for metadata
        if func.__doc__:
            lines = func.__doc__.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('@category:'):
                    try:
                        category = TestCategory(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith('@tags:'):
                    tags = [t.strip() for t in line.split(':', 1)[1].split(',')]
                elif line.startswith('@timeout:'):
                    try:
                        timeout = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
        
        return TestCase(
            name=name,
            test_function=func,
            category=category,
            tags=tags,
            timeout=timeout
        )
    
    def _discover_from_test_class(self, test_class: Type) -> List[TestCase]:
        """Discover tests from a test class."""
        tests = []
        instance = test_class()
        
        # Look for setup/teardown methods
        setup_method = getattr(instance, 'setUp', None) or getattr(instance, 'setup', None)
        teardown_method = getattr(instance, 'tearDown', None) or getattr(instance, 'teardown', None)
        
        for name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if self._is_test_function(name, method):
                test_case = TestCase(
                    name=f"{test_class.__name__}.{name}",
                    test_function=method,
                    setup_function=setup_method,
                    teardown_function=teardown_method
                )
                tests.append(test_case)
        
        return tests


class TestRunner:
    """
    Advanced test runner with parallel execution and reporting.
    """
    
    def __init__(self, parallel: bool = False, max_workers: int = 4):
        self.parallel = parallel
        self.max_workers = max_workers
        self.executions: List[TestExecution] = []
        self.start_time = 0.0
        self.end_time = 0.0
        
        # Statistics
        self.stats = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def run_test_suite(self, test_suite: TestSuite) -> Dict[str, Any]:
        """Run a complete test suite."""
        self.logger.info(f"Running test suite: {test_suite.name}")
        self.start_time = time.time()
        
        # Suite setup
        if test_suite.setup_suite:
            try:
                test_suite.setup_suite()
            except Exception as e:
                self.logger.error(f"Suite setup failed: {e}")
                return self._create_suite_result(test_suite, setup_failed=True)
        
        try:
            # Run tests
            if test_suite.parallel and self.parallel:
                executions = self._run_tests_parallel(test_suite.test_cases)
            else:
                executions = self._run_tests_sequential(test_suite.test_cases)
            
            self.executions.extend(executions)
            
        finally:
            # Suite teardown
            if test_suite.teardown_suite:
                try:
                    test_suite.teardown_suite()
                except Exception as e:
                    self.logger.error(f"Suite teardown failed: {e}")
        
        self.end_time = time.time()
        return self._create_suite_result(test_suite)
    
    def run_single_test(self, test_case: TestCase) -> TestExecution:
        """Run a single test case."""
        start_time = time.time()
        
        try:
            # Setup
            if test_case.setup_function:
                test_case.setup_function()
            
            # Execute test with timeout
            result = self._execute_with_timeout(
                test_case.test_function,
                test_case.timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Determine result
            if result is None:
                test_result = TestResult.PASS
                error_message = None
                traceback_str = None
            elif isinstance(result, Exception):
                test_result = TestResult.FAIL
                error_message = str(result)
                traceback_str = traceback.format_exc()
            else:
                test_result = TestResult.PASS
                error_message = None
                traceback_str = None
            
        except TimeoutError:
            test_result = TestResult.ERROR
            error_message = f"Test timed out after {test_case.timeout}s"
            traceback_str = None
            end_time = time.time()
            duration = end_time - start_time
            
        except Exception as e:
            test_result = TestResult.ERROR
            error_message = str(e)
            traceback_str = traceback.format_exc()
            end_time = time.time()
            duration = end_time - start_time
        
        finally:
            # Teardown
            if test_case.teardown_function:
                try:
                    test_case.teardown_function()
                except Exception as e:
                    self.logger.warning(f"Teardown failed for {test_case.name}: {e}")
        
        # Create execution record
        execution = TestExecution(
            test_name=test_case.name,
            result=test_result,
            duration=duration,
            start_time=start_time,
            end_time=end_time,
            error_message=error_message,
            traceback=traceback_str,
            metadata={
                'category': test_case.category.value,
                'tags': test_case.tags,
                'expected_duration': test_case.expected_duration,
                'timeout': test_case.timeout,
                'criticality': test_case.criticality
            }
        )
        
        # Update statistics
        self.stats['total'] += 1
        if test_result == TestResult.PASS:
            self.stats['passed'] += 1
        elif test_result == TestResult.FAIL:
            self.stats['failed'] += 1
        elif test_result == TestResult.SKIP:
            self.stats['skipped'] += 1
        else:
            self.stats['errors'] += 1
        
        # Log result
        status_icon = {
            TestResult.PASS: "✅",
            TestResult.FAIL: "❌", 
            TestResult.SKIP: "⏭️",
            TestResult.ERROR: "💥"
        }
        
        self.logger.info(
            f"{status_icon[test_result]} {test_case.name} "
            f"({duration:.3f}s) - {test_result.value.upper()}"
        )
        
        if error_message:
            self.logger.error(f"   Error: {error_message}")
        
        return execution
    
    def _run_tests_sequential(self, test_cases: List[TestCase]) -> List[TestExecution]:
        """Run tests sequentially."""
        executions = []
        for test_case in test_cases:
            execution = self.run_single_test(test_case)
            executions.append(execution)
        return executions
    
    def _run_tests_parallel(self, test_cases: List[TestCase]) -> List[TestExecution]:
        """Run tests in parallel using threading."""
        import concurrent.futures
        
        executions = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(self.run_single_test, test_case): test_case
                for test_case in test_cases
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_test):
                execution = future.result()
                executions.append(execution)
        
        return executions
    
    def _execute_with_timeout(self, func: Callable, timeout: float) -> Any:
        """Execute function with timeout."""
        result = None
        exception = None
        
        def target():
            nonlocal result, exception
            try:
                result = func()
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Function timed out after {timeout}s")
        
        if exception:
            return exception
        
        return result
    
    def _create_suite_result(self, test_suite: TestSuite, setup_failed: bool = False) -> Dict[str, Any]:
        """Create test suite result summary."""
        duration = self.end_time - self.start_time if self.end_time > self.start_time else 0
        
        if setup_failed:
            return {
                'suite_name': test_suite.name,
                'setup_failed': True,
                'duration': duration,
                'stats': {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 1}
            }
        
        return {
            'suite_name': test_suite.name,
            'duration': duration,
            'stats': self.stats.copy(),
            'success_rate': self.stats['passed'] / max(self.stats['total'], 1),
            'executions': len(self.executions)
        }


class PerformanceBenchmark:
    """
    Performance benchmarking and regression testing.
    """
    
    def __init__(self):
        self.benchmarks = {}
        self.baseline_results = {}
        self.current_results = {}
    
    def register_benchmark(self, name: str, benchmark_func: Callable,
                          baseline_value: Optional[float] = None,
                          tolerance: float = 0.1):
        """Register a performance benchmark."""
        self.benchmarks[name] = {
            'function': benchmark_func,
            'baseline': baseline_value,
            'tolerance': tolerance
        }
    
    def run_benchmark(self, name: str, iterations: int = 5) -> Dict[str, Any]:
        """Run a specific benchmark multiple times."""
        if name not in self.benchmarks:
            raise ValueError(f"Benchmark {name} not registered")
        
        benchmark_func = self.benchmarks[name]['function']
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            try:
                benchmark_func()
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                logging.error(f"Benchmark {name} failed: {e}")
                return {'name': name, 'success': False, 'error': str(e)}
        
        if not times:
            return {'name': name, 'success': False, 'error': 'No successful runs'}
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        result = {
            'name': name,
            'success': True,
            'iterations': iterations,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'all_times': times
        }
        
        # Check against baseline
        baseline = self.benchmarks[name]['baseline']
        if baseline is not None:
            tolerance = self.benchmarks[name]['tolerance']
            regression = (avg_time - baseline) / baseline
            
            result['baseline'] = baseline
            result['regression'] = regression
            result['within_tolerance'] = abs(regression) <= tolerance
            
            if regression > tolerance:
                result['status'] = 'REGRESSION'
            elif regression < -tolerance:
                result['status'] = 'IMPROVEMENT'
            else:
                result['status'] = 'STABLE'
        
        self.current_results[name] = result
        return result
    
    def run_all_benchmarks(self, iterations: int = 5) -> Dict[str, Any]:
        """Run all registered benchmarks."""
        results = {}
        
        for benchmark_name in self.benchmarks:
            result = self.run_benchmark(benchmark_name, iterations)
            results[benchmark_name] = result
        
        # Summary
        successful = sum(1 for r in results.values() if r['success'])
        regressions = sum(1 for r in results.values() 
                         if r.get('status') == 'REGRESSION')
        
        return {
            'benchmarks': results,
            'summary': {
                'total': len(results),
                'successful': successful,
                'failed': len(results) - successful,
                'regressions': regressions
            }
        }


class QualityGate:
    """
    Implements quality gates for continuous integration.
    """
    
    def __init__(self):
        self.gates = {}
        self.gate_results = {}
    
    def add_gate(self, name: str, gate_function: Callable[[Dict], bool],
                description: str = "", required: bool = True):
        """Add a quality gate."""
        self.gates[name] = {
            'function': gate_function,
            'description': description,
            'required': required
        }
    
    def evaluate_gates(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all quality gates."""
        gate_results = {}
        all_passed = True
        
        for gate_name, gate_info in self.gates.items():
            try:
                passed = gate_info['function'](test_results)
                gate_results[gate_name] = {
                    'passed': passed,
                    'required': gate_info['required'],
                    'description': gate_info['description']
                }
                
                if gate_info['required'] and not passed:
                    all_passed = False
                    
            except Exception as e:
                gate_results[gate_name] = {
                    'passed': False,
                    'required': gate_info['required'],
                    'description': gate_info['description'],
                    'error': str(e)
                }
                
                if gate_info['required']:
                    all_passed = False
        
        self.gate_results = gate_results
        
        return {
            'all_passed': all_passed,
            'gates': gate_results,
            'required_gates_passed': all(
                result['passed'] for result in gate_results.values()
                if result['required']
            )
        }


# Predefined quality gates
def minimum_test_coverage_gate(results: Dict[str, Any], min_coverage: float = 0.8) -> bool:
    """Quality gate for minimum test coverage."""
    stats = results.get('stats', {})
    total_tests = stats.get('total', 0)
    passed_tests = stats.get('passed', 0)
    
    if total_tests == 0:
        return False
    
    coverage = passed_tests / total_tests
    return coverage >= min_coverage


def no_critical_failures_gate(results: Dict[str, Any]) -> bool:
    """Quality gate ensuring no critical test failures."""
    executions = results.get('executions', [])
    
    for execution in executions:
        metadata = execution.get('metadata', {})
        if (metadata.get('criticality') == 'critical' and
            execution.get('result') in ['fail', 'error']):
            return False
    
    return True


def performance_regression_gate(results: Dict[str, Any]) -> bool:
    """Quality gate for performance regression."""
    benchmarks = results.get('benchmarks', {})
    
    for benchmark_result in benchmarks.values():
        if benchmark_result.get('status') == 'REGRESSION':
            return False
    
    return True


# Example usage and testing
if __name__ == "__main__":
    print("Testing Quality Framework...")
    
    # Sample test functions
    def test_addition():
        """
        @category: unit
        @tags: math, basic
        @timeout: 1.0
        """
        assert 1 + 1 == 2
    
    def test_subtraction():
        """Test subtraction operation."""
        assert 5 - 3 == 2
    
    def test_slow_operation():
        """
        @category: performance
        @timeout: 3.0
        """
        time.sleep(0.1)  # Simulate slow operation
        assert True
    
    def test_failing():
        """Test that should fail."""
        assert 1 + 1 == 3  # This will fail
    
    # Test discovery
    print("\n1. Testing Test Discovery:")
    discovery = TestDiscovery()
    
    # Mock module for discovery
    class MockModule:
        @staticmethod
        def test_mock1():
            pass
        
        @staticmethod  
        def test_mock2():
            pass
    
    discovered = discovery.discover_from_module(MockModule)
    print(f"  ✓ Discovered {len(discovered)} tests")
    
    # Create test suite
    print("\n2. Testing Test Runner:")
    test_suite = TestSuite(
        name="Basic Tests",
        description="Basic functionality tests"
    )
    
    test_suite.test_cases = [
        TestCase("test_addition", test_addition),
        TestCase("test_subtraction", test_subtraction), 
        TestCase("test_slow_operation", test_slow_operation, category=TestCategory.PERFORMANCE),
        TestCase("test_failing", test_failing)
    ]
    
    # Run tests
    runner = TestRunner(parallel=False)
    suite_result = runner.run_test_suite(test_suite)
    
    print(f"  ✓ Test Results: {suite_result['stats']}")
    print(f"  ✓ Success Rate: {suite_result['success_rate']:.1%}")
    
    # Test performance benchmarks
    print("\n3. Testing Performance Benchmarks:")
    benchmark = PerformanceBenchmark()
    
    def simple_computation():
        return sum(range(1000))
    
    benchmark.register_benchmark("simple_computation", simple_computation, 
                                baseline_value=0.001, tolerance=0.2)
    
    bench_result = benchmark.run_benchmark("simple_computation", iterations=3)
    print(f"  ✓ Benchmark: {bench_result['avg_time']:.6f}s average")
    
    # Test quality gates
    print("\n4. Testing Quality Gates:")
    gate = QualityGate()
    
    gate.add_gate("coverage", lambda r: minimum_test_coverage_gate(r, 0.5))
    gate.add_gate("no_critical_failures", no_critical_failures_gate)
    
    gate_results = gate.evaluate_gates(suite_result)
    print(f"  ✓ Quality Gates: {gate_results['all_passed']}")
    
    for gate_name, gate_result in gate_results['gates'].items():
        status = "✅" if gate_result['passed'] else "❌"
        print(f"    {status} {gate_name}: {gate_result['passed']}")
    
    print("\n✓ Quality framework test completed!")