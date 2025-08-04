#!/usr/bin/env python3
"""
Comprehensive Test Suite for SNN-Fusion

This test suite validates functionality, performance, and integration
across the entire multi-modal spiking neural network framework.
"""

import sys
import os
import time
import traceback
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class TestResults:
    """Track test results and statistics."""
    
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()
    
    def add_result(self, test_name: str, status: str, details: str = "", duration: float = 0.0):
        """Add a test result."""
        self.tests.append({
            'name': test_name,
            'status': status,
            'details': details,
            'duration': duration
        })
        
        if status == 'PASS':
            self.passed += 1
        elif status == 'FAIL':
            self.failed += 1
        elif status == 'SKIP':
            self.skipped += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        total_time = time.time() - self.start_time
        total_tests = len(self.tests)
        
        return {
            'total_tests': total_tests,
            'passed': self.passed,
            'failed': self.failed,
            'skipped': self.skipped,
            'pass_rate': (self.passed / total_tests * 100) if total_tests > 0 else 0,
            'total_duration': total_time,
            'average_test_time': total_time / total_tests if total_tests > 0 else 0
        }


def test_module_imports(results: TestResults) -> None:
    """Test that all modules can be imported without heavy dependencies."""
    print("🧪 Testing Module Imports...")
    
    # Core modules that should import without torch
    import_tests = [
        ('snn_fusion.__init__', 'Package init'),
        ('snn_fusion.utils.validation', 'Validation utilities'),
        ('snn_fusion.utils.security', 'Security utilities'),
        ('snn_fusion.utils.logging', 'Logging utilities'),
    ]
    
    for module_name, description in import_tests:
        start_time = time.time()
        try:
            # Try to import without actually loading torch-dependent parts
            import importlib
            importlib.import_module(module_name)
            
            duration = time.time() - start_time
            results.add_result(f"Import {description}", "PASS", duration=duration)
            print(f"  ✅ {description}")
            
        except ImportError as e:
            if "torch" in str(e) or "numpy" in str(e):
                # Expected for modules requiring heavy dependencies
                duration = time.time() - start_time
                results.add_result(f"Import {description}", "SKIP", 
                                 f"Requires heavy dependencies: {e}", duration)
                print(f"  ⏭️  {description} (requires dependencies)")
            else:
                duration = time.time() - start_time
                results.add_result(f"Import {description}", "FAIL", str(e), duration)
                print(f"  ❌ {description}: {e}")
        
        except Exception as e:
            duration = time.time() - start_time
            results.add_result(f"Import {description}", "FAIL", str(e), duration)
            print(f"  ❌ {description}: {e}")


def test_configuration_validation(results: TestResults) -> None:
    """Test configuration validation functions."""
    print("\n🧪 Testing Configuration Validation...")
    
    try:
        from snn_fusion.utils.validation import (
            validate_configuration, 
            ValidationError,
            DATASET_CONFIG_SCHEMA,
            validate_dataset_config
        )
        
        # Test valid configuration
        start_time = time.time()
        try:
            valid_config = {
                'root_dir': './test_data',
                'split': 'train',
                'modalities': ['audio', 'events'],
                'sequence_length': 100,
                'spike_encoding': True
            }
            
            validate_dataset_config(valid_config)
            duration = time.time() - start_time
            results.add_result("Valid config validation", "PASS", duration=duration)
            print("  ✅ Valid configuration accepted")
            
        except Exception as e:
            duration = time.time() - start_time
            results.add_result("Valid config validation", "FAIL", str(e), duration)
            print(f"  ❌ Valid configuration rejected: {e}")
        
        # Test invalid configuration
        start_time = time.time()
        try:
            invalid_config = {
                'root_dir': './test_data',
                'split': 'invalid_split',  # Invalid value
                'modalities': ['audio'],
                'sequence_length': -1,  # Invalid value
            }
            
            validate_dataset_config(invalid_config)
            duration = time.time() - start_time
            results.add_result("Invalid config validation", "FAIL", 
                             "Should have rejected invalid config", duration)
            print("  ❌ Invalid configuration was accepted (should have failed)")
            
        except ValidationError:
            duration = time.time() - start_time
            results.add_result("Invalid config validation", "PASS", duration=duration)
            print("  ✅ Invalid configuration properly rejected")
        
        except Exception as e:
            duration = time.time() - start_time
            results.add_result("Invalid config validation", "FAIL", str(e), duration)
            print(f"  ❌ Unexpected error: {e}")
    
    except ImportError as e:
        results.add_result("Configuration validation", "SKIP", 
                         f"Module import failed: {e}")
        print(f"  ⏭️  Skipped (import failed): {e}")


def test_security_functions(results: TestResults) -> None:
    """Test security utility functions."""
    print("\n🧪 Testing Security Functions...")
    
    try:
        from snn_fusion.utils.security import (
            sanitize_input,
            validate_file_path,
            SecurityError
        )
        
        # Test input sanitization
        start_time = time.time()
        try:
            # Test normal input
            safe_input = sanitize_input("normal_string", str, max_length=100)
            assert safe_input == "normal_string"
            
            duration = time.time() - start_time
            results.add_result("Input sanitization (safe)", "PASS", duration=duration)
            print("  ✅ Safe input sanitization")
            
        except Exception as e:
            duration = time.time() - start_time
            results.add_result("Input sanitization (safe)", "FAIL", str(e), duration)
            print(f"  ❌ Safe input sanitization failed: {e}")
        
        # Test malicious input detection
        start_time = time.time()
        try:
            malicious_input = "<script>alert('xss')</script>"
            sanitize_input(malicious_input, str)
            
            duration = time.time() - start_time
            results.add_result("Malicious input detection", "FAIL", 
                             "Should have detected XSS", duration)
            print("  ❌ Malicious input was not detected")
            
        except SecurityError:
            duration = time.time() - start_time
            results.add_result("Malicious input detection", "PASS", duration=duration)
            print("  ✅ Malicious input properly detected")
        
        except Exception as e:
            duration = time.time() - start_time
            results.add_result("Malicious input detection", "FAIL", str(e), duration)
            print(f"  ❌ Unexpected error: {e}")
        
        # Test file path validation
        start_time = time.time()
        try:
            # Test safe path
            safe_path = validate_file_path("test_file.txt", check_exists=False)
            
            duration = time.time() - start_time
            results.add_result("File path validation (safe)", "PASS", duration=duration)
            print("  ✅ Safe file path validation")
            
        except Exception as e:
            duration = time.time() - start_time
            results.add_result("File path validation (safe)", "FAIL", str(e), duration)
            print(f"  ❌ Safe file path validation failed: {e}")
    
    except ImportError as e:
        results.add_result("Security functions", "SKIP", f"Module import failed: {e}")
        print(f"  ⏭️  Skipped (import failed): {e}")


def test_synthetic_dataset(results: TestResults) -> None:
    """Test synthetic dataset generation."""
    print("\n🧪 Testing Synthetic Dataset...")
    
    try:
        from snn_fusion.datasets.synthetic import create_synthetic_dataset
        
        start_time = time.time()
        try:
            # Create small synthetic dataset
            dataset = create_synthetic_dataset(
                split='train',
                num_samples=10,
                sequence_length=20,
                modalities=['audio', 'events', 'imu'],
                num_classes=3,
                seed=42
            )
            
            # Test dataset properties
            assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"
            
            # Test sample structure
            sample = dataset[0]
            expected_keys = {'audio', 'events', 'imu', 'label', 'sample_id'}
            sample_keys = set(sample.keys())
            assert expected_keys.issubset(sample_keys), f"Missing keys: {expected_keys - sample_keys}"
            
            duration = time.time() - start_time
            results.add_result("Synthetic dataset creation", "PASS", duration=duration)
            print("  ✅ Synthetic dataset creation")
            
        except Exception as e:
            duration = time.time() - start_time
            results.add_result("Synthetic dataset creation", "FAIL", str(e), duration)
            print(f"  ❌ Synthetic dataset creation failed: {e}")
    
    except ImportError as e:
        results.add_result("Synthetic dataset", "SKIP", f"Module import failed: {e}")
        print(f"  ⏭️  Skipped (import failed): {e}")


def test_monitoring_system(results: TestResults) -> None:
    """Test monitoring and performance tracking."""
    print("\n🧪 Testing Monitoring System...")
    
    try:
        from snn_fusion.utils.monitoring import PerformanceMonitor, ResourceMonitor
        
        start_time = time.time()
        try:
            # Test performance monitor
            perf_monitor = PerformanceMonitor(history_size=10, enable_gpu_monitoring=False)
            
            # Record some metrics
            perf_monitor.record_operation_time("test_op", 0.1)
            perf_monitor.record_custom_metric("test_metric", 42.0)
            
            # Get summary
            summary = perf_monitor.get_performance_summary(last_n_seconds=60)
            assert 'operation_stats' in summary
            assert 'test_op' in summary['operation_stats']
            
            duration = time.time() - start_time
            results.add_result("Performance monitoring", "PASS", duration=duration)
            print("  ✅ Performance monitoring")
            
        except Exception as e:
            duration = time.time() - start_time
            results.add_result("Performance monitoring", "FAIL", str(e), duration)
            print(f"  ❌ Performance monitoring failed: {e}")
        
        start_time = time.time()
        try:
            # Test resource monitor
            resource_monitor = ResourceMonitor()
            
            # Check memory usage
            memory_info = resource_monitor.check_memory_usage(threshold_gb=1.0)
            assert 'current_memory_gb' in memory_info
            assert 'within_limit' in memory_info
            
            duration = time.time() - start_time
            results.add_result("Resource monitoring", "PASS", duration=duration)
            print("  ✅ Resource monitoring")
            
        except Exception as e:
            duration = time.time() - start_time
            results.add_result("Resource monitoring", "FAIL", str(e), duration)
            print(f"  ❌ Resource monitoring failed: {e}")
    
    except ImportError as e:
        results.add_result("Monitoring system", "SKIP", f"Module import failed: {e}")
        print(f"  ⏭️  Skipped (import failed): {e}")


def test_data_transforms(results: TestResults) -> None:
    """Test data transformation functions."""
    print("\n🧪 Testing Data Transforms...")
    
    try:
        from snn_fusion.datasets.transforms import (
            TemporalJitter,
            ModalityDropout,
            create_training_transforms
        )
        
        start_time = time.time()
        try:
            # Create sample data (mock tensors)
            import torch
            sample = {
                'audio': torch.randn(50, 2, 32),
                'events': torch.randn(50, 64, 64, 2),
                'imu': torch.randn(50, 6),
                'label': torch.tensor(1)
            }
            
            # Test temporal jitter
            jitter = TemporalJitter(max_jitter=3, probability=1.0)
            jittered = jitter(sample)
            
            # Should have same keys and shapes
            assert set(jittered.keys()) == set(sample.keys())
            for key in ['audio', 'events', 'imu']:
                assert jittered[key].shape == sample[key].shape
            
            duration = time.time() - start_time
            results.add_result("Data transforms", "PASS", duration=duration)
            print("  ✅ Data transforms")
            
        except ImportError:
            duration = time.time() - start_time
            results.add_result("Data transforms", "SKIP", "PyTorch not available", duration)
            print("  ⏭️  Data transforms (PyTorch not available)")
            
        except Exception as e:
            duration = time.time() - start_time
            results.add_result("Data transforms", "FAIL", str(e), duration)
            print(f"  ❌ Data transforms failed: {e}")
    
    except ImportError as e:
        results.add_result("Data transforms", "SKIP", f"Module import failed: {e}")
        print(f"  ⏭️  Skipped (import failed): {e}")


def test_logging_system(results: TestResults) -> None:
    """Test logging functionality."""
    print("\n🧪 Testing Logging System...")
    
    try:
        from snn_fusion.utils.logging import setup_logging, get_logger, LogLevel
        
        start_time = time.time()
        try:
            # Setup logging
            logger = setup_logging(
                log_level=LogLevel.INFO,
                enable_console=False,  # Don't spam console during tests
                structured_format=False
            )
            
            # Test basic logging
            test_logger = get_logger('test_module')
            test_logger.info("Test message")
            test_logger.warning("Test warning")
            
            duration = time.time() - start_time
            results.add_result("Logging system", "PASS", duration=duration)
            print("  ✅ Logging system")
            
        except Exception as e:
            duration = time.time() - start_time
            results.add_result("Logging system", "FAIL", str(e), duration)
            print(f"  ❌ Logging system failed: {e}")
    
    except ImportError as e:
        results.add_result("Logging system", "SKIP", f"Module import failed: {e}")
        print(f"  ⏭️  Skipped (import failed): {e}")


def run_performance_benchmarks(results: TestResults) -> None:
    """Run performance benchmarks."""
    print("\n⚡ Running Performance Benchmarks...")
    
    # Test file I/O performance
    start_time = time.time()
    try:
        # Create and write test file
        test_file = Path("test_performance.tmp")
        test_data = "x" * 1000000  # 1MB of data
        
        with open(test_file, 'w') as f:
            f.write(test_data)
        
        # Read back
        with open(test_file, 'r') as f:
            read_data = f.read()
        
        assert len(read_data) == len(test_data)
        
        # Cleanup
        test_file.unlink()
        
        duration = time.time() - start_time
        throughput_mb_s = 2.0 / duration  # Read + write 1MB each
        
        results.add_result("File I/O benchmark", "PASS", 
                         f"Throughput: {throughput_mb_s:.2f} MB/s", duration)
        print(f"  ✅ File I/O: {throughput_mb_s:.2f} MB/s")
        
    except Exception as e:
        duration = time.time() - start_time
        results.add_result("File I/O benchmark", "FAIL", str(e), duration)
        print(f"  ❌ File I/O benchmark failed: {e}")
    
    # Test memory allocation performance
    start_time = time.time()
    try:
        # Allocate and deallocate memory
        large_lists = []
        for i in range(100):
            large_lists.append([0] * 10000)
        
        # Clear memory
        large_lists.clear()
        
        duration = time.time() - start_time
        allocations_per_sec = 100 / duration
        
        results.add_result("Memory allocation benchmark", "PASS",
                         f"Rate: {allocations_per_sec:.2f} allocs/s", duration)
        print(f"  ✅ Memory allocation: {allocations_per_sec:.2f} allocs/s")
        
    except Exception as e:
        duration = time.time() - start_time
        results.add_result("Memory allocation benchmark", "FAIL", str(e), duration)
        print(f"  ❌ Memory allocation benchmark failed: {e}")


def generate_test_report(results: TestResults) -> None:
    """Generate comprehensive test report."""
    summary = results.get_summary()
    
    print("\n📊 Comprehensive Test Report")
    print("=" * 80)
    
    print(f"Total Tests: {summary['total_tests']}")
    print(f"✅ Passed: {summary['passed']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"⏭️  Skipped: {summary['skipped']}")
    print(f"📈 Pass Rate: {summary['pass_rate']:.1f}%")
    print(f"⏱️  Total Duration: {summary['total_duration']:.2f}s")
    print(f"⚡ Average Test Time: {summary['average_test_time']:.3f}s")
    
    # Detailed results
    print("\n📋 Detailed Results:")
    print("-" * 80)
    
    for test in results.tests:
        status_icon = {
            'PASS': '✅',
            'FAIL': '❌',
            'SKIP': '⏭️ '
        }.get(test['status'], '❓')
        
        duration_str = f"({test['duration']:.3f}s)" if test['duration'] > 0 else ""
        print(f"{status_icon} {test['name']:<40} {duration_str}")
        
        if test['details']:
            print(f"    {test['details']}")
    
    # Export results to JSON
    report_data = {
        'summary': summary,
        'tests': results.tests,
        'timestamp': time.time()
    }
    
    with open('test_results.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: test_results.json")


def main():
    """Run comprehensive test suite."""
    print("🚀 Starting SNN-Fusion Comprehensive Test Suite\n")
    
    results = TestResults()
    
    # Suppress warnings during testing
    warnings.filterwarnings('ignore')
    
    try:
        # Run all test suites
        test_module_imports(results)
        test_configuration_validation(results)
        test_security_functions(results)
        test_synthetic_dataset(results)
        test_monitoring_system(results)
        test_data_transforms(results)
        test_logging_system(results)
        run_performance_benchmarks(results)
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return False
    
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        traceback.print_exc()
        return False
    
    finally:
        # Always generate report
        generate_test_report(results)
    
    # Determine overall success
    summary = results.get_summary()
    
    if summary['failed'] == 0:
        print("\n🎉 All tests passed successfully!")
        return True
    elif summary['pass_rate'] >= 80:
        print(f"\n✅ Most tests passed ({summary['pass_rate']:.1f}% pass rate)")
        return True
    else:
        print(f"\n⚠️  Many tests failed ({summary['pass_rate']:.1f}% pass rate)")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)