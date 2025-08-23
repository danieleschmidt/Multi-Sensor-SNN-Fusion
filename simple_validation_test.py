#!/usr/bin/env python3
"""
Simple Validation Test for Enhanced Neuromorphic Algorithms

Tests the key enhanced algorithms directly without full framework dependencies.

Authors: Terry (Terragon Labs) - Simple Validation Framework
"""

import sys
import os
sys.path.append('src')

import numpy as np
import torch
import time
import logging
from datetime import datetime

# Direct imports of enhanced algorithms
try:
    from snn_fusion.algorithms.advanced_neuromorphic_enhancement import (
        UltraFastTemporalProcessor,
        RealTimeAdaptiveLearningEngine,
        OptimizationTarget,
        AdaptationStrategy
    )
    ENHANCED_ALGORITHMS_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced algorithms not available: {e}")
    ENHANCED_ALGORITHMS_AVAILABLE = False

try:
    from snn_fusion.algorithms.production_optimization_engine import (
        RealTimeMonitoringSystem,
        IntelligentAutoScaler,
        ScalingPolicy
    )
    PRODUCTION_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"Production optimization engine not available: {e}")
    PRODUCTION_ENGINE_AVAILABLE = False

try:
    from snn_fusion.research.advanced_research_validation_framework import (
        AdvancedStatisticalAnalyzer,
        ReproducibilityValidator,
        StatisticalTest
    )
    RESEARCH_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"Research validation framework not available: {e}")
    RESEARCH_FRAMEWORK_AVAILABLE = False


def test_ultra_fast_processor():
    """Test ultra-fast temporal processor."""
    print("🚀 Testing Ultra-Fast Temporal Processor...")
    
    if not ENHANCED_ALGORITHMS_AVAILABLE:
        print("❌ Enhanced algorithms not available")
        return False
    
    try:
        # Create processor
        processor = UltraFastTemporalProcessor(
            num_neurons=1000,
            temporal_resolution_us=1.0,
            cache_size=5000,
            parallel_workers=4
        )
        
        # Generate test data
        test_spikes = np.random.binomial(1, 0.1, (500, 1000))  # 500 time steps, 1000 neurons
        test_timestamps = np.arange(500) * 100  # 100μs intervals
        
        # Test processing
        start_time = time.perf_counter()
        result = processor.process_spike_batch(test_spikes, test_timestamps)
        processing_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Validate results
        if result and 'active_neurons' in result and 'temporal_patterns' in result:
            print(f"✅ Processing successful in {processing_time:.2f}ms")
            print(f"   - Active neurons: {len(result['active_neurons'])}")
            print(f"   - Temporal patterns detected: {result.get('temporal_patterns', {}).get('total_bursts', 0)}")
            
            # Test optimization
            optimization_result = processor.optimize_for_target(
                OptimizationTarget.LATENCY,
                1.0  # Target 1ms
            )
            print(f"   - Optimization applied: {len(optimization_result.get('optimizations_applied', []))} optimizations")
            
            return True
        else:
            print("❌ Processing failed - invalid result")
            return False
            
    except Exception as e:
        print(f"❌ Ultra-fast processor test failed: {e}")
        return False


def test_adaptive_learning_engine():
    """Test real-time adaptive learning engine."""
    print("🧠 Testing Real-Time Adaptive Learning Engine...")
    
    if not ENHANCED_ALGORITHMS_AVAILABLE:
        print("❌ Enhanced algorithms not available")
        return False
    
    try:
        # Create learning engine
        learning_engine = RealTimeAdaptiveLearningEngine(
            base_learning_rate=0.001,
            adaptation_strategy=AdaptationStrategy.CONSCIOUSNESS_DRIVEN,
            emergence_threshold=0.8
        )
        
        # Test adaptation
        performance_data = {
            'accuracy': 0.75,
            'latency_ms': 5.0,
            'throughput_ops_sec': 2000.0
        }
        
        context = {
            'consciousness_level': 0.9,
            'attention_focus': ['accuracy'],
            'target_accuracy': 0.85
        }
        
        adaptation_result = learning_engine.adapt_learning_parameters(performance_data, context)
        
        if adaptation_result and 'strategy_used' in adaptation_result:
            print(f"✅ Adaptation successful using {adaptation_result['strategy_used']}")
            print(f"   - Parameters changed: {adaptation_result.get('parameters_changed', 0)}")
            print(f"   - Consciousness influence: {adaptation_result.get('consciousness_influence', 0):.3f}")
            
            # Test learning summary
            summary = learning_engine.get_learning_summary()
            intelligence_metrics = summary.get('intelligence_metrics', {})
            print(f"   - Intelligence metrics: {len(intelligence_metrics)} measured")
            
            return True
        else:
            print("❌ Adaptation failed - invalid result")
            return False
            
    except Exception as e:
        print(f"❌ Adaptive learning engine test failed: {e}")
        return False


def test_production_monitoring():
    """Test production monitoring system."""
    print("📊 Testing Production Monitoring System...")
    
    if not PRODUCTION_ENGINE_AVAILABLE:
        print("❌ Production optimization engine not available")
        return False
    
    try:
        # Create monitoring system
        monitoring = RealTimeMonitoringSystem(
            monitoring_interval_seconds=0.5,
            retention_hours=1
        )
        
        # Test monitoring startup
        monitoring.start_monitoring()
        time.sleep(1.0)  # Let it collect some data
        
        status = monitoring.get_current_status()
        monitoring.stop_monitoring()
        
        if status and 'deployment_id' in status:
            print(f"✅ Monitoring successful")
            print(f"   - Deployment ID: {status['deployment_id'][:8]}...")
            print(f"   - System metrics collected: {len(status.get('system_metrics', {}))} metrics")
            print(f"   - Health checks: {len(status.get('health_checks', {}))} checks")
            
            return True
        else:
            print("❌ Monitoring failed - invalid status")
            return False
            
    except Exception as e:
        print(f"❌ Production monitoring test failed: {e}")
        return False


def test_intelligent_autoscaler():
    """Test intelligent auto-scaler."""
    print("📈 Testing Intelligent Auto-Scaler...")
    
    if not PRODUCTION_ENGINE_AVAILABLE:
        print("❌ Production optimization engine not available")
        return False
    
    try:
        # Create auto-scaler
        autoscaler = IntelligentAutoScaler(
            min_instances=1,
            max_instances=10,
            scaling_policy=ScalingPolicy.PREDICTIVE
        )
        
        # Test scaling decision with high load
        high_load_metrics = {
            'cpu_percent': 85.0,  # High CPU
            'memory_percent': 80.0,  # High memory
            'requests_per_second': 1500
        }
        
        scaling_decision = autoscaler.evaluate_scaling_decision(high_load_metrics)
        
        if scaling_decision and 'action' in scaling_decision:
            print(f"✅ Scaling decision successful")
            print(f"   - Action: {scaling_decision['action']}")
            print(f"   - Target instances: {scaling_decision.get('target_instances', 'N/A')}")
            print(f"   - Reason: {scaling_decision.get('reason', 'N/A')}")
            
            # Apply decision (dry run)
            scaling_result = autoscaler.apply_scaling_decision(scaling_decision, dry_run=True)
            print(f"   - Dry run result: {scaling_result.get('action_taken', 'N/A')}")
            
            return True
        else:
            print("❌ Scaling decision failed - invalid result")
            return False
            
    except Exception as e:
        print(f"❌ Intelligent auto-scaler test failed: {e}")
        return False


def test_statistical_analyzer():
    """Test statistical analyzer."""
    print("📉 Testing Statistical Analyzer...")
    
    if not RESEARCH_FRAMEWORK_AVAILABLE:
        print("❌ Research validation framework not available")
        return False
    
    try:
        # Create statistical analyzer
        stats_analyzer = AdvancedStatisticalAnalyzer(
            significance_level=0.05,
            target_power=0.8
        )
        
        # Generate test data
        np.random.seed(42)
        baseline_data = np.random.normal(0.75, 0.05, 30)  # Baseline performance
        improved_data = np.random.normal(0.82, 0.04, 30)  # Improved performance
        
        # Perform statistical test
        stat_result = stats_analyzer.perform_statistical_test(
            baseline_data,
            improved_data,
            test_type=StatisticalTest.T_TEST_INDEPENDENT,
            test_id="baseline_vs_improved"
        )
        
        if stat_result and stat_result.p_value is not None:
            print(f"✅ Statistical analysis successful")
            print(f"   - Test: {stat_result.test_name}")
            print(f"   - p-value: {stat_result.p_value:.6f}")
            print(f"   - Effect size: {stat_result.effect_size:.3f} ({stat_result.effect_size_magnitude})")
            print(f"   - Significant: {stat_result.is_significant}")
            
            return True
        else:
            print("❌ Statistical analysis failed - invalid result")
            return False
            
    except Exception as e:
        print(f"❌ Statistical analyzer test failed: {e}")
        return False


def test_reproducibility_validator():
    """Test reproducibility validator."""
    print("🔁 Testing Reproducibility Validator...")
    
    if not RESEARCH_FRAMEWORK_AVAILABLE:
        print("❌ Research validation framework not available")
        return False
    
    try:
        # Create reproducibility validator
        repro_validator = ReproducibilityValidator(
            num_replications=3,
            reproducibility_threshold=0.8,
            random_seeds=[42, 123, 456]
        )
        
        # Mock experiment
        def mock_experiment(random_seed: int = 42) -> dict:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            
            # Simulate consistent results with small noise
            base_accuracy = 0.85
            noise = np.random.normal(0, 0.01)
            return {'accuracy': base_accuracy + noise}
        
        # Validate reproducibility
        repro_result = repro_validator.validate_reproducibility(
            experiment_function=mock_experiment,
            result_key='accuracy',
            experiment_id='test_experiment'
        )
        
        if repro_result and 'reproducible' in repro_result:
            print(f"✅ Reproducibility validation successful")
            print(f"   - Reproducible: {repro_result['reproducible']}")
            print(f"   - Mean accuracy: {repro_result['reproducibility_metrics']['mean']:.4f}")
            print(f"   - Coefficient of variation: {repro_result['reproducibility_metrics']['cv']:.4f}")
            print(f"   - Successful replications: {repro_result['successful_replications']}/{repro_result['num_replications']}")
            
            return True
        else:
            print("❌ Reproducibility validation failed - invalid result")
            return False
            
    except Exception as e:
        print(f"❌ Reproducibility validator test failed: {e}")
        return False


def main():
    """Run simple validation tests."""
    print("🚀 Advanced Neuromorphic Enhancement Framework - Simple Validation")
    print("=" * 70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run tests
    tests = [
        ("Ultra-Fast Temporal Processor", test_ultra_fast_processor),
        ("Real-Time Adaptive Learning Engine", test_adaptive_learning_engine),
        ("Production Monitoring System", test_production_monitoring),
        ("Intelligent Auto-Scaler", test_intelligent_autoscaler),
        ("Statistical Analyzer", test_statistical_analyzer),
        ("Reproducibility Validator", test_reproducibility_validator)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 50}")
        result = test_func()
        results.append((test_name, result))
        print(f"Result: {'PASS' if result else 'FAIL'}")
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    print(f"Total Tests: {len(results)}")
    print(f"Tests Passed: {passed}")
    print(f"Tests Failed: {failed}")
    print(f"Pass Rate: {(passed / len(results)) * 100:.1f}%")
    print(f"Total Time: {total_time:.2f} seconds")
    
    print("\nDetailed Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Enhanced algorithms are working correctly.")
        overall_status = "SUCCESS"
    elif passed > len(results) // 2:
        print("\n⚠️  Most tests passed, but some components need attention.")
        overall_status = "PARTIAL_SUCCESS"
    else:
        print("\n❌ Multiple tests failed. System needs debugging.")
        overall_status = "FAILURE"
    
    print(f"\nOverall Status: {overall_status}")
    
    # Additional system info
    print("\n💻 System Information:")
    print(f"  - PyTorch version: {torch.__version__}")
    print(f"  - NumPy version: {np.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA devices: {torch.cuda.device_count()}")
    
    return overall_status == "SUCCESS"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
