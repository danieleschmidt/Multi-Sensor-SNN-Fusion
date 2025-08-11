#!/usr/bin/env python3
"""
Simple structure test for Generation 1 SNN Fusion framework.
Tests basic imports and code structure without heavy dependencies.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test that core modules can be imported."""
    print("=== Testing Basic Imports ===")
    
    try:
        # Test utils imports (should work without torch)
        from snn_fusion.utils.config import DatasetConfig, ModelConfig, TrainingConfig
        print("✓ Config classes imported")
        
        from snn_fusion.utils.logging import setup_logging, get_logger
        print("✓ Logging utilities imported")
        
        # Test algorithm imports
        from snn_fusion.algorithms.encoding import SpikeEncoder, RateEncoder
        print("✓ Encoding classes imported")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config_creation():
    """Test configuration creation."""
    print("\n=== Testing Configuration Creation ===")
    
    try:
        from snn_fusion.utils.config import (
            DatasetConfig, ModelConfig, TrainingConfig, STDPConfig, SNNFusionConfig
        )
        
        # Create configurations
        dataset_config = DatasetConfig(root_dir="test_data")
        model_config = ModelConfig(n_neurons=500)
        training_config = TrainingConfig(epochs=50, learning_rate=0.001)
        stdp_config = STDPConfig(enabled=True, tau_pre=15.0)
        
        # Create main config
        config = SNNFusionConfig(
            dataset=dataset_config,
            model=model_config,
            training=training_config,
            stdp=stdp_config,
            experiment_name="test_experiment"
        )
        
        print(f"✓ Config created with {config.model.n_neurons} neurons")
        print(f"✓ Experiment name: {config.experiment_name}")
        print(f"✓ Dataset path: {config.dataset.root_dir}")
        print(f"✓ Training epochs: {config.training.epochs}")
        print(f"✓ STDP enabled: {config.stdp.enabled}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False

def test_logging_setup():
    """Test logging setup."""
    print("\n=== Testing Logging Setup ===")
    
    try:
        from snn_fusion.utils.logging import setup_logging, get_logger, LogLevel
        
        # Setup logging
        logger = setup_logging(
            log_level=LogLevel.INFO,
            enable_console=True,
            structured_format=False
        )
        
        print("✓ Main logger created")
        
        # Test module logger
        module_logger = get_logger("test_module")
        module_logger.info("Test log message")
        print("✓ Module logger working")
        
        return True
        
    except Exception as e:
        print(f"✗ Logging setup failed: {e}")
        return False

def test_encoding_basic():
    """Test basic encoding functionality."""
    print("\n=== Testing Basic Encoding ===")
    
    try:
        import numpy as np
        from snn_fusion.algorithms.encoding import RateEncoder, TemporalEncoder
        
        # Test rate encoder
        rate_encoder = RateEncoder(
            n_neurons=32,
            duration=100.0,
            max_rate=50.0
        )
        
        # Test data
        test_data = np.random.randn(100)
        spike_data = rate_encoder.encode(test_data)
        
        print(f"✓ Rate encoder: {len(spike_data.spike_times)} spikes generated")
        print(f"✓ Spike data has {spike_data.n_neurons} neurons")
        print(f"✓ Duration: {spike_data.duration} ms")
        
        # Test temporal encoder
        temporal_encoder = TemporalEncoder(
            n_neurons=32,
            duration=100.0,
            encoding_window=50.0
        )
        
        spike_data_temp = temporal_encoder.encode(test_data)
        print(f"✓ Temporal encoder: {len(spike_data_temp.spike_times)} spikes generated")
        
        return True
        
    except Exception as e:
        print(f"✗ Encoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that expected files exist."""
    print("\n=== Testing File Structure ===")
    
    src_dir = Path(__file__).parent / "src" / "snn_fusion"
    
    expected_files = [
        "models/lsm.py",
        "models/multimodal_lsm.py", 
        "models/hierarchical_fusion.py",
        "models/neurons.py",
        "models/attention.py",
        "models/readouts.py",
        "training/trainer.py",
        "training/losses.py",
        "training/plasticity.py",
        "datasets/maven_dataset.py",
        "datasets/loaders.py",
        "utils/config.py",
        "utils/logging.py",
        "utils/metrics.py",
        "algorithms/encoding.py",
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in expected_files:
        full_path = src_dir / file_path
        if full_path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    print(f"✓ {len(existing_files)} core files exist")
    
    if missing_files:
        print(f"⚠️  {len(missing_files)} files missing:")
        for file_path in missing_files:
            print(f"   - {file_path}")
    
    return len(missing_files) == 0

def main():
    """Run all basic structure tests."""
    print("🧠 SNN Fusion Framework - Basic Structure Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Basic Imports", test_basic_imports),
        ("Configuration Creation", test_config_creation),
        ("Logging Setup", test_logging_setup),
        ("Basic Encoding", test_encoding_basic),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"🧠 Structure Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL BASIC TESTS PASSED!")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())