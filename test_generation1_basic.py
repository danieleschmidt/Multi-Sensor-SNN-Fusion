#!/usr/bin/env python3
"""
Generation 1 Basic Functionality Test

Test the core pipeline without PyTorch dependencies to verify basic functionality.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test basic imports work."""
    print("Testing imports...")
    
    try:
        from snn_fusion.algorithms.encoding import MultiModalEncoder, RateEncoder
        from snn_fusion.datasets.synthetic import create_synthetic_dataset
        from snn_fusion.utils.logging import setup_logging
        from snn_fusion.utils.config import create_debug_config
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_encoding():
    """Test spike encoding functionality."""
    print("Testing spike encoding...")
    
    try:
        from snn_fusion.algorithms.encoding import RateEncoder, MultiModalEncoder
        
        # Test rate encoder
        encoder = RateEncoder(n_neurons=64, duration=100.0, dt=1.0)
        test_data = np.random.rand(64)
        spike_data = encoder.encode(test_data)
        
        print(f"  Rate encoder: {len(spike_data.spike_times)} spikes generated")
        
        # Test multi-modal encoder
        mm_encoder = MultiModalEncoder(
            audio_config={'n_neurons': 32},
            visual_config={'width': 16, 'height': 16},
            tactile_config={'n_sensors': 8},
            duration=50.0
        )
        
        # Test encoding
        audio_data = np.random.randn(800)  # 50ms at 16kHz
        visual_data = np.random.rand(5, 16, 16)  # 5 frames
        tactile_data = np.random.rand(50, 8)  # 50 timesteps
        
        combined_spikes = mm_encoder.encode(
            audio_data=audio_data,
            visual_data=visual_data,
            tactile_data=tactile_data
        )
        
        print(f"  Multi-modal encoder: {len(combined_spikes.spike_times)} spikes from {combined_spikes.n_neurons} neurons")
        print("✓ Encoding tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Encoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_logging():
    """Test logging setup."""
    print("Testing logging...")
    
    try:
        from snn_fusion.utils.logging import setup_logging, LogLevel
        
        logger = setup_logging(
            log_level=LogLevel.INFO,
            enable_console=False  # Don't spam console
        )
        
        logger.info("Test log message")
        print("✓ Logging setup successful")
        return True
        
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False

def test_config():
    """Test configuration system."""
    print("Testing configuration...")
    
    try:
        from snn_fusion.utils.config import create_debug_config, validate_config
        
        config = create_debug_config()
        validate_config(config)
        
        print(f"  Config model type: {config.model.model_type}")
        print(f"  Config neurons: {config.model.n_neurons}")
        print("✓ Configuration tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Generation 1 tests."""
    print("=" * 60)
    print("SNN-FUSION GENERATION 1 BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Spike Encoding", test_encoding),
        ("Logging", test_logging),
        ("Configuration", test_config),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("GENERATION 1 TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} [{status}]")
        if result:
            passed += 1
    
    print(f"\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Generation 1 basic functionality COMPLETE!")
        return True
    else:
        print("❌ Some tests failed - need fixes")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)