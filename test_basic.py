#!/usr/bin/env python3
"""
Basic functionality test for SNN-Fusion package.
Tests core functionality without requiring heavy dependencies.
"""

import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test basic package imports."""
    print("🧪 Testing Package Imports...")
    
    test_results = []
    
    # Test 1: Basic package import
    try:
        from snn_fusion import __version__, __author__
        print(f"✅ Package info: v{__version__} by {__author__}")
        test_results.append(("Package import", True))
    except Exception as e:
        print(f"❌ Package import failed: {e}")
        test_results.append(("Package import", False))
        return test_results
    
    # Test 2: Model imports
    try:
        from snn_fusion.models.hierarchical_fusion import HierarchicalFusionSNN
        from snn_fusion.models.multimodal_lsm import MultiModalLSM
        print("✅ Model imports successful")
        test_results.append(("Model imports", True))
    except Exception as e:
        print(f"❌ Model imports failed: {e}")
        traceback.print_exc()
        test_results.append(("Model imports", False))
    
    # Test 3: Training imports  
    try:
        from snn_fusion.training.plasticity import STDPPlasticity
        from snn_fusion.training.losses import TemporalLoss
        print("✅ Training imports successful")
        test_results.append(("Training imports", True))
    except Exception as e:
        print(f"❌ Training imports failed: {e}")
        traceback.print_exc()
        test_results.append(("Training imports", False))
    
    # Test 4: Dataset imports
    try:
        from snn_fusion.datasets.maven_dataset import MAVENDataset, MAVENConfig
        from snn_fusion.datasets.loaders import MultiModalDataLoader
        print("✅ Dataset imports successful")
        test_results.append(("Dataset imports", True))
    except Exception as e:
        print(f"❌ Dataset imports failed: {e}")
        traceback.print_exc()
        test_results.append(("Dataset imports", False))
    
    return test_results

def test_basic_functionality():
    """Test basic functionality without heavy computations."""
    print("\n🧪 Testing Basic Functionality...")
    
    test_results = []
    
    try:
        # Test dataset configuration
        from snn_fusion.datasets.maven_dataset import MAVENConfig
        
        config = MAVENConfig(
            root_dir="./test_data",
            modalities=['audio', 'events', 'imu'],
            split='train',
            sequence_length=50
        )
        
        print("✅ Dataset configuration creation successful")
        test_results.append(("Dataset config", True))
        
    except Exception as e:
        print(f"❌ Dataset configuration failed: {e}")
        test_results.append(("Dataset config", False))
    
    try:
        # Test model configuration (without torch)
        from snn_fusion.models.hierarchical_fusion import create_default_hierarchical_config
        
        input_shapes = {
            'audio': (64,),
            'events': (346, 260, 2),
            'imu': (6,)
        }
        
        config = create_default_hierarchical_config(
            input_shapes=input_shapes,
            num_levels=2,
            base_hidden_dim=128
        )
        
        print("✅ Model configuration creation successful")
        test_results.append(("Model config", True))
        
    except Exception as e:
        print(f"❌ Model configuration failed: {e}")
        test_results.append(("Model config", False))
    
    return test_results

def test_cli_availability():
    """Test CLI command availability."""
    print("\n🧪 Testing CLI Availability...")
    
    test_results = []
    
    try:
        from snn_fusion import cli
        print("✅ CLI module import successful")
        test_results.append(("CLI import", True))
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        test_results.append(("CLI import", False))
    
    # Test if entry points are properly configured
    try:
        import pkg_resources
        entry_points = pkg_resources.get_entry_map('snn-fusion')
        if 'console_scripts' in entry_points:
            scripts = entry_points['console_scripts']
            print(f"✅ Entry points configured: {list(scripts.keys())}")
            test_results.append(("Entry points", True))
        else:
            print("⚠️  No console scripts configured")
            test_results.append(("Entry points", False))
    except Exception as e:
        print(f"⚠️  Could not check entry points: {e}")
        test_results.append(("Entry points", False))
    
    return test_results

def run_all_tests():
    """Run all tests and summarize results."""
    print("🚀 Starting SNN-Fusion Basic Tests\n")
    
    all_results = []
    
    # Run test suites
    all_results.extend(test_imports())
    all_results.extend(test_basic_functionality())
    all_results.extend(test_cli_availability())
    
    # Summarize results
    print("\n📊 Test Summary:")
    print("=" * 50)
    
    passed = sum(1 for _, result in all_results if result)
    total = len(all_results)
    
    for test_name, result in all_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print("=" * 50)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The basic functionality is working.")
        return True
    else:
        print("⚠️  Some tests failed. See details above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)