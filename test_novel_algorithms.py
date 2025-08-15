#!/usr/bin/env python3
"""
Test Novel Neuromorphic Algorithms - Structural Validation

Tests the structural integrity and basic functionality of our novel algorithms
without requiring heavy dependencies. Validates:

1. Algorithm instantiation and configuration
2. Basic fusion pipeline execution
3. Output format validation
4. Error handling and edge cases
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all novel algorithms can be imported."""
    print("Testing imports...")
    
    try:
        from snn_fusion.algorithms.novel_ttfs_tsa_fusion import (
            NovelTTFSTSAFusion, create_novel_ttfs_tsa_fusion, TTFSEncodingMode
        )
        print("✓ TTFS-TSA Hybrid import successful")
    except Exception as e:
        print(f"❌ TTFS-TSA Hybrid import failed: {e}")
        return False
    
    try:
        from snn_fusion.algorithms.temporal_reversible_attention import (
            TemporalReversibleAttention, create_temporal_reversible_attention, ReversibilityConfig
        )
        print("✓ Temporal Reversible Attention import successful")
    except Exception as e:
        print(f"❌ Temporal Reversible Attention import failed: {e}")
        return False
    
    try:
        from snn_fusion.algorithms.hardware_aware_adaptive_attention import (
            HardwareAwareAdaptiveAttention, create_hardware_aware_adaptive_attention, HardwareType
        )
        print("✓ Hardware-Aware Adaptive Attention import successful")
    except Exception as e:
        print(f"❌ Hardware-Aware Adaptive Attention import failed: {e}")
        return False
    
    try:
        from snn_fusion.research.comprehensive_research_validation import (
            ComprehensiveResearchValidator, ValidationConfig
        )
        print("✓ Research Validation Framework import successful")
    except Exception as e:
        print(f"❌ Research Validation Framework import failed: {e}")
        return False
    
    return True


def test_algorithm_instantiation():
    """Test algorithm instantiation with various configurations."""
    print("\nTesting algorithm instantiation...")
    
    # Mock imports for testing
    try:
        from snn_fusion.algorithms.novel_ttfs_tsa_fusion import create_novel_ttfs_tsa_fusion, TTFSEncodingMode
        from snn_fusion.algorithms.temporal_reversible_attention import create_temporal_reversible_attention
        from snn_fusion.algorithms.hardware_aware_adaptive_attention import create_hardware_aware_adaptive_attention, HardwareType
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    modalities = ['audio', 'vision']
    
    # Test TTFS-TSA Hybrid
    try:
        ttfs_config = {
            'ttfs_encoding_mode': TTFSEncodingMode.ADAPTIVE_THRESHOLD,
            'target_sparsity': 0.95,
            'enable_cross_modal_sync': True,
        }
        ttfs_algorithm = create_novel_ttfs_tsa_fusion(modalities, config=ttfs_config)
        print("✓ TTFS-TSA Hybrid instantiation successful")
    except Exception as e:
        print(f"❌ TTFS-TSA Hybrid instantiation failed: {e}")
        return False
    
    # Test Temporal Reversible Attention
    try:
        reversible_algorithm = create_temporal_reversible_attention(modalities)
        print("✓ Temporal Reversible Attention instantiation successful")
    except Exception as e:
        print(f"❌ Temporal Reversible Attention instantiation failed: {e}")
        return False
    
    # Test Hardware-Aware Adaptive Attention
    try:
        hardware_algorithm = create_hardware_aware_adaptive_attention(
            modalities, hardware_type=HardwareType.CPU_EMULATION
        )
        print("✓ Hardware-Aware Adaptive Attention instantiation successful")
    except Exception as e:
        print(f"❌ Hardware-Aware Adaptive Attention instantiation failed: {e}")
        return False
    
    return True


def create_mock_modality_data():
    """Create mock modality data for testing."""
    try:
        from snn_fusion.algorithms.fusion import ModalityData
    except:
        # Create a simple mock class if import fails
        class ModalityData:
            def __init__(self, modality_name, spike_times, neuron_ids, features=None):
                self.modality_name = modality_name
                self.spike_times = spike_times
                self.neuron_ids = neuron_ids
                self.features = features
    
    # Simple mock data arrays (using lists instead of numpy arrays)
    audio_data = ModalityData(
        modality_name='audio',
        spike_times=[10.0, 25.0, 40.0, 55.0, 70.0],
        neuron_ids=[1, 5, 3, 7, 2],
        features=[0.8, 1.2, 0.9, 1.1, 0.7]
    )
    
    vision_data = ModalityData(
        modality_name='vision',
        spike_times=[15.0, 30.0, 45.0, 60.0],
        neuron_ids=[2, 4, 6, 8],
        features=[1.0, 0.9, 1.3, 0.8]
    )
    
    return {
        'audio': audio_data,
        'vision': vision_data
    }


def test_basic_fusion_pipeline():
    """Test basic fusion pipeline execution."""
    print("\nTesting fusion pipeline...")
    
    try:
        from snn_fusion.algorithms.novel_ttfs_tsa_fusion import create_novel_ttfs_tsa_fusion
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    modalities = ['audio', 'vision']
    
    try:
        # Create algorithm
        algorithm = create_novel_ttfs_tsa_fusion(modalities)
        print("✓ Algorithm created")
        
        # Create mock data
        modality_data = create_mock_modality_data()
        print("✓ Mock data created")
        
        # Test fusion
        fusion_result = algorithm.fuse_modalities(modality_data)
        print("✓ Fusion executed successfully")
        
        # Validate result structure
        assert hasattr(fusion_result, 'fused_spikes'), "Missing fused_spikes attribute"
        assert hasattr(fusion_result, 'fusion_weights'), "Missing fusion_weights attribute"
        assert hasattr(fusion_result, 'confidence_scores'), "Missing confidence_scores attribute"
        assert hasattr(fusion_result, 'metadata'), "Missing metadata attribute"
        print("✓ Result structure validated")
        
        # Check metadata contains expected TTFS info
        metadata = fusion_result.metadata
        assert 'fusion_type' in metadata, "Missing fusion_type in metadata"
        assert metadata['fusion_type'] == 'novel_ttfs_tsa', "Incorrect fusion_type"
        print("✓ Metadata validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Fusion pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    try:
        from snn_fusion.algorithms.novel_ttfs_tsa_fusion import create_novel_ttfs_tsa_fusion
        from snn_fusion.algorithms.fusion import ModalityData
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    modalities = ['audio', 'vision']
    algorithm = create_novel_ttfs_tsa_fusion(modalities)
    
    # Test empty data
    try:
        empty_data = {
            'audio': ModalityData('audio', [], [], []),
            'vision': ModalityData('vision', [], [], [])
        }
        result = algorithm.fuse_modalities(empty_data)
        print("✓ Empty data handling successful")
    except Exception as e:
        print(f"❌ Empty data handling failed: {e}")
        return False
    
    # Test single modality
    try:
        single_modality_data = {
            'audio': ModalityData('audio', [10.0, 20.0], [1, 2], [0.5, 0.8])
        }
        result = algorithm.fuse_modalities(single_modality_data)
        print("✓ Single modality handling successful")
    except Exception as e:
        print(f"❌ Single modality handling failed: {e}")
        return False
    
    return True


def test_algorithm_analysis():
    """Test algorithm analysis and statistics."""
    print("\nTesting algorithm analysis...")
    
    try:
        from snn_fusion.algorithms.novel_ttfs_tsa_fusion import create_novel_ttfs_tsa_fusion
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    modalities = ['audio', 'vision']
    algorithm = create_novel_ttfs_tsa_fusion(modalities)
    
    # Test analysis methods
    try:
        analysis = algorithm.get_research_analysis()
        print("✓ Research analysis successful")
        
        # Validate analysis structure
        assert 'ttfs_metrics' in analysis, "Missing ttfs_metrics"
        assert 'encoder_adaptation' in analysis, "Missing encoder_adaptation"
        assert 'hardware_efficiency' in analysis, "Missing hardware_efficiency"
        print("✓ Analysis structure validated")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("NOVEL NEUROMORPHIC ALGORITHMS - STRUCTURAL VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Instantiation Tests", test_algorithm_instantiation),
        ("Fusion Pipeline Tests", test_basic_fusion_pipeline),
        ("Edge Case Tests", test_edge_cases),
        ("Analysis Tests", test_algorithm_analysis),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - ALGORITHMS STRUCTURALLY VALID")
        return True
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW IMPLEMENTATIONS")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)