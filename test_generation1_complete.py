#!/usr/bin/env python3
"""
Generation 1 Complete Test Suite
Tests all basic functionality implemented in Generation 1
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import snn_fusion
from snn_fusion.models import LiquidStateMachine, MultiModalLSM
from snn_fusion.datasets import MAVENDataset, MAVENConfig, create_maven_config
from snn_fusion.models.neurons import AdaptiveLIF
from snn_fusion.models.readouts import LinearReadout, TemporalReadout
from snn_fusion.models.attention import CrossModalAttention

def test_basic_components():
    """Test basic SNN components."""
    print("🧪 Testing Basic Components...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test AdaptiveLIF neurons
    print("  Testing AdaptiveLIF neurons...")
    neurons = AdaptiveLIF(n_neurons=100, device=device)
    input_current = torch.randn(32, 100, device=device)  # batch_size=32
    spikes, states = neurons(input_current)
    assert spikes.shape == (32, 100), f"Expected (32, 100), got {spikes.shape}"
    assert 'v_mem' in states
    print(f"    ✓ Neurons firing rate: {spikes.mean().item():.3f}")
    
    # Test Linear Readout
    print("  Testing LinearReadout...")
    readout = LinearReadout(input_size=400, output_size=10, device=device)
    liquid_state = torch.randn(32, 400, device=device)
    outputs = readout(liquid_state)
    assert outputs.shape == (32, 10), f"Expected (32, 10), got {outputs.shape}"
    print(f"    ✓ Readout output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
    
    print("✅ Basic Components Test Passed")

def test_liquid_state_machine():
    """Test core LSM functionality."""
    print("🧪 Testing Liquid State Machine...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create LSM
    lsm = LiquidStateMachine(
        n_inputs=64,
        n_reservoir=500,
        n_outputs=10,
        connectivity=0.1,
        spectral_radius=0.9,
        device=device
    )
    
    # Test forward pass
    batch_size, time_steps, n_inputs = 16, 100, 64
    input_spikes = torch.rand(batch_size, time_steps, n_inputs, device=device) > 0.7
    input_spikes = input_spikes.float()
    
    outputs, states = lsm(input_spikes, return_states=True)
    
    assert outputs.shape == (batch_size, 10), f"Expected (16, 10), got {outputs.shape}"
    assert 'reservoir_states' in states
    assert 'spike_history' in states
    
    print(f"    ✓ LSM output shape: {outputs.shape}")
    print(f"    ✓ Reservoir activity: {states['spike_history'].mean().item():.3f}")
    
    # Test statistics
    stats = lsm.get_reservoir_statistics()
    print(f"    ✓ Connectivity: {stats['actual_connectivity']:.3f}")
    print(f"    ✓ Spectral radius: {stats['spectral_radius']:.3f}")
    
    print("✅ Liquid State Machine Test Passed")

def test_maven_dataset():
    """Test MAVEN dataset functionality."""
    print("🧪 Testing MAVEN Dataset...")
    
    # Create dataset config
    config = create_maven_config(
        root_dir="./data/test_maven",
        modalities=['audio', 'events', 'imu'],
        split='train',
        sequence_length=50
    )
    
    # Initialize dataset (will create synthetic data)
    dataset = MAVENDataset(config)
    
    assert len(dataset) > 0, "Dataset should have samples"
    
    # Test sample loading
    sample = dataset[0]
    
    assert 'audio' in sample, "Sample should have audio data"
    assert 'events' in sample, "Sample should have events data"
    assert 'imu' in sample, "Sample should have IMU data"
    assert 'label' in sample, "Sample should have label"
    
    print(f"    ✓ Dataset size: {len(dataset)}")
    print(f"    ✓ Sample keys: {list(sample.keys())}")
    
    for modality in ['audio', 'events', 'imu']:
        data = sample[modality]
        print(f"    ✓ {modality} shape: {data.shape}")
    
    # Test statistics
    stats = dataset.get_modality_statistics()
    print(f"    ✓ Computed statistics for {len(stats)} modalities")
    
    print("✅ MAVEN Dataset Test Passed")

def test_cross_modal_attention():
    """Test cross-modal attention mechanism."""
    print("🧪 Testing Cross-Modal Attention...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    modalities = ['audio', 'vision', 'tactile']
    modality_dims = {'audio': 64, 'vision': 128, 'tactile': 32}
    
    attention = CrossModalAttention(
        modalities=modalities,
        modality_dims=modality_dims,
        hidden_dim=128,
        num_heads=4,
        device=device
    )
    
    # Create mock modality outputs
    batch_size = 16
    modality_outputs = {
        'audio': torch.randn(batch_size, 64, device=device),
        'vision': torch.randn(batch_size, 128, device=device),
        'tactile': torch.randn(batch_size, 32, device=device),
    }
    
    fused_output, attention_dict = attention(modality_outputs)
    
    assert fused_output.shape == (batch_size, 128), f"Expected (16, 128), got {fused_output.shape}"
    assert 'cross_modal_attention' in attention_dict
    assert 'modality_importance' in attention_dict
    
    print(f"    ✓ Fused output shape: {fused_output.shape}")
    print(f"    ✓ Available modalities: {attention_dict['available_modalities']}")
    
    # Test with missing modalities
    partial_outputs = {'audio': modality_outputs['audio'], 'vision': modality_outputs['vision']}
    fused_partial, _ = attention(partial_outputs, available_modalities=['audio', 'vision'])
    
    assert fused_partial.shape == (batch_size, 128), "Should handle missing modalities"
    print(f"    ✓ Partial fusion (2/3 modalities) successful")
    
    print("✅ Cross-Modal Attention Test Passed")

def test_multimodal_lsm():
    """Test MultiModal LSM integration."""
    print("🧪 Testing MultiModal LSM...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configure modalities
    modality_configs = {
        'audio': {'n_inputs': 64, 'n_reservoir': 300, 'n_outputs': 32},
        'events': {'n_inputs': 256, 'n_reservoir': 400, 'n_outputs': 32},
        'imu': {'n_inputs': 6, 'n_reservoir': 200, 'n_outputs': 32}
    }
    
    # Create MultiModal LSM
    multi_lsm = MultiModalLSM(
        modality_configs=modality_configs,
        n_outputs=10,
        fusion_type="attention",
        device=device
    )
    
    # Create test inputs
    batch_size, time_steps = 8, 50
    inputs = {
        'audio': torch.rand(batch_size, time_steps, 64, device=device) > 0.5,
        'events': torch.rand(batch_size, time_steps, 256, device=device) > 0.3,
        'imu': torch.randn(batch_size, time_steps, 6, device=device)
    }
    
    # Convert to float
    for key in inputs:
        inputs[key] = inputs[key].float()
    
    # Forward pass
    outputs, states = multi_lsm(inputs, return_states=True)
    
    assert outputs.shape == (batch_size, 10), f"Expected (8, 10), got {outputs.shape}"
    assert 'modality_states' in states
    assert 'fusion_attention' in states
    
    print(f"    ✓ MultiModal LSM output shape: {outputs.shape}")
    print(f"    ✓ Processed modalities: {states['available_modalities']}")
    
    # Test single modality
    audio_only = multi_lsm.forward_single_modality('audio', inputs['audio'])
    assert audio_only.shape == (batch_size, 10), "Single modality should work"
    print(f"    ✓ Single modality (audio) output: {audio_only.shape}")
    
    # Test missing modalities
    partial_inputs = {'audio': inputs['audio'], 'imu': inputs['imu']}
    partial_outputs, _ = multi_lsm(partial_inputs, missing_modalities=['events'])
    assert partial_outputs.shape == (batch_size, 10), "Should handle missing modalities"
    print(f"    ✓ Partial inputs (2/3 modalities) successful")
    
    # Get statistics
    stats = multi_lsm.get_modality_statistics()
    print(f"    ✓ Statistics for {len(stats)} modality LSMs")
    
    print("✅ MultiModal LSM Test Passed")

def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline."""
    print("🧪 Testing End-to-End Pipeline...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Create dataset
    config = create_maven_config(
        root_dir="./data/test_e2e",
        modalities=['audio', 'events', 'imu'],
        split='train',
        sequence_length=100
    )
    dataset = MAVENDataset(config)
    
    # 2. Create model
    modality_configs = {
        'audio': {'n_inputs': 128, 'n_reservoir': 300},  # Audio cochleagram channels
        'events': {'n_inputs': 1794, 'n_reservoir': 400},  # Flattened event frames
        'imu': {'n_inputs': 6, 'n_reservoir': 200}
    }
    
    model = MultiModalLSM(
        modality_configs=modality_configs,
        n_outputs=10,
        fusion_type="attention",
        device=device
    )
    
    # 3. Process samples
    sample = dataset[0]
    
    # Reshape data for model input
    inputs = {}
    for modality in ['audio', 'events', 'imu']:
        data = sample[modality].unsqueeze(0).to(device)  # Add batch dimension
        
        if modality == 'audio':
            # Reshape audio: [batch, time, channels, freq] -> [batch, time, channels*freq]
            if data.dim() == 4:
                b, t, c, f = data.shape
                data = data.view(b, t, c * f)
        elif modality == 'events':
            # Reshape events: [batch, time, H, W, pol] -> [batch, time, H*W*pol]
            if data.dim() == 5:
                b, t, h, w, p = data.shape
                data = data.view(b, t, h * w * p)
        
        inputs[modality] = data
    
    # 4. Forward pass
    outputs, states = model(inputs, return_states=True)
    
    assert outputs.shape[0] == 1, "Batch size should be 1"
    assert outputs.shape[1] == 10, "Should have 10 output classes"
    
    print(f"    ✓ End-to-end pipeline successful")
    print(f"    ✓ Input shapes: {[inputs[mod].shape for mod in inputs]}")
    print(f"    ✓ Output shape: {outputs.shape}")
    print(f"    ✓ Prediction: {torch.argmax(outputs, dim=1).item()}")
    print(f"    ✓ True label: {sample['label'].item()}")
    
    print("✅ End-to-End Pipeline Test Passed")

def main():
    """Run all Generation 1 tests."""
    print("🚀 Starting Generation 1 Complete Test Suite")
    print("=" * 60)
    
    try:
        test_basic_components()
        print()
        
        test_liquid_state_machine()
        print()
        
        test_maven_dataset()
        print()
        
        test_cross_modal_attention()
        print()
        
        test_multimodal_lsm()
        print()
        
        test_end_to_end_pipeline()
        print()
        
        print("🎉 ALL GENERATION 1 TESTS PASSED!")
        print("✅ Basic functionality is working correctly")
        print("✅ Multi-modal fusion is functional")
        print("✅ End-to-end pipeline is operational")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)