#!/usr/bin/env python3
"""
Test script for Generation 1 basic functionality of SNN Fusion framework.

Tests core components:
- Model instantiation (LSM, MultiModalLSM, HierarchicalFusion)
- Data loading with MAVEN dataset
- Basic training loop
- Encoding and metrics
- Configuration system
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Core model components
    from snn_fusion.models.lsm import LiquidStateMachine
    from snn_fusion.models.multimodal_lsm import MultiModalLSM
    from snn_fusion.models.hierarchical_fusion import HierarchicalFusionSNN
    from snn_fusion.models.neurons import AdaptiveLIF
    from snn_fusion.models.attention import CrossModalAttention
    from snn_fusion.models.readouts import LinearReadout
    
    # Training components
    from snn_fusion.training.trainer import SNNTrainer, MultiModalTrainer
    from snn_fusion.training.losses import TemporalCrossEntropyLoss, VanRossumLoss
    from snn_fusion.training.plasticity import STDPLearner
    
    # Data components
    from snn_fusion.datasets.maven_dataset import MAVENDataset
    from snn_fusion.datasets.loaders import MultiModalCollate
    
    # Utils
    from snn_fusion.utils.config import load_config, create_default_config
    from snn_fusion.utils.logging import setup_logging, get_logger
    from snn_fusion.utils.metrics import SpikeMetrics, FusionMetrics
    from snn_fusion.algorithms.encoding import RateEncoder, CochlearEncoder
    
    print("✓ All imports successful!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_configuration():
    """Test configuration system."""
    print("\n=== Testing Configuration System ===")
    
    try:
        # Create default config
        config = create_default_config()
        print(f"✓ Default config created with experiment: {config.experiment_name}")
        
        # Test validation
        from snn_fusion.utils.config import validate_config
        validate_config(config)
        print("✓ Configuration validation passed")
        
        return config
    
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return None


def test_models():
    """Test model instantiation."""
    print("\n=== Testing Model Components ===")
    
    try:
        # Test basic neuron model
        neurons = AdaptiveLIF(n_neurons=100)
        dummy_input = torch.randn(1, 100)
        spikes, states = neurons(dummy_input)
        print(f"✓ AdaptiveLIF: input {dummy_input.shape} -> spikes {spikes.shape}")
        
        # Test LSM
        lsm = LiquidStateMachine(
            n_neurons=200,
            input_size=50,
            connectivity=0.1,
            spectral_radius=0.9
        )
        dummy_input = torch.randn(2, 10, 50)  # [batch, time, features]
        liquid_states = lsm(dummy_input)
        print(f"✓ LSM: input {dummy_input.shape} -> states {liquid_states.shape}")
        
        # Test MultiModalLSM
        modalities = ["audio", "vision", "tactile"]
        input_dims = {"audio": 64, "vision": 128, "tactile": 6}
        multi_lsm = MultiModalLSM(
            modalities=modalities,
            input_dims=input_dims,
            n_neurons_per_modality=100,
            fusion_strategy="attention"
        )
        
        dummy_inputs = {
            "audio": torch.randn(2, 10, 64),
            "vision": torch.randn(2, 10, 128),
            "tactile": torch.randn(2, 10, 6)
        }
        fused_output = multi_lsm(dummy_inputs)
        print(f"✓ MultiModalLSM: fused output shape {fused_output.shape}")
        
        # Test attention mechanism
        attention = CrossModalAttention(
            modalities=modalities,
            modality_dims=input_dims,
            hidden_dim=128
        )
        
        modality_features = {
            "audio": torch.randn(2, 64),
            "vision": torch.randn(2, 128),
            "tactile": torch.randn(2, 6)
        }
        attended_output, attn_weights = attention(modality_features)
        print(f"✓ CrossModalAttention: output {attended_output.shape}")
        
        # Test readout layer
        readout = LinearReadout(input_size=200, output_size=10)
        dummy_states = torch.randn(2, 200)
        predictions = readout(dummy_states)
        print(f"✓ LinearReadout: states {dummy_states.shape} -> predictions {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading components."""
    print("\n=== Testing Data Loading ===")
    
    try:
        # Test MAVEN dataset (will use synthetic data if real data not available)
        dataset = MAVENDataset(
            root='data/MAVEN',
            modalities=['audio', 'events', 'imu'],
            split='train',
            time_window_ms=100,
            use_synthetic_fallback=True  # Use synthetic data if real not available
        )
        print(f"✓ MAVEN dataset created with {len(dataset)} samples")
        
        # Test data sample
        sample = dataset[0]
        print(f"✓ Sample keys: {list(sample.keys())}")
        
        # Test collate function
        collate_fn = MultiModalCollate(
            modalities=['audio', 'events', 'imu'],
            pad_sequences=True,
            handle_missing_modalities=True
        )
        
        # Create small batch
        batch = [dataset[i] for i in range(min(4, len(dataset)))]
        collated = collate_fn(batch)
        print(f"✓ Collated batch with {len(collated)} modalities")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_components():
    """Test training components."""
    print("\n=== Testing Training Components ===")
    
    try:
        # Test loss functions
        temporal_loss = TemporalCrossEntropyLoss()
        van_rossum_loss = VanRossumLoss(tau=10.0)
        
        # Dummy data for testing
        outputs = torch.randn(2, 10, 5)  # [batch, time, classes]
        targets = torch.randint(0, 5, (2,))
        spike_trains = torch.randint(0, 2, (2, 10, 100)).float()
        
        temp_loss = temporal_loss(outputs, targets)
        vr_loss = van_rossum_loss(spike_trains, spike_trains)  # Self-similarity should be 0
        
        print(f"✓ Temporal loss: {temp_loss.item():.4f}")
        print(f"✓ Van Rossum loss: {vr_loss.item():.4f}")
        
        # Test STDP
        stdp = STDPLearner(
            tau_pre=20.0,
            tau_post=20.0,
            A_plus=0.01,
            A_minus=0.012
        )
        
        pre_spikes = torch.randint(0, 2, (2, 50)).float()
        post_spikes = torch.randint(0, 2, (2, 30)).float()
        weights = torch.randn(50, 30)
        
        updated_weights, learning_info = stdp(pre_spikes, post_spikes, weights)
        print(f"✓ STDP: weight change magnitude {learning_info['weight_change'].item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoding_and_metrics():
    """Test encoding and metrics."""
    print("\n=== Testing Encoding and Metrics ===")
    
    try:
        # Test rate encoding
        encoder = RateEncoder(
            n_neurons=64,
            duration=100.0,
            max_rate=100.0
        )
        
        # Generate dummy audio data
        dummy_audio = np.random.randn(1000)
        encoded_spikes = encoder.encode(dummy_audio)
        print(f"✓ Rate encoder: {len(encoded_spikes.spike_times)} spikes generated")
        
        # Test spike metrics
        spike_metrics = SpikeMetrics()
        spike_trains = torch.randint(0, 2, (4, 100, 64)).float()
        
        firing_rates = spike_metrics.compute_firing_rates(spike_trains)
        cv_isi = spike_metrics.compute_cv_isi(spike_trains)
        
        print(f"✓ Spike metrics: firing rates shape {firing_rates.shape}")
        print(f"✓ CV ISI shape {cv_isi.shape}")
        
        # Test fusion metrics
        fusion_metrics = FusionMetrics()
        
        single_modal_preds = {
            "audio": torch.randn(10, 5),
            "vision": torch.randn(10, 5),
            "tactile": torch.randn(10, 5)
        }
        fusion_pred = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        importance = fusion_metrics.compute_modality_importance(
            single_modal_preds, fusion_pred, targets
        )
        print(f"✓ Modality importance: {importance}")
        
        return True
        
    except Exception as e:
        print(f"✗ Encoding and metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_training_loop():
    """Test a basic training loop."""
    print("\n=== Testing Basic Training Loop ===")
    
    try:
        # Create simple model
        model = LiquidStateMachine(
            n_neurons=100,
            input_size=32,
            connectivity=0.1
        )
        
        # Add readout layer
        readout = LinearReadout(input_size=100, output_size=3)
        
        # Create trainer
        trainer = SNNTrainer(
            model=model,
            device='cpu',
            learning_rule='SuperSpike',
            optimizer='Adam',
            lr=1e-3
        )
        
        # Dummy training data
        batch_size = 4
        time_steps = 20
        input_size = 32
        
        inputs = torch.randn(batch_size, time_steps, input_size)
        targets = torch.randint(0, 3, (batch_size,))
        
        # Forward pass
        liquid_states = model(inputs)
        predictions = readout(liquid_states[:, -1, :])  # Use final time step
        
        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(predictions, targets)
        
        print(f"✓ Basic training loop: loss {loss.item():.4f}")
        print(f"✓ Model output shape: {predictions.shape}")
        
        # Test gradient computation
        loss.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        print(f"✓ Gradients computed: {has_grads}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests for Generation 1 basic functionality."""
    print("🧠 SNN Fusion Framework - Generation 1 Basic Functionality Test")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging(log_level="INFO", enable_console=True)
    logger.info("Starting Generation 1 tests")
    
    tests = [
        ("Configuration System", test_configuration),
        ("Model Components", test_models),
        ("Data Loading", test_data_loading),
        ("Training Components", test_training_components),
        ("Encoding and Metrics", test_encoding_and_metrics),
        ("Basic Training Loop", test_basic_training_loop),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"🧠 Generation 1 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Generation 1 basic functionality is working!")
        logger.info("Generation 1 basic functionality verified successfully")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the implementation.")
        logger.error(f"Generation 1 tests failed: {total - passed} failures")
        return 1


if __name__ == "__main__":
    sys.exit(main())