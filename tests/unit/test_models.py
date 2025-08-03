"""
Unit tests for neuromorphic models.

Tests core functionality of LSMs, neurons, and multi-modal architectures
with focus on correctness and biological plausibility.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any

from snn_fusion.models import (
    LiquidStateMachine,
    MultiModalLSM,
    AdaptiveLIF,
    LinearReadout,
    TemporalReadout
)


class TestAdaptiveLIF:
    """Test adaptive leaky integrate-and-fire neurons."""
    
    def test_initialization(self, device):
        """Test neuron initialization."""
        n_neurons = 100
        neuron = AdaptiveLIF(
            n_neurons=n_neurons,
            tau_mem=20.0,
            tau_adapt=100.0,
            device=device
        )
        
        assert neuron.n_neurons == n_neurons
        assert neuron.tau_mem == 20.0
        assert neuron.tau_adapt == 100.0
        assert neuron.device == device
        
        # Check initial states
        assert neuron.v_mem.shape == (n_neurons,)
        assert neuron.v_adapt.shape == (n_neurons,)
        assert torch.all(neuron.v_mem == 0.0)
        assert torch.all(neuron.v_adapt == 0.0)
    
    def test_forward_pass(self, device):
        """Test forward pass through neurons."""
        n_neurons = 50
        batch_size = 4
        neuron = AdaptiveLIF(n_neurons=n_neurons, device=device)
        
        # Test single input
        input_current = torch.randn(n_neurons, device=device)
        spikes, states = neuron(input_current)
        
        assert spikes.shape == (1, n_neurons)
        assert torch.all((spikes == 0) | (spikes == 1))  # Binary spikes
        
        # Test batch input
        batch_input = torch.randn(batch_size, n_neurons, device=device)
        spikes, states = neuron(batch_input)
        
        assert spikes.shape == (batch_size, n_neurons)
        assert 'v_mem' in states
        assert 'v_adapt' in states
        assert states['v_mem'].shape == (batch_size, n_neurons)
    
    def test_adaptation_mechanism(self, device):
        """Test threshold adaptation mechanism."""
        n_neurons = 10
        neuron = AdaptiveLIF(
            n_neurons=n_neurons,
            v_threshold=1.0,
            adapt_increment=0.1,
            device=device
        )
        
        # Strong input to trigger spikes
        strong_input = torch.ones(n_neurons, device=device) * 2.0
        
        # First spike
        spikes1, states1 = neuron(strong_input)
        initial_adapt = states1['v_adapt'].clone()
        
        # Second spike should have higher threshold
        spikes2, states2 = neuron(strong_input)
        
        # Adaptation should increase for spiking neurons
        spiked_neurons = spikes1.squeeze().bool()
        if spiked_neurons.any():
            assert torch.all(states2['v_adapt'][0, spiked_neurons] > initial_adapt[0, spiked_neurons])
    
    def test_reset_state(self, device):
        """Test state reset functionality."""
        neuron = AdaptiveLIF(n_neurons=20, device=device)
        
        # Apply input to change states
        input_current = torch.randn(20, device=device)
        neuron(input_current)
        
        # States should be non-zero
        assert not torch.all(neuron.v_mem == 0.0)
        
        # Reset states
        neuron.reset_state()
        
        # States should be zero
        assert torch.all(neuron.v_mem == 0.0)
        assert torch.all(neuron.v_adapt == 0.0)
        assert torch.all(neuron.spike_count == 0.0)
    
    def test_firing_rates(self, device):
        """Test firing rate computation."""
        n_neurons = 100
        neuron = AdaptiveLIF(n_neurons=n_neurons, device=device)
        
        # Simulate for multiple time steps
        n_steps = 1000
        for _ in range(n_steps):
            input_current = torch.randn(n_neurons, device=device) * 0.5
            neuron(input_current)
        
        # Compute firing rates
        firing_rates = neuron.get_firing_rates(time_window=n_steps)
        
        assert firing_rates.shape == (n_neurons,)
        assert torch.all(firing_rates >= 0.0)
        assert torch.all(firing_rates <= 1000.0)  # Max 1kHz for 1ms timesteps


class TestLiquidStateMachine:
    """Test liquid state machine functionality."""
    
    def test_initialization(self, simple_lsm):
        """Test LSM initialization."""
        lsm = simple_lsm
        
        assert lsm.n_inputs == 10
        assert lsm.n_reservoir == 50
        assert lsm.n_outputs == 3
        assert lsm.connectivity == 0.1
        
        # Check weight matrices
        assert lsm.W_input.shape == (50, 10)
        assert lsm.W_reservoir.shape == (50, 50)
        
        # Check connectivity
        input_connections = (lsm.W_input != 0).sum().item()
        reservoir_connections = (lsm.W_reservoir != 0).sum().item()
        
        # Should have sparse connectivity
        assert input_connections < 50 * 10  # Less than fully connected
        assert reservoir_connections < 50 * 50
    
    def test_forward_pass(self, simple_lsm, device):
        """Test forward pass through LSM."""
        lsm = simple_lsm
        batch_size = 2
        time_steps = 20
        
        # Create input spike trains
        input_spikes = torch.randn(batch_size, time_steps, 10, device=device)
        
        # Forward pass
        outputs, states = lsm(input_spikes, return_states=True)
        
        assert outputs.shape == (batch_size, 3)
        assert states is not None
        assert 'reservoir_states' in states
        assert 'spike_history' in states
        assert states['reservoir_states'].shape == (batch_size, time_steps, 50)
        assert states['spike_history'].shape == (batch_size, time_steps, 50)
    
    def test_temporal_processing(self, simple_lsm, device):
        """Test temporal dynamics of LSM."""
        lsm = simple_lsm
        time_steps = 50
        
        # Create temporally structured input
        input_spikes = torch.zeros(1, time_steps, 10, device=device)
        input_spikes[0, 10:15, :] = 1.0  # Burst at specific time
        
        outputs, states = lsm(input_spikes, return_states=True, reset_state=True)
        
        reservoir_activity = states['reservoir_states'].squeeze()  # [time_steps, n_reservoir]
        
        # Activity should propagate through time
        early_activity = reservoir_activity[:10].sum()
        burst_activity = reservoir_activity[10:20].sum()
        late_activity = reservoir_activity[20:30].sum()
        
        # Should see increased activity during and after burst
        assert burst_activity > early_activity
        # Late activity should be non-zero due to reservoir dynamics
        assert late_activity > 0
    
    def test_reservoir_statistics(self, simple_lsm):
        """Test reservoir connectivity statistics."""
        lsm = simple_lsm
        stats = lsm.get_reservoir_statistics()
        
        assert 'actual_connectivity' in stats
        assert 'spectral_radius' in stats
        assert 'input_connectivity' in stats
        
        # Connectivity should be close to target
        assert abs(stats['actual_connectivity'] - lsm.connectivity) < 0.05
        
        # Spectral radius should be close to target
        assert abs(stats['spectral_radius'] - lsm.spectral_radius) < 0.1
    
    def test_plasticity_enable_disable(self, simple_lsm):
        """Test plasticity control."""
        lsm = simple_lsm
        
        # Initially disabled
        assert not lsm.plasticity_enabled
        
        # Enable plasticity
        lsm.enable_plasticity("stdp")
        assert lsm.plasticity_enabled
        
        # Disable plasticity
        lsm.disable_plasticity()
        assert not lsm.plasticity_enabled
    
    @pytest.mark.slow
    def test_state_persistence(self, simple_lsm, temp_dir, device):
        """Test saving and loading reservoir states."""
        lsm = simple_lsm
        
        # Run LSM to generate states
        input_spikes = torch.randn(2, 30, 10, device=device)
        lsm(input_spikes)
        
        # Save states
        save_path = temp_dir / "lsm_states.pt"
        lsm.save_liquid_states(str(save_path))
        
        assert save_path.exists()
        
        # Create new LSM and load states
        new_lsm = LiquidStateMachine(
            n_inputs=10, n_reservoir=50, n_outputs=3, device=device
        )
        new_lsm.load_liquid_states(str(save_path))
        
        # Should have loaded states
        assert len(new_lsm.reservoir_states) > 0
        assert len(new_lsm.spike_history) > 0


class TestMultiModalLSM:
    """Test multi-modal liquid state machine."""
    
    def test_initialization(self, multimodal_lsm):
        """Test multi-modal LSM initialization."""
        mm_lsm = multimodal_lsm
        
        assert len(mm_lsm.modalities) == 3
        assert "audio" in mm_lsm.modality_lsms
        assert "vision" in mm_lsm.modality_lsms
        assert "tactile" in mm_lsm.modality_lsms
        
        # Check modality-specific LSMs
        assert mm_lsm.modality_lsms["audio"].n_inputs == 32
        assert mm_lsm.modality_lsms["vision"].n_inputs == 64
        assert mm_lsm.modality_lsms["tactile"].n_inputs == 6
    
    def test_multimodal_forward_pass(self, multimodal_lsm, synthetic_multimodal_data, device):
        """Test forward pass with multiple modalities."""
        mm_lsm = multimodal_lsm
        
        outputs, states = mm_lsm(synthetic_multimodal_data, return_states=True)
        
        assert outputs.shape == (2, 5)  # batch_size, n_outputs
        assert states is not None
        assert 'modality_states' in states
        assert 'fusion_attention' in states
        
        # Check modality-specific states
        mod_states = states['modality_states']
        assert 'audio' in mod_states
        assert 'vision' in mod_states
        assert 'tactile' in mod_states
    
    def test_missing_modality_robustness(self, multimodal_lsm, synthetic_multimodal_data):
        """Test robustness to missing modalities."""
        mm_lsm = multimodal_lsm
        
        # Test with missing tactile modality
        partial_data = {
            "audio": synthetic_multimodal_data["audio"],
            "vision": synthetic_multimodal_data["vision"]
        }
        
        outputs, states = mm_lsm(
            partial_data, 
            return_states=True,
            missing_modalities=["tactile"]
        )
        
        assert outputs.shape == (2, 5)
        assert 'tactile' not in states['available_modalities']
    
    def test_single_modality_processing(self, multimodal_lsm, synthetic_multimodal_data):
        """Test single modality processing."""
        mm_lsm = multimodal_lsm
        
        audio_output = mm_lsm.forward_single_modality(
            "audio", 
            synthetic_multimodal_data["audio"]
        )
        
        assert audio_output.shape == (2, 5)
    
    def test_fusion_types(self, test_config, device):
        """Test different fusion strategies."""
        modality_configs = test_config["model"]["modality_configs"]
        
        fusion_types = ["attention", "concatenation", "weighted_sum"]
        
        for fusion_type in fusion_types:
            mm_lsm = MultiModalLSM(
                modality_configs=modality_configs,
                n_outputs=5,
                fusion_type=fusion_type,
                device=device
            )
            
            # Test that model can be created and used
            test_input = {
                "audio": torch.randn(1, 10, 32, device=device),
                "vision": torch.randn(1, 10, 64, device=device)
            }
            
            outputs, _ = mm_lsm(test_input)
            assert outputs.shape == (1, 5)
    
    def test_modality_statistics(self, multimodal_lsm):
        """Test modality-specific statistics."""
        mm_lsm = multimodal_lsm
        stats = mm_lsm.get_modality_statistics()
        
        assert 'audio' in stats
        assert 'vision' in stats
        assert 'tactile' in stats
        
        for modality_stats in stats.values():
            assert 'actual_connectivity' in modality_stats
            assert 'spectral_radius' in modality_stats
    
    def test_modality_plasticity_control(self, multimodal_lsm):
        """Test modality-specific plasticity control."""
        mm_lsm = multimodal_lsm
        
        # Enable plasticity for specific modalities
        mm_lsm.enable_modality_plasticity(["audio", "vision"])
        
        # Check that specified modalities have plasticity enabled
        # (This would require accessing internal plasticity state)
        # For now, just test that the method executes without error
        
        # Disable all plasticity
        mm_lsm.disable_modality_plasticity()
    
    def test_state_reset(self, multimodal_lsm, synthetic_multimodal_data):
        """Test state reset functionality."""
        mm_lsm = multimodal_lsm
        
        # Process data to change internal states
        mm_lsm(synthetic_multimodal_data)
        
        # Reset all states
        mm_lsm.reset_all_states()
        
        # Test that states are cleared (implementation dependent)
        # This test verifies the method executes without error


class TestReadoutLayers:
    """Test readout layer functionality."""
    
    def test_linear_readout(self, device):
        """Test linear readout layer."""
        input_size = 100
        output_size = 10
        batch_size = 4
        
        readout = LinearReadout(
            input_size=input_size,
            output_size=output_size,
            dropout=0.2,
            device=device
        )
        
        # Test forward pass
        liquid_state = torch.randn(batch_size, input_size, device=device)
        outputs = readout(liquid_state)
        
        assert outputs.shape == (batch_size, output_size)
    
    def test_temporal_readout(self, device):
        """Test temporal readout with attention."""
        input_size = 50
        output_size = 5
        sequence_length = 20
        batch_size = 2
        
        readout = TemporalReadout(
            input_size=input_size,
            output_size=output_size,
            hidden_size=64,
            num_heads=4,
            device=device
        )
        
        # Test forward pass
        reservoir_sequence = torch.randn(
            batch_size, sequence_length, input_size, device=device
        )
        outputs, attention_weights = readout(reservoir_sequence)
        
        assert outputs.shape == (batch_size, output_size)
        assert attention_weights.shape == (batch_size, 4, sequence_length, sequence_length)
    
    def test_temporal_readout_with_mask(self, device):
        """Test temporal readout with attention mask."""
        readout = TemporalReadout(
            input_size=30,
            output_size=3,
            device=device
        )
        
        batch_size = 2
        sequence_length = 15
        
        reservoir_sequence = torch.randn(batch_size, sequence_length, 30, device=device)
        
        # Create mask (True for positions to mask)
        attention_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)
        attention_mask[0, 10:] = True  # Mask last 5 positions of first sample
        
        outputs, attention_weights = readout(reservoir_sequence, attention_mask)
        
        assert outputs.shape == (batch_size, 3)
        assert attention_weights.shape[0] == batch_size
    
    def test_readout_history_reset(self, device):
        """Test readout history reset."""
        readout = LinearReadout(
            input_size=50,
            output_size=10,
            temporal_integration=True,
            device=device
        )
        
        # Process some data
        for _ in range(10):
            liquid_state = torch.randn(2, 50, device=device)
            readout(liquid_state)
        
        # Reset history
        readout.reset_history()
        
        # Should execute without error
        # (Detailed state verification would require access to internal buffers)


@pytest.mark.performance
class TestModelPerformance:
    """Performance tests for neuromorphic models."""
    
    def test_lsm_forward_latency(self, simple_lsm, device, performance_monitor):
        """Test LSM forward pass latency."""
        lsm = simple_lsm
        batch_size = 32
        time_steps = 100
        
        input_spikes = torch.randn(batch_size, time_steps, 10, device=device)
        
        # Warm up
        for _ in range(10):
            lsm(input_spikes)
        
        # Measure latency
        performance_monitor.start()
        
        for _ in range(100):
            outputs, _ = lsm(input_spikes)
        
        performance_monitor.stop()
        
        avg_latency_ms = (performance_monitor.get_elapsed_time() / 100) * 1000
        
        # Should be under 10ms per forward pass for this model size
        assert avg_latency_ms < 10.0, f"Average latency: {avg_latency_ms:.2f}ms"
    
    @pytest.mark.gpu
    def test_multimodal_memory_usage(self, device, performance_monitor):
        """Test memory usage of multi-modal LSM."""
        if device.type != 'cuda':
            pytest.skip("GPU memory test requires CUDA")
        
        modality_configs = {
            "audio": {"n_inputs": 128, "n_reservoir": 500},
            "vision": {"n_inputs": 1024, "n_reservoir": 800},
            "tactile": {"n_inputs": 12, "n_reservoir": 200}
        }
        
        performance_monitor.start()
        
        mm_lsm = MultiModalLSM(
            modality_configs=modality_configs,
            n_outputs=50,
            device=device
        )
        
        # Large batch
        batch_size = 64
        time_steps = 200
        
        test_input = {
            "audio": torch.randn(batch_size, time_steps, 128, device=device),
            "vision": torch.randn(batch_size, time_steps, 1024, device=device),
            "tactile": torch.randn(batch_size, time_steps, 12, device=device)
        }
        
        outputs, states = mm_lsm(test_input, return_states=True)
        
        performance_monitor.stop()
        
        peak_memory_mb = performance_monitor.get_peak_memory_mb()
        
        # Should use less than 2GB for this configuration
        assert peak_memory_mb < 2048, f"Peak memory usage: {peak_memory_mb:.2f}MB"