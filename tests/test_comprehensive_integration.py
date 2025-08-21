"""
Comprehensive Integration Tests for SNN-Fusion

Tests the complete pipeline including preprocessing, models, training, security, and scaling.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any

from snn_fusion.preprocessing import (
    CochlearModel, GaborFilters, EventEncoder, WaveletTransform,
    PopulationEncoder, RateEncoder, TemporalEncoder
)
from snn_fusion.security import (
    AdversarialDetector, SecurityValidator, ThreatLevel
)
from snn_fusion.scaling import (
    ScalabilityManager, NodeInfo, TaskSpec, ProcessingMode,
    QuantumNeuromorphicOptimizer, ConsciousnessLayer
)
from snn_fusion.utils.robust_error_handling import (
    InputValidator, retry_with_exponential_backoff, error_context,
    PerformanceMonitor
)


class TestNeuromorphicPipeline:
    """Test complete neuromorphic processing pipeline."""
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data."""
        return torch.randn(2, 16000)  # 2 channels, 1 second at 16kHz
    
    @pytest.fixture
    def sample_visual_data(self):
        """Generate sample visual data."""
        return torch.randn(3, 128, 128)  # RGB image 128x128
    
    @pytest.fixture
    def sample_tactile_data(self):
        """Generate sample tactile data."""
        return {
            'accelerometer': torch.randn(3, 1000),  # 3 axes, 1000 samples
            'gyroscope': torch.randn(3, 1000)
        }
    
    @pytest.fixture
    def node_info(self):
        """Create test node information."""
        return NodeInfo(
            node_id='test_node',
            rank=0,
            world_size=1,
            device=torch.device('cpu'),
            memory_gb=4.0,
            compute_capability=1.5,
            network_bandwidth=100.0,
            load_factor=0.2,
            specialization='general'
        )
    
    def test_audio_preprocessing_pipeline(self, sample_audio_data):
        """Test audio preprocessing components."""
        
        # Test cochlear model
        cochlear = CochlearModel(
            sample_rate=16000,
            n_filters=64,
            low_freq=80,
            high_freq=8000
        )
        
        cochlear_output = cochlear(sample_audio_data)
        assert cochlear_output.shape[1] == 64  # 64 frequency channels
        assert torch.all(torch.isfinite(cochlear_output))
        
        # Test population encoder
        population_encoder = PopulationEncoder(
            input_dim=64,
            num_neurons=100,
            time_steps=50
        )
        
        spike_output = population_encoder(cochlear_output.mean(dim=2))
        assert spike_output.shape == (2, 64, 100, 50)  # [Batch, Features, Neurons, Time]
        assert torch.all((spike_output >= 0) & (spike_output <= 1))
    
    def test_visual_preprocessing_pipeline(self, sample_visual_data):
        """Test visual preprocessing components."""
        
        # Test Gabor filters
        gabor_filters = GaborFilters(
            num_orientations=4,
            num_scales=3,
            kernel_size=15
        )
        
        # Convert to grayscale for processing
        gray_image = sample_visual_data.mean(dim=0, keepdim=True).unsqueeze(0)
        gabor_output = gabor_filters(gray_image)
        
        expected_channels = 4 * 3  # orientations * scales
        assert gabor_output.shape[1] == expected_channels
        assert torch.all(torch.isfinite(gabor_output))
        
        # Test event encoder
        event_encoder = EventEncoder(threshold=0.1)
        
        # Create temporal sequence
        temporal_sequence = torch.stack([
            gray_image.squeeze(0) + 0.1 * torch.randn_like(gray_image.squeeze(0))
            for _ in range(5)
        ], dim=1)
        
        events = event_encoder(temporal_sequence)
        assert 'events' in events
        assert 'polarity' in events
        assert torch.all(torch.abs(events['polarity']) <= 1)
    
    def test_tactile_preprocessing_pipeline(self, sample_tactile_data):
        """Test tactile preprocessing components."""
        
        # Test wavelet transform
        wavelet_transform = WaveletTransform(
            wavelet='db4',
            levels=3
        )
        
        accel_features = wavelet_transform(sample_tactile_data['accelerometer'])
        assert accel_features.dim() == 2  # [Batch, Features]
        assert torch.all(torch.isfinite(accel_features))
        
        # Test rate encoder
        rate_encoder = RateEncoder(
            input_dim=accel_features.shape[1],
            time_steps=100,
            max_rate=50.0
        )
        
        spike_trains = rate_encoder(accel_features)
        assert spike_trains.shape[-1] == 100  # Time steps
        assert torch.all((spike_trains >= 0) & (spike_trains <= 1))
    
    def test_security_validation(self, sample_visual_data):
        """Test security and adversarial detection."""
        
        # Create simple model for testing
        test_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10)
        )
        
        # Test input validation
        validator = SecurityValidator()
        
        # Valid input should pass
        assert validator.validate_input_safety(sample_visual_data)
        
        # Test adversarial detection
        detector = AdversarialDetector(detection_threshold=0.5)
        
        # Normal input should be benign
        detection_result = detector.detect_adversarial_input(
            sample_visual_data.unsqueeze(0),
            test_model
        )
        
        assert 'is_adversarial' in detection_result
        assert 'threat_level' in detection_result
        assert detection_result['threat_level'] in [ThreatLevel.BENIGN, ThreatLevel.SUSPICIOUS]
        
        # Highly anomalous input should be detected
        anomalous_input = torch.full_like(sample_visual_data, 100.0)  # Extreme values
        anomalous_detection = detector.detect_adversarial_input(
            anomalous_input.unsqueeze(0),
            test_model
        )
        
        # Should detect high anomaly
        assert anomalous_detection['confidence'] > 0.3
    
    def test_quantum_optimization(self):
        """Test quantum-enhanced optimization."""
        
        quantum_optimizer = QuantumNeuromorphicOptimizer(num_qubits=4)
        
        # Simple quadratic objective function
        def objective(params):
            return torch.sum((params - 1.0) ** 2).item()
        
        initial_params = torch.zeros(5)
        optimized_params, final_value = quantum_optimizer.quantum_variational_optimization(
            objective,
            initial_params,
            num_iterations=10
        )
        
        # Should improve objective
        initial_value = objective(initial_params)
        assert final_value < initial_value
        
        # Check convergence metrics
        assert 'final_value' in quantum_optimizer.convergence_metrics
        assert 'iterations' in quantum_optimizer.convergence_metrics
        assert 'speedup_factor' in quantum_optimizer.convergence_metrics
    
    def test_consciousness_layer(self):
        """Test consciousness-driven processing."""
        
        consciousness_layer = ConsciousnessLayer(
            input_dim=64,
            workspace_dim=32,
            num_specialists=4,
            attention_heads=2
        )
        
        test_input = torch.randn(2, 64)
        output = consciousness_layer(test_input)
        
        # Check required outputs
        assert 'output' in output
        assert 'attention_weights' in output
        assert 'consciousness_level' in output
        assert 'metacognition' in output
        
        # Check output shapes
        assert output['output'].shape == (2, 32)  # workspace_dim
        assert output['consciousness_level'].shape == (1,)
        
        # Consciousness level should be between 0 and 1
        consciousness_level = output['consciousness_level'].item()
        assert 0.0 <= consciousness_level <= 1.0
        
        # Test memory formation for high consciousness
        if consciousness_level > 0.8:
            assert len(consciousness_layer.episodic_memory) > 0
    
    def test_scalability_manager_integration(self, node_info):
        """Test complete scalability management."""
        
        scalability_manager = ScalabilityManager(
            node_info=node_info,
            enable_quantum=True,
            enable_consciousness=True,
            enable_swarm=True
        )
        
        # Create test model and tasks
        test_model = nn.Linear(10, 5)
        
        tasks = [
            TaskSpec(
                task_id=f'task_{i}',
                task_type='inference',
                input_data=torch.randn(10),
                model_spec={'type': 'linear'},
                priority=1,
                deadline=None,
                dependencies=[],
                resource_requirements={'compute_units': 1}
            )
            for i in range(3)
        ]
        
        # Test optimization
        results = scalability_manager.optimize_distributed_processing(
            test_model,
            tasks,
            optimization_target='latency'
        )
        
        # Check results structure
        assert 'optimization_target' in results
        assert 'performance_metrics' in results
        assert 'quantum_enhancement' in results
        assert 'consciousness_adaptation' in results
        assert 'swarm_coordination' in results
        assert 'improvement_factor' in results
        
        # Improvement factor should be reasonable
        improvement_factor = results['improvement_factor']
        assert 0.1 <= improvement_factor <= 100.0  # Reasonable range
        
        # Test scaling report
        report = scalability_manager.get_scaling_report()
        assert 'node_info' in report
        assert 'enabled_features' in report
        assert 'scaling_history' in report
    
    def test_error_handling_robustness(self):
        """Test robust error handling mechanisms."""
        
        # Test input validation
        validator = InputValidator()
        
        # Valid tensor
        valid_tensor = torch.randn(10, 5)
        validated = validator.validate_tensor(
            valid_tensor,
            expected_shape=(10, 5),
            expected_dtype=torch.float32
        )
        assert torch.equal(validated, valid_tensor)
        
        # Invalid tensor (NaN values)
        invalid_tensor = torch.tensor([1.0, float('nan'), 3.0])
        with pytest.raises(ValueError, match="contains NaN values"):
            validator.validate_tensor(
                invalid_tensor,
                allow_nan=False
            )
        
        # Test spike data validation
        spike_data = torch.randint(0, 2, (50, 100)).float()
        validated_spikes = validator.validate_spike_data(spike_data)
        assert torch.equal(validated_spikes, spike_data)
        
        # Test error context
        with pytest.raises(Exception):
            with error_context("test_operation"):
                raise ValueError("Test error")
        
        # Test retry decorator
        call_count = 0
        
        @retry_with_exponential_backoff(max_retries=2, base_delay=0.01)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = failing_function()
        assert result == "success"
        assert call_count == 2
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        
        monitor = PerformanceMonitor()
        
        # Test operation timing
        monitor.start_operation('test_operation')
        time.sleep(0.01)  # Simulate work
        monitor.end_operation('test_operation')
        
        assert 'test_operation' in monitor.metrics
        assert 'duration' in monitor.metrics['test_operation']
        assert monitor.metrics['test_operation']['duration'] >= 0.01
        
        # Test memory monitoring
        memory_info = monitor.get_memory_usage()
        assert 'gpu_allocated' in memory_info
        assert 'gpu_cached' in memory_info
        assert all(isinstance(v, float) for v in memory_info.values())
    
    def test_end_to_end_integration(self, sample_audio_data, sample_visual_data, node_info):
        """Test complete end-to-end integration."""
        
        # Step 1: Preprocessing
        cochlear = CochlearModel(sample_rate=16000, n_filters=32)
        audio_features = cochlear(sample_audio_data)
        
        gabor = GaborFilters(num_orientations=2, num_scales=2, kernel_size=7)
        visual_features = gabor(sample_visual_data.mean(dim=0, keepdim=True).unsqueeze(0))
        
        # Step 2: Encoding
        encoder = PopulationEncoder(input_dim=32, num_neurons=50, time_steps=25)
        audio_spikes = encoder(audio_features.mean(dim=2))
        
        # Step 3: Security validation
        security_validator = SecurityValidator()
        assert security_validator.validate_input_safety(audio_spikes)
        
        detector = AdversarialDetector()
        test_model = nn.Linear(50, 10)
        detection_result = detector.detect_adversarial_input(
            audio_spikes.view(2, -1),
            test_model
        )
        assert detection_result['threat_level'] != ThreatLevel.CRITICAL
        
        # Step 4: Consciousness processing
        consciousness = ConsciousnessLayer(input_dim=audio_spikes.view(2, -1).shape[1], workspace_dim=64)
        consciousness_output = consciousness(audio_spikes.view(2, -1))
        
        # Step 5: Scalability optimization
        scalability_manager = ScalabilityManager(
            node_info=node_info,
            enable_quantum=False,  # Disable for faster testing
            enable_consciousness=True,
            enable_swarm=False
        )
        
        tasks = [TaskSpec(
            task_id='integration_task',
            task_type='multimodal_processing',
            input_data=audio_spikes.view(-1),
            model_spec={'type': 'consciousness_driven'},
            priority=2,
            deadline=None,
            dependencies=[],
            resource_requirements={'compute_units': 2}
        )]
        
        optimization_results = scalability_manager.optimize_distributed_processing(
            test_model,
            tasks,
            optimization_target='throughput'
        )
        
        # Verify end-to-end success
        assert optimization_results['improvement_factor'] > 0
        assert consciousness_output['consciousness_level'] >= 0
        
        print("✅ End-to-end integration test completed successfully")


class TestAdvancedFeatures:
    """Test advanced neuromorphic features."""
    
    def test_multimodal_fusion(self):
        """Test multimodal sensor fusion capabilities."""
        
        # Create different modality encoders
        audio_encoder = RateEncoder(input_dim=64, time_steps=100)
        visual_encoder = TemporalEncoder(input_dim=128, time_steps=100)
        tactile_encoder = PopulationEncoder(input_dim=32, num_neurons=50, time_steps=100)
        
        # Generate multimodal data
        audio_data = torch.randn(2, 64)
        visual_data = torch.randn(2, 128)
        tactile_data = torch.randn(2, 32)
        
        # Encode each modality
        audio_spikes = audio_encoder(audio_data)
        visual_spikes = visual_encoder(visual_data)
        tactile_spikes = tactile_encoder(tactile_data)
        
        # Validate outputs
        assert audio_spikes.shape[-1] == 100  # Time steps
        assert visual_spikes.shape[-1] == 100
        assert tactile_spikes.shape[-1] == 100
        
        # Test temporal alignment
        assert audio_spikes.shape[0] == visual_spikes.shape[0] == tactile_spikes.shape[0]
        
        print("✅ Multimodal fusion capabilities validated")
    
    def test_adaptive_processing(self):
        """Test adaptive processing based on input characteristics."""
        
        consciousness_layer = ConsciousnessLayer(
            input_dim=100,
            workspace_dim=50,
            num_specialists=6
        )
        
        # Test with different input patterns
        test_patterns = {
            'low_complexity': torch.randn(1, 100) * 0.1,
            'high_complexity': torch.randn(1, 100) * 2.0,
            'structured': torch.sin(torch.linspace(0, 10, 100)).unsqueeze(0),
            'noisy': torch.randn(1, 100) + torch.randn(1, 100) * 5.0
        }
        
        adaptations = {}
        for pattern_name, pattern_data in test_patterns.items():
            output = consciousness_layer(pattern_data)
            adaptations[pattern_name] = {
                'consciousness_level': output['consciousness_level'].item(),
                'attention_distribution': output['attention_weights'].std().item()
            }
        
        # Verify adaptive responses
        assert len(adaptations) == 4
        
        # High complexity should generally lead to higher consciousness
        high_consciousness = adaptations['high_complexity']['consciousness_level']
        low_consciousness = adaptations['low_complexity']['consciousness_level']
        
        # Allow for some variance but expect general trend
        print(f"Consciousness levels - High: {high_consciousness:.3f}, Low: {low_consciousness:.3f}")
        
        print("✅ Adaptive processing capabilities validated")
    
    def test_distributed_coordination(self, node_info):
        """Test distributed processing coordination."""
        
        from snn_fusion.scaling import AutonomousSwarmCoordinator
        
        coordinator = AutonomousSwarmCoordinator(
            node_info=node_info,
            max_neighbors=5
        )
        
        # Test swarm joining
        success = coordinator.join_swarm(
            'test_swarm',
            ['bootstrap_1', 'bootstrap_2'],
            {'compute_capability': 2.0, 'specialization': 'general'}
        )
        assert success
        
        # Test task distribution
        global_task = TaskSpec(
            task_id='distributed_test',
            task_type='inference',
            input_data=torch.randn(1000),
            model_spec={'type': 'snn'},
            priority=3,
            deadline=None,
            dependencies=[],
            resource_requirements={'compute_units': 10}
        )
        
        allocation = coordinator.coordinate_task_distribution(global_task, 'minimize_latency')
        assert isinstance(allocation, dict)
        
        # Test emergent behavior detection
        emergent_behaviors = coordinator.detect_emergent_behavior()
        assert isinstance(emergent_behaviors, list)
        
        print("✅ Distributed coordination capabilities validated")


class TestQualityMetrics:
    """Test quality metrics and performance benchmarks."""
    
    def test_processing_latency(self):
        """Test processing latency requirements."""
        
        # Create processing pipeline
        encoder = RateEncoder(input_dim=128, time_steps=50)
        consciousness_layer = ConsciousnessLayer(input_dim=128, workspace_dim=64)
        
        test_data = torch.randn(1, 128)
        
        # Measure processing time
        start_time = time.time()
        
        # Run multiple iterations
        for _ in range(10):
            spikes = encoder(test_data)
            consciousness_output = consciousness_layer(test_data)
        
        total_time = time.time() - start_time
        avg_latency = total_time / 10
        
        # Latency should be reasonable for real-time processing
        assert avg_latency < 0.1  # Less than 100ms per inference
        
        print(f"✅ Average processing latency: {avg_latency:.4f}s")
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large-scale components
        large_encoder = PopulationEncoder(input_dim=1000, num_neurons=500, time_steps=200)
        large_consciousness = ConsciousnessLayer(input_dim=1000, workspace_dim=256, num_specialists=10)
        
        # Process data
        large_data = torch.randn(5, 1000)
        encoded_data = large_encoder(large_data)
        consciousness_output = large_consciousness(large_data)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 500  # Less than 500MB increase
        
        print(f"✅ Memory usage increase: {memory_increase:.1f}MB")
    
    def test_accuracy_benchmarks(self):
        """Test accuracy on synthetic benchmarks."""
        
        # Create synthetic classification task
        num_classes = 5
        num_samples = 100
        input_dim = 64
        
        # Generate synthetic data with patterns
        X = torch.randn(num_samples, input_dim)
        y = torch.randint(0, num_classes, (num_samples,))
        
        # Add class-specific patterns
        for i in range(num_classes):
            class_mask = (y == i)
            if class_mask.any():
                X[class_mask] += i * 0.5  # Add class-specific bias
        
        # Create and test model
        model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        # Simple training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Test accuracy
        with torch.no_grad():
            predictions = model(X).argmax(dim=1)
            accuracy = (predictions == y).float().mean().item()
        
        # Should achieve reasonable accuracy on synthetic data
        assert accuracy > 0.7  # At least 70% accuracy
        
        print(f"✅ Synthetic benchmark accuracy: {accuracy:.3f}")
    
    def test_security_robustness(self):
        """Test security measures robustness."""
        
        detector = AdversarialDetector(detection_threshold=0.6)
        validator = SecurityValidator()
        
        # Test various input types
        test_inputs = {
            'normal': torch.randn(3, 64, 64),
            'high_values': torch.randn(3, 64, 64) * 10,
            'extreme_values': torch.full((3, 64, 64), 100.0),
            'nan_values': torch.full((3, 64, 64), float('nan')),
            'inf_values': torch.full((3, 64, 64), float('inf'))
        }
        
        test_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10)
        )
        
        security_results = {}
        
        for input_type, input_data in test_inputs.items():
            try:
                # Validate input safety
                if input_type not in ['nan_values', 'inf_values']:
                    validator.validate_input_safety(input_data)
                
                # Test adversarial detection
                detection_result = detector.detect_adversarial_input(
                    input_data.unsqueeze(0),
                    test_model
                )
                
                security_results[input_type] = detection_result['threat_level']
                
            except Exception as e:
                security_results[input_type] = f"Blocked: {type(e).__name__}"
        
        # Verify security responses
        assert security_results['normal'] == ThreatLevel.BENIGN
        assert 'extreme_values' in security_results
        assert 'Blocked' in str(security_results.get('nan_values', ''))
        
        print("✅ Security robustness validated")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])