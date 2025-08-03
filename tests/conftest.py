"""
Pytest configuration and fixtures for SNN-Fusion testing.

Provides comprehensive test fixtures for neuromorphic models, datasets,
and hardware simulation environments.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator, Optional
import json
import h5py

from snn_fusion.models import MultiModalLSM, LiquidStateMachine
from snn_fusion.preprocessing import AudioSpikeEncoder
from snn_fusion.database import DatabaseManager
from snn_fusion.cache import CacheManager


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Basic test configuration."""
    return {
        "model": {
            "modality_configs": {
                "audio": {"n_inputs": 32, "n_reservoir": 100},
                "vision": {"n_inputs": 64, "n_reservoir": 150},
                "tactile": {"n_inputs": 6, "n_reservoir": 50}
            },
            "n_outputs": 5,
            "fusion_type": "attention"
        },
        "training": {
            "learning_rule": "bptt",
            "temporal_window": 50,
            "batch_size": 4,
            "learning_rate": 1e-3
        }
    }


@pytest.fixture
def simple_lsm(device: torch.device) -> LiquidStateMachine:
    """Create simple LSM for testing."""
    return LiquidStateMachine(
        n_inputs=10,
        n_reservoir=50,
        n_outputs=3,
        connectivity=0.1,
        device=device
    )


@pytest.fixture
def multimodal_lsm(test_config: Dict[str, Any], device: torch.device) -> MultiModalLSM:
    """Create multi-modal LSM for testing."""
    return MultiModalLSM(
        modality_configs=test_config["model"]["modality_configs"],
        n_outputs=test_config["model"]["n_outputs"],
        fusion_type=test_config["model"]["fusion_type"],
        device=device
    )


@pytest.fixture
def audio_encoder(device: torch.device) -> AudioSpikeEncoder:
    """Create audio spike encoder for testing."""
    return AudioSpikeEncoder(
        sample_rate=16000,
        n_cochlear_channels=32,
        enable_binaural=True,
        device=device
    )


@pytest.fixture
def synthetic_audio_data() -> torch.Tensor:
    """Generate synthetic audio data for testing."""
    # Generate 1 second of stereo audio at 16kHz
    sample_rate = 16000
    duration = 1.0
    n_samples = int(sample_rate * duration)
    
    # Create stereo sine waves
    t = torch.linspace(0, duration, n_samples)
    left_channel = 0.5 * torch.sin(2 * np.pi * 440 * t)  # 440 Hz
    right_channel = 0.3 * torch.sin(2 * np.pi * 523 * t)  # 523 Hz (C5)
    
    audio = torch.stack([left_channel, right_channel], dim=0)  # [2, n_samples]
    return audio.unsqueeze(0)  # [1, 2, n_samples]


@pytest.fixture
def synthetic_multimodal_data() -> Dict[str, torch.Tensor]:
    """Generate synthetic multi-modal data for testing."""
    batch_size = 2
    time_steps = 50
    
    return {
        "audio": torch.randn(batch_size, time_steps, 32),
        "vision": torch.randn(batch_size, time_steps, 64),
        "tactile": torch.randn(batch_size, time_steps, 6)
    }


@pytest.fixture
def synthetic_spike_data() -> torch.Tensor:
    """Generate synthetic spike data for testing."""
    batch_size = 4
    time_steps = 100
    n_neurons = 200
    spike_prob = 0.1
    
    # Generate sparse spike data
    spikes = torch.rand(batch_size, time_steps, n_neurons) < spike_prob
    return spikes.float()


@pytest.fixture
def test_database(temp_dir: Path) -> DatabaseManager:
    """Create test database for integration tests."""
    db_path = temp_dir / "test.db"
    db = DatabaseManager(
        db_type="sqlite",
        db_path=str(db_path),
        auto_migrate=True
    )
    return db


@pytest.fixture
def test_cache(temp_dir: Path) -> CacheManager:
    """Create test cache manager."""
    cache_dir = temp_dir / "cache"
    return CacheManager(
        memory_cache_size=100,
        memory_cache_mb=50,
        disk_cache_dir=str(cache_dir),
        disk_cache_gb=1.0,
        enable_compression=False  # Faster for tests
    )


@pytest.fixture
def sample_experiment_data() -> Dict[str, Any]:
    """Sample experiment data for database testing."""
    return {
        "name": "test_experiment_001",
        "description": "Test experiment for unit testing",
        "config_json": json.dumps({
            "model_type": "MultiModalLSM",
            "modalities": ["audio", "vision"],
            "parameters": {"n_reservoir": 500}
        }),
        "status": "created",
        "tags": json.dumps(["test", "unit_test", "multimodal"])
    }


@pytest.fixture
def sample_spike_patterns(temp_dir: Path) -> Path:
    """Create sample spike pattern file for testing."""
    file_path = temp_dir / "sample_spikes.h5"
    
    # Generate sample spike data
    n_trials = 10
    n_neurons = 100
    n_timesteps = 1000
    
    with h5py.File(file_path, 'w') as f:
        # Create datasets
        spikes = f.create_dataset(
            'spikes', 
            (n_trials, n_timesteps, n_neurons), 
            dtype=np.float32
        )
        
        # Generate sparse spike patterns
        for trial in range(n_trials):
            trial_spikes = np.random.random((n_timesteps, n_neurons)) < 0.05
            spikes[trial] = trial_spikes.astype(np.float32)
        
        # Add metadata
        f.attrs['sample_rate'] = 1000.0
        f.attrs['duration_ms'] = 1000.0
        f.attrs['n_neurons'] = n_neurons
        f.attrs['spike_threshold'] = 0.05
    
    return file_path


@pytest.fixture
def hardware_sim_config() -> Dict[str, Any]:
    """Configuration for hardware simulation tests."""
    return {
        "loihi2": {
            "cores_used": 4,
            "neurons_per_core": 128,
            "timestep_us": 1000,
            "voltage_decay": 4096,
            "current_decay": 2048
        },
        "akida": {
            "quantization_bits": 8,
            "sparsity_level": 0.9,
            "batch_size": 1
        },
        "spinnaker": {
            "machine_width": 2,
            "machine_height": 2,
            "timestep_ms": 1.0
        }
    }


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_training_data():
    """Mock training data for trainer tests."""
    class MockDataset:
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Return mock multi-modal data and labels
            data = {
                "audio": torch.randn(50, 32),
                "vision": torch.randn(50, 64),
                "tactile": torch.randn(50, 6)
            }
            label = torch.randint(0, 5, (1,)).item()
            return data, label
    
    return MockDataset()


@pytest.fixture
def performance_benchmark_config() -> Dict[str, Any]:
    """Configuration for performance benchmarking."""
    return {
        "latency_targets": {
            "forward_pass_ms": 10.0,
            "training_step_ms": 100.0,
            "inference_batch_ms": 5.0
        },
        "memory_targets": {
            "model_size_mb": 100.0,
            "peak_memory_mb": 1000.0,
            "cache_efficiency": 0.8
        },
        "accuracy_targets": {
            "min_accuracy": 0.7,
            "convergence_epochs": 50
        }
    }


# Hardware-specific fixtures
@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def skip_if_no_neuromorphic_hw():
    """Skip test if neuromorphic hardware is not available."""
    # Check for hardware availability
    hardware_available = False
    
    try:
        # Check for Loihi
        import lava
        hardware_available = True
    except ImportError:
        pass
    
    try:
        # Check for Akida
        import akida
        hardware_available = True
    except ImportError:
        pass
    
    if not hardware_available:
        pytest.skip("No neuromorphic hardware or simulators available")


# Parameterized fixtures for different model configurations
@pytest.fixture(params=[
    {"fusion_type": "attention", "n_outputs": 5},
    {"fusion_type": "concatenation", "n_outputs": 10},
    {"fusion_type": "weighted_sum", "n_outputs": 3}
])
def fusion_config(request):
    """Parameterized fusion configuration for comprehensive testing."""
    return request.param


@pytest.fixture(params=[0.05, 0.1, 0.15])
def connectivity_values(request):
    """Parameterized connectivity values for LSM testing."""
    return request.param


@pytest.fixture(params=["bptt", "stdp"])
def learning_rules(request):
    """Parameterized learning rules for training tests."""
    return request.param


# Performance monitoring fixtures
@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.peak_memory = 0
            self.metrics = {}
        
        def start(self):
            import time
            self.start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        
        def stop(self):
            import time
            self.end_time = time.time()
            if torch.cuda.is_available():
                self.peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        def get_elapsed_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
        
        def get_peak_memory_mb(self):
            return self.peak_memory
    
    return PerformanceMonitor()


# Custom markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests that require neuromorphic hardware"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests that benchmark performance"
    )