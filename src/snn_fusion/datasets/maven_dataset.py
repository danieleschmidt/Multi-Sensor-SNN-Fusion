"""
MAVEN Dataset: Multi-modal Asynchronous Vision-Audio-Tactile Dataset

This module implements the MAVEN dataset loader for synchronized multi-modal
sensory data including binaural audio, event camera streams, and tactile IMU data.
"""

import os
import json
import torch
import torch.utils.data as data
import numpy as np
import h5py
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings
from pathlib import Path


@dataclass
class MAVENConfig:
    """Configuration for MAVEN dataset loading."""
    
    # Data paths
    root_dir: str
    split: str = "train"  # train, val, test
    
    # Modalities to load
    modalities: List[str] = None  # ['audio', 'events', 'imu']
    
    # Temporal parameters
    time_window_ms: float = 100.0
    temporal_resolution_ms: float = 1.0
    sequence_length: int = 100
    
    # Audio parameters
    audio_sample_rate: int = 48000
    audio_channels: int = 2
    cochlear_channels: int = 64
    
    # Event camera parameters
    event_resolution: Tuple[int, int] = (346, 260)
    event_accumulation_ms: float = 10.0
    
    # IMU parameters
    imu_sample_rate: int = 1000
    imu_channels: int = 6  # 3 accel + 3 gyro
    
    # Preprocessing
    normalize: bool = True
    spike_encoding: bool = True
    temporal_jitter: bool = False
    
    # Dataset filtering
    min_sequence_length: int = 50
    max_sequence_length: int = 1000
    filter_inactive: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.modalities is None:
            self.modalities = ['audio', 'events', 'imu']
        
        valid_splits = ['train', 'val', 'test']
        if self.split not in valid_splits:
            raise ValueError(f"Split must be one of {valid_splits}, got {self.split}")
        
        valid_modalities = ['audio', 'events', 'imu']
        for modality in self.modalities:
            if modality not in valid_modalities:
                raise ValueError(f"Unknown modality: {modality}")


class MAVENDataset(data.Dataset):
    """
    MAVEN Dataset for multi-modal spiking neural network training.
    
    Loads synchronized multi-modal sensory data with spike encoding for
    training liquid state machines and hierarchical fusion networks.
    """
    
    def __init__(self, config: MAVENConfig):
        """
        Initialize MAVEN dataset.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.root_dir = Path(config.root_dir)
        
        # Validate dataset structure
        self._validate_dataset()
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        
        # Build sample index
        self.samples = self._build_sample_index()
        
        # Initialize preprocessing components
        self._initialize_preprocessors()
        
        print(f"Loaded MAVEN dataset: {len(self.samples)} samples for split '{config.split}'")
    
    def _validate_dataset(self) -> None:
        """Validate dataset directory structure."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")
        
        # Check for required subdirectories
        required_dirs = ['raw_data', 'processed_data', 'metadata']
        for dir_name in required_dirs:
            dir_path = self.root_dir / dir_name
            if not dir_path.exists():
                warnings.warn(f"Expected directory not found: {dir_path}")
        
        # Check for metadata file
        metadata_file = self.root_dir / 'metadata' / 'dataset_info.json'
        if not metadata_file.exists():
            warnings.warn(f"Metadata file not found: {metadata_file}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata."""
        metadata_file = self.root_dir / 'metadata' / 'dataset_info.json'
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            # Create default metadata
            metadata = {
                'version': '1.0',
                'description': 'MAVEN Multi-Modal Dataset',
                'modalities': self.config.modalities,
                'splits': {
                    'train': 0.7,
                    'val': 0.15,
                    'test': 0.15
                }
            }
            
            # Save default metadata
            os.makedirs(metadata_file.parent, exist_ok=True)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return metadata
    
    def _build_sample_index(self) -> List[Dict[str, Any]]:
        """Build index of available samples."""
        samples = []
        
        # Look for processed data first
        processed_dir = self.root_dir / 'processed_data' / self.config.split
        if processed_dir.exists():
            samples.extend(self._index_processed_data(processed_dir))
        
        # Fallback to raw data if no processed data found
        if not samples:
            raw_dir = self.root_dir / 'raw_data' / self.config.split
            if raw_dir.exists():
                samples.extend(self._index_raw_data(raw_dir))
        
        # Generate synthetic data if no real data found
        if not samples:
            warnings.warn("No dataset files found. Generating synthetic samples for testing.")
            samples = self._create_synthetic_samples()
        
        # Filter samples based on configuration
        samples = self._filter_samples(samples)
        
        return samples
    
    def _index_processed_data(self, data_dir: Path) -> List[Dict[str, Any]]:
        """Index processed HDF5 data files."""
        samples = []
        
        for file_path in data_dir.glob('*.h5'):
            try:
                with h5py.File(file_path, 'r') as f:
                    # Get basic sample info
                    sample_info = {
                        'file_path': str(file_path),
                        'file_type': 'processed',
                        'sample_id': file_path.stem,
                    }
                    
                    # Check available modalities
                    available_modalities = []
                    for modality in self.config.modalities:
                        if modality in f:
                            available_modalities.append(modality)
                    
                    sample_info['modalities'] = available_modalities
                    
                    # Get temporal information
                    if 'metadata' in f:
                        metadata = dict(f['metadata'].attrs)
                        sample_info.update(metadata)
                    
                    samples.append(sample_info)
                    
            except Exception as e:
                warnings.warn(f"Error reading file {file_path}: {e}")
        
        return samples
    
    def _index_raw_data(self, data_dir: Path) -> List[Dict[str, Any]]:
        """Index raw data files (to be implemented based on actual data format)."""
        samples = []
        
        # This would be implemented based on the actual raw data format
        # For now, return empty list to trigger synthetic data generation
        warnings.warn("Raw data indexing not implemented. Use processed data or synthetic data.")
        
        return samples
    
    def _create_synthetic_samples(self) -> List[Dict[str, Any]]:
        """Create synthetic samples for testing when no real data is available."""
        num_samples = 1000 if self.config.split == 'train' else 200
        
        samples = []
        for i in range(num_samples):
            sample_info = {
                'sample_id': f'synthetic_{self.config.split}_{i:04d}',
                'file_type': 'synthetic',
                'modalities': self.config.modalities.copy(),
                'sequence_length': np.random.randint(
                    self.config.min_sequence_length,
                    min(self.config.max_sequence_length, self.config.sequence_length * 2)
                ),
                'label': np.random.randint(0, 10),  # 10 action classes
                'quality_score': np.random.uniform(0.7, 1.0)
            }
            samples.append(sample_info)
        
        return samples
    
    def _filter_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter samples based on configuration criteria."""
        filtered_samples = []
        
        for sample in samples:
            # Check modality availability
            available_modalities = sample.get('modalities', [])
            if not all(mod in available_modalities for mod in self.config.modalities):
                continue
            
            # Check sequence length
            seq_len = sample.get('sequence_length', self.config.sequence_length)
            if seq_len < self.config.min_sequence_length or seq_len > self.config.max_sequence_length:
                continue
            
            # Filter inactive samples if requested
            if self.config.filter_inactive:
                quality_score = sample.get('quality_score', 1.0)
                if quality_score < 0.5:
                    continue
            
            filtered_samples.append(sample)
        
        return filtered_samples
    
    def _initialize_preprocessors(self) -> None:
        """Initialize preprocessing components."""
        from ..preprocessing.audio import CochlearModel
        
        # Audio preprocessing
        if 'audio' in self.config.modalities:
            self.cochlear_model = CochlearModel(
                sample_rate=self.config.audio_sample_rate,
                n_channels=self.config.cochlear_channels,
                frequency_range=(80, 8000)
            )
        
        # Event preprocessing (placeholder)
        if 'events' in self.config.modalities:
            self.event_processor = None  # To be implemented
        
        # IMU preprocessing (placeholder)
        if 'imu' in self.config.modalities:
            self.imu_processor = None  # To be implemented
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing multi-modal data and labels
        """
        sample_info = self.samples[idx]
        
        if sample_info['file_type'] == 'processed':
            return self._load_processed_sample(sample_info)
        elif sample_info['file_type'] == 'synthetic':
            return self._generate_synthetic_sample(sample_info)
        else:
            raise ValueError(f"Unknown file type: {sample_info['file_type']}")
    
    def _load_processed_sample(self, sample_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Load a processed sample from HDF5 file."""
        file_path = sample_info['file_path']
        
        try:
            with h5py.File(file_path, 'r') as f:
                sample = {}
                
                # Load each requested modality
                for modality in self.config.modalities:
                    if modality in f:
                        data = torch.from_numpy(f[modality][:]).float()
                        
                        # Apply modality-specific preprocessing
                        if modality == 'audio':
                            data = self._preprocess_audio(data)
                        elif modality == 'events':
                            data = self._preprocess_events(data)
                        elif modality == 'imu':
                            data = self._preprocess_imu(data)
                        
                        sample[modality] = data
                
                # Load label if available
                if 'label' in f:
                    sample['label'] = torch.from_numpy(f['label'][:]).long()
                else:
                    sample['label'] = torch.tensor(sample_info.get('label', 0)).long()
                
                # Add metadata
                sample['sample_id'] = sample_info['sample_id']
                sample['sequence_length'] = torch.tensor(data.shape[0] if len(data.shape) > 1 else self.config.sequence_length)
                
                return sample
                
        except Exception as e:
            warnings.warn(f"Error loading sample {sample_info['sample_id']}: {e}")
            # Fallback to synthetic sample
            return self._generate_synthetic_sample(sample_info)
    
    def _generate_synthetic_sample(self, sample_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generate a synthetic multi-modal sample."""
        seq_len = sample_info.get('sequence_length', self.config.sequence_length)
        sample = {}
        
        # Generate synthetic data for each modality
        for modality in self.config.modalities:
            if modality == 'audio':
                # Synthetic binaural cochleagram
                data = self._generate_synthetic_audio(seq_len)
            elif modality == 'events':
                # Synthetic event stream
                data = self._generate_synthetic_events(seq_len)
            elif modality == 'imu':
                # Synthetic IMU data
                data = self._generate_synthetic_imu(seq_len)
            else:
                # Default synthetic data
                data = torch.randn(seq_len, 32)
            
            sample[modality] = data
        
        # Add label and metadata
        sample['label'] = torch.tensor(sample_info.get('label', 0)).long()
        sample['sample_id'] = sample_info['sample_id']
        sample['sequence_length'] = torch.tensor(seq_len)
        
        return sample
    
    def _generate_synthetic_audio(self, seq_len: int) -> torch.Tensor:
        """Generate synthetic cochleagram data."""
        # Create synthetic cochleagram with temporal structure
        n_channels = self.config.cochlear_channels
        
        # Generate frequency-dependent activity
        freqs = torch.linspace(0, 1, n_channels)
        temporal_pattern = torch.sin(torch.linspace(0, 4 * np.pi, seq_len))
        
        # Create cochleagram with both temporal and spectral structure
        cochleagram = torch.zeros(seq_len, 2, n_channels)  # [time, channels(L/R), frequencies]
        
        for c in range(n_channels):
            # Frequency-specific modulation
            freq_mod = torch.sin(freqs[c] * 2 * np.pi + temporal_pattern)
            noise = torch.randn(seq_len) * 0.1
            
            # Binaural data (slight phase difference)
            cochleagram[:, 0, c] = torch.relu(freq_mod + noise)  # Left channel
            cochleagram[:, 1, c] = torch.relu(freq_mod + noise + 0.1 * temporal_pattern)  # Right channel
        
        # Apply spike encoding if requested
        if self.config.spike_encoding:
            cochleagram = self._encode_spikes(cochleagram, threshold=0.3)
        
        return cochleagram
    
    def _generate_synthetic_events(self, seq_len: int) -> torch.Tensor:
        """Generate synthetic event camera data."""
        height, width = self.config.event_resolution
        
        # Generate sparse event data
        event_density = 0.01  # 1% of pixels active per time step
        n_events_per_step = int(height * width * event_density)
        
        events = torch.zeros(seq_len, height, width, 2)  # [time, H, W, polarity]
        
        for t in range(seq_len):
            # Random event locations
            y_coords = torch.randint(0, height, (n_events_per_step,))
            x_coords = torch.randint(0, width, (n_events_per_step,))
            polarities = torch.randint(0, 2, (n_events_per_step,))
            
            # Set events
            events[t, y_coords, x_coords, polarities] = 1.0
        
        return events
    
    def _generate_synthetic_imu(self, seq_len: int) -> torch.Tensor:
        """Generate synthetic IMU data."""
        n_channels = self.config.imu_channels
        
        # Generate smooth IMU signals with some dynamics
        t = torch.linspace(0, seq_len * 0.001, seq_len)  # Time in seconds
        
        imu_data = torch.zeros(seq_len, n_channels)
        
        # Accelerometer channels (0-2)
        for i in range(3):
            # Gravity + motion
            gravity = torch.tensor([0.0, 0.0, 9.81])[i]
            motion = torch.sin(2 * np.pi * (i + 1) * t) * 2.0
            noise = torch.randn(seq_len) * 0.1
            imu_data[:, i] = gravity + motion + noise
        
        # Gyroscope channels (3-5) 
        for i in range(3, 6):
            # Rotational motion
            rotation = torch.cos(2 * np.pi * 0.5 * t) * 0.5
            noise = torch.randn(seq_len) * 0.05
            imu_data[:, i] = rotation + noise
        
        # Apply spike encoding if requested
        if self.config.spike_encoding:
            # Normalize to [0, 1] first
            imu_normalized = (imu_data - imu_data.min()) / (imu_data.max() - imu_data.min() + 1e-6)
            imu_data = self._encode_spikes(imu_normalized, threshold=0.5)
        
        return imu_data
    
    def _preprocess_audio(self, audio_data: torch.Tensor) -> torch.Tensor:
        """Preprocess audio data."""
        if self.config.normalize:
            audio_data = (audio_data - audio_data.mean()) / (audio_data.std() + 1e-6)
        
        # Additional preprocessing would go here
        return audio_data
    
    def _preprocess_events(self, event_data: torch.Tensor) -> torch.Tensor:
        """Preprocess event camera data."""
        if self.config.normalize:
            # Events are already binary/sparse, minimal preprocessing needed
            pass
        
        return event_data
    
    def _preprocess_imu(self, imu_data: torch.Tensor) -> torch.Tensor:
        """Preprocess IMU data."""
        if self.config.normalize:
            imu_data = (imu_data - imu_data.mean(dim=0)) / (imu_data.std(dim=0) + 1e-6)
        
        return imu_data
    
    def _encode_spikes(self, data: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convert continuous data to spike trains using threshold encoding."""
        # Simple threshold-based spike encoding
        spikes = (data > threshold).float()
        
        # Add some temporal dynamics
        if data.dim() > 1:
            # Apply temporal filtering to create more realistic spike patterns
            kernel = torch.tensor([0.1, 0.8, 0.1]).view(1, 1, -1)
            if spikes.dim() == 3:
                # Handle 3D data (time, channels, features)
                original_shape = spikes.shape
                spikes_flat = spikes.view(-1, 1, spikes.shape[-1])
                filtered = torch.nn.functional.conv1d(
                    spikes_flat.transpose(1, 2), 
                    kernel, 
                    padding=1
                ).transpose(1, 2)
                spikes = filtered.view(original_shape)
            
        return spikes
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample."""
        return self.samples[idx].copy()
    
    def get_modality_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for each modality across the dataset."""
        stats = {}
        
        # Sample a subset of data for statistics
        sample_indices = np.random.choice(len(self), min(100, len(self)), replace=False)
        
        for modality in self.config.modalities:
            modality_data = []
            
            for idx in sample_indices:
                sample = self[idx]
                if modality in sample:
                    modality_data.append(sample[modality])
            
            if modality_data:
                # Stack and compute statistics
                stacked_data = torch.stack(modality_data)
                stats[modality] = {
                    'mean': stacked_data.mean().item(),
                    'std': stacked_data.std().item(),
                    'min': stacked_data.min().item(),
                    'max': stacked_data.max().item(),
                    'shape': list(stacked_data.shape[1:]),  # Exclude batch dimension
                    'sparsity': (stacked_data == 0).float().mean().item()
                }
        
        return stats


# Utility functions

def create_maven_config(
    root_dir: str,
    modalities: List[str] = None,
    **kwargs
) -> MAVENConfig:
    """
    Create MAVEN dataset configuration with sensible defaults.
    
    Args:
        root_dir: Path to dataset root directory
        modalities: List of modalities to load
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured MAVENConfig instance
    """
    if modalities is None:
        modalities = ['audio', 'events', 'imu']
    
    return MAVENConfig(
        root_dir=root_dir,
        modalities=modalities,
        **kwargs
    )


def download_maven_dataset(root_dir: str, split: str = "all") -> None:
    """
    Download MAVEN dataset (placeholder for actual download implementation).
    
    Args:
        root_dir: Directory to download dataset to
        split: Which split to download ('train', 'val', 'test', 'all')
    """
    # This would implement actual dataset download
    # For now, just create directory structure for synthetic data
    
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    for subdir in ['raw_data', 'processed_data', 'metadata']:
        (root_path / subdir).mkdir(exist_ok=True)
        
        if subdir in ['raw_data', 'processed_data']:
            for split_name in ['train', 'val', 'test']:
                (root_path / subdir / split_name).mkdir(exist_ok=True)
    
    print(f"Created MAVEN dataset directory structure at {root_dir}")
    print("Note: This is a placeholder. Implement actual download logic for real dataset.")


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset implementation
    config = create_maven_config(
        root_dir="./data/maven",
        modalities=['audio', 'events', 'imu'],
        split='train',
        sequence_length=100
    )
    
    # Create dataset directory if it doesn't exist
    download_maven_dataset(config.root_dir)
    
    # Initialize dataset
    dataset = MAVENDataset(config)
    
    # Test loading samples
    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
    
    # Get statistics
    stats = dataset.get_modality_statistics()
    print("\nDataset statistics:")
    for modality, modality_stats in stats.items():
        print(f"{modality}: {modality_stats}")