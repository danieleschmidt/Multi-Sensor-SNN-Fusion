"""
Synthetic Multi-Modal Dataset Generator

This module provides synthetic data generation for testing and development
of multi-modal spiking neural networks when real data is not available.
"""

import torch
import torch.utils.data as data
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import math


class SyntheticMultiModalDataset(data.Dataset):
    """
    Synthetic multi-modal dataset for SNN testing and development.
    
    Generates synthetic binaural audio, event camera, and IMU data with
    configurable patterns and noise characteristics.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        sequence_length: int = 100,
        modalities: List[str] = None,
        num_classes: int = 10,
        spike_encoding: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate
            sequence_length: Length of temporal sequences
            modalities: List of modalities to include
            num_classes: Number of output classes
            spike_encoding: Whether to encode data as spikes
            seed: Random seed for reproducibility
        """
        if modalities is None:
            modalities = ['audio', 'events', 'imu']
        
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.modalities = modalities
        self.num_classes = num_classes
        self.spike_encoding = spike_encoding
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Generate all samples upfront for consistency
        self.samples = self._generate_all_samples()
    
    def _generate_all_samples(self) -> List[Dict[str, torch.Tensor]]:
        """Generate all samples upfront."""
        samples = []
        
        for i in range(self.num_samples):
            sample = {'sample_id': f'synthetic_{i:04d}'}
            
            # Generate label
            label = i % self.num_classes
            sample['label'] = torch.tensor(label).long()
            
            # Generate each modality
            for modality in self.modalities:
                if modality == 'audio':
                    sample[modality] = self._generate_audio(label, i)
                elif modality == 'events':
                    sample[modality] = self._generate_events(label, i)
                elif modality == 'imu':
                    sample[modality] = self._generate_imu(label, i)
            
            samples.append(sample)
        
        return samples
    
    def _generate_audio(self, label: int, sample_idx: int) -> torch.Tensor:
        """Generate synthetic binaural cochleagram data."""
        n_channels = 64  # Cochlear channels
        
        # Create label-dependent frequency pattern
        dominant_freq_idx = label * (n_channels // self.num_classes)
        
        # Generate temporal pattern
        t = torch.linspace(0, 1, self.sequence_length)
        
        # Base pattern with class-specific frequency
        base_freq = 2 + label * 0.5  # Different temporal frequencies per class
        temporal_pattern = torch.sin(2 * np.pi * base_freq * t)
        
        # Create cochleagram
        cochleagram = torch.zeros(self.sequence_length, 2, n_channels)
        
        for c in range(n_channels):
            # Frequency response (stronger around dominant frequency)
            freq_response = torch.exp(-0.1 * (c - dominant_freq_idx) ** 2)
            
            # Temporal modulation
            temporal_mod = temporal_pattern * freq_response
            
            # Add noise
            noise = torch.randn(self.sequence_length) * 0.1
            
            # Left and right channels (slight difference for binaural)
            cochleagram[:, 0, c] = torch.relu(temporal_mod + noise)
            cochleagram[:, 1, c] = torch.relu(temporal_mod + noise + 0.05 * temporal_pattern)
        
        # Apply spike encoding
        if self.spike_encoding:
            cochleagram = self._encode_spikes(cochleagram, threshold=0.3)
        
        return cochleagram
    
    def _generate_events(self, label: int, sample_idx: int) -> torch.Tensor:
        """Generate synthetic event camera data."""
        height, width = 128, 128  # Reduced size for efficiency
        
        # Create label-dependent spatial patterns
        center_x = width // 2 + (label - self.num_classes // 2) * 5
        center_y = height // 2
        
        events = torch.zeros(self.sequence_length, height, width, 2)
        
        for t in range(self.sequence_length):
            # Moving pattern based on class
            t_norm = t / self.sequence_length
            
            # Circular motion pattern
            angle = 2 * np.pi * t_norm + label * np.pi / 4
            radius = 20 + 10 * torch.sin(4 * np.pi * t_norm)
            
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            
            # Add events around the moving center
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    event_x = max(0, min(width - 1, x + dx))
                    event_y = max(0, min(height - 1, y + dy))
                    
                    # Distance-based probability
                    dist = np.sqrt(dx*dx + dy*dy)
                    prob = np.exp(-dist / 2)
                    
                    if np.random.random() < prob * 0.5:
                        polarity = 1 if np.random.random() > 0.5 else 0
                        events[t, event_y, event_x, polarity] = 1.0
            
            # Add some random noise events
            noise_events = int(0.01 * height * width)  # 1% noise
            for _ in range(noise_events):
                y_noise = np.random.randint(0, height)
                x_noise = np.random.randint(0, width)
                pol_noise = np.random.randint(0, 2)
                events[t, y_noise, x_noise, pol_noise] = 1.0
        
        return events
    
    def _generate_imu(self, label: int, sample_idx: int) -> torch.Tensor:
        """Generate synthetic IMU data."""
        n_channels = 6  # 3 accel + 3 gyro
        
        t = torch.linspace(0, 1, self.sequence_length)
        imu_data = torch.zeros(self.sequence_length, n_channels)
        
        # Class-dependent motion patterns
        motion_freq = 1 + label * 0.2  # Different motion frequencies
        
        # Accelerometer channels (0-2)
        for i in range(3):
            # Gravity component
            gravity = torch.tensor([0.0, 0.0, 9.81])[i]
            
            # Motion component (class-dependent)
            if i == 0:  # X-axis
                motion = torch.sin(2 * np.pi * motion_freq * t) * 2.0
            elif i == 1:  # Y-axis  
                motion = torch.cos(2 * np.pi * motion_freq * t) * 1.5
            else:  # Z-axis
                motion = torch.sin(4 * np.pi * motion_freq * t) * 1.0
            
            # Noise
            noise = torch.randn(self.sequence_length) * 0.1
            
            imu_data[:, i] = gravity + motion + noise
        
        # Gyroscope channels (3-5)
        for i in range(3, 6):
            # Rotational motion (class-dependent)
            if i == 3:  # Roll
                rotation = torch.sin(2 * np.pi * motion_freq * t + label * np.pi / 4) * 0.5
            elif i == 4:  # Pitch
                rotation = torch.cos(2 * np.pi * motion_freq * t + label * np.pi / 4) * 0.3
            else:  # Yaw
                rotation = torch.sin(4 * np.pi * motion_freq * t) * 0.2
            
            # Noise
            noise = torch.randn(self.sequence_length) * 0.05
            
            imu_data[:, i] = rotation + noise
        
        # Apply spike encoding
        if self.spike_encoding:
            # Normalize to [0, 1] range first
            imu_normalized = (imu_data - imu_data.min()) / (imu_data.max() - imu_data.min() + 1e-6)
            imu_data = self._encode_spikes(imu_normalized, threshold=0.5)
        
        return imu_data
    
    def _encode_spikes(self, data: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convert continuous data to spike trains."""
        # Threshold encoding
        spikes = (data > threshold).float()
        
        # Add temporal dynamics with refractory period
        if data.dim() >= 2:
            # Simple refractory period simulation
            refractory_mask = torch.ones_like(spikes)
            
            for t in range(1, spikes.shape[0]):
                # If neuron spiked in previous time step, reduce probability
                prev_spike_mask = spikes[t-1] > 0
                refractory_mask[t] = torch.where(
                    prev_spike_mask,
                    torch.rand_like(refractory_mask[t]) > 0.5,  # 50% chance during refractory
                    refractory_mask[t]
                )
            
            spikes = spikes * refractory_mask
        
        return spikes
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        return self.samples[idx].copy()
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes in the dataset."""
        class_counts = {}
        for sample in self.samples:
            label = sample['label'].item()
            class_counts[label] = class_counts.get(label, 0) + 1
        
        return class_counts
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get dataset statistics."""
        stats = {}
        
        for modality in self.modalities:
            modality_data = []
            for sample in self.samples:
                if modality in sample:
                    modality_data.append(sample[modality])
            
            if modality_data:
                stacked_data = torch.stack(modality_data)
                stats[modality] = {
                    'mean': stacked_data.mean().item(),
                    'std': stacked_data.std().item(),
                    'min': stacked_data.min().item(),
                    'max': stacked_data.max().item(),
                    'shape': list(stacked_data.shape[1:]),
                    'sparsity': (stacked_data == 0).float().mean().item()
                }
        
        return stats


# Factory function
def create_synthetic_dataset(
    split: str = "train",
    num_classes: int = 10,
    **kwargs
) -> SyntheticMultiModalDataset:
    """
    Create synthetic dataset with appropriate size for split.
    
    Args:
        split: Dataset split ('train', 'val', 'test')
        num_classes: Number of classes
        **kwargs: Additional arguments for SyntheticMultiModalDataset
        
    Returns:
        Configured synthetic dataset
    """
    # Default sizes based on split
    default_sizes = {
        'train': 1000,
        'val': 200,
        'test': 200
    }
    
    num_samples = kwargs.pop('num_samples', default_sizes.get(split, 1000))
    
    return SyntheticMultiModalDataset(
        num_samples=num_samples,
        num_classes=num_classes,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Test synthetic dataset
    dataset = create_synthetic_dataset(
        split='train',
        num_samples=100,
        sequence_length=50,
        modalities=['audio', 'events', 'imu'],
        num_classes=5,
        seed=42
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test sample
    sample = dataset[0]
    print("\nSample structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {value}")
    
    # Test statistics
    stats = dataset.get_statistics()
    print("\nDataset statistics:")
    for modality, modality_stats in stats.items():
        print(f"  {modality}:")
        for stat_name, stat_value in modality_stats.items():
            if isinstance(stat_value, list):
                print(f"    {stat_name}: {stat_value}")
            else:
                print(f"    {stat_name}: {stat_value:.4f}")
    
    # Test class distribution
    class_dist = dataset.get_class_distribution()
    print(f"\nClass distribution: {class_dist}")