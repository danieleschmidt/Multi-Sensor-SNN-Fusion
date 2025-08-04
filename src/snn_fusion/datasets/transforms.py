"""
Transforms for Multi-Modal Spiking Neural Network Data

This module implements data augmentation and preprocessing transforms
specifically designed for multi-modal spike-based data.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import random
import math


class SpikeTransform:
    """
    Base class for spike-based data transforms.
    
    Provides common functionality for transforming spike trains and
    multi-modal temporal data.
    """
    
    def __init__(self, probability: float = 1.0):
        """
        Initialize transform.
        
        Args:
            probability: Probability of applying the transform
        """
        self.probability = probability
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply transform to sample."""
        if random.random() < self.probability:
            return self.apply_transform(sample)
        return sample
    
    def apply_transform(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Override this method in subclasses."""
        raise NotImplementedError


class TemporalJitter(SpikeTransform):
    """
    Apply temporal jittering to spike data.
    
    Randomly shifts spike timing within a small window to improve
    temporal robustness.
    """
    
    def __init__(
        self,
        max_jitter: int = 5,
        probability: float = 0.5,
        modalities: Optional[List[str]] = None
    ):
        """
        Initialize temporal jitter transform.
        
        Args:
            max_jitter: Maximum jitter in time steps (±max_jitter)
            probability: Probability of applying transform
            modalities: List of modalities to apply jitter to (None = all)
        """
        super().__init__(probability)
        self.max_jitter = max_jitter
        self.modalities = modalities
    
    def apply_transform(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply temporal jittering."""
        transformed_sample = sample.copy()
        
        # Generate random jitter amount
        jitter = random.randint(-self.max_jitter, self.max_jitter)
        
        for key, tensor in sample.items():
            # Skip non-tensor data
            if not isinstance(tensor, torch.Tensor):
                continue
            
            # Skip if modality not in target list
            if self.modalities is not None and key not in self.modalities:
                continue
            
            # Skip if not temporal data
            if tensor.dim() < 2:
                continue
            
            # Apply circular shift along temporal dimension (assumed to be dim 0)
            if jitter != 0:
                transformed_sample[key] = torch.roll(tensor, jitter, dims=0)
        
        return transformed_sample


class ModalityDropout(SpikeTransform):
    """
    Randomly drop entire modalities during training.
    
    Helps improve robustness to missing modalities and prevents
    over-reliance on specific sensors.
    """
    
    def __init__(
        self,
        dropout_probability: float = 0.1,
        modalities: Optional[List[str]] = None,
        min_modalities: int = 1
    ):
        """
        Initialize modality dropout.
        
        Args:
            dropout_probability: Probability of dropping each modality
            modalities: List of modalities that can be dropped
            min_modalities: Minimum number of modalities to keep
        """
        super().__init__(probability=1.0)  # Always apply (internal logic handles probability)
        self.dropout_probability = dropout_probability
        self.modalities = modalities
        self.min_modalities = min_modalities
    
    def apply_transform(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply modality dropout."""
        transformed_sample = sample.copy()
        
        # Identify tensor modalities
        tensor_keys = [key for key, value in sample.items() 
                      if isinstance(value, torch.Tensor) and key not in ['label', 'sample_id']]
        
        # Filter to target modalities if specified
        if self.modalities is not None:
            tensor_keys = [key for key in tensor_keys if key in self.modalities]
        
        # Determine which modalities to drop
        dropped_modalities = []
        for key in tensor_keys:
            if random.random() < self.dropout_probability:
                dropped_modalities.append(key)
        
        # Ensure minimum number of modalities remain
        remaining_modalities = len(tensor_keys) - len(dropped_modalities)
        if remaining_modalities < self.min_modalities:
            # Keep some modalities
            num_to_keep = self.min_modalities - remaining_modalities
            to_keep = random.sample(dropped_modalities, min(num_to_keep, len(dropped_modalities)))
            dropped_modalities = [mod for mod in dropped_modalities if mod not in to_keep]
        
        # Apply dropout by zeroing out modalities
        for modality in dropped_modalities:
            transformed_sample[modality] = torch.zeros_like(sample[modality])
        
        return transformed_sample


class SpikeMasking(SpikeTransform):
    """
    Randomly mask portions of spike data.
    
    Similar to masking in language models but for temporal spike data.
    """
    
    def __init__(
        self,
        mask_probability: float = 0.15,
        mask_length_range: Tuple[int, int] = (5, 20),
        probability: float = 0.3,
        modalities: Optional[List[str]] = None
    ):
        """
        Initialize spike masking.
        
        Args:
            mask_probability: Probability of masking any given time window
            mask_length_range: (min_length, max_length) of masked segments
            probability: Probability of applying transform
            modalities: List of modalities to apply masking to
        """
        super().__init__(probability)
        self.mask_probability = mask_probability
        self.mask_length_range = mask_length_range
        self.modalities = modalities
    
    def apply_transform(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply spike masking."""
        transformed_sample = sample.copy()
        
        for key, tensor in sample.items():
            # Skip non-tensor data
            if not isinstance(tensor, torch.Tensor):
                continue
            
            # Skip if modality not in target list
            if self.modalities is not None and key not in self.modalities:
                continue
            
            # Skip non-temporal data
            if tensor.dim() < 2:
                continue
            
            # Apply masking
            sequence_length = tensor.shape[0]
            
            # Determine number of masks
            num_masks = int(sequence_length * self.mask_probability / 
                           np.mean(self.mask_length_range))
            
            for _ in range(num_masks):
                # Random mask length
                mask_length = random.randint(*self.mask_length_range)
                
                # Random start position
                start_pos = random.randint(0, max(0, sequence_length - mask_length))
                end_pos = min(start_pos + mask_length, sequence_length)
                
                # Apply mask (zero out)
                transformed_sample[key][start_pos:end_pos] = 0.0
        
        return transformed_sample


class SpikeNoise(SpikeTransform):
    """
    Add noise to spike data.
    
    Adds random spikes or removes existing spikes with low probability
    to improve robustness.
    """
    
    def __init__(
        self,
        noise_probability: float = 0.01,
        add_noise: bool = True,
        remove_spikes: bool = True,
        probability: float = 0.4,
        modalities: Optional[List[str]] = None
    ):
        """
        Initialize spike noise.
        
        Args:
            noise_probability: Probability of adding/removing each spike
            add_noise: Whether to add random spikes
            remove_spikes: Whether to randomly remove spikes
            probability: Probability of applying transform
            modalities: List of modalities to apply noise to
        """
        super().__init__(probability)
        self.noise_probability = noise_probability
        self.add_noise = add_noise
        self.remove_spikes = remove_spikes
        self.modalities = modalities
    
    def apply_transform(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply spike noise."""
        transformed_sample = sample.copy()
        
        for key, tensor in sample.items():
            # Skip non-tensor data
            if not isinstance(tensor, torch.Tensor):
                continue
            
            # Skip if modality not in target list
            if self.modalities is not None and key not in self.modalities:
                continue
            
            # Skip non-spike data (assume spike data is binary/sparse)
            if tensor.max() > 1.0 or tensor.dtype not in [torch.float32, torch.bool]:
                continue
            
            transformed_tensor = tensor.clone()
            
            # Add random spikes
            if self.add_noise:
                noise_mask = torch.rand_like(transformed_tensor) < self.noise_probability
                transformed_tensor = torch.clamp(transformed_tensor + noise_mask.float(), 0.0, 1.0)
            
            # Remove existing spikes
            if self.remove_spikes:
                removal_mask = torch.rand_like(transformed_tensor) < self.noise_probability
                spike_locations = transformed_tensor > 0
                remove_locations = spike_locations & removal_mask
                transformed_tensor[remove_locations] = 0.0
            
            transformed_sample[key] = transformed_tensor
        
        return transformed_sample


class ModalityScaling(SpikeTransform):
    """
    Apply random scaling to modality data.
    
    Scales the amplitude of continuous data or firing rates of spike data
    to simulate sensor variations.
    """
    
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        probability: float = 0.3,
        modalities: Optional[List[str]] = None
    ):
        """
        Initialize modality scaling.
        
        Args:
            scale_range: (min_scale, max_scale) for random scaling
            probability: Probability of applying transform
            modalities: List of modalities to apply scaling to
        """
        super().__init__(probability)
        self.scale_range = scale_range
        self.modalities = modalities
    
    def apply_transform(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply modality scaling."""
        transformed_sample = sample.copy()
        
        for key, tensor in sample.items():
            # Skip non-tensor data
            if not isinstance(tensor, torch.Tensor):
                continue
            
            # Skip if modality not in target list
            if self.modalities is not None and key not in self.modalities:
                continue
            
            # Generate random scale factor
            scale_factor = random.uniform(*self.scale_range)
            
            # Apply scaling
            transformed_sample[key] = tensor * scale_factor
            
            # Clamp spike data to [0, 1] range
            if tensor.max() <= 1.0 and tensor.min() >= 0.0:
                transformed_sample[key] = torch.clamp(transformed_sample[key], 0.0, 1.0)
        
        return transformed_sample


class Compose:
    """
    Compose multiple transforms together.
    
    Applies transforms in sequence to create complex augmentation pipelines.
    """
    
    def __init__(self, transforms: List[Callable]):
        """
        Initialize composed transform.
        
        Args:
            transforms: List of transform functions/objects
        """
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomChoice:
    """
    Randomly choose one transform from a list.
    
    Applies exactly one transform from the provided list.
    """
    
    def __init__(self, transforms: List[Callable], probabilities: Optional[List[float]] = None):
        """
        Initialize random choice transform.
        
        Args:
            transforms: List of transform functions/objects
            probabilities: Optional probabilities for each transform (must sum to 1)
        """
        self.transforms = transforms
        self.probabilities = probabilities
        
        if probabilities is not None:
            if len(probabilities) != len(transforms):
                raise ValueError("Number of probabilities must match number of transforms")
            if abs(sum(probabilities) - 1.0) > 1e-6:
                raise ValueError("Probabilities must sum to 1.0")
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply randomly chosen transform."""
        if self.probabilities is not None:
            transform = np.random.choice(self.transforms, p=self.probabilities)
        else:
            transform = random.choice(self.transforms)
        
        return transform(sample)


# Factory functions for common transform pipelines

def create_training_transforms(
    temporal_jitter: bool = True,
    modality_dropout: bool = True,
    spike_masking: bool = True,
    spike_noise: bool = True,
    modality_scaling: bool = True
) -> Compose:
    """
    Create standard training transform pipeline.
    
    Args:
        temporal_jitter: Whether to include temporal jittering
        modality_dropout: Whether to include modality dropout
        spike_masking: Whether to include spike masking
        spike_noise: Whether to include spike noise
        modality_scaling: Whether to include modality scaling
        
    Returns:
        Composed transform pipeline
    """
    transforms = []
    
    if temporal_jitter:
        transforms.append(TemporalJitter(max_jitter=5, probability=0.3))
    
    if modality_dropout:
        transforms.append(ModalityDropout(dropout_probability=0.1, min_modalities=1))
    
    if spike_masking:
        transforms.append(SpikeMasking(
            mask_probability=0.1,
            mask_length_range=(3, 15),
            probability=0.2
        ))
    
    if spike_noise:
        transforms.append(SpikeNoise(
            noise_probability=0.01,
            probability=0.3
        ))
    
    if modality_scaling:
        transforms.append(ModalityScaling(
            scale_range=(0.9, 1.1),
            probability=0.2
        ))
    
    return Compose(transforms)


def create_validation_transforms() -> Compose:
    """
    Create minimal validation transform pipeline.
    
    Returns:
        Composed transform pipeline (minimal augmentation)
    """
    # Minimal augmentation for validation
    transforms = [
        ModalityScaling(scale_range=(0.95, 1.05), probability=0.1)
    ]
    
    return Compose(transforms)


def create_test_transforms() -> Compose:
    """
    Create test transform pipeline (usually no augmentation).
    
    Returns:
        Composed transform pipeline (no augmentation)
    """
    # No augmentation for testing
    return Compose([])


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic sample for testing
    sample = {
        'audio': torch.rand(100, 2, 64),  # [time, channels, frequencies]
        'events': torch.rand(100, 128, 128, 2),  # [time, height, width, polarity]
        'imu': torch.rand(100, 6),  # [time, channels]
        'label': torch.tensor(3)
    }
    
    print("Original sample shapes:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test individual transforms
    print("\nTesting individual transforms:")
    
    # Temporal jitter
    jitter_transform = TemporalJitter(max_jitter=3, probability=1.0)
    jittered_sample = jitter_transform(sample)
    print("✓ Temporal jitter applied")
    
    # Modality dropout
    dropout_transform = ModalityDropout(dropout_probability=0.5, min_modalities=2)
    dropout_sample = dropout_transform(sample)
    print("✓ Modality dropout applied")
    
    # Spike masking
    masking_transform = SpikeMasking(mask_probability=0.2, probability=1.0)
    masked_sample = masking_transform(sample)
    print("✓ Spike masking applied")
    
    # Test composed transforms
    print("\nTesting composed transforms:")
    
    training_transforms = create_training_transforms()
    augmented_sample = training_transforms(sample)
    
    print("Training transforms applied:")
    for key, value in augmented_sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} (mean: {value.mean():.4f})")
    
    # Test random choice
    choice_transforms = RandomChoice([
        TemporalJitter(probability=1.0),
        SpikeMasking(probability=1.0),
        SpikeNoise(probability=1.0)
    ])
    
    choice_sample = choice_transforms(sample)
    print("✓ Random choice transform applied")