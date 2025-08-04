"""
Multi-Modal Data Loaders for SNN Training

This module implements specialized data loaders for multi-modal spiking neural
network training, with support for temporal synchronization, missing modalities,
and efficient batch processing.
"""

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings
from collections import defaultdict
import random


class MultiModalCollate:
    """
    Custom collate function for multi-modal data with variable sequence lengths.
    
    Handles padding, temporal synchronization, and missing modalities across
    samples in a batch.
    """
    
    def __init__(
        self,
        padding_value: float = 0.0,
        max_sequence_length: Optional[int] = None,
        temporal_alignment: str = "start",  # "start", "center", "end"
        handle_missing_modalities: str = "zero_pad",  # "zero_pad", "skip", "interpolate"
    ):
        """
        Initialize collate function.
        
        Args:
            padding_value: Value to use for padding sequences
            max_sequence_length: Maximum sequence length (truncate if longer)
            temporal_alignment: How to align sequences of different lengths
            handle_missing_modalities: How to handle missing modalities
        """
        self.padding_value = padding_value
        self.max_sequence_length = max_sequence_length
        self.temporal_alignment = temporal_alignment
        self.handle_missing_modalities = handle_missing_modalities
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of multi-modal samples.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched and padded multi-modal data
        """
        if not batch:
            return {}
        
        # Determine all modalities present in the batch
        all_modalities = set()
        for sample in batch:
            all_modalities.update(sample.keys())
        
        # Remove non-tensor keys
        tensor_keys = set()
        for sample in batch:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    tensor_keys.add(key)
        
        all_modalities = all_modalities.intersection(tensor_keys)
        
        # Determine sequence lengths
        sequence_lengths = []
        for sample in batch:
            max_len = 0
            for modality in all_modalities:
                if modality in sample and len(sample[modality].shape) > 0:
                    max_len = max(max_len, sample[modality].shape[0])
            sequence_lengths.append(max_len)
        
        # Determine target sequence length
        target_length = max(sequence_lengths) if sequence_lengths else 100
        if self.max_sequence_length is not None:
            target_length = min(target_length, self.max_sequence_length)
        
        # Collate each modality
        collated_batch = {}
        
        for modality in all_modalities:
            modality_data = []
            
            for sample in batch:
                if modality in sample:
                    data = sample[modality]
                    
                    # Handle sequence alignment and padding
                    if len(data.shape) > 0 and data.shape[0] != target_length:
                        data = self._align_and_pad(data, target_length)
                    
                    modality_data.append(data)
                else:
                    # Handle missing modality
                    if self.handle_missing_modalities == "zero_pad":
                        # Create zero tensor with appropriate shape
                        ref_shape = self._get_reference_shape(batch, modality)
                        if ref_shape is not None:
                            zero_data = torch.full(ref_shape, self.padding_value)
                            modality_data.append(zero_data)
                    elif self.handle_missing_modalities == "skip":
                        # Skip this sample for this modality
                        continue
            
            if modality_data:
                try:
                    collated_batch[modality] = torch.stack(modality_data)
                except RuntimeError as e:
                    warnings.warn(f"Error stacking modality {modality}: {e}")
                    # Fallback to padding
                    collated_batch[modality] = self._stack_with_padding(modality_data)
        
        # Handle non-tensor data (labels, metadata)
        for key in batch[0].keys():
            if key not in tensor_keys and key not in collated_batch:
                values = [sample.get(key, None) for sample in batch]
                if all(isinstance(v, (int, float, torch.Tensor)) for v in values):
                    if all(isinstance(v, torch.Tensor) for v in values):
                        collated_batch[key] = torch.stack(values)
                    else:
                        collated_batch[key] = torch.tensor(values)
                else:
                    collated_batch[key] = values
        
        return collated_batch
    
    def _align_and_pad(self, data: torch.Tensor, target_length: int) -> torch.Tensor:
        """Align and pad sequence to target length."""
        current_length = data.shape[0]
        
        if current_length == target_length:
            return data
        
        if current_length > target_length:
            # Truncate based on alignment
            if self.temporal_alignment == "start":
                return data[:target_length]
            elif self.temporal_alignment == "end":
                return data[-target_length:]
            elif self.temporal_alignment == "center":
                start_idx = (current_length - target_length) // 2
                return data[start_idx:start_idx + target_length]
        else:
            # Pad based on alignment
            pad_amount = target_length - current_length
            
            if self.temporal_alignment == "start":
                # Pad at the end
                pad_shape = list(data.shape)
                pad_shape[0] = pad_amount
                padding = torch.full(pad_shape, self.padding_value, dtype=data.dtype)
                return torch.cat([data, padding], dim=0)
            elif self.temporal_alignment == "end":
                # Pad at the beginning
                pad_shape = list(data.shape)
                pad_shape[0] = pad_amount
                padding = torch.full(pad_shape, self.padding_value, dtype=data.dtype)
                return torch.cat([padding, data], dim=0)
            elif self.temporal_alignment == "center":
                # Pad on both sides
                pad_before = pad_amount // 2
                pad_after = pad_amount - pad_before
                
                pad_shape = list(data.shape)
                
                if pad_before > 0:
                    pad_shape[0] = pad_before
                    padding_before = torch.full(pad_shape, self.padding_value, dtype=data.dtype)
                    data = torch.cat([padding_before, data], dim=0)
                
                if pad_after > 0:
                    pad_shape[0] = pad_after
                    padding_after = torch.full(pad_shape, self.padding_value, dtype=data.dtype)
                    data = torch.cat([data, padding_after], dim=0)
                
                return data
        
        return data
    
    def _get_reference_shape(self, batch: List[Dict], modality: str) -> Optional[Tuple[int, ...]]:
        """Get reference shape for missing modality from other samples."""
        for sample in batch:
            if modality in sample:
                return sample[modality].shape
        return None
    
    def _stack_with_padding(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Stack tensors with different shapes by padding."""
        if not tensors:
            return torch.tensor([])
        
        # Find maximum dimensions
        max_dims = []
        for dim in range(max(t.dim() for t in tensors)):
            max_size = max(t.shape[dim] if dim < t.dim() else 1 for t in tensors)
            max_dims.append(max_size)
        
        # Pad all tensors to maximum dimensions
        padded_tensors = []
        for tensor in tensors:
            # Calculate padding needed
            padding = []
            for dim in reversed(range(len(max_dims))):
                current_size = tensor.shape[dim] if dim < tensor.dim() else 1
                pad_size = max_dims[dim] - current_size
                padding.extend([0, pad_size])
            
            # Apply padding
            if any(p > 0 for p in padding):
                padded_tensor = torch.nn.functional.pad(
                    tensor, padding, value=self.padding_value
                )
            else:
                padded_tensor = tensor
            
            padded_tensors.append(padded_tensor)
        
        return torch.stack(padded_tensors)


class MultiModalDataLoader(DataLoader):
    """
    Specialized DataLoader for multi-modal spiking neural network data.
    
    Extends PyTorch DataLoader with multi-modal specific functionality
    including missing modality handling and temporal synchronization.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        modality_dropout: float = 0.0,
        temporal_jitter: bool = False,
        **kwargs
    ):
        """
        Initialize multi-modal data loader.
        
        Args:
            dataset: Multi-modal dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            collate_fn: Custom collate function (uses MultiModalCollate if None)
            modality_dropout: Probability of dropping modalities during training
            temporal_jitter: Whether to apply temporal jittering
            **kwargs: Additional arguments for DataLoader
        """
        # Use custom collate function if none provided
        if collate_fn is None:
            collate_fn = MultiModalCollate()
        
        self.modality_dropout = modality_dropout
        self.temporal_jitter = temporal_jitter
        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **kwargs
        )
    
    def __iter__(self):
        """Override iterator to apply augmentations."""
        for batch in super().__iter__():
            # Apply modality dropout during training
            if self.modality_dropout > 0.0 and hasattr(self.dataset, 'config'):
                batch = self._apply_modality_dropout(batch)
            
            # Apply temporal jittering
            if self.temporal_jitter:
                batch = self._apply_temporal_jitter(batch)
            
            yield batch
    
    def _apply_modality_dropout(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply modality dropout augmentation."""
        # Identify modality keys (exclude labels and metadata)
        modality_keys = []
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor) and key not in ['label', 'sample_id', 'sequence_length']:
                modality_keys.append(key)
        
        # Randomly drop modalities
        for key in modality_keys:
            if random.random() < self.modality_dropout:
                # Replace with zeros
                batch[key] = torch.zeros_like(batch[key])
        
        return batch
    
    def _apply_temporal_jitter(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply temporal jittering augmentation."""
        # Apply small random temporal shifts
        jitter_amount = random.randint(-5, 5)  # ±5 time steps
        
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                # Apply circular shift along time dimension
                if jitter_amount != 0:
                    batch[key] = torch.roll(tensor, jitter_amount, dims=1)
        
        return batch


class BalancedMultiModalSampler(data.Sampler):
    """
    Balanced sampler for multi-modal data that ensures equal representation
    of different classes and modality combinations.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        samples_per_class: Optional[int] = None,
        replacement: bool = True
    ):
        """
        Initialize balanced sampler.
        
        Args:
            dataset: Dataset to sample from
            samples_per_class: Number of samples per class (None for balanced)
            replacement: Whether to sample with replacement
        """
        self.dataset = dataset
        self.replacement = replacement
        
        # Analyze dataset to get class distribution
        self.class_indices = self._build_class_indices()
        self.num_classes = len(self.class_indices)
        
        if samples_per_class is None:
            # Use minimum class size for balance
            min_class_size = min(len(indices) for indices in self.class_indices.values())
            self.samples_per_class = min_class_size
        else:
            self.samples_per_class = samples_per_class
        
        self.total_samples = self.num_classes * self.samples_per_class
    
    def _build_class_indices(self) -> Dict[int, List[int]]:
        """Build mapping from class labels to sample indices."""
        class_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                label = sample.get('label', 0)
                if isinstance(label, torch.Tensor):
                    label = label.item()
                class_indices[int(label)].append(idx)
            except Exception as e:
                warnings.warn(f"Error getting label for sample {idx}: {e}")
        
        return dict(class_indices)
    
    def __iter__(self):
        """Generate balanced sample indices."""
        for _ in range(self.total_samples):
            # Randomly select a class
            class_label = random.choice(list(self.class_indices.keys()))
            
            # Randomly select a sample from that class
            class_samples = self.class_indices[class_label]
            sample_idx = random.choice(class_samples)
            
            yield sample_idx
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return self.total_samples


def create_dataloaders(
    dataset_config: Dict[str, Any],
    batch_size: int = 32,
    num_workers: int = 4,
    train_shuffle: bool = True,
    val_shuffle: bool = False,
    test_shuffle: bool = False,
    modality_dropout: float = 0.1,
    use_balanced_sampling: bool = False,
    **loader_kwargs
) -> Dict[str, MultiModalDataLoader]:
    """
    Create train, validation, and test data loaders for multi-modal SNN training.
    
    Args:
        dataset_config: Configuration for dataset creation
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes
        train_shuffle: Whether to shuffle training data
        val_shuffle: Whether to shuffle validation data
        test_shuffle: Whether to shuffle test data
        modality_dropout: Probability of modality dropout during training
        use_balanced_sampling: Whether to use balanced sampling
        **loader_kwargs: Additional arguments for DataLoader
        
    Returns:
        Dictionary of data loaders for each split
    """
    from .maven_dataset import MAVENDataset, MAVENConfig
    
    data_loaders = {}
    
    # Create datasets for each split
    splits = ['train', 'val', 'test']
    shuffle_settings = [train_shuffle, val_shuffle, test_shuffle]
    
    for split, shuffle in zip(splits, shuffle_settings):
        # Create dataset config for this split
        split_config = MAVENConfig(**{**dataset_config, 'split': split})
        
        try:
            # Create dataset
            dataset = MAVENDataset(split_config)
            
            # Create sampler if requested
            sampler = None
            if use_balanced_sampling and split == 'train':
                sampler = BalancedMultiModalSampler(dataset)
                shuffle = False  # Sampler handles shuffling
            
            # Apply modality dropout only to training set
            dropout_prob = modality_dropout if split == 'train' else 0.0
            
            # Create data loader
            loader = MultiModalDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                sampler=sampler,
                modality_dropout=dropout_prob,
                temporal_jitter=(split == 'train'),
                **loader_kwargs
            )
            
            data_loaders[split] = loader
            
        except Exception as e:
            warnings.warn(f"Failed to create {split} dataloader: {e}")
    
    return data_loaders


def create_inference_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 1,
    **kwargs
) -> MultiModalDataLoader:
    """
    Create a data loader optimized for inference.
    
    Args:
        dataset: Dataset for inference
        batch_size: Batch size (typically 1 for inference)
        num_workers: Number of workers
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Configured data loader for inference
    """
    return MultiModalDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        modality_dropout=0.0,
        temporal_jitter=False,
        **kwargs
    )


# Utility functions for data loading

def analyze_dataloader(dataloader: MultiModalDataLoader) -> Dict[str, Any]:
    """
    Analyze a data loader to understand data characteristics.
    
    Args:
        dataloader: Data loader to analyze
        
    Returns:
        Dictionary of analysis results
    """
    analysis = {
        'num_batches': len(dataloader),
        'batch_size': dataloader.batch_size,
        'dataset_size': len(dataloader.dataset),
        'modalities': set(),
        'shapes': {},
        'dtypes': {},
        'value_ranges': {}
    }
    
    # Analyze first few batches
    sample_batches = []
    for i, batch in enumerate(dataloader):
        sample_batches.append(batch)
        if i >= 2:  # Sample first 3 batches
            break
    
    if sample_batches:
        # Analyze modalities and shapes
        for key in sample_batches[0].keys():
            if isinstance(sample_batches[0][key], torch.Tensor):
                analysis['modalities'].add(key)
                analysis['shapes'][key] = list(sample_batches[0][key].shape)
                analysis['dtypes'][key] = str(sample_batches[0][key].dtype)
                
                # Compute value ranges
                values = torch.cat([batch[key].flatten() for batch in sample_batches])
                analysis['value_ranges'][key] = {
                    'min': values.min().item(),
                    'max': values.max().item(),
                    'mean': values.mean().item(),
                    'std': values.std().item()
                }
    
    return analysis


def test_dataloader_performance(
    dataloader: MultiModalDataLoader,
    num_batches: int = 10
) -> Dict[str, float]:
    """
    Test data loader performance and identify bottlenecks.
    
    Args:
        dataloader: Data loader to test
        num_batches: Number of batches to test
        
    Returns:
        Performance metrics
    """
    import time
    
    times = []
    batch_sizes = []
    
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        batch_start = time.time()
        
        # Simulate processing time
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                _ = tensor.sum()  # Simple operation to ensure data is loaded
        
        batch_end = time.time()
        times.append(batch_end - batch_start)
        
        # Track actual batch size (may vary for last batch)
        if 'label' in batch:
            batch_sizes.append(len(batch['label']))
        
        if i >= num_batches - 1:
            break
    
    total_time = time.time() - start_time
    
    performance = {
        'total_time': total_time,
        'avg_batch_time': np.mean(times),
        'std_batch_time': np.std(times),
        'batches_per_second': len(times) / total_time,
        'samples_per_second': sum(batch_sizes) / total_time if batch_sizes else 0,
        'num_batches_tested': len(times)
    }
    
    return performance


# Example usage
if __name__ == "__main__":
    # Test data loader creation
    config = {
        'root_dir': './data/maven',
        'modalities': ['audio', 'events', 'imu'],
        'sequence_length': 100,
        'spike_encoding': True
    }
    
    try:
        dataloaders = create_dataloaders(
            dataset_config=config,
            batch_size=8,
            num_workers=2,
            modality_dropout=0.1
        )
        
        if 'train' in dataloaders:
            print("Created data loaders successfully!")
            
            # Analyze the training data loader
            analysis = analyze_dataloader(dataloaders['train'])
            print("Data loader analysis:")
            for key, value in analysis.items():
                print(f"  {key}: {value}")
            
            # Test performance
            performance = test_dataloader_performance(dataloaders['train'])
            print("\nPerformance metrics:")
            for key, value in performance.items():
                print(f"  {key}: {value:.4f}")
    
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()