"""
Validation Utilities for SNN-Fusion

This module provides comprehensive validation functions for data, configurations,
and model parameters to ensure robustness and prevent errors.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Type
import warnings
from pathlib import Path
import json


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[Optional[int], ...],
    tensor_name: str = "tensor",
    allow_batch: bool = True
) -> bool:
    """
    Validate tensor shape against expected dimensions.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (None for variable dimensions)
        tensor_name: Name of tensor for error messages
        allow_batch: Whether to allow additional batch dimension
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If shape is invalid
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(
            f"{tensor_name} must be a torch.Tensor, got {type(tensor)}",
            error_code="INVALID_TYPE"
        )
    
    actual_shape = tensor.shape
    
    # Handle batch dimension
    if allow_batch and len(actual_shape) == len(expected_shape) + 1:
        # Skip batch dimension for comparison
        actual_shape = actual_shape[1:]
    
    if len(actual_shape) != len(expected_shape):
        raise ValidationError(
            f"{tensor_name} has {len(actual_shape)} dimensions, expected {len(expected_shape)}",
            error_code="DIMENSION_MISMATCH",
            details={
                "actual_shape": list(tensor.shape),
                "expected_shape": list(expected_shape)
            }
        )
    
    # Check each dimension
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValidationError(
                f"{tensor_name} dimension {i} has size {actual}, expected {expected}",
                error_code="SIZE_MISMATCH",
                details={
                    "dimension": i,
                    "actual_size": actual,
                    "expected_size": expected
                }
            )
    
    return True


def validate_modality_data(
    modality_data: Dict[str, torch.Tensor],
    expected_modalities: List[str],
    allow_missing: bool = False,
    validate_shapes: bool = True
) -> bool:
    """
    Validate multi-modal data dictionary.
    
    Args:
        modality_data: Dictionary of modality tensors
        expected_modalities: List of expected modality names
        allow_missing: Whether missing modalities are allowed
        validate_shapes: Whether to validate tensor shapes
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(modality_data, dict):
        raise ValidationError(
            f"Modality data must be a dictionary, got {type(modality_data)}",
            error_code="INVALID_TYPE"
        )
    
    # Check for missing modalities
    missing_modalities = set(expected_modalities) - set(modality_data.keys())
    if missing_modalities and not allow_missing:
        raise ValidationError(
            f"Missing required modalities: {list(missing_modalities)}",
            error_code="MISSING_MODALITIES",
            details={"missing": list(missing_modalities)}
        )
    
    # Check for unexpected modalities
    unexpected_modalities = set(modality_data.keys()) - set(expected_modalities)
    if unexpected_modalities:
        warnings.warn(f"Unexpected modalities found: {list(unexpected_modalities)}")
    
    # Validate tensor types and basic properties
    for modality, data in modality_data.items():
        if not isinstance(data, torch.Tensor):
            raise ValidationError(
                f"Modality '{modality}' data must be a torch.Tensor, got {type(data)}",
                error_code="INVALID_MODALITY_TYPE",
                details={"modality": modality}
            )
        
        # Check for NaN or Inf values
        if torch.isnan(data).any():
            raise ValidationError(
                f"Modality '{modality}' contains NaN values",
                error_code="NAN_VALUES",
                details={"modality": modality}
            )
        
        if torch.isinf(data).any():
            raise ValidationError(
                f"Modality '{modality}' contains infinite values",
                error_code="INF_VALUES",
                details={"modality": modality}
            )
    
    # Validate temporal consistency (all modalities should have same sequence length)
    if validate_shapes and len(modality_data) > 1:
        sequence_lengths = {}
        for modality, data in modality_data.items():
            if data.dim() > 0:
                sequence_lengths[modality] = data.shape[0]
        
        if len(set(sequence_lengths.values())) > 1:
            warnings.warn(
                f"Modalities have different sequence lengths: {sequence_lengths}. "
                "This may cause issues during processing."
            )
    
    return True


def validate_spike_data(
    spike_tensor: torch.Tensor,
    tensor_name: str = "spike_data",
    check_binary: bool = True,
    max_firing_rate: float = 1.0
) -> bool:
    """
    Validate spike train data.
    
    Args:
        spike_tensor: Tensor containing spike data
        tensor_name: Name for error messages
        check_binary: Whether to check if data is binary (0/1)
        max_firing_rate: Maximum allowed firing rate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    validate_tensor_shape(spike_tensor, (None,), tensor_name, allow_batch=True)
    
    # Check value range
    min_val = spike_tensor.min().item()
    max_val = spike_tensor.max().item()
    
    if min_val < 0:
        raise ValidationError(
            f"{tensor_name} contains negative values (min: {min_val})",
            error_code="NEGATIVE_SPIKES"
        )
    
    if check_binary and max_val > 1.0:
        raise ValidationError(
            f"{tensor_name} contains values > 1.0 (max: {max_val}), expected binary spikes",
            error_code="NON_BINARY_SPIKES"
        )
    
    # Check firing rate
    if spike_tensor.dim() >= 2:
        firing_rate = spike_tensor.mean().item()
        if firing_rate > max_firing_rate:
            warnings.warn(
                f"{tensor_name} has high firing rate ({firing_rate:.3f} > {max_firing_rate}). "
                "This may indicate encoding issues."
            )
    
    return True


def validate_configuration(
    config: Dict[str, Any],
    schema: Dict[str, Dict[str, Any]],
    config_name: str = "configuration"
) -> bool:
    """
    Validate configuration dictionary against schema.
    
    Args:
        config: Configuration dictionary to validate
        schema: Schema defining expected structure and types
        config_name: Name for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(config, dict):
        raise ValidationError(
            f"{config_name} must be a dictionary, got {type(config)}",
            error_code="INVALID_CONFIG_TYPE"
        )
    
    # Check required fields
    for field_name, field_schema in schema.items():
        required = field_schema.get('required', False)
        
        if required and field_name not in config:
            raise ValidationError(
                f"Required field '{field_name}' missing from {config_name}",
                error_code="MISSING_REQUIRED_FIELD",
                details={"field": field_name}
            )
        
        if field_name in config:
            value = config[field_name]
            
            # Type validation
            expected_type = field_schema.get('type')
            if expected_type and not isinstance(value, expected_type):
                raise ValidationError(
                    f"Field '{field_name}' has type {type(value)}, expected {expected_type}",
                    error_code="INVALID_FIELD_TYPE",
                    details={"field": field_name, "expected_type": expected_type.__name__}
                )
            
            # Range validation
            min_val = field_schema.get('min')
            max_val = field_schema.get('max')
            
            if min_val is not None and isinstance(value, (int, float)) and value < min_val:
                raise ValidationError(
                    f"Field '{field_name}' value {value} is below minimum {min_val}",
                    error_code="VALUE_BELOW_MINIMUM",
                    details={"field": field_name, "value": value, "minimum": min_val}
                )
            
            if max_val is not None and isinstance(value, (int, float)) and value > max_val:
                raise ValidationError(
                    f"Field '{field_name}' value {value} is above maximum {max_val}",
                    error_code="VALUE_ABOVE_MAXIMUM",
                    details={"field": field_name, "value": value, "maximum": max_val}
                )
            
            # Choice validation
            choices = field_schema.get('choices')
            if choices and value not in choices:
                raise ValidationError(
                    f"Field '{field_name}' value '{value}' not in allowed choices: {choices}",
                    error_code="INVALID_CHOICE",
                    details={"field": field_name, "value": value, "choices": choices}
                )
    
    return True


def validate_model_parameters(
    model: torch.nn.Module,
    expected_param_count: Optional[int] = None,
    check_gradients: bool = True
) -> bool:
    """
    Validate model parameters and gradients.
    
    Args:
        model: PyTorch model to validate
        expected_param_count: Expected number of parameters
        check_gradients: Whether to check gradient properties
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(model, torch.nn.Module):
        raise ValidationError(
            f"Model must be a torch.nn.Module, got {type(model)}",
            error_code="INVALID_MODEL_TYPE"
        )
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    
    if expected_param_count is not None and param_count != expected_param_count:
        raise ValidationError(
            f"Model has {param_count} parameters, expected {expected_param_count}",
            error_code="PARAMETER_COUNT_MISMATCH",
            details={"actual": param_count, "expected": expected_param_count}
        )
    
    # Check for NaN or Inf in parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            raise ValidationError(
                f"Parameter '{name}' contains NaN values",
                error_code="NAN_PARAMETERS",
                details={"parameter": name}
            )
        
        if torch.isinf(param).any():
            raise ValidationError(
                f"Parameter '{name}' contains infinite values",
                error_code="INF_PARAMETERS",
                details={"parameter": name}
            )
        
        # Check gradients if available
        if check_gradients and param.grad is not None:
            if torch.isnan(param.grad).any():
                raise ValidationError(
                    f"Gradient for parameter '{name}' contains NaN values",
                    error_code="NAN_GRADIENTS",
                    details={"parameter": name}
                )
            
            if torch.isinf(param.grad).any():
                raise ValidationError(
                    f"Gradient for parameter '{name}' contains infinite values",
                    error_code="INF_GRADIENTS",
                    details={"parameter": name}
                )
    
    return True


def validate_training_data(
    data_loader,
    expected_batch_size: Optional[int] = None,
    sample_validation: bool = True,
    max_samples_to_check: int = 10
) -> bool:
    """
    Validate training data loader.
    
    Args:
        data_loader: DataLoader to validate
        expected_batch_size: Expected batch size
        sample_validation: Whether to validate sample data
        max_samples_to_check: Maximum number of batches to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not hasattr(data_loader, '__iter__'):
        raise ValidationError(
            "Data loader must be iterable",
            error_code="NON_ITERABLE_DATALOADER"
        )
    
    # Check a few batches
    batch_count = 0
    for batch in data_loader:
        if batch_count >= max_samples_to_check:
            break
        
        if not isinstance(batch, dict):
            raise ValidationError(
                f"Expected batch to be a dictionary, got {type(batch)}",
                error_code="INVALID_BATCH_TYPE"
            )
        
        # Check batch size
        if expected_batch_size is not None:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    actual_batch_size = value.shape[0]
                    if actual_batch_size != expected_batch_size:
                        warnings.warn(
                            f"Batch size mismatch for '{key}': got {actual_batch_size}, "
                            f"expected {expected_batch_size}"
                        )
                    break
        
        # Sample-level validation
        if sample_validation:
            # Check for required keys
            if 'label' not in batch:
                warnings.warn("Batch missing 'label' key")
            
            # Validate tensor types
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    validate_tensor_shape(value, (None,), f"batch['{key}']", allow_batch=True)
        
        batch_count += 1
    
    if batch_count == 0:
        raise ValidationError(
            "Data loader is empty",
            error_code="EMPTY_DATALOADER"
        )
    
    return True


def validate_file_integrity(file_path: Union[str, Path]) -> bool:
    """
    Validate file exists and is readable.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    path = Path(file_path)
    
    if not path.exists():
        raise ValidationError(
            f"File does not exist: {path}",
            error_code="FILE_NOT_FOUND"
        )
    
    if not path.is_file():
        raise ValidationError(
            f"Path is not a file: {path}",
            error_code="NOT_A_FILE"
        )
    
    try:
        with open(path, 'rb') as f:
            # Try to read first few bytes
            f.read(1024)
    except PermissionError:
        raise ValidationError(
            f"No read permission for file: {path}",
            error_code="NO_READ_PERMISSION"
        )
    except Exception as e:
        raise ValidationError(
            f"Cannot read file {path}: {e}",
            error_code="FILE_READ_ERROR"
        )
    
    return True


# Schema definitions for common configurations

DATASET_CONFIG_SCHEMA = {
    'root_dir': {'type': str, 'required': True},
    'split': {'type': str, 'required': True, 'choices': ['train', 'val', 'test']},
    'modalities': {'type': list, 'required': True},
    'sequence_length': {'type': int, 'required': True, 'min': 1, 'max': 10000},
    'batch_size': {'type': int, 'required': False, 'min': 1, 'max': 1024},
    'spike_encoding': {'type': bool, 'required': False},
}

MODEL_CONFIG_SCHEMA = {
    'input_shapes': {'type': dict, 'required': True},
    'hidden_dim': {'type': int, 'required': True, 'min': 1, 'max': 10000},
    'num_layers': {'type': int, 'required': False, 'min': 1, 'max': 100},
    'dropout': {'type': float, 'required': False, 'min': 0.0, 'max': 1.0},
    'learning_rate': {'type': float, 'required': False, 'min': 1e-6, 'max': 1.0},
}

TRAINING_CONFIG_SCHEMA = {
    'epochs': {'type': int, 'required': True, 'min': 1, 'max': 10000},
    'learning_rate': {'type': float, 'required': True, 'min': 1e-6, 'max': 1.0},
    'batch_size': {'type': int, 'required': True, 'min': 1, 'max': 1024},
    'device': {'type': str, 'required': False, 'choices': ['cpu', 'cuda', 'auto']},
    'save_interval': {'type': int, 'required': False, 'min': 1},
}


# Convenience functions

def validate_dataset_config(config: Dict[str, Any]) -> bool:
    """Validate dataset configuration."""
    return validate_configuration(config, DATASET_CONFIG_SCHEMA, "dataset config")


def validate_model_config(config: Dict[str, Any]) -> bool:
    """Validate model configuration."""
    return validate_configuration(config, MODEL_CONFIG_SCHEMA, "model config")


def validate_training_config(config: Dict[str, Any]) -> bool:
    """Validate training configuration."""
    return validate_configuration(config, TRAINING_CONFIG_SCHEMA, "training config")


# Example usage and testing
if __name__ == "__main__":
    print("Testing validation functions...")
    
    # Test tensor validation
    try:
        test_tensor = torch.randn(32, 100, 64)
        validate_tensor_shape(test_tensor, (100, 64), "test_tensor", allow_batch=True)
        print("✓ Tensor validation passed")
    except ValidationError as e:
        print(f"✗ Tensor validation failed: {e}")
    
    # Test modality validation
    try:
        modality_data = {
            'audio': torch.randn(100, 2, 64),
            'events': torch.randn(100, 128, 128, 2),
            'imu': torch.randn(100, 6)
        }
        validate_modality_data(modality_data, ['audio', 'events', 'imu'])
        print("✓ Modality validation passed")
    except ValidationError as e:
        print(f"✗ Modality validation failed: {e}")
    
    # Test configuration validation
    try:
        config = {
            'root_dir': './data',
            'split': 'train',
            'modalities': ['audio', 'events'],
            'sequence_length': 100,
            'spike_encoding': True
        }
        validate_dataset_config(config)
        print("✓ Configuration validation passed")
    except ValidationError as e:
        print(f"✗ Configuration validation failed: {e}")
    
    print("Validation tests completed!")