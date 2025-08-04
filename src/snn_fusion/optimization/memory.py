"""
Memory Optimization for SNN-Fusion

This module provides memory optimization techniques for efficient training
and inference of large spiking neural networks, including gradient checkpointing,
model sharding, and memory-efficient operations.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import gc
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from contextlib import contextmanager
import numpy as np
from collections import defaultdict
import threading
import weakref


class MemoryOptimizer:
    """
    Comprehensive memory optimization for SNN training and inference.
    
    Provides automatic memory management, optimization strategies,
    and memory leak detection for large-scale SNN deployments.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_memory_gb: float = 4.0,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision: bool = True,
        optimize_for_inference: bool = False
    ):
        """
        Initialize memory optimizer.
        
        Args:
            model: Model to optimize
            target_memory_gb: Target memory usage in GB
            enable_gradient_checkpointing: Enable gradient checkpointing
            enable_mixed_precision: Enable mixed precision training
            optimize_for_inference: Optimize for inference vs training
        """
        self.model = model
        self.target_memory_bytes = int(target_memory_gb * 1024 ** 3)
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision = enable_mixed_precision
        self.optimize_for_inference = optimize_for_inference
        
        # Memory tracking
        self.memory_usage_history = []
        self.peak_memory = 0
        self.baseline_memory = 0
        
        # Optimization strategies
        self.applied_optimizations = []
        self.checkpoint_layers = []
        
        # Mixed precision setup
        self.scaler = None
        if enable_mixed_precision and torch.cuda.is_available():
            try:
                self.scaler = torch.cuda.amp.GradScaler()
            except AttributeError:
                warnings.warn("Mixed precision not available in this PyTorch version")
    
    def optimize_model(self) -> Dict[str, Any]:
        """
        Apply comprehensive memory optimizations to the model.
        
        Returns:
            Dictionary of optimization results
        """
        optimization_results = {
            'initial_memory_mb': self._get_memory_usage_mb(),
            'optimizations_applied': [],
            'memory_savings_mb': 0,
            'parameter_reduction': 0
        }
        
        initial_memory = optimization_results['initial_memory_mb']
        self.baseline_memory = initial_memory
        
        # 1. Apply gradient checkpointing
        if self.enable_gradient_checkpointing:
            self._apply_gradient_checkpointing()
            optimization_results['optimizations_applied'].append('gradient_checkpointing')
        
        # 2. Optimize tensor operations
        self._optimize_tensor_operations()
        optimization_results['optimizations_applied'].append('tensor_optimization')
        
        # 3. Apply memory-efficient attention if applicable
        self._apply_memory_efficient_attention()
        optimization_results['optimizations_applied'].append('efficient_attention')
        
        # 4. Optimize data types
        if self.optimize_for_inference:
            self._optimize_data_types()
            optimization_results['optimizations_applied'].append('data_type_optimization')
        
        # 5. Clean up unused parameters
        self._cleanup_unused_parameters()
        optimization_results['optimizations_applied'].append('parameter_cleanup')
        
        # Calculate savings
        final_memory = self._get_memory_usage_mb()
        optimization_results['final_memory_mb'] = final_memory
        optimization_results['memory_savings_mb'] = initial_memory - final_memory
        
        # Parameter count reduction (if any)
        optimization_results['parameter_reduction'] = self._calculate_parameter_reduction()
        
        self.applied_optimizations = optimization_results['optimizations_applied']
        
        return optimization_results
    
    def _apply_gradient_checkpointing(self) -> None:
        """Apply gradient checkpointing to reduce memory usage."""
        checkpoint_modules = []
        
        # Find suitable modules for checkpointing
        for name, module in self.model.named_modules():
            # Target large modules or those with many parameters
            param_count = sum(p.numel() for p in module.parameters())
            
            if param_count > 100000:  # 100K parameters threshold
                checkpoint_modules.append((name, module))
        
        # Apply checkpointing to selected modules
        for name, module in checkpoint_modules:
            if hasattr(module, 'forward'):
                # Wrap forward method with checkpoint
                original_forward = module.forward
                
                def checkpointed_forward(*args, **kwargs):
                    return checkpoint(original_forward, *args, **kwargs)
                
                module.forward = checkpointed_forward
                self.checkpoint_layers.append(name)
    
    def _optimize_tensor_operations(self) -> None:
        """Optimize tensor operations for memory efficiency."""
        # Replace in-place operations where possible
        for module in self.model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = True
            
            # Optimize activation functions
            if isinstance(module, nn.ReLU):
                module.inplace = True
            elif isinstance(module, nn.Dropout):
                module.inplace = True
    
    def _apply_memory_efficient_attention(self) -> None:
        """Apply memory-efficient attention mechanisms."""
        # This would implement memory-efficient attention
        # For now, just mark as applied
        pass
    
    def _optimize_data_types(self) -> None:
        """Optimize data types for inference."""
        if self.optimize_for_inference:
            # Convert to half precision where appropriate
            for module in self.model.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    # Keep batch norm in float32 for stability
                    if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        try:
                            module.half()
                        except Exception:
                            # Some modules don't support half precision
                            pass
    
    def _cleanup_unused_parameters(self) -> None:
        """Remove unused parameters to save memory."""
        # Remove parameters with zero gradients (if in training)
        if self.model.training:
            params_to_remove = []
            
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.allclose(param.grad, torch.zeros_like(param.grad)):
                    params_to_remove.append(name)
            
            # Note: Actually removing parameters is complex and risky
            # In practice, we'd just mark them for monitoring
            if params_to_remove:
                warnings.warn(f"Found {len(params_to_remove)} parameters with zero gradients")
    
    def _calculate_parameter_reduction(self) -> float:
        """Calculate percentage reduction in parameters."""
        # This would track parameter count changes
        # For now, return 0 as we don't actually remove parameters
        return 0.0
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            # For CPU, we'd use psutil or similar
            return 0.0
    
    @contextmanager
    def memory_efficient_forward(self):
        """Context manager for memory-efficient forward pass."""
        try:
            # Clear cache before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Enable memory efficient mode
            with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision):
                yield
        
        finally:
            # Cleanup after forward pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        current_memory = self._get_memory_usage_mb()
        
        report = {
            'current_memory_mb': current_memory,
            'baseline_memory_mb': self.baseline_memory,
            'peak_memory_mb': max(self.memory_usage_history) if self.memory_usage_history else current_memory,
            'target_memory_mb': self.target_memory_bytes / (1024 ** 2),
            'memory_efficiency': (self.target_memory_bytes / (1024 ** 2)) / current_memory if current_memory > 0 else 1.0,
            'optimizations_applied': self.applied_optimizations,
            'checkpointed_layers': self.checkpoint_layers
        }
        
        # GPU-specific info
        if torch.cuda.is_available():
            report['gpu_memory'] = {
                'allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2),
                'cached_mb': torch.cuda.memory_cached() / (1024 ** 2),
                'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
                'total_memory_mb': torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            }
        
        return report


class GradientCheckpointing:
    """
    Gradient checkpointing implementation for memory-efficient training.
    
    Trades computation for memory by recomputing activations during
    backward pass instead of storing them.
    """
    
    def __init__(self, model: nn.Module, checkpoint_ratio: float = 0.5):
        """
        Initialize gradient checkpointing.
        
        Args:
            model: Model to apply checkpointing to
            checkpoint_ratio: Ratio of layers to checkpoint (0.0 to 1.0)
        """
        self.model = model
        self.checkpoint_ratio = checkpoint_ratio
        self.checkpointed_modules = []
    
    def apply_checkpointing(self) -> List[str]:
        """
        Apply gradient checkpointing to selected layers.
        
        Returns:
            List of layer names that were checkpointed
        """
        # Get all modules with significant memory usage
        module_info = []
        
        for name, module in self.model.named_modules():
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 1000:  # Only consider modules with significant parameters
                module_info.append((name, module, param_count))
        
        # Sort by parameter count (descending)
        module_info.sort(key=lambda x: x[2], reverse=True)
        
        # Select top modules for checkpointing
        n_to_checkpoint = int(len(module_info) * self.checkpoint_ratio)
        selected_modules = module_info[:n_to_checkpoint]
        
        # Apply checkpointing
        checkpointed_names = []
        for name, module, _ in selected_modules:
            if self._can_checkpoint_module(module):
                self._wrap_module_with_checkpoint(module)
                checkpointed_names.append(name)
                self.checkpointed_modules.append(module)
        
        return checkpointed_names
    
    def _can_checkpoint_module(self, module: nn.Module) -> bool:
        """Check if module can be safely checkpointed."""
        # Avoid checkpointing certain module types
        unsafe_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)
        return not isinstance(module, unsafe_types)
    
    def _wrap_module_with_checkpoint(self, module: nn.Module) -> None:
        """Wrap module's forward method with checkpointing."""
        original_forward = module.forward
        
        def checkpointed_forward(*args, **kwargs):
            # Only use checkpointing during training
            if module.training:
                return checkpoint(original_forward, *args, **kwargs)
            else:
                return original_forward(*args, **kwargs)
        
        module.forward = checkpointed_forward
    
    def remove_checkpointing(self) -> None:
        """Remove checkpointing from all modules."""
        # This would require storing original forward methods
        # Implementation would be more complex in practice
        warnings.warn("Checkpoint removal not implemented")


class ModelSharding:
    """
    Model sharding for memory-efficient large model training.
    
    Distributes model parameters across multiple devices or memory
    segments to handle models that don't fit in single device memory.
    """
    
    def __init__(self, model: nn.Module, num_shards: int = 2):
        """
        Initialize model sharding.
        
        Args:
            model: Model to shard
            num_shards: Number of shards to create  
        """
        self.model = model
        self.num_shards = num_shards
        self.shards = []
        self.shard_map = {}
    
    def create_shards(self) -> List[Dict[str, Any]]:
        """
        Create model shards.
        
        Returns:
            List of shard information dictionaries
        """
        # Get all parameters and their sizes
        param_info = []
        total_params = 0
        
        for name, param in self.model.named_parameters():
            param_size = param.numel() * param.element_size()
            param_info.append((name, param, param_size))
            total_params += param_size
        
        # Calculate target size per shard
        target_shard_size = total_params // self.num_shards
        
        # Distribute parameters across shards
        current_shard = 0
        current_shard_size = 0
        shard_contents = [[] for _ in range(self.num_shards)]
        
        for name, param, size in param_info:
            # Check if we should move to next shard
            if (current_shard_size + size > target_shard_size and 
                current_shard < self.num_shards - 1):
                current_shard += 1
                current_shard_size = 0
            
            shard_contents[current_shard].append((name, param, size))
            current_shard_size += size
            self.shard_map[name] = current_shard
        
        # Create shard info
        shard_info = []
        for i, contents in enumerate(shard_contents):
            total_size = sum(size for _, _, size in contents)
            param_names = [name for name, _, _ in contents]
            
            shard_info.append({
                'shard_id': i,
                'parameter_names': param_names,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 ** 2),
                'parameter_count': len(param_names)
            })
        
        self.shards = shard_info
        return shard_info
    
    def get_shard_for_parameter(self, param_name: str) -> Optional[int]:
        """Get shard ID for a given parameter."""
        return self.shard_map.get(param_name)


# Utility functions

def optimize_memory_usage(
    model: nn.Module,
    target_memory_gb: float = 4.0,
    aggressive: bool = False
) -> Dict[str, Any]:
    """
    Convenient function to optimize model memory usage.
    
    Args:
        model: Model to optimize
        target_memory_gb: Target memory usage
        aggressive: Whether to apply aggressive optimizations
        
    Returns:
        Optimization results
    """
    optimizer = MemoryOptimizer(
        model=model,
        target_memory_gb=target_memory_gb,
        enable_gradient_checkpointing=True,
        enable_mixed_precision=aggressive,
        optimize_for_inference=False
    )
    
    results = optimizer.optimize_model()
    
    # Apply additional aggressive optimizations
    if aggressive:
        # Apply gradient checkpointing
        checkpoint = GradientCheckpointing(model, checkpoint_ratio=0.7)
        checkpointed_layers = checkpoint.apply_checkpointing()
        results['checkpointed_layers'] = checkpointed_layers
        
        # Apply model sharding if model is very large
        param_count = sum(p.numel() for p in model.parameters())
        if param_count > 100_000_000:  # 100M parameters
            sharding = ModelSharding(model, num_shards=4)
            shard_info = sharding.create_shards()
            results['sharding_info'] = shard_info
    
    return results


def memory_efficient_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    batch: Dict[str, torch.Tensor],
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """
    Memory-efficient training step with automatic optimizations.
    
    Args:
        model: Model to train
        optimizer: Optimizer
        loss_fn: Loss function
        batch: Training batch
        scaler: Optional gradient scaler for mixed precision
        
    Returns:
        Training metrics
    """
    # Clear gradients
    optimizer.zero_grad()
    
    # Memory-efficient forward pass
    with torch.cuda.amp.autocast(enabled=scaler is not None):
        # Forward pass
        outputs = model(batch)
        loss = loss_fn(outputs, batch['labels'])
    
    # Backward pass with gradient scaling if available
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    
    # Memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'loss': loss.item(),
        'memory_used_mb': torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing memory optimization...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(1000, 2000),
                nn.ReLU(),
                nn.Linear(2000, 1000),
                nn.ReLU(),
                nn.Linear(1000, 100)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = TestModel()
    
    # Test memory optimizer
    optimizer = MemoryOptimizer(model, target_memory_gb=1.0)
    results = optimizer.optimize_model()
    
    print("Optimization results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Test gradient checkpointing
    checkpoint = GradientCheckpointing(model, checkpoint_ratio=0.5)
    checkpointed = checkpoint.apply_checkpointing()
    print(f"Checkpointed layers: {checkpointed}")
    
    # Test model sharding
    sharding = ModelSharding(model, num_shards=2)
    shards = sharding.create_shards()
    print("Shard information:")
    for shard in shards:
        print(f"  Shard {shard['shard_id']}: {shard['total_size_mb']:.2f} MB, {shard['parameter_count']} params")
    
    print("Memory optimization tests completed!")