"""
SNN Training Infrastructure

Implements specialized training loops and optimization strategies
for spiking neural networks with temporal credit assignment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
from pathlib import Path

from .losses import TemporalLoss, SpikeLoss
from .plasticity import STDPLearner
from ..utils.metrics import SpikeMetrics, FusionMetrics


class SNNTrainer:
    """
    Trainer for spiking neural networks with temporal processing.
    
    Handles backpropagation through time, spike-based regularization,
    and neuromorphic-specific optimization techniques.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu'),
        learning_rule: str = "bptt",
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        temporal_window: int = 100,
        surrogate_gradient: str = "fast_sigmoid",
        regularization_weight: float = 1e-4,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize SNN trainer.
        
        Args:
            model: Spiking neural network model
            device: Computation device
            learning_rule: Learning algorithm (bptt, stdp, etc.)
            optimizer: Optimizer type
            learning_rate: Base learning rate
            temporal_window: BPTT temporal window
            surrogate_gradient: Surrogate gradient function
            regularization_weight: L2 regularization weight
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rule = learning_rule
        self.temporal_window = temporal_window
        self.surrogate_gradient = surrogate_gradient
        self.regularization_weight = regularization_weight
        
        # Setup optimizer
        self.optimizer = self._create_optimizer(optimizer, learning_rate)
        
        # Setup loss functions
        self.loss_fn = TemporalLoss(
            loss_type="cross_entropy",
            temporal_weighting=True,
            device=device,
        )
        self.spike_loss = SpikeLoss(
            regularization_type="l1",
            target_firing_rate=0.1,
            device=device,
        )
        
        # STDP learning (if enabled)
        if learning_rule == "stdp":
            self.stdp_learner = STDPLearner(
                tau_pre=20.0,
                tau_post=20.0,
                A_plus=0.01,
                A_minus=0.012,
            )
        
        # Metrics tracking
        self.spike_metrics = SpikeMetrics()
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'spike_rate': [],
            'learning_rate': [],
        }
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
    def _create_optimizer(self, optimizer_type: str, learning_rate: float) -> optim.Optimizer:
        """Create optimizer with SNN-specific configurations."""
        if optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.regularization_weight,
            )
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.regularization_weight,
            )
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=self.regularization_weight,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        log_interval: int = 100,
    ) -> Dict[str, float]:
        """
        Train model for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            log_interval: Logging interval
            
        Returns:
            metrics: Training metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_spike_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        total_spikes = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Reset model states for new sequence
            if hasattr(self.model, 'reset_state'):
                self.model.reset_state()
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.learning_rule == "bptt":
                outputs, states = self._forward_bptt(data)
            elif self.learning_rule == "stdp":
                outputs, states = self._forward_stdp(data)
            else:
                outputs, states = self.model(data, return_states=True)
            
            # Compute losses
            main_loss = self.loss_fn(outputs, target)
            
            # Spike regularization
            if states and 'spike_history' in states:
                spike_reg_loss = self.spike_loss(states['spike_history'])
            else:
                spike_reg_loss = torch.tensor(0.0, device=self.device)
            
            total_loss_batch = main_loss + 0.1 * spike_reg_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += main_loss.item()
            total_spike_loss += spike_reg_loss.item()
            
            # Compute accuracy
            with torch.no_grad():
                pred = outputs.argmax(dim=1)
                correct_predictions += pred.eq(target).sum().item()
                total_samples += target.size(0)
                
                # Track spike statistics
                if states and 'spike_history' in states:
                    total_spikes += states['spike_history'].sum().item()
            
            # Logging
            if batch_idx % log_interval == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: '
                    f'Loss: {main_loss.item():.6f}, '
                    f'Spike Loss: {spike_reg_loss.item():.6f}, '
                    f'Accuracy: {correct_predictions/total_samples:.4f}'
                )
        
        # Epoch metrics
        epoch_metrics = {
            'loss': total_loss / len(train_loader),
            'spike_loss': total_spike_loss / len(train_loader),
            'accuracy': correct_predictions / total_samples,
            'avg_spike_rate': total_spikes / (total_samples * self.temporal_window),
            'epoch_time': time.time() - start_time,
        }
        
        # Update training history
        for key, value in epoch_metrics.items():
            if key in self.training_history:
                self.training_history[key].append(value)
        
        return epoch_metrics
    
    def _forward_bptt(self, data: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with backpropagation through time."""
        # Split temporal sequence into windows for BPTT
        batch_size, total_time, *input_dims = data.shape
        
        if total_time <= self.temporal_window:
            # Process entire sequence
            return self.model(data, return_states=True)
        
        # Process in temporal windows
        outputs_list = []
        all_states = {'spike_history': []}
        
        for t_start in range(0, total_time, self.temporal_window):
            t_end = min(t_start + self.temporal_window, total_time)
            window_data = data[:, t_start:t_end]
            
            window_outputs, window_states = self.model(window_data, return_states=True)
            outputs_list.append(window_outputs)
            
            if window_states and 'spike_history' in window_states:
                all_states['spike_history'].append(window_states['spike_history'])
        
        # Combine outputs (use final window output)
        outputs = outputs_list[-1]
        
        # Combine spike history
        if all_states['spike_history']:
            all_states['spike_history'] = torch.cat(all_states['spike_history'], dim=1)
        
        return outputs, all_states
    
    def _forward_stdp(self, data: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with STDP learning."""
        # Forward pass
        outputs, states = self.model(data, return_states=True)
        
        # Apply STDP updates
        if hasattr(self.model, 'W_reservoir') and states and 'spike_history' in states:
            spike_history = states['spike_history']
            batch_size, time_steps, n_neurons = spike_history.shape
            
            # Apply STDP for each time step
            for t in range(1, time_steps):
                pre_spikes = spike_history[:, t-1, :]  # Pre-synaptic spikes
                post_spikes = spike_history[:, t, :]   # Post-synaptic spikes
                
                # Update weights using STDP
                weight_update = self.stdp_learner.compute_weight_update(
                    pre_spikes, post_spikes
                )
                
                # Apply update to reservoir weights
                with torch.no_grad():
                    self.model.W_reservoir += 0.001 * weight_update.mean(dim=0)
        
        return outputs, states
    
    def evaluate(
        self,
        val_loader: DataLoader,
        return_detailed_metrics: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            return_detailed_metrics: Whether to return detailed spike metrics
            
        Returns:
            metrics: Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_spike_trains = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Reset model states
                if hasattr(self.model, 'reset_state'):
                    self.model.reset_state()
                
                # Forward pass
                outputs, states = self.model(data, return_states=True)
                
                # Compute loss
                loss = self.loss_fn(outputs, target)
                total_loss += loss.item()
                
                # Track predictions
                pred = outputs.argmax(dim=1)
                correct_predictions += pred.eq(target).sum().item()
                total_samples += target.size(0)
                
                # Store for detailed analysis
                if return_detailed_metrics:
                    all_predictions.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                    
                    if states and 'spike_history' in states:
                        all_spike_trains.append(states['spike_history'].cpu())
        
        # Basic metrics
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': correct_predictions / total_samples,
        }
        
        # Detailed spike analysis
        if return_detailed_metrics and all_spike_trains:
            spike_history = torch.cat(all_spike_trains, dim=0)
            spike_stats = self.spike_metrics.compute_spike_statistics(
                spike_history, time_window=self.temporal_window
            )
            
            metrics.update({
                f'spike_{key}': value.mean().item() if isinstance(value, torch.Tensor) else value
                for key, value in spike_stats.items()
            })
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_best: bool = True,
        early_stopping: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            save_best: Whether to save best model
            early_stopping: Early stopping patience
            
        Returns:
            history: Training history
        """
        best_val_acc = 0.0
        patience_counter = 0
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Acc: {val_metrics['val_accuracy']:.4f}"
                )
                
                # Update history
                self.training_history['val_loss'] = self.training_history.get('val_loss', [])
                self.training_history['val_accuracy'] = self.training_history.get('val_accuracy', [])
                self.training_history['val_loss'].append(val_metrics['val_loss'])
                self.training_history['val_accuracy'].append(val_metrics['val_accuracy'])
                
                # Save best model
                if save_best and val_metrics['val_accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['val_accuracy']
                    self.save_checkpoint(epoch, is_best=True)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if early_stopping and patience_counter >= early_stopping:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}"
                )
            
            # Save regular checkpoint
            if self.checkpoint_dir and epoch % 10 == 0:
                self.save_checkpoint(epoch)
        
        return self.training_history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        epoch = checkpoint['epoch']
        self.logger.info(f"Loaded checkpoint from epoch {epoch}")
        
        return epoch


class MultiModalTrainer(SNNTrainer):
    """
    Specialized trainer for multi-modal spiking neural networks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        modality_weights: Optional[Dict[str, float]] = None,
        cross_modal_loss_weight: float = 0.1,
        **kwargs
    ):
        """
        Initialize multi-modal trainer.
        
        Args:
            model: Multi-modal SNN model
            modality_weights: Weights for modality-specific losses
            cross_modal_loss_weight: Weight for cross-modal consistency loss
            **kwargs: Arguments for base trainer
        """
        super().__init__(model, **kwargs)
        
        self.modality_weights = modality_weights or {}
        self.cross_modal_loss_weight = cross_modal_loss_weight
        
        # Multi-modal metrics
        self.fusion_metrics = FusionMetrics()
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        log_interval: int = 100,
    ) -> Dict[str, float]:
        """Train multi-modal model for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_cross_modal_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        modality_losses = {}
        
        start_time = time.time()
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Unpack multi-modal batch
            if isinstance(batch_data, dict):
                inputs = {k: v.to(self.device) for k, v in batch_data['inputs'].items()}
                target = batch_data['target'].to(self.device)
            else:
                inputs, target = batch_data
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                target = target.to(self.device)
            
            # Reset model states
            if hasattr(self.model, 'reset_all_states'):
                self.model.reset_all_states()
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, states = self.model(inputs, return_states=True)
            
            # Main classification loss
            main_loss = self.loss_fn(outputs, target)
            
            # Modality-specific losses
            modality_loss_total = 0.0
            if states and 'modality_states' in states:
                for modality, mod_states in states['modality_states'].items():
                    if modality in self.modality_weights:
                        # Single modality prediction
                        mod_output = self.model.forward_single_modality(
                            modality, inputs[modality]
                        )
                        mod_loss = self.loss_fn(mod_output, target)
                        modality_loss_total += self.modality_weights[modality] * mod_loss
                        
                        # Track modality-specific losses
                        if modality not in modality_losses:
                            modality_losses[modality] = 0.0
                        modality_losses[modality] += mod_loss.item()
            
            # Cross-modal consistency loss
            cross_modal_loss = torch.tensor(0.0, device=self.device)
            if states and 'fusion_attention' in states:
                attention_weights = states['fusion_attention']
                if attention_weights is not None:
                    # Encourage diverse attention (entropy regularization)
                    attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum()
                    cross_modal_loss = -0.1 * attention_entropy  # Negative for maximization
            
            # Total loss
            total_loss_batch = (
                main_loss + 
                modality_loss_total + 
                self.cross_modal_loss_weight * cross_modal_loss
            )
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += main_loss.item()
            total_cross_modal_loss += cross_modal_loss.item()
            
            with torch.no_grad():
                pred = outputs.argmax(dim=1)
                correct_predictions += pred.eq(target).sum().item()
                total_samples += target.size(0)
            
            # Logging
            if batch_idx % log_interval == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}: '
                    f'Loss: {main_loss.item():.6f}, '
                    f'Cross-modal: {cross_modal_loss.item():.6f}, '
                    f'Accuracy: {correct_predictions/total_samples:.4f}'
                )
        
        # Epoch metrics
        epoch_metrics = {
            'loss': total_loss / len(train_loader),
            'cross_modal_loss': total_cross_modal_loss / len(train_loader),
            'accuracy': correct_predictions / total_samples,
            'epoch_time': time.time() - start_time,
        }
        
        # Add modality-specific metrics
        for modality, loss_sum in modality_losses.items():
            epoch_metrics[f'{modality}_loss'] = loss_sum / len(train_loader)
        
        return epoch_metrics