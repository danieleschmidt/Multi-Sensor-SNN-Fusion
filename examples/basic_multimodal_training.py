#!/usr/bin/env python3
"""
Basic Multi-Modal SNN Training Example

This example demonstrates end-to-end training of a multi-modal spiking neural
network using synthetic data. It showcases the complete pipeline from data
loading to model training and evaluation.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from snn_fusion.models.multimodal_lsm import MultiModalLSM
from snn_fusion.algorithms.encoding import MultiModalEncoder
from snn_fusion.training.trainer import MultiModalTrainer
from snn_fusion.datasets.synthetic import SyntheticMultiModalDataset
from snn_fusion.datasets.loaders import MultiModalDataLoader
from snn_fusion.utils.config import SNNFusionConfig, create_debug_config
from snn_fusion.utils.validation import validate_config
from snn_fusion.utils.logging import setup_logging


def create_synthetic_data(config: SNNFusionConfig, num_samples: int = 1000):
    """Create synthetic multi-modal dataset for testing."""
    print("Creating synthetic multi-modal dataset...")
    
    # Initialize multi-modal encoder
    encoder = MultiModalEncoder(
        audio_config={'n_neurons': 128, 'sample_rate': 16000},
        visual_config={'width': 64, 'height': 64, 'threshold': 0.1},
        tactile_config={'n_sensors': 16},
        duration=config.dataset.time_window_ms,
        dt=1.0
    )
    
    # Generate synthetic data samples
    samples = []
    for i in range(num_samples):
        # Generate random multi-modal data
        audio_data = np.random.randn(int(16000 * config.dataset.time_window_ms / 1000))
        visual_data = np.random.rand(10, 64, 64)  # 10 frames
        tactile_data = np.random.rand(int(config.dataset.time_window_ms), 16)
        
        # Encode to spikes
        spike_data = encoder.encode(
            audio_data=audio_data,
            visual_data=visual_data,
            tactile_data=tactile_data
        )
        
        # Create tensor format expected by model
        audio_spikes = torch.zeros(int(config.dataset.time_window_ms), 128)
        visual_spikes = torch.zeros(int(config.dataset.time_window_ms), 64*64*2)
        tactile_spikes = torch.zeros(int(config.dataset.time_window_ms), 32)
        
        # Fill in actual spike data
        for spike_time, neuron_id in zip(spike_data.spike_times, spike_data.neuron_ids):
            timestep = int(spike_time)
            if timestep < config.dataset.time_window_ms:
                if neuron_id < 128:  # Audio neurons
                    audio_spikes[timestep, neuron_id] = 1
                elif neuron_id < 128 + 64*64*2:  # Visual neurons
                    visual_spikes[timestep, neuron_id - 128] = 1
                else:  # Tactile neurons
                    tactile_idx = neuron_id - 128 - 64*64*2
                    if tactile_idx < 32:
                        tactile_spikes[timestep, tactile_idx] = 1
        
        sample = {
            'audio': audio_spikes,
            'events': visual_spikes,
            'imu': tactile_spikes,
            'label': torch.tensor(i % config.model.num_readouts, dtype=torch.long)
        }
        samples.append(sample)
    
    print(f"Generated {len(samples)} synthetic samples")
    return samples


def create_model(config: SNNFusionConfig):
    """Create multi-modal LSM model."""
    print("Creating Multi-Modal LSM model...")
    
    model = MultiModalLSM(
        # Reservoir parameters
        n_neurons=config.model.n_neurons,
        connectivity=config.model.connectivity,
        spectral_radius=config.model.spectral_radius,
        
        # Input dimensions
        audio_channels=128,
        event_size=(64, 64),
        imu_dims=32,
        
        # Neuron parameters
        tau_mem=config.model.tau_mem,
        tau_adapt=config.model.tau_adapt,
        
        # Output
        n_outputs=config.model.num_readouts,
        fusion_type='attention'
    )
    
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def train_model(model, train_loader, val_loader, config: SNNFusionConfig):
    """Train the multi-modal SNN model."""
    print("Starting model training...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = MultiModalTrainer(
        model=model,
        device=device,
        learning_rule='bptt',
        optimizer='adam',
        learning_rate=config.training.learning_rate,
        temporal_window=int(config.dataset.time_window_ms),
        checkpoint_dir=f"{config.output_dir}/{config.experiment_name}/checkpoints"
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.epochs,
        save_best=True,
        early_stopping=config.training.early_stopping_patience
    )
    
    print("Training completed!")
    return trainer, history


def evaluate_model(trainer, test_loader):
    """Evaluate the trained model."""
    print("Evaluating model...")
    
    metrics = trainer.evaluate(test_loader, return_detailed_metrics=True)
    
    print("Evaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return metrics


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("SNN-Fusion Basic Multi-Modal Training Example")
    print("=" * 60)
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        config = create_debug_config()
        config.experiment_name = "basic_multimodal_example"
        config.dataset.batch_size = 8
        config.training.epochs = 10
        config.training.learning_rate = 1e-3
        
        # Validate configuration
        print("Validating configuration...")
        validate_config(config)
        print("✓ Configuration valid")
        
        # Create output directory
        output_dir = Path(config.output_dir) / config.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic dataset
        train_data = create_synthetic_data(config, num_samples=800)
        val_data = create_synthetic_data(config, num_samples=100)
        test_data = create_synthetic_data(config, num_samples=100)
        
        # Create data loaders
        train_loader = MultiModalDataLoader(
            train_data, 
            batch_size=config.dataset.batch_size, 
            shuffle=True
        )
        val_loader = MultiModalDataLoader(
            val_data, 
            batch_size=config.dataset.batch_size, 
            shuffle=False
        )
        test_loader = MultiModalDataLoader(
            test_data, 
            batch_size=config.dataset.batch_size, 
            shuffle=False
        )
        
        print(f"Created data loaders:")
        print(f"  Train: {len(train_loader)} batches")
        print(f"  Val:   {len(val_loader)} batches")
        print(f"  Test:  {len(test_loader)} batches")
        
        # Create model
        model = create_model(config)
        
        # Train model
        trainer, history = train_model(model, train_loader, val_loader, config)
        
        # Evaluate model
        test_metrics = evaluate_model(trainer, test_loader)
        
        # Save results
        results = {
            'config': config.__dict__,
            'training_history': history,
            'test_metrics': test_metrics,
        }
        
        results_path = output_dir / 'results.pt'
        torch.save(results, results_path)
        print(f"Saved results to: {results_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Model: {config.model.model_type}")
        print(f"Neurons: {config.model.n_neurons}")
        print(f"Epochs: {config.training.epochs}")
        print(f"Final Train Accuracy: {history['accuracy'][-1]:.4f}")
        if 'val_accuracy' in history:
            print(f"Final Val Accuracy: {history['val_accuracy'][-1]:.4f}")
        print(f"Test Accuracy: {test_metrics['val_accuracy']:.4f}")
        print(f"Average Spike Rate: {test_metrics.get('spike_rate', 'N/A')}")
        
        print("\n✓ Training pipeline completed successfully!")
        
        return trainer, results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    trainer, results = main()