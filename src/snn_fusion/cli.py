"""
Command Line Interface for SNN-Fusion

Provides comprehensive CLI tools for training, evaluation, and deployment
of multi-modal spiking neural networks.
"""

import click
import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys

from .models import MultiModalLSM
from .training import SNNTrainer, MultiModalTrainer
from .datasets import create_dataloaders
from .utils.config import load_config, validate_config
from .utils.logging import setup_logging


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """
    SNN-Fusion: Multi-Modal Spiking Neural Network Framework
    
    Train and deploy neuromorphic sensor fusion systems for real-time applications.
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(log_level)
    
    # Load configuration
    if config:
        ctx.ensure_object(dict)
        ctx.obj['config'] = load_config(config)
    else:
        ctx.obj = {}


@cli.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True,
              help='Path to training data directory')
@click.option('--model-config', type=click.Path(exists=True),
              help='Model configuration file')
@click.option('--output-dir', type=click.Path(), default='./outputs',
              help='Output directory for models and logs')
@click.option('--epochs', type=int, default=100,
              help='Number of training epochs')
@click.option('--batch-size', type=int, default=32,
              help='Training batch size')
@click.option('--learning-rate', type=float, default=1e-3,
              help='Initial learning rate')
@click.option('--device', type=str, default='auto',
              help='Device for training (cpu/cuda/auto)')
@click.option('--resume', type=click.Path(exists=True),
              help='Resume from checkpoint')
@click.pass_context
def train(
    ctx: click.Context,
    data_dir: str,
    model_config: Optional[str],
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    resume: Optional[str],
) -> None:
    """Train a multi-modal SNN fusion model."""
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    click.echo(f"Training on device: {device}")
    
    # Load model configuration
    if model_config:
        with open(model_config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'model': {
                'modality_configs': {
                    'audio': {'n_inputs': 128, 'n_reservoir': 300},
                    'vision': {'n_inputs': 1024, 'n_reservoir': 400},
                    'tactile': {'n_inputs': 6, 'n_reservoir': 200}
                },
                'n_outputs': 10,
                'fusion_type': 'attention',
            },
            'training': {
                'learning_rule': 'bptt',
                'temporal_window': 100,
            }
        }
    
    # Validate configuration
    validate_config(config)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = MultiModalLSM(
        modality_configs=config['model']['modality_configs'],
        n_outputs=config['model']['n_outputs'],
        fusion_type=config['model']['fusion_type'],
        device=device,
    )
    
    click.echo(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        modalities=list(config['model']['modality_configs'].keys()),
    )
    
    click.echo(f"Data loaded: {len(train_loader)} train, {len(val_loader)} val batches")
    
    # Initialize trainer
    trainer = MultiModalTrainer(
        model=model,
        device=device,
        learning_rule=config['training']['learning_rule'],
        learning_rate=learning_rate,
        temporal_window=config['training']['temporal_window'],
        checkpoint_dir=str(output_path / 'checkpoints'),
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume:
        start_epoch = trainer.load_checkpoint(resume)
        click.echo(f"Resumed training from epoch {start_epoch}")
    
    # Train model
    click.echo("Starting training...")
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            save_best=True,
            early_stopping=20,
        )
        
        # Save training history
        history_path = output_path / 'training_history.yaml'
        with open(history_path, 'w') as f:
            yaml.dump(history, f)
        
        click.echo(f"Training completed. Best validation accuracy: {max(history.get('val_accuracy', [0])):.4f}")
        
    except KeyboardInterrupt:
        click.echo("Training interrupted by user")
        trainer.save_checkpoint(epochs, is_best=False)
    except Exception as e:
        click.echo(f"Training failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to trained model checkpoint')
@click.option('--data-dir', type=click.Path(exists=True), required=True,
              help='Path to test data directory')
@click.option('--output-file', type=click.Path(), default='evaluation_results.yaml',
              help='Output file for evaluation results')
@click.option('--batch-size', type=int, default=32,
              help='Evaluation batch size')
@click.option('--device', type=str, default='auto',
              help='Device for evaluation')
@click.pass_context
def evaluate(
    ctx: click.Context,
    model_path: str,
    data_dir: str,
    output_file: str,
    batch_size: int,
    device: str,
) -> None:
    """Evaluate a trained model on test data."""
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    click.echo(f"Evaluating on device: {device}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct model (simplified - in practice, would load from config)
    model_config = {
        'modality_configs': {
            'audio': {'n_inputs': 128, 'n_reservoir': 300},
            'vision': {'n_inputs': 1024, 'n_reservoir': 400},
            'tactile': {'n_inputs': 6, 'n_reservoir': 200}
        },
        'n_outputs': 10,
        'fusion_type': 'attention',
    }
    
    model = MultiModalLSM(
        modality_configs=model_config['modality_configs'],
        n_outputs=model_config['n_outputs'],
        fusion_type=model_config['fusion_type'],
        device=device,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    click.echo("Model loaded successfully")
    
    # Load test data
    _, _, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        modalities=list(model_config['modality_configs'].keys()),
    )
    
    # Initialize trainer for evaluation
    trainer = MultiModalTrainer(model=model, device=device)
    
    # Run evaluation
    click.echo("Running evaluation...")
    results = trainer.evaluate(test_loader, return_detailed_metrics=True)
    
    # Save results
    with open(output_file, 'w') as f:
        yaml.dump(results, f)
    
    # Display summary
    click.echo(f"Evaluation Results:")
    click.echo(f"  Test Accuracy: {results['val_accuracy']:.4f}")
    click.echo(f"  Test Loss: {results['val_loss']:.4f}")
    
    if 'spike_firing_rates' in results:
        click.echo(f"  Average Firing Rate: {results['spike_firing_rates']:.4f} Hz")
    
    click.echo(f"Detailed results saved to: {output_file}")


@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to trained model checkpoint')
@click.option('--hardware', type=click.Choice(['loihi2', 'akida', 'spinnaker']),
              required=True, help='Target neuromorphic hardware')
@click.option('--output-dir', type=click.Path(), default='./deployment',
              help='Output directory for deployment files')
@click.option('--optimize', is_flag=True,
              help='Apply hardware-specific optimizations')
@click.pass_context
def deploy(
    ctx: click.Context,
    model_path: str,
    hardware: str,
    output_dir: str,
    optimize: bool,
) -> None:
    """Deploy model to neuromorphic hardware."""
    
    click.echo(f"Deploying to {hardware} hardware...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        if hardware == 'loihi2':
            from .hardware.loihi_deployer import LoihiDeployer
            deployer = LoihiDeployer()
        elif hardware == 'akida':
            from .hardware.akida_deployer import AkidaDeployer
            deployer = AkidaDeployer()
        elif hardware == 'spinnaker':
            from .hardware.spinnaker_deployer import SpiNNakerDeployer
            deployer = SpiNNakerDeployer()
        
        # Load and convert model
        deployment_files = deployer.deploy_model(
            model_path=model_path,
            output_dir=str(output_path),
            optimize=optimize,
        )
        
        click.echo(f"Deployment successful!")
        click.echo(f"Generated files:")
        for file_path in deployment_files:
            click.echo(f"  - {file_path}")
            
    except ImportError as e:
        click.echo(f"Hardware support not available: {e}")
        click.echo(f"Install hardware-specific dependencies for {hardware}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Deployment failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--config-template', type=click.Choice(['basic', 'advanced', 'research']),
              default='basic', help='Configuration template type')
@click.option('--output-file', type=click.Path(), default='snn_config.yaml',
              help='Output configuration file')
def init_config(config_template: str, output_file: str) -> None:
    """Generate a configuration file template."""
    
    templates = {
        'basic': {
            'model': {
                'modality_configs': {
                    'audio': {'n_inputs': 64, 'n_reservoir': 200},
                    'vision': {'n_inputs': 512, 'n_reservoir': 300},
                },
                'n_outputs': 10,
                'fusion_type': 'concatenation',
            },
            'training': {
                'learning_rule': 'bptt',
                'temporal_window': 50,
                'batch_size': 16,
                'learning_rate': 1e-3,
                'epochs': 50,
            },
            'data': {
                'modalities': ['audio', 'vision'],
                'sample_rate': 1000,
            }
        },
        'advanced': {
            'model': {
                'modality_configs': {
                    'audio': {'n_inputs': 128, 'n_reservoir': 400, 'connectivity': 0.15},
                    'vision': {'n_inputs': 1024, 'n_reservoir': 500, 'connectivity': 0.12},
                    'tactile': {'n_inputs': 6, 'n_reservoir': 150, 'connectivity': 0.20},
                },
                'n_outputs': 50,
                'fusion_type': 'attention',
                'fusion_config': {'hidden_dim': 256, 'num_heads': 8},
                'global_reservoir_size': 600,
            },
            'training': {
                'learning_rule': 'bptt',
                'temporal_window': 100,
                'batch_size': 32,
                'learning_rate': 1e-3,
                'optimizer': 'adamw',
                'epochs': 200,
                'early_stopping': 20,
            },
            'data': {
                'modalities': ['audio', 'vision', 'tactile'],
                'sample_rate': 2000,
                'augmentation': True,
            }
        },
        'research': {
            'model': {
                'modality_configs': {
                    'audio': {
                        'n_inputs': 128, 'n_reservoir': 800, 'connectivity': 0.1,
                        'tau_mem': 20.0, 'tau_adapt': 200.0, 'spectral_radius': 0.95
                    },
                    'vision': {
                        'n_inputs': 2048, 'n_reservoir': 1200, 'connectivity': 0.08,
                        'tau_mem': 15.0, 'tau_adapt': 150.0, 'spectral_radius': 0.9
                    },
                    'tactile': {
                        'n_inputs': 12, 'n_reservoir': 300, 'connectivity': 0.15,
                        'tau_mem': 25.0, 'tau_adapt': 100.0, 'spectral_radius': 0.85
                    },
                },
                'n_outputs': 100,
                'fusion_type': 'attention',
                'fusion_config': {'hidden_dim': 512, 'num_heads': 16, 'dropout': 0.1},
                'global_reservoir_size': 1000,
                'enable_cross_modal_plasticity': True,
            },
            'training': {
                'learning_rule': 'stdp',
                'temporal_window': 200,
                'batch_size': 64,
                'learning_rate': 5e-4,
                'optimizer': 'adamw',
                'epochs': 500,
                'early_stopping': 50,
                'gradient_clipping': 1.0,
                'regularization_weight': 1e-5,
            },
            'data': {
                'modalities': ['audio', 'vision', 'tactile'],
                'sample_rate': 5000,
                'augmentation': True,
                'noise_injection': 0.1,
            },
            'hardware': {
                'target_platform': 'loihi2',
                'quantization': True,
                'pruning_ratio': 0.3,
            }
        }
    }
    
    config = templates[config_template]
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    click.echo(f"Configuration template '{config_template}' saved to {output_file}")
    click.echo("Customize the configuration file before training.")


@cli.command()
@click.option('--port', type=int, default=8080, help='Dashboard port')
@click.option('--host', type=str, default='localhost', help='Dashboard host')
@click.option('--data-dir', type=click.Path(exists=True),
              help='Data directory for monitoring')
def dashboard(port: int, host: str, data_dir: Optional[str]) -> None:
    """Launch real-time monitoring dashboard."""
    
    try:
        from .visualization.dashboard import launch_dashboard
        
        click.echo(f"Launching dashboard at http://{host}:{port}")
        launch_dashboard(host=host, port=port, data_dir=data_dir)
        
    except ImportError as e:
        click.echo(f"Dashboard dependencies not available: {e}")
        click.echo("Install with: pip install snn-fusion[viz]")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main()