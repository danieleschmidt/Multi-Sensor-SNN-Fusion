"""
Configuration Management for SNN-Fusion

This module provides utilities for loading, validating, and managing
configuration files and parameters across the SNN-Fusion framework.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from dataclasses import dataclass, asdict
from .validation import validate_configuration, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset parameters."""
    root_dir: str
    split: str = 'train'
    modalities: List[str] = None
    sequence_length: int = 100
    batch_size: int = 32
    spike_encoding: bool = True
    num_workers: int = 4
    shuffle: bool = True
    time_window_ms: float = 100.0
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ['audio', 'events', 'imu']


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    model_type: str = 'MultiModalLSM'
    input_shapes: Dict[str, tuple] = None
    hidden_dim: int = 512
    n_neurons: int = 1000
    connectivity: float = 0.1
    spectral_radius: float = 0.9
    tau_mem: float = 20.0
    tau_adapt: float = 100.0
    num_readouts: int = 10
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.input_shapes is None:
            self.input_shapes = {
                'audio': (2, 64),
                'events': (128, 128, 2),
                'imu': (6,)
            }


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int = 100
    learning_rate: float = 1e-3
    optimizer: str = 'Adam'
    scheduler: str = 'StepLR'
    scheduler_params: Dict[str, Any] = None
    loss_function: str = 'CrossEntropy'
    device: str = 'auto'
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    save_interval: int = 10
    validate_interval: int = 5
    early_stopping_patience: int = 20
    
    def __post_init__(self):
        if self.scheduler_params is None:
            self.scheduler_params = {'step_size': 30, 'gamma': 0.1}


@dataclass
class STDPConfig:
    """Configuration for STDP plasticity."""
    enabled: bool = True
    tau_pre: float = 20.0
    tau_post: float = 20.0
    A_plus: float = 0.01
    A_minus: float = 0.012
    weight_bounds: tuple = (0.0, 1.0)
    multiplicative: bool = False
    reward_modulated: bool = False
    tau_reward: float = 1000.0


@dataclass
class SNNFusionConfig:
    """Complete configuration for SNN-Fusion system."""
    dataset: DatasetConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    stdp: STDPConfig = None
    experiment_name: str = 'snn_fusion_experiment'
    output_dir: str = './experiments'
    random_seed: int = 42
    log_level: str = 'INFO'
    
    def __post_init__(self):
        if self.dataset is None:
            self.dataset = DatasetConfig(root_dir='./data')
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.stdp is None:
            self.stdp = STDPConfig()


def load_config(config_path: Union[str, Path]) -> SNNFusionConfig:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        logger.info(f"Loaded configuration from {config_path}")
        
        return dict_to_config(config_dict)
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def save_config(config: SNNFusionConfig, config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration object to save
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = asdict(config)
    
    try:
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        logger.info(f"Saved configuration to {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise


def dict_to_config(config_dict: Dict[str, Any]) -> SNNFusionConfig:
    """
    Convert dictionary to configuration object.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configuration object
    """
    # Extract sub-configurations
    dataset_dict = config_dict.get('dataset', {})
    model_dict = config_dict.get('model', {})
    training_dict = config_dict.get('training', {})
    stdp_dict = config_dict.get('stdp', {})
    
    # Create sub-configurations
    dataset_config = DatasetConfig(**dataset_dict) if dataset_dict else DatasetConfig(root_dir='./data')
    model_config = ModelConfig(**model_dict) if model_dict else ModelConfig()
    training_config = TrainingConfig(**training_dict) if training_dict else TrainingConfig()
    stdp_config = STDPConfig(**stdp_dict) if stdp_dict else STDPConfig()
    
    # Create main configuration
    main_config = {
        'dataset': dataset_config,
        'model': model_config,
        'training': training_config,
        'stdp': stdp_config,
        'experiment_name': config_dict.get('experiment_name', 'snn_fusion_experiment'),
        'output_dir': config_dict.get('output_dir', './experiments'),
        'random_seed': config_dict.get('random_seed', 42),
        'log_level': config_dict.get('log_level', 'INFO')
    }
    
    return SNNFusionConfig(**main_config)


def create_default_config(
    output_path: Optional[Union[str, Path]] = None,
    experiment_name: str = 'snn_fusion_default'
) -> SNNFusionConfig:
    """
    Create a default configuration.
    
    Args:
        output_path: Optional path to save the default config
        experiment_name: Name for the experiment
        
    Returns:
        Default configuration
    """
    config = SNNFusionConfig(experiment_name=experiment_name)
    
    if output_path is not None:
        save_config(config, output_path)
        logger.info(f"Created default configuration: {output_path}")
    
    return config


def validate_config(config: SNNFusionConfig) -> bool:
    """
    Validate configuration object.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Convert to dict for validation
        config_dict = asdict(config)
        
        # Define validation schemas
        dataset_schema = {
            'root_dir': {'type': str, 'required': True},
            'split': {'type': str, 'choices': ['train', 'val', 'test'], 'required': True},
            'modalities': {'type': list, 'required': True},
            'sequence_length': {'type': int, 'min': 1, 'max': 10000, 'required': True},
            'batch_size': {'type': int, 'min': 1, 'max': 1024, 'required': True},
        }
        
        model_schema = {
            'model_type': {'type': str, 'required': True},
            'hidden_dim': {'type': int, 'min': 1, 'max': 10000, 'required': True},
            'n_neurons': {'type': int, 'min': 1, 'max': 100000, 'required': True},
            'connectivity': {'type': float, 'min': 0.0, 'max': 1.0, 'required': True},
            'dropout': {'type': float, 'min': 0.0, 'max': 1.0, 'required': False},
        }
        
        training_schema = {
            'epochs': {'type': int, 'min': 1, 'max': 10000, 'required': True},
            'learning_rate': {'type': float, 'min': 1e-6, 'max': 1.0, 'required': True},
            'optimizer': {'type': str, 'choices': ['Adam', 'SGD', 'AdamW'], 'required': True},
            'device': {'type': str, 'choices': ['cpu', 'cuda', 'auto'], 'required': True},
        }
        
        # Validate each section
        validate_configuration(config_dict['dataset'], dataset_schema, 'dataset config')
        validate_configuration(config_dict['model'], model_schema, 'model config')  
        validate_configuration(config_dict['training'], training_schema, 'training config')
        
        # Additional validation
        if not os.path.exists(config.dataset.root_dir):
            logger.warning(f"Dataset root directory does not exist: {config.dataset.root_dir}")
        
        logger.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ValidationError(f"Invalid configuration: {e}")


def merge_configs(base_config: SNNFusionConfig, override_config: Dict[str, Any]) -> SNNFusionConfig:
    """
    Merge base configuration with override values.
    
    Args:
        base_config: Base configuration object
        override_config: Dictionary of values to override
        
    Returns:
        Merged configuration
    """
    base_dict = asdict(base_config)
    
    # Deep merge override values
    def deep_merge(base: Dict, override: Dict) -> Dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged_dict = deep_merge(base_dict, override_config)
    return dict_to_config(merged_dict)


def get_config_from_env(env_prefix: str = 'SNN_FUSION') -> Dict[str, Any]:
    """
    Get configuration values from environment variables.
    
    Args:
        env_prefix: Prefix for environment variables
        
    Returns:
        Dictionary of configuration values from environment
    """
    config_dict = {}
    
    # Define environment variable mappings
    env_mappings = {
        f'{env_prefix}_DATA_DIR': ['dataset', 'root_dir'],
        f'{env_prefix}_BATCH_SIZE': ['dataset', 'batch_size'],
        f'{env_prefix}_EPOCHS': ['training', 'epochs'],
        f'{env_prefix}_LEARNING_RATE': ['training', 'learning_rate'],
        f'{env_prefix}_DEVICE': ['training', 'device'],
        f'{env_prefix}_OUTPUT_DIR': ['output_dir'],
        f'{env_prefix}_EXPERIMENT_NAME': ['experiment_name'],
        f'{env_prefix}_LOG_LEVEL': ['log_level'],
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Navigate to the correct nested position
            current = config_dict
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Convert value to appropriate type
            final_key = config_path[-1]
            if env_var.endswith('_BATCH_SIZE') or env_var.endswith('_EPOCHS'):
                current[final_key] = int(value)
            elif env_var.endswith('_LEARNING_RATE'):
                current[final_key] = float(value)
            elif env_var.endswith('_SHUFFLE') or env_var.endswith('_ENCODING'):
                current[final_key] = value.lower() in ('true', '1', 'yes', 'on')
            else:
                current[final_key] = value
    
    return config_dict


# Example configuration templates
def create_research_config() -> SNNFusionConfig:
    """Create configuration optimized for research experiments."""
    config = SNNFusionConfig()
    
    # Research-oriented settings
    config.model.n_neurons = 2000
    config.model.connectivity = 0.15
    config.training.epochs = 200
    config.training.learning_rate = 5e-4
    config.training.early_stopping_patience = 30
    config.stdp.enabled = True
    config.stdp.reward_modulated = True
    
    return config


def create_production_config() -> SNNFusionConfig:
    """Create configuration optimized for production deployment."""
    config = SNNFusionConfig()
    
    # Production-oriented settings
    config.model.n_neurons = 1000
    config.model.connectivity = 0.1
    config.model.dropout = 0.2
    config.training.epochs = 50
    config.training.mixed_precision = True
    config.training.gradient_clip = 0.5
    config.stdp.enabled = False  # Faster inference
    
    return config


def create_debug_config() -> SNNFusionConfig:
    """Create configuration for debugging and testing."""
    config = SNNFusionConfig()
    
    # Debug-oriented settings
    config.dataset.batch_size = 4
    config.dataset.sequence_length = 50
    config.model.n_neurons = 100
    config.model.hidden_dim = 128
    config.training.epochs = 5
    config.training.validate_interval = 1
    config.log_level = 'DEBUG'
    
    return config


# Example usage
if __name__ == "__main__":
    print("Testing configuration management...")
    
    # Create and save default config
    config = create_default_config("./test_config.yaml")
    print(f"✓ Created default config with {config.model.n_neurons} neurons")
    
    # Load config back
    loaded_config = load_config("./test_config.yaml")
    print(f"✓ Loaded config with experiment name: {loaded_config.experiment_name}")
    
    # Validate config
    try:
        validate_config(loaded_config)
        print("✓ Configuration validation passed")
    except ValidationError as e:
        print(f"✗ Configuration validation failed: {e}")
    
    # Test environment config
    os.environ['SNN_FUSION_EPOCHS'] = '100'
    os.environ['SNN_FUSION_BATCH_SIZE'] = '64'
    env_config = get_config_from_env()
    print(f"✓ Environment config: epochs={env_config.get('training', {}).get('epochs')}")
    
    # Create specialized configs
    research_config = create_research_config()
    production_config = create_production_config()
    debug_config = create_debug_config()
    
    print(f"✓ Research config: {research_config.model.n_neurons} neurons, STDP enabled: {research_config.stdp.enabled}")
    print(f"✓ Production config: {production_config.model.n_neurons} neurons, mixed precision: {production_config.training.mixed_precision}")
    print(f"✓ Debug config: {debug_config.dataset.batch_size} batch size, {debug_config.training.epochs} epochs")
    
    # Cleanup
    if os.path.exists("./test_config.yaml"):
        os.remove("./test_config.yaml")
    
    print("Configuration management tests completed!")