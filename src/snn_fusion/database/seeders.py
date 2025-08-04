"""
Database Seeders for SNN-Fusion

Provides sample data seeding for development and testing environments
with neuromorphic-specific experimental configurations.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
import random

from .connection import DatabaseManager, get_database


logger = logging.getLogger(__name__)


class DatabaseSeeder:
    """Main database seeder for SNN-Fusion development data."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize seeder with database manager."""
        self.db = db_manager or get_database()
        self.logger = logging.getLogger(__name__)
    
    def seed_all(self, clear_existing: bool = False) -> None:
        """Seed all tables with sample data."""
        if clear_existing:
            self.clear_all_data()
        
        self.logger.info("Starting database seeding...")
        
        # Seed in dependency order
        self.seed_experiments()
        self.seed_datasets() 
        self.seed_models()
        self.seed_training_runs()
        self.seed_hardware_profiles()
        self.seed_performance_metrics()
        self.seed_spike_data()
        self.seed_model_artifacts()
        
        self.logger.info("Database seeding completed successfully")
    
    def clear_all_data(self) -> None:
        """Clear all data from tables (for development only)."""
        tables = [
            'model_artifacts',
            'spike_data', 
            'performance_metrics',
            'hardware_profiles',
            'training_runs',
            'models',
            'datasets',
            'experiments'
        ]
        
        for table in tables:
            try:
                self.db.execute_query(f"DELETE FROM {table}")
                self.logger.info(f"Cleared data from {table}")
            except Exception as e:
                self.logger.warning(f"Failed to clear {table}: {e}")
    
    def seed_experiments(self) -> List[int]:
        """Seed experiments table with sample experiments."""
        experiments = [
            {
                'name': 'Multi-Modal Audio-Visual Fusion',
                'description': 'Investigating cross-modal attention mechanisms for synchronized audio-visual processing using liquid state machines.',
                'config_json': json.dumps({
                    'modalities': ['audio', 'vision'],
                    'fusion_type': 'attention',
                    'temporal_window_ms': 100,
                    'learning_rate': 0.001,
                    'batch_size': 32
                }),
                'status': 'completed',
                'tags': json.dumps(['audio', 'vision', 'attention', 'lsm']),
                'metadata_json': json.dumps({
                    'research_goal': 'Real-time audiovisual processing',
                    'hardware_target': 'loihi2',
                    'expected_latency_ms': 5
                }),
                'created_at': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
                'started_at': (datetime.now(timezone.utc) - timedelta(days=25)).isoformat(),
                'completed_at': (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
            },
            {
                'name': 'Tactile-Audio Sensor Fusion',
                'description': 'Exploring haptic feedback integration with auditory processing for robotic manipulation tasks.',
                'config_json': json.dumps({
                    'modalities': ['audio', 'tactile'],
                    'fusion_type': 'hierarchical',
                    'temporal_window_ms': 50,
                    'learning_rate': 0.0005,
                    'batch_size': 16
                }),
                'status': 'running',
                'tags': json.dumps(['tactile', 'audio', 'robotics', 'manipulation']),
                'metadata_json': json.dumps({
                    'research_goal': 'Robotic object manipulation',
                    'hardware_target': 'akida',
                    'expected_accuracy': 0.85
                }),
                'created_at': (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
                'started_at': (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
            },
            {
                'name': 'Event-Based Vision Processing',
                'description': 'Pure event camera processing for dynamic scene understanding with minimal latency.',
                'config_json': json.dumps({
                    'modalities': ['event_vision'],
                    'fusion_type': None,
                    'temporal_window_ms': 20,
                    'learning_rate': 0.002,
                    'batch_size': 64
                }),
                'status': 'created',
                'tags': json.dumps(['event_camera', 'vision', 'low_latency']),
                'metadata_json': json.dumps({
                    'research_goal': 'Ultra-low latency vision',
                    'hardware_target': 'spinnaker',
                    'target_latency_ms': 1
                }),
                'created_at': (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
            },
            {
                'name': 'Tri-Modal Sensor Integration',
                'description': 'Comprehensive fusion of audio, vision, and tactile modalities for autonomous navigation.',
                'config_json': json.dumps({
                    'modalities': ['audio', 'vision', 'tactile'],
                    'fusion_type': 'cross_modal_attention',
                    'temporal_window_ms': 150,
                    'learning_rate': 0.0008,
                    'batch_size': 24
                }),
                'status': 'created',
                'tags': json.dumps(['trimodal', 'navigation', 'autonomous', 'attention']),
                'metadata_json': json.dumps({
                    'research_goal': 'Autonomous navigation',
                    'hardware_target': 'loihi2',
                    'complexity_level': 'high'
                }),
                'created_at': datetime.now(timezone.utc).isoformat()
            }
        ]
        
        experiment_ids = []
        for exp in experiments:
            exp_id = self.db.insert_record('experiments', exp)
            experiment_ids.append(exp_id)
            self.logger.debug(f"Seeded experiment: {exp['name']} (ID: {exp_id})")
        
        return experiment_ids
    
    def seed_datasets(self) -> List[int]:
        """Seed datasets table with sample datasets."""
        datasets = [
            {
                'name': 'MAVEN-AudioVisual-v1',
                'path': '/data/maven/audiovisual_v1',
                'modalities': json.dumps(['audio', 'vision']),
                'n_samples': 50000,
                'sample_rate': 48000.0,
                'sequence_length': 1000,
                'format': 'hdf5',
                'size_bytes': 25600000000,  # 25.6GB
                'checksum': 'sha256:a1b2c3d4e5f6789012345678901234567890',
                'metadata_json': json.dumps({
                    'recording_conditions': 'indoor, controlled lighting',
                    'subjects': 20,
                    'sessions_per_subject': 50,
                    'annotation_type': 'action_labels'
                })
            },
            {
                'name': 'TactileAudio-Manipulation-v2',
                'path': '/data/tactile/manipulation_v2',
                'modalities': json.dumps(['audio', 'tactile']),
                'n_samples': 30000,
                'sample_rate': 1000.0,
                'sequence_length': 500,
                'format': 'numpy',
                'size_bytes': 8500000000,  # 8.5GB
                'checksum': 'sha256:f6e5d4c3b2a1098765432109876543210987',
                'metadata_json': json.dumps({
                    'manipulation_tasks': ['grasp', 'lift', 'place', 'push'],
                    'objects': 15,
                    'surface_types': ['wood', 'metal', 'plastic', 'fabric'],
                    'sensor_resolution': {'tactile': '16x16', 'audio': '48kHz'}
                })
            },
            {
                'name': 'EventCamera-Dynamic-v1',
                'path': '/data/events/dynamic_v1',
                'modalities': json.dumps(['event_vision']),
                'n_samples': 75000,
                'sample_rate': None,
                'sequence_length': 2000,
                'format': 'aedat4',
                'size_bytes': 12000000000,  # 12GB
                'checksum': 'sha256:123456789abcdef0123456789abcdef012345678',
                'metadata_json': json.dumps({
                    'camera_resolution': '346x260',
                    'scenes': ['indoor', 'outdoor', 'mixed_lighting'],
                    'motion_types': ['translation', 'rotation', 'scaling'],
                    'event_rate': 'high'
                })
            }
        ]
        
        dataset_ids = []
        for dataset in datasets:
            dataset_id = self.db.insert_record('datasets', dataset)
            dataset_ids.append(dataset_id)
            self.logger.debug(f"Seeded dataset: {dataset['name']} (ID: {dataset_id})")
        
        return dataset_ids
    
    def seed_models(self) -> List[int]:
        """Seed models table with sample models."""
        # Get experiment IDs
        experiments = self.db.search_records('experiments', order_by='id ASC')
        if not experiments:
            self.logger.warning("No experiments found for model seeding")
            return []
        
        models = []
        for i, exp in enumerate(experiments[:3]):  # First 3 experiments
            base_models = [
                {
                    'experiment_id': exp['id'],
                    'name': f'MultiModalLSM-{exp["name"][:20]}-v1',
                    'architecture': 'MultiModalLSM',
                    'parameters_count': random.randint(50000, 500000),
                    'model_config_json': json.dumps({
                        'n_outputs': random.choice([10, 15, 20]),
                        'modality_configs': {
                            'audio': {'n_inputs': 64, 'n_reservoir': 200},
                            'vision': {'n_inputs': 128, 'n_reservoir': 300}
                        },
                        'fusion_type': 'attention',
                        'temporal_dynamics': {
                            'tau_mem': 20.0,
                            'tau_adapt': 100.0
                        }
                    }),
                    'best_accuracy': 0.78 + random.uniform(0, 0.15),
                    'best_loss': 0.5 + random.uniform(-0.3, 0.3),
                    'training_epochs': random.randint(50, 200),
                    'checkpoint_path': f'/models/checkpoints/lsm_exp{exp["id"]}_v1.pt',
                    'metadata_json': json.dumps({
                        'optimizer': 'Adam',
                        'scheduler': 'CosineAnnealingLR',
                        'regularization': 'dropout_0.2'
                    })
                },
                {
                    'experiment_id': exp['id'],
                    'name': f'HierarchicalSNN-{exp["name"][:20]}-v1',
                    'architecture': 'HierarchicalFusionSNN',
                    'parameters_count': random.randint(100000, 800000),
                    'model_config_json': json.dumps({
                        'n_outputs': random.choice([10, 15, 20]),
                        'hierarchy_levels': 3,
                        'fusion_layers': [
                            {'type': 'attention', 'heads': 4},
                            {'type': 'gated_fusion', 'hidden': 256}
                        ],
                        'task_heads': {
                            'classification': random.choice([10, 15, 20]),
                            'localization': 4
                        }
                    }),
                    'best_accuracy': 0.82 + random.uniform(0, 0.12),
                    'best_loss': 0.4 + random.uniform(-0.2, 0.3),
                    'training_epochs': random.randint(75, 250),
                    'checkpoint_path': f'/models/checkpoints/hsnn_exp{exp["id"]}_v1.pt',
                    'metadata_json': json.dumps({
                        'optimizer': 'AdamW',
                        'scheduler': 'MultiStepLR',
                        'weight_decay': 0.01
                    })
                }
            ]
            models.extend(base_models)
        
        model_ids = []
        for model in models:
            model_id = self.db.insert_record('models', model)
            model_ids.append(model_id)
            self.logger.debug(f"Seeded model: {model['name']} (ID: {model_id})")
        
        return model_ids
    
    def seed_training_runs(self) -> List[int]:
        """Seed training_runs table with sample training data."""
        models = self.db.search_records('models', order_by='id ASC')
        datasets = self.db.search_records('datasets', order_by='id ASC')
        
        if not models or not datasets:
            self.logger.warning("No models or datasets found for training run seeding")
            return []
        
        training_runs = []
        statuses = ['completed', 'running', 'failed', 'stopped']
        
        for model in models:
            for i in range(random.randint(1, 3)):  # 1-3 runs per model
                dataset = random.choice(datasets)
                status = random.choice(statuses)
                
                start_time = datetime.now(timezone.utc) - timedelta(
                    days=random.randint(1, 30),
                    hours=random.randint(0, 23)
                )
                
                training_run = {
                    'model_id': model['id'],
                    'dataset_id': dataset['id'],
                    'status': status,
                    'config_json': json.dumps({
                        'epochs': random.randint(50, 200),
                        'batch_size': random.choice([16, 32, 64]),
                        'learning_rate': random.uniform(0.0001, 0.01),
                        'temporal_window': random.randint(50, 200),
                        'optimizer': random.choice(['Adam', 'AdamW', 'SGD']),
                        'scheduler': random.choice(['StepLR', 'CosineAnnealingLR'])
                    }),
                    'current_epoch': random.randint(0, 200) if status == 'running' else 200,
                    'total_epochs': 200,
                    'started_at': start_time.isoformat(),
                    'log_file_path': f'/logs/training/run_{model["id"]}_{i+1}.log',
                    'checkpoint_dir': f'/checkpoints/model_{model["id"]}/run_{i+1}/'
                }
                
                if status == 'completed':
                    training_run.update({
                        'best_val_accuracy': 0.7 + random.uniform(0, 0.25),
                        'best_val_loss': 0.3 + random.uniform(0, 0.4),
                        'final_train_accuracy': 0.8 + random.uniform(0, 0.15),
                        'final_train_loss': 0.2 + random.uniform(0, 0.3),
                        'training_time_seconds': random.uniform(1800, 14400),  # 30min - 4hrs
                        'completed_at': (start_time + timedelta(hours=random.randint(2, 12))).isoformat(),
                        'metrics_json': json.dumps({
                            'spike_rate_avg': random.uniform(0.1, 0.3),
                            'synchrony_index': random.uniform(0.4, 0.8),
                            'convergence_epoch': random.randint(20, 100)
                        })
                    })
                elif status == 'failed':
                    training_run.update({
                        'completed_at': (start_time + timedelta(hours=random.randint(1, 6))).isoformat(),
                        'metrics_json': json.dumps({
                            'error_type': random.choice(['out_of_memory', 'gradient_explosion', 'nan_loss']),
                            'failed_epoch': random.randint(1, 50)
                        })
                    })
                
                training_runs.append(training_run)
        
        run_ids = []
        for run in training_runs:
            run_id = self.db.insert_record('training_runs', run)
            run_ids.append(run_id)
            self.logger.debug(f"Seeded training run for model {run['model_id']} (ID: {run_id})")
        
        return run_ids
    
    def seed_hardware_profiles(self) -> List[int]:
        """Seed hardware_profiles table with deployment data."""
        models = self.db.search_records('models', order_by='id ASC')
        
        if not models:
            self.logger.warning("No models found for hardware profile seeding")
            return []
        
        hardware_types = ['loihi2', 'akida', 'spinnaker', 'gpu', 'cpu']
        profiles = []
        
        for model in models[:4]:  # Deploy first 4 models
            for hw_type in random.sample(hardware_types, random.randint(1, 3)):
                profile = {
                    'name': f"{model['name'][:30]}_on_{hw_type}_{random.randint(1000, 9999)}",
                    'hardware_type': hw_type,
                    'model_id': model['id'],
                    'deployment_config_json': json.dumps({
                        'quantization': random.choice(['int8', 'int16', 'fp16']),
                        'optimization_level': random.choice(['O1', 'O2', 'O3']),
                        'batch_size': random.choice([1, 4, 8, 16]),
                        'memory_limit_mb': random.randint(100, 2000)
                    }),
                    'inference_latency_ms': random.uniform(0.5, 10.0),
                    'power_consumption_mw': random.uniform(50, 500),
                    'accuracy_deployed': random.uniform(0.75, 0.95),
                    'memory_usage_mb': random.uniform(50, 500),
                    'throughput_samples_per_sec': random.uniform(100, 2000),
                    'optimization_applied': random.choice([True, False]),
                    'deployed_at': (datetime.now(timezone.utc) - timedelta(
                        days=random.randint(0, 20)
                    )).isoformat(),
                    'benchmark_results_json': json.dumps({
                        'latency_p50': random.uniform(1.0, 5.0),
                        'latency_p95': random.uniform(5.0, 15.0),
                        'latency_p99': random.uniform(10.0, 25.0),
                        'accuracy_vs_baseline': random.uniform(0.95, 1.02),
                        'energy_per_inference_uj': random.uniform(10, 1000)
                    })
                }
                profiles.append(profile)
        
        profile_ids = []
        for profile in profiles:
            profile_id = self.db.insert_record('hardware_profiles', profile)
            profile_ids.append(profile_id)
            self.logger.debug(f"Seeded hardware profile: {profile['name']} (ID: {profile_id})")
        
        return profile_ids
    
    def seed_performance_metrics(self) -> List[int]:
        """Seed performance_metrics table with sample metrics."""
        training_runs = self.db.search_records('training_runs', {'status': 'completed'})
        hardware_profiles = self.db.search_records('hardware_profiles')
        
        if not training_runs and not hardware_profiles:
            self.logger.warning("No training runs or hardware profiles found for metrics seeding")
            return []
        
        metrics = []
        metric_types = [
            ('accuracy', '%', 'validation_epoch_{}'),
            ('loss', 'scalar', 'training_epoch_{}'),
            ('spike_rate', 'Hz', 'batch_{}'),
            ('firing_rate', 'Hz', 'layer_{}'),
            ('synchrony_index', 'scalar', 'network_analysis'),
            ('inference_latency', 'ms', 'hardware_test'),
            ('power_consumption', 'mW', 'hardware_test'),
            ('memory_usage', 'MB', 'runtime_analysis')
        ]
        
        # Training metrics
        for run in training_runs:
            for _ in range(random.randint(20, 100)):  # Multiple metrics per run
                metric_name, unit, context_template = random.choice(metric_types[:5])
                
                metrics.append({
                    'training_run_id': run['id'],
                    'hardware_profile_id': None,
                    'metric_name': metric_name,
                    'metric_value': self._generate_metric_value(metric_name),
                    'metric_unit': unit,
                    'measurement_context': context_template.format(random.randint(1, 200)),
                    'recorded_at': (datetime.now(timezone.utc) - timedelta(
                        days=random.randint(0, 30),
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59)
                    )).isoformat(),
                    'metadata_json': json.dumps({
                        'measurement_method': 'automated',
                        'confidence_interval': random.uniform(0.95, 0.99)
                    })
                })
        
        # Hardware metrics
        for profile in hardware_profiles:
            for _ in range(random.randint(10, 50)):
                metric_name, unit, context_template = random.choice(metric_types[5:])
                
                metrics.append({
                    'training_run_id': None,
                    'hardware_profile_id': profile['id'],
                    'metric_name': metric_name,
                    'metric_value': self._generate_metric_value(metric_name),
                    'metric_unit': unit,
                    'measurement_context': context_template,
                    'recorded_at': (datetime.now(timezone.utc) - timedelta(
                        days=random.randint(0, 10),
                        hours=random.randint(0, 23)
                    )).isoformat(),
                    'metadata_json': json.dumps({
                        'hardware_state': 'nominal',
                        'temperature_c': random.uniform(25, 45)
                    })
                })
        
        metric_ids = []
        for metric in metrics:
            metric_id = self.db.insert_record('performance_metrics', metric)
            metric_ids.append(metric_id)
        
        self.logger.debug(f"Seeded {len(metric_ids)} performance metrics")
        return metric_ids
    
    def seed_spike_data(self) -> List[int]:
        """Seed spike_data table with neuromorphic spike information."""
        training_runs = self.db.search_records('training_runs', {'status': 'completed'})
        
        if not training_runs:
            self.logger.warning("No completed training runs found for spike data seeding")
            return []
        
        spike_records = []
        modalities = ['audio', 'vision', 'tactile', 'fusion']
        
        for run in training_runs:
            epochs = random.randint(50, 200)
            for epoch in range(0, epochs, 10):  # Every 10th epoch
                for modality in random.sample(modalities, random.randint(2, 4)):
                    for batch_idx in range(random.randint(5, 20)):
                        spike_records.append({
                            'training_run_id': run['id'],
                            'epoch': epoch,
                            'batch_idx': batch_idx,
                            'modality': modality,
                            'firing_rate': random.uniform(0.05, 0.4),
                            'spike_count': random.randint(100, 5000),
                            'synchrony_index': random.uniform(0.3, 0.9),
                            'cv_isi': random.uniform(0.5, 2.0),
                            'recorded_at': (datetime.now(timezone.utc) - timedelta(
                                days=random.randint(0, 30)
                            )).isoformat(),
                            'spike_patterns_path': f'/spike_data/run_{run["id"]}/epoch_{epoch}/batch_{batch_idx}_{modality}.npz'
                        })
        
        spike_ids = []
        # Batch insert for performance
        for i in range(0, len(spike_records), 100):
            batch = spike_records[i:i+100]
            for record in batch:
                spike_id = self.db.insert_record('spike_data', record)
                spike_ids.append(spike_id)
        
        self.logger.debug(f"Seeded {len(spike_ids)} spike data records")
        return spike_ids
    
    def seed_model_artifacts(self) -> List[int]:
        """Seed model_artifacts table with model files and visualizations."""
        models = self.db.search_records('models', order_by='id ASC')
        
        if not models:
            self.logger.warning("No models found for artifact seeding")
            return []
        
        artifacts = []
        artifact_types = ['checkpoint', 'weights', 'config', 'visualization', 'onnx', 'tensorrt']
        
        for model in models:
            # Each model has multiple artifacts
            for artifact_type in random.sample(artifact_types, random.randint(3, 6)):
                size_ranges = {
                    'checkpoint': (50_000_000, 500_000_000),    # 50MB - 500MB
                    'weights': (10_000_000, 100_000_000),       # 10MB - 100MB
                    'config': (1_000, 10_000),                  # 1KB - 10KB
                    'visualization': (500_000, 5_000_000),      # 500KB - 5MB
                    'onnx': (20_000_000, 200_000_000),          # 20MB - 200MB
                    'tensorrt': (5_000_000, 50_000_000)         # 5MB - 50MB
                }
                
                min_size, max_size = size_ranges[artifact_type]
                
                artifacts.append({
                    'model_id': model['id'],
                    'artifact_type': artifact_type,
                    'file_path': f'/artifacts/model_{model["id"]}/{artifact_type}_{random.randint(1000, 9999)}.{self._get_extension(artifact_type)}',
                    'file_size_bytes': random.randint(min_size, max_size),
                    'checksum': f'sha256:{random.randint(10**63, 10**64-1):064x}',
                    'description': f'{artifact_type.title()} for {model["name"]}',
                    'created_at': (datetime.now(timezone.utc) - timedelta(
                        days=random.randint(0, 20)
                    )).isoformat(),
                    'metadata_json': json.dumps({
                        'format_version': '1.0',
                        'compression': artifact_type in ['checkpoint', 'weights'],
                        'validation_passed': True
                    })
                })
        
        artifact_ids = []
        for artifact in artifacts:
            artifact_id = self.db.insert_record('model_artifacts', artifact)
            artifact_ids.append(artifact_id)
        
        self.logger.debug(f"Seeded {len(artifact_ids)} model artifacts")
        return artifact_ids
    
    def _generate_metric_value(self, metric_name: str) -> float:
        """Generate realistic metric values based on metric type."""
        value_ranges = {
            'accuracy': (0.5, 0.95),
            'loss': (0.01, 2.0),
            'spike_rate': (0.05, 0.5),
            'firing_rate': (1.0, 100.0),
            'synchrony_index': (0.3, 0.9),
            'inference_latency': (0.5, 20.0),
            'power_consumption': (50.0, 800.0),
            'memory_usage': (50.0, 1000.0)
        }
        
        min_val, max_val = value_ranges.get(metric_name, (0.0, 1.0))
        return random.uniform(min_val, max_val)
    
    def _get_extension(self, artifact_type: str) -> str:
        """Get file extension for artifact type."""
        extensions = {
            'checkpoint': 'pt',
            'weights': 'pth',
            'config': 'json',
            'visualization': 'png',
            'onnx': 'onnx',
            'tensorrt': 'engine'
        }
        return extensions.get(artifact_type, 'bin')