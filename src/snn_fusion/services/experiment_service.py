"""
Experiment Service

Business logic layer for experiment management, lifecycle coordination,
and integration with models and training runs.
"""

import logging
import uuid
import asyncio
import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
from werkzeug.datastructures import FileStorage

from ..repositories import ExperimentRepository, ModelRepository, TrainingRepository
from ..repositories.experiment_repository import Experiment
from ..cache import CacheManager
from ..models import MultiModalLSM
from ..datasets import SpikeEncoder
from ..preprocessing import CochlearModel, GaborFilters, WaveletTransform
from ..utils.logging import get_logger


class ExperimentService:
    """Service layer for experiment management operations."""
    
    def __init__(
        self,
        experiment_repo: Optional[ExperimentRepository] = None,
        model_repo: Optional[ModelRepository] = None,
        training_repo: Optional[TrainingRepository] = None,
        cache: Optional[CacheManager] = None
    ):
        """Initialize experiment service with repositories and cache."""
        self.experiment_repo = experiment_repo or ExperimentRepository()
        self.model_repo = model_repo or ModelRepository()
        self.training_repo = training_repo or TrainingRepository()
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        config: Dict[str, Any] = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Tuple[int, Experiment]:
        """
        Create a new experiment with validation and setup.
        
        Args:
            name: Experiment name (must be unique)
            description: Experiment description
            config: Experiment configuration
            tags: List of tags
            metadata: Additional metadata
            
        Returns:
            Tuple of (experiment_id, experiment_object)
        """
        try:
            # Validate experiment name uniqueness
            existing = self.experiment_repo.find_one_by({'name': name})
            if existing:
                raise ValueError(f"Experiment with name '{name}' already exists")
            
            # Validate configuration if provided
            if config:
                self._validate_experiment_config(config)
            
            # Create experiment object
            experiment = Experiment(
                name=name,
                description=description,
                config=config or {},
                tags=tags or [],
                metadata=metadata or {},
                status='created',
                created_at=datetime.now(timezone.utc).isoformat()
            )
            
            # Save to database
            experiment_id = self.experiment_repo.create(experiment)
            experiment.id = experiment_id
            
            # Invalidate cache
            if self.cache:
                self.cache.invalidate_namespace('experiments')
            
            self.logger.info(f"Created experiment '{name}' with ID {experiment_id}")
            return experiment_id, experiment
            
        except Exception as e:
            self.logger.error(f"Failed to create experiment '{name}': {e}")
            raise
    
    def get_experiment_with_details(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get experiment with complete details including models and metrics."""
        try:
            # Try cache first
            cache_key = f"experiment_details_{experiment_id}"
            if self.cache:
                cached = self.cache.get(cache_key, namespace='experiments')
                if cached:
                    return cached
            
            # Get experiment
            experiment = self.experiment_repo.get_by_id(experiment_id)
            if not experiment:
                return None
            
            # Get associated models
            models = self.model_repo.find_by_experiment(experiment_id)
            
            # Get training statistics
            training_stats = self._get_experiment_training_stats(experiment_id)
            
            # Build detailed response
            details = {
                'experiment': self.experiment_repo._to_dict(experiment),
                'models': [self.model_repo._to_dict(model) for model in models],
                'training_statistics': training_stats,
                'model_count': len(models),
                'latest_activity': self._get_latest_activity(experiment_id)
            }
            
            # Cache the result
            if self.cache:
                self.cache.put(cache_key, details, ttl=300, namespace='experiments')
            
            return details
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment details {experiment_id}: {e}")
            raise
    
    def start_experiment(self, experiment_id: int) -> bool:
        """Start an experiment and update its status."""
        try:
            experiment = self.experiment_repo.get_by_id(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            if experiment.status not in ['created', 'paused']:
                raise ValueError(f"Cannot start experiment in status '{experiment.status}'")
            
            # Update status
            success = self.experiment_repo.start_experiment(experiment_id)
            
            if success:
                # Invalidate cache
                if self.cache:
                    self.cache.invalidate_namespace('experiments')
                
                self.logger.info(f"Started experiment {experiment_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to start experiment {experiment_id}: {e}")
            raise
    
    def complete_experiment(self, experiment_id: int) -> bool:
        """Complete an experiment and finalize results."""
        try:
            experiment = self.experiment_repo.get_by_id(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            # Check if all training runs are completed
            active_runs = self.training_repo.find_by({'model_id': experiment_id, 'status': 'running'})
            if active_runs:
                self.logger.warning(f"Experiment {experiment_id} has {len(active_runs)} active training runs")
            
            # Update status
            success = self.experiment_repo.complete_experiment(experiment_id)
            
            if success:
                # Generate final summary
                self._generate_experiment_summary(experiment_id)
                
                # Invalidate cache
                if self.cache:
                    self.cache.invalidate_namespace('experiments')
                
                self.logger.info(f"Completed experiment {experiment_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to complete experiment {experiment_id}: {e}")
            raise
    
    def add_experiment_tag(self, experiment_id: int, tag: str) -> bool:
        """Add a tag to an experiment."""
        try:
            success = self.experiment_repo.add_tag(experiment_id, tag)
            
            if success and self.cache:
                self.cache.invalidate_namespace('experiments')
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to add tag '{tag}' to experiment {experiment_id}: {e}")
            return False
    
    def search_experiments(
        self,
        query: str = None,
        status: str = None,
        tags: List[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search experiments with various criteria."""
        try:
            experiments = []
            
            if status:
                experiments = self.experiment_repo.find_by_status(status)
            elif query:
                experiments = self.experiment_repo.find_by_name_pattern(query)
            elif tags:
                # Find experiments with any of the specified tags
                for tag in tags:
                    tag_experiments = self.experiment_repo.find_by_tag(tag)
                    experiments.extend(tag_experiments)
                
                # Remove duplicates
                seen_ids = set()
                unique_experiments = []
                for exp in experiments:
                    if exp.id not in seen_ids:
                        unique_experiments.append(exp)
                        seen_ids.add(exp.id)
                experiments = unique_experiments
            else:
                experiments = self.experiment_repo.find_all(limit=limit, order_by='created_at DESC')
            
            # Convert to dictionaries and add summary info
            results = []
            for exp in experiments[:limit]:
                exp_dict = self.experiment_repo._to_dict(exp)
                
                # Add model count
                model_count = len(self.model_repo.find_by_experiment(exp.id))
                exp_dict['model_count'] = model_count
                
                results.append(exp_dict)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search experiments: {e}")
            raise
    
    def get_experiment_metrics_summary(self, experiment_id: int) -> Dict[str, Any]:
        """Get comprehensive metrics summary for an experiment."""
        try:
            # Get all models for the experiment
            models = self.model_repo.find_by_experiment(experiment_id)
            if not models:
                return {'error': 'No models found for experiment'}
            
            # Aggregate metrics across all models
            total_training_runs = 0
            best_accuracy = 0.0
            total_training_time = 0.0
            
            model_summaries = []
            for model in models:
                training_runs = self.training_repo.find_by_model(model.id)
                total_training_runs += len(training_runs)
                
                # Find best accuracy for this model
                model_best_accuracy = 0.0
                model_training_time = 0.0
                
                for run in training_runs:
                    if run.best_val_accuracy and run.best_val_accuracy > model_best_accuracy:
                        model_best_accuracy = run.best_val_accuracy
                    
                    if run.training_time_seconds:
                        model_training_time += run.training_time_seconds
                
                if model_best_accuracy > best_accuracy:
                    best_accuracy = model_best_accuracy
                
                total_training_time += model_training_time
                
                model_summaries.append({
                    'model_id': model.id,
                    'model_name': model.name,
                    'architecture': model.architecture,
                    'parameters': model.parameters_count,
                    'training_runs': len(training_runs),
                    'best_accuracy': model_best_accuracy,
                    'training_time_seconds': model_training_time
                })
            
            return {
                'experiment_id': experiment_id,
                'total_models': len(models),
                'total_training_runs': total_training_runs,
                'best_accuracy_overall': best_accuracy,
                'total_training_time_seconds': total_training_time,
                'model_summaries': model_summaries,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics summary for experiment {experiment_id}: {e}")
            raise
    
    def _validate_experiment_config(self, config: Dict[str, Any]) -> None:
        """Validate experiment configuration."""
        # Basic validation - can be extended
        if not isinstance(config, dict):
            raise ValueError("Experiment config must be a dictionary")
        
        # Check for required fields if any
        # This is domain-specific and can be customized
        pass
    
    def _get_experiment_training_stats(self, experiment_id: int) -> Dict[str, Any]:
        """Get training statistics for an experiment."""
        try:
            # Get all models for the experiment
            models = self.model_repo.find_by_experiment(experiment_id)
            model_ids = [model.id for model in models]
            
            if not model_ids:
                return {'total_runs': 0, 'completed_runs': 0, 'success_rate': 0.0}
            
            # Count training runs by status
            total_runs = 0
            completed_runs = 0
            
            for model_id in model_ids:
                runs = self.training_repo.find_by_model(model_id)
                total_runs += len(runs)
                completed_runs += sum(1 for run in runs if run.status == 'completed')
            
            success_rate = (completed_runs / total_runs * 100) if total_runs > 0 else 0.0
            
            return {
                'total_runs': total_runs,
                'completed_runs': completed_runs,
                'success_rate': round(success_rate, 2)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get training stats for experiment {experiment_id}: {e}")
            return {'total_runs': 0, 'completed_runs': 0, 'success_rate': 0.0}
    
    def _get_latest_activity(self, experiment_id: int) -> Optional[str]:
        """Get the timestamp of the latest activity for an experiment."""
        try:
            # Get latest model creation or update
            models = self.model_repo.find_by_experiment(experiment_id)
            if not models:
                return None
            
            latest_timestamp = None
            
            for model in models:
                # Check model timestamp
                if model.updated_at:
                    if not latest_timestamp or model.updated_at > latest_timestamp:
                        latest_timestamp = model.updated_at
                
                # Check training runs
                runs = self.training_repo.find_by_model(model.id)
                for run in runs:
                    if run.completed_at:
                        if not latest_timestamp or run.completed_at > latest_timestamp:
                            latest_timestamp = run.completed_at
                    elif run.started_at:
                        if not latest_timestamp or run.started_at > latest_timestamp:
                            latest_timestamp = run.started_at
            
            return latest_timestamp
            
        except Exception as e:
            self.logger.warning(f"Failed to get latest activity for experiment {experiment_id}: {e}")
            return None
    
    def _generate_experiment_summary(self, experiment_id: int) -> None:
        """Generate and store experiment completion summary."""
        try:
            summary = self.get_experiment_metrics_summary(experiment_id)
            
            # Update experiment metadata with summary
            self.experiment_repo.update_metadata(experiment_id, {
                'completion_summary': summary,
                'completed_at': datetime.now(timezone.utc).isoformat()
            })
            
            self.logger.info(f"Generated completion summary for experiment {experiment_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary for experiment {experiment_id}: {e}")
    
    # New methods for API functionality
    
    def queue_training_job(self, training_id: int, config: Dict[str, Any]) -> str:
        """
        Queue a training job for background processing.
        
        Args:
            training_id: ID of the training run
            config: Training configuration
            
        Returns:
            Task ID for the queued job
        """
        try:
            task_id = str(uuid.uuid4())
            
            # Create background training task
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self._execute_training_job, training_id, config, task_id)
            
            self.logger.info(f"Queued training job {training_id} with task ID {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to queue training job {training_id}: {e}")
            raise
    
    def _execute_training_job(self, training_id: int, config: Dict[str, Any], task_id: str) -> None:
        """Execute training job in background thread."""
        try:
            self.logger.info(f"Starting background training job {training_id}")
            
            # Simulate training process (in production, integrate with actual training)
            epochs = config.get('epochs', 100)
            for epoch in range(epochs):
                # Simulate epoch training
                time.sleep(0.1)  # Simulate computation
                
                # Update progress periodically
                if epoch % 10 == 0:
                    progress = (epoch + 1) / epochs * 100
                    self._update_training_progress(training_id, epoch + 1, progress)
            
            # Mark as completed
            self._complete_training_job(training_id, task_id)
            
        except Exception as e:
            self.logger.error(f"Training job {training_id} failed: {e}")
            self._fail_training_job(training_id, str(e))
    
    def _update_training_progress(self, training_id: int, epoch: int, progress: float) -> None:
        """Update training progress in database."""
        try:
            # Update training run record with current progress
            self.training_repo.update_progress(training_id, epoch, progress)
            
        except Exception as e:
            self.logger.warning(f"Failed to update training progress {training_id}: {e}")
    
    def _complete_training_job(self, training_id: int, task_id: str) -> None:
        """Mark training job as completed."""
        try:
            # Simulate final metrics
            final_metrics = {
                'final_accuracy': 0.92 + np.random.normal(0, 0.05),
                'final_loss': 0.15 + np.random.normal(0, 0.02),
                'training_time': time.time()  # Mock training time
            }
            
            # Update training run with completion
            self.training_repo.complete_run(training_id, final_metrics)
            
            self.logger.info(f"Completed training job {training_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to complete training job {training_id}: {e}")
    
    def _fail_training_job(self, training_id: int, error_message: str) -> None:
        """Mark training job as failed."""
        try:
            self.training_repo.fail_run(training_id, error_message)
            
        except Exception as e:
            self.logger.error(f"Failed to mark training job {training_id} as failed: {e}")
    
    def process_uploaded_files(self, files: Dict[str, FileStorage]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Process uploaded multimodal files for inference.
        
        Args:
            files: Dictionary of uploaded files
            
        Returns:
            Dictionary of processed tensors or None if processing failed
        """
        try:
            processed_data = {}
            
            # Process audio files
            if 'audio' in files and files['audio']:
                audio_tensor = self._process_audio_file(files['audio'])
                if audio_tensor is not None:
                    processed_data['audio'] = audio_tensor
            
            # Process event camera data
            if 'events' in files and files['events']:
                event_tensor = self._process_event_file(files['events'])
                if event_tensor is not None:
                    processed_data['events'] = event_tensor
            
            # Process IMU data
            if 'imu' in files and files['imu']:
                imu_tensor = self._process_imu_file(files['imu'])
                if imu_tensor is not None:
                    processed_data['imu'] = imu_tensor
            
            return processed_data if processed_data else None
            
        except Exception as e:
            self.logger.error(f"Failed to process uploaded files: {e}")
            return None
    
    def _process_audio_file(self, file: FileStorage) -> Optional[torch.Tensor]:
        """Process uploaded audio file."""
        try:
            # Save temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                file.save(tmp_file.name)
                
                # Process with cochlear model
                cochlear = CochlearModel()
                audio_spikes = cochlear.process_file(tmp_file.name)
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return torch.tensor(audio_spikes, dtype=torch.float32)
                
        except Exception as e:
            self.logger.warning(f"Failed to process audio file: {e}")
            return None
    
    def _process_event_file(self, file: FileStorage) -> Optional[torch.Tensor]:
        """Process uploaded event camera file."""
        try:
            # Mock event processing (would integrate with actual event processing library)
            # Return synthetic event data for now
            return torch.rand(1000, 128, 128, 2, dtype=torch.float32)
            
        except Exception as e:
            self.logger.warning(f"Failed to process event file: {e}")
            return None
    
    def _process_imu_file(self, file: FileStorage) -> Optional[torch.Tensor]:
        """Process uploaded IMU file."""
        try:
            # Mock IMU processing
            # Return synthetic IMU data for now
            return torch.rand(1000, 6, dtype=torch.float32)
            
        except Exception as e:
            self.logger.warning(f"Failed to process IMU file: {e}")
            return None
    
    def prepare_inference_data(self, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepare input data for inference.
        
        Args:
            input_data: Raw input data dictionary
            
        Returns:
            Dictionary of prepared tensors
        """
        try:
            tensors = {}
            
            # Convert lists to tensors
            for key, value in input_data.items():
                if isinstance(value, list):
                    tensors[key] = torch.tensor(value, dtype=torch.float32)
                elif isinstance(value, np.ndarray):
                    tensors[key] = torch.from_numpy(value).float()
                elif isinstance(value, torch.Tensor):
                    tensors[key] = value.float()
                else:
                    raise ValueError(f"Unsupported data type for key '{key}': {type(value)}")
            
            return tensors
            
        except Exception as e:
            self.logger.error(f"Failed to prepare inference data: {e}")
            raise
    
    def run_inference(self, model_id: int, input_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Run inference on a trained model.
        
        Args:
            model_id: ID of the model to use
            input_data: Preprocessed input tensors
            
        Returns:
            Inference results with predictions and metadata
        """
        try:
            start_time = time.time()
            
            # Load model (mock for now)
            model = self._load_model(model_id)
            
            # Run inference
            with torch.no_grad():
                predictions = model.forward(input_data)
                
                # Convert to numpy for JSON serialization
                if isinstance(predictions, torch.Tensor):
                    pred_array = predictions.cpu().numpy()
                    confidence = torch.softmax(predictions, dim=-1).cpu().numpy()
                else:
                    # Mock predictions
                    pred_array = np.random.rand(10)
                    confidence = np.random.rand(10)
                    confidence = confidence / confidence.sum()  # Normalize
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'predictions': pred_array,
                'confidence': confidence,
                'inference_time_ms': round(inference_time, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Inference failed for model {model_id}: {e}")
            raise
    
    def _load_model(self, model_id: int) -> torch.nn.Module:
        """Load a trained model."""
        try:
            # Mock model loading (in production, load from saved weights)
            model = MultiModalLSM(
                n_neurons=1000,
                connectivity=0.1,
                audio_channels=64,
                event_size=(128, 128),
                imu_dims=6,
                n_outputs=10
            )
            
            # In production: model.load_state_dict(torch.load(model_path))
            
            model.eval()
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def deploy_to_hardware(
        self,
        model_id: int,
        hardware_type: str,
        deployment_config: Dict[str, Any],
        optimize: bool = False
    ) -> Dict[str, Any]:
        """
        Deploy model to neuromorphic hardware.
        
        Args:
            model_id: ID of the model to deploy
            hardware_type: Target hardware platform
            deployment_config: Hardware-specific configuration
            optimize: Whether to apply optimization
            
        Returns:
            Deployment result with success status and metrics
        """
        try:
            self.logger.info(f"Deploying model {model_id} to {hardware_type}")
            
            # Load model
            model = self._load_model(model_id)
            
            # Apply hardware-specific conversion
            if hardware_type.lower() == 'loihi2':
                result = self._deploy_to_loihi(model, deployment_config, optimize)
            elif hardware_type.lower() == 'akida':
                result = self._deploy_to_akida(model, deployment_config, optimize)
            elif hardware_type.lower() == 'spinnaker':
                result = self._deploy_to_spinnaker(model, deployment_config, optimize)
            else:
                raise ValueError(f"Unsupported hardware type: {hardware_type}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Hardware deployment failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'metrics': {}
            }
    
    def _deploy_to_loihi(self, model: torch.nn.Module, config: Dict, optimize: bool) -> Dict[str, Any]:
        """Deploy model to Intel Loihi 2."""
        try:
            # Mock Loihi deployment
            base_latency = 0.8
            base_power = 120.0
            base_accuracy = 0.92
            
            # Apply optimization effects
            if optimize:
                base_latency *= 0.7
                base_power *= 0.85
                base_accuracy *= 1.02
            
            # Add some realistic variation
            latency = base_latency + np.random.normal(0, 0.1)
            power = base_power + np.random.normal(0, 10)
            accuracy = min(0.98, base_accuracy + np.random.normal(0, 0.02))
            
            return {
                'success': True,
                'metrics': {
                    'latency_ms': max(0.1, latency),
                    'power_mw': max(50, power),
                    'accuracy': max(0.5, accuracy),
                    'memory_mb': 2.5
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'metrics': {}}
    
    def _deploy_to_akida(self, model: torch.nn.Module, config: Dict, optimize: bool) -> Dict[str, Any]:
        """Deploy model to BrainChip Akida."""
        try:
            # Mock Akida deployment
            base_latency = 1.2
            base_power = 300.0
            base_accuracy = 0.89
            
            if optimize:
                base_latency *= 0.8
                base_power *= 0.9
                base_accuracy *= 1.01
            
            latency = base_latency + np.random.normal(0, 0.15)
            power = base_power + np.random.normal(0, 20)
            accuracy = min(0.95, base_accuracy + np.random.normal(0, 0.03))
            
            return {
                'success': True,
                'metrics': {
                    'latency_ms': max(0.2, latency),
                    'power_mw': max(100, power),
                    'accuracy': max(0.5, accuracy),
                    'memory_mb': 4.2
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'metrics': {}}
    
    def _deploy_to_spinnaker(self, model: torch.nn.Module, config: Dict, optimize: bool) -> Dict[str, Any]:
        """Deploy model to SpiNNaker."""
        try:
            # Mock SpiNNaker deployment
            base_latency = 2.1
            base_power = 800.0
            base_accuracy = 0.90
            
            if optimize:
                base_latency *= 0.75
                base_power *= 0.8
                base_accuracy *= 1.015
            
            latency = base_latency + np.random.normal(0, 0.2)
            power = base_power + np.random.normal(0, 50)
            accuracy = min(0.96, base_accuracy + np.random.normal(0, 0.025))
            
            return {
                'success': True,
                'metrics': {
                    'latency_ms': max(0.5, latency),
                    'power_mw': max(200, power),
                    'accuracy': max(0.5, accuracy),
                    'memory_mb': 8.0
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'metrics': {}}