"""
Experiment Service

Business logic layer for experiment management, lifecycle coordination,
and integration with models and training runs.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

from ..repositories import ExperimentRepository, ModelRepository, TrainingRepository
from ..repositories.experiment_repository import Experiment
from ..cache import CacheManager


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