"""
Model Repository Implementation

Provides data access operations for SNN model management with support
for model lifecycle, checkpoints, and performance tracking.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

from .base import BaseRepository


@dataclass
class SNNModel:
    """SNN Model model class."""
    id: Optional[int] = None
    experiment_id: int = None
    name: str = ""
    architecture: str = ""
    parameters_count: int = 0
    model_config: Dict[str, Any] = None
    checkpoint_path: Optional[str] = None
    best_accuracy: Optional[float] = None
    best_loss: Optional[float] = None
    training_epochs: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = {}
        if self.metadata is None:
            self.metadata = {}


class ModelRepository(BaseRepository[SNNModel]):
    """Repository for SNN model data access operations."""
    
    def _get_table_name(self) -> str:
        return "models"
    
    def _to_dict(self, model: SNNModel) -> Dict[str, Any]:
        """Convert model object to dictionary for database storage."""
        return {
            'id': model.id,
            'experiment_id': model.experiment_id,
            'name': model.name,
            'architecture': model.architecture,
            'parameters_count': model.parameters_count,
            'model_config_json': self._serialize_json_field(model.model_config),
            'checkpoint_path': model.checkpoint_path,
            'best_accuracy': model.best_accuracy,
            'best_loss': model.best_loss,
            'training_epochs': model.training_epochs,
            'created_at': model.created_at,
            'updated_at': model.updated_at,
            'metadata_json': self._serialize_json_field(model.metadata)
        }
    
    def _from_dict(self, data: Dict[str, Any]) -> SNNModel:
        """Convert dictionary from database to model object."""
        return SNNModel(
            id=data.get('id'),
            experiment_id=data.get('experiment_id'),
            name=data.get('name', ''),
            architecture=data.get('architecture', ''),
            parameters_count=data.get('parameters_count', 0),
            model_config=self._deserialize_json_field(data.get('model_config_json')) or {},
            checkpoint_path=data.get('checkpoint_path'),
            best_accuracy=data.get('best_accuracy'),
            best_loss=data.get('best_loss'),
            training_epochs=data.get('training_epochs', 0),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            metadata=self._deserialize_json_field(data.get('metadata_json')) or {}
        )
    
    def find_by_experiment(self, experiment_id: int) -> List[SNNModel]:
        """Find all models for a specific experiment."""
        return self.find_by({'experiment_id': experiment_id}, order_by='created_at DESC')
    
    def find_by_architecture(self, architecture: str) -> List[SNNModel]:
        """Find models by architecture type."""
        return self.find_by({'architecture': architecture}, order_by='created_at DESC')
    
    def find_best_models_by_accuracy(self, limit: int = 10) -> List[SNNModel]:
        """Find models with highest accuracy."""
        try:
            query = f"""
            SELECT * FROM {self.table_name}
            WHERE best_accuracy IS NOT NULL
            ORDER BY best_accuracy DESC
            LIMIT ?
            """
            
            records = self.db.execute_query(query, [limit], fetch="all")
            return [self._from_dict(record) for record in records or []]
            
        except Exception as e:
            self.logger.error(f"Failed to get best models by accuracy: {e}")
            raise
    
    def find_models_with_checkpoints(self) -> List[SNNModel]:
        """Find models that have saved checkpoints."""
        try:
            query = f"""
            SELECT * FROM {self.table_name}
            WHERE checkpoint_path IS NOT NULL AND checkpoint_path != ''
            ORDER BY updated_at DESC
            """
            
            records = self.db.execute_query(query, fetch="all")
            return [self._from_dict(record) for record in records or []]
            
        except Exception as e:
            self.logger.error(f"Failed to get models with checkpoints: {e}")
            raise
    
    def update_performance_metrics(
        self,
        model_id: int,
        accuracy: Optional[float] = None,
        loss: Optional[float] = None,
        epochs: Optional[int] = None
    ) -> bool:
        """Update model performance metrics."""
        updates = {}
        
        if accuracy is not None:
            updates['best_accuracy'] = accuracy
        if loss is not None:
            updates['best_loss'] = loss
        if epochs is not None:
            updates['training_epochs'] = epochs
        
        if updates:
            return self.update(model_id, updates)
        return True
    
    def update_checkpoint_path(self, model_id: int, checkpoint_path: str) -> bool:
        """Update model checkpoint path."""
        return self.update(model_id, {'checkpoint_path': checkpoint_path})
    
    def get_model_performance_history(self, model_id: int) -> List[Dict[str, Any]]:
        """Get performance history for a model from training runs."""
        try:
            query = """
            SELECT 
                tr.id as training_run_id,
                tr.current_epoch,
                tr.total_epochs,
                tr.best_val_accuracy,
                tr.best_val_loss,
                tr.final_train_accuracy,
                tr.final_train_loss,
                tr.training_time_seconds,
                tr.status,
                tr.started_at,
                tr.completed_at
            FROM training_runs tr
            WHERE tr.model_id = ?
            ORDER BY tr.started_at DESC
            """
            
            records = self.db.execute_query(query, [model_id], fetch="all")
            return records or []
            
        except Exception as e:
            self.logger.error(f"Failed to get performance history for model {model_id}: {e}")
            raise
    
    def find_similar_models(
        self,
        model_id: int,
        similarity_threshold: float = 0.1
    ) -> List[Tuple[SNNModel, float]]:
        """Find models with similar parameter counts and architecture."""
        try:
            model = self.get_by_id(model_id)
            if not model:
                return []
            
            # Find models with similar parameter counts (within threshold)
            param_range = model.parameters_count * similarity_threshold
            min_params = model.parameters_count - param_range
            max_params = model.parameters_count + param_range
            
            query = f"""
            SELECT * FROM {self.table_name}
            WHERE architecture = ? 
            AND parameters_count BETWEEN ? AND ?
            AND id != ?
            ORDER BY ABS(parameters_count - ?) ASC
            """
            
            records = self.db.execute_query(
                query,
                [model.architecture, min_params, max_params, model_id, model.parameters_count],
                fetch="all"
            )
            
            # Calculate similarity scores
            similar_models = []
            for record in records or []:
                similar_model = self._from_dict(record)
                param_diff = abs(similar_model.parameters_count - model.parameters_count)
                similarity_score = 1.0 - (param_diff / model.parameters_count)
                similar_models.append((similar_model, similarity_score))
            
            return similar_models
            
        except Exception as e:
            self.logger.error(f"Failed to find similar models for {model_id}: {e}")
            raise
    
    def get_architecture_statistics(self) -> Dict[str, Any]:
        """Get statistics about model architectures."""
        try:
            arch_query = f"""
            SELECT 
                architecture,
                COUNT(*) as count,
                AVG(parameters_count) as avg_params,
                MAX(best_accuracy) as best_accuracy,
                AVG(best_accuracy) as avg_accuracy
            FROM {self.table_name}
            GROUP BY architecture
            ORDER BY count DESC
            """
            
            arch_stats = self.db.execute_query(arch_query, fetch="all") or []
            
            # Get total model count
            total_count = self.count()
            
            return {
                'total_models': total_count,
                'architecture_breakdown': [
                    {
                        'architecture': row['architecture'],
                        'count': row['count'],
                        'avg_parameters': int(row['avg_params']) if row['avg_params'] else 0,
                        'best_accuracy': row['best_accuracy'],
                        'avg_accuracy': row['avg_accuracy']
                    }
                    for row in arch_stats
                ],
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get architecture statistics: {e}")
            raise
    
    def find_models_by_parameter_range(
        self,
        min_params: Optional[int] = None,
        max_params: Optional[int] = None
    ) -> List[SNNModel]:
        """Find models within parameter count range."""
        try:
            conditions = []
            params = []
            
            if min_params is not None:
                conditions.append("parameters_count >= ?")
                params.append(min_params)
            
            if max_params is not None:
                conditions.append("parameters_count <= ?")
                params.append(max_params)
            
            if not conditions:
                return self.find_all(order_by='parameters_count ASC')
            
            query = f"""
            SELECT * FROM {self.table_name}
            WHERE {' AND '.join(conditions)}
            ORDER BY parameters_count ASC
            """
            
            records = self.db.execute_query(query, params, fetch="all")
            return [self._from_dict(record) for record in records or []]
            
        except Exception as e:
            self.logger.error(f"Failed to find models by parameter range: {e}")
            raise
    
    def get_model_training_summary(self, model_id: int) -> Dict[str, Any]:
        """Get comprehensive training summary for a model."""
        try:
            model = self.get_by_id(model_id)
            if not model:
                return {}
            
            # Get training runs summary
            training_query = """
            SELECT 
                COUNT(*) as total_runs,
                AVG(training_time_seconds) as avg_training_time,
                MAX(best_val_accuracy) as best_val_accuracy,
                MIN(best_val_loss) as best_val_loss,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_runs,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_runs
            FROM training_runs
            WHERE model_id = ?
            """
            
            training_stats = self.db.execute_query(training_query, [model_id], fetch="one")
            
            # Get hardware deployment info
            hardware_query = """
            SELECT 
                COUNT(*) as total_deployments,
                hardware_type,
                AVG(inference_latency_ms) as avg_latency,
                AVG(power_consumption_mw) as avg_power
            FROM hardware_profiles
            WHERE model_id = ?
            GROUP BY hardware_type
            """
            
            hardware_stats = self.db.execute_query(hardware_query, [model_id], fetch="all")
            
            return {
                'model': self._to_dict(model),
                'training_summary': training_stats or {},
                'hardware_deployments': hardware_stats or [],
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get training summary for model {model_id}: {e}")
            raise
    
    def find_models_by_name_pattern(self, pattern: str) -> List[SNNModel]:
        """Find models by name pattern (case-insensitive)."""
        try:
            query = f"""
            SELECT * FROM {self.table_name}
            WHERE LOWER(name) LIKE LOWER(?)
            ORDER BY created_at DESC
            """
            
            search_pattern = f'%{pattern}%'
            records = self.db.execute_query(query, [search_pattern], fetch="all")
            
            return [self._from_dict(record) for record in records or []]
            
        except Exception as e:
            self.logger.error(f"Failed to search models by name pattern '{pattern}': {e}")
            raise
    
    def update_metadata(self, model_id: int, metadata_updates: Dict[str, Any]) -> bool:
        """Update model metadata."""
        try:
            model = self.get_by_id(model_id)
            if not model:
                return False
            
            # Merge metadata
            updated_metadata = model.metadata.copy()
            updated_metadata.update(metadata_updates)
            
            return self.update(model_id, {
                'metadata_json': self._serialize_json_field(updated_metadata)
            })
            
        except Exception as e:
            self.logger.error(f"Failed to update metadata for model {model_id}: {e}")
            return False