"""
Training Repository Implementation

Provides data access operations for training run management with support
for metrics tracking, spike data analysis, and training lifecycle.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

from .base import BaseRepository


@dataclass
class TrainingRun:
    """Training Run model class."""
    id: Optional[int] = None
    model_id: int = None
    dataset_id: int = None
    status: str = "queued"
    config: Dict[str, Any] = None
    current_epoch: int = 0
    total_epochs: int = 100
    best_val_accuracy: Optional[float] = None
    best_val_loss: Optional[float] = None
    final_train_accuracy: Optional[float] = None
    final_train_loss: Optional[float] = None
    training_time_seconds: Optional[float] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    log_file_path: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}
        if self.metrics is None:
            self.metrics = {}


class TrainingRepository(BaseRepository[TrainingRun]):
    """Repository for training run data access operations."""
    
    def _get_table_name(self) -> str:
        return "training_runs"
    
    def _to_dict(self, training_run: TrainingRun) -> Dict[str, Any]:
        """Convert training run object to dictionary for database storage."""
        return {
            'id': training_run.id,
            'model_id': training_run.model_id,
            'dataset_id': training_run.dataset_id,
            'status': training_run.status,
            'config_json': self._serialize_json_field(training_run.config),
            'current_epoch': training_run.current_epoch,
            'total_epochs': training_run.total_epochs,
            'best_val_accuracy': training_run.best_val_accuracy,
            'best_val_loss': training_run.best_val_loss,
            'final_train_accuracy': training_run.final_train_accuracy,
            'final_train_loss': training_run.final_train_loss,
            'training_time_seconds': training_run.training_time_seconds,
            'started_at': training_run.started_at,
            'completed_at': training_run.completed_at,
            'log_file_path': training_run.log_file_path,
            'checkpoint_dir': training_run.checkpoint_dir,
            'metrics_json': self._serialize_json_field(training_run.metrics)
        }
    
    def _from_dict(self, data: Dict[str, Any]) -> TrainingRun:
        """Convert dictionary from database to training run object."""
        return TrainingRun(
            id=data.get('id'),
            model_id=data.get('model_id'),
            dataset_id=data.get('dataset_id'),
            status=data.get('status', 'queued'),
            config=self._deserialize_json_field(data.get('config_json')) or {},
            current_epoch=data.get('current_epoch', 0),
            total_epochs=data.get('total_epochs', 100),
            best_val_accuracy=data.get('best_val_accuracy'),
            best_val_loss=data.get('best_val_loss'),
            final_train_accuracy=data.get('final_train_accuracy'),
            final_train_loss=data.get('final_train_loss'),
            training_time_seconds=data.get('training_time_seconds'),
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            log_file_path=data.get('log_file_path'),
            checkpoint_dir=data.get('checkpoint_dir'),
            metrics=self._deserialize_json_field(data.get('metrics_json')) or {}
        )
    
    def find_by_model(self, model_id: int) -> List[TrainingRun]:
        """Find all training runs for a specific model."""
        return self.find_by({'model_id': model_id}, order_by='started_at DESC')
    
    def find_by_status(self, status: str) -> List[TrainingRun]:
        """Find training runs by status."""
        return self.find_by({'status': status}, order_by='started_at DESC')
    
    def find_active_runs(self) -> List[TrainingRun]:
        """Find currently active training runs."""
        return self.find_by({'status': 'running'}, order_by='started_at DESC')
    
    def find_completed_runs(self, limit: Optional[int] = None) -> List[TrainingRun]:
        """Find completed training runs."""
        return self.find_by(
            {'status': 'completed'}, 
            limit=limit, 
            order_by='completed_at DESC'
        )
    
    def update_progress(
        self,
        run_id: int,
        current_epoch: int,
        metrics_update: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update training progress."""
        updates = {'current_epoch': current_epoch}
        
        if metrics_update:
            # Get current training run to merge metrics
            training_run = self.get_by_id(run_id)
            if training_run:
                updated_metrics = training_run.metrics.copy()
                updated_metrics.update(metrics_update)
                updates['metrics_json'] = self._serialize_json_field(updated_metrics)
        
        return self.update(run_id, updates)
    
    def start_training(self, run_id: int) -> bool:
        """Mark training run as started."""
        return self.update(run_id, {
            'status': 'running',
            'started_at': datetime.now(timezone.utc).isoformat()
        })
    
    def complete_training(
        self,
        run_id: int,
        final_metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Mark training run as completed."""
        updates = {
            'status': 'completed',
            'completed_at': datetime.now(timezone.utc).isoformat()
        }
        
        if final_metrics:
            for key, value in final_metrics.items():
                if hasattr(TrainingRun, key):
                    updates[key] = value
        
        return self.update(run_id, updates)
    
    def fail_training(self, run_id: int, error_message: Optional[str] = None) -> bool:
        """Mark training run as failed."""
        updates = {
            'status': 'failed',
            'completed_at': datetime.now(timezone.utc).isoformat()
        }
        
        if error_message:
            # Add error to metrics
            training_run = self.get_by_id(run_id)
            if training_run:
                updated_metrics = training_run.metrics.copy()
                updated_metrics['error_message'] = error_message
                updates['metrics_json'] = self._serialize_json_field(updated_metrics)
        
        return self.update(run_id, updates)
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training run statistics."""
        try:
            stats_query = f"""
            SELECT 
                status,
                COUNT(*) as count,
                AVG(training_time_seconds) as avg_time,
                AVG(best_val_accuracy) as avg_accuracy
            FROM {self.table_name}
            GROUP BY status
            """
            
            status_stats = self.db.execute_query(stats_query, fetch="all") or []
            
            # Get completion rate
            total_count = self.count()
            completed_count = self.count({'status': 'completed'})
            completion_rate = (completed_count / total_count * 100) if total_count > 0 else 0
            
            return {
                'total_runs': total_count,
                'completion_rate': round(completion_rate, 2),
                'status_breakdown': [
                    {
                        'status': row['status'],
                        'count': row['count'],
                        'avg_training_time': row['avg_time'],
                        'avg_accuracy': row['avg_accuracy']
                    }
                    for row in status_stats
                ],
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get training statistics: {e}")
            raise