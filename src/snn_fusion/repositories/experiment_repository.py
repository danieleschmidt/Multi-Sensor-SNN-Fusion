"""
Experiment Repository Implementation

Provides data access operations for experiment management in SNN-Fusion
framework with support for complex queries and experiment lifecycle.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

from .base import BaseRepository


@dataclass
class Experiment:
    """Experiment model class."""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    config: Dict[str, Any] = None
    status: str = "created"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class ExperimentRepository(BaseRepository[Experiment]):
    """Repository for experiment data access operations."""
    
    def _get_table_name(self) -> str:
        return "experiments"
    
    def _to_dict(self, experiment: Experiment) -> Dict[str, Any]:
        """Convert experiment object to dictionary for database storage."""
        return {
            'id': experiment.id,
            'name': experiment.name,
            'description': experiment.description,
            'config_json': self._serialize_json_field(experiment.config),
            'status': experiment.status,
            'created_at': experiment.created_at,
            'updated_at': experiment.updated_at,
            'started_at': experiment.started_at,
            'completed_at': experiment.completed_at,
            'tags': self._serialize_json_field(experiment.tags),
            'metadata_json': self._serialize_json_field(experiment.metadata)
        }
    
    def _from_dict(self, data: Dict[str, Any]) -> Experiment:
        """Convert dictionary from database to experiment object."""
        return Experiment(
            id=data.get('id'),
            name=data.get('name', ''),
            description=data.get('description', ''),
            config=self._deserialize_json_field(data.get('config_json')) or {},
            status=data.get('status', 'created'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at'),
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            tags=self._deserialize_json_field(data.get('tags')) or [],
            metadata=self._deserialize_json_field(data.get('metadata_json')) or {}
        )
    
    def find_by_status(self, status: str) -> List[Experiment]:
        """Find experiments by status."""
        return self.find_by({'status': status})
    
    def find_by_tag(self, tag: str) -> List[Experiment]:
        """Find experiments containing a specific tag."""
        try:
            # For JSON field searches, we need custom SQL
            query = f"""
            SELECT * FROM {self.table_name}
            WHERE tags LIKE ?
            ORDER BY created_at DESC
            """
            
            # Use LIKE to search within JSON array
            search_pattern = f'%"{tag}"%'
            records = self.db.execute_query(query, [search_pattern], fetch="all")
            
            return [self._from_dict(record) for record in records or []]
            
        except Exception as e:
            self.logger.error(f"Failed to search experiments by tag '{tag}': {e}")
            raise
    
    def find_active_experiments(self) -> List[Experiment]:
        """Find experiments that are currently running."""
        return self.find_by({'status': 'running'})
    
    def find_recent_experiments(self, limit: int = 10) -> List[Experiment]:
        """Find most recently created experiments."""
        return self.find_all(limit=limit, order_by='created_at DESC')
    
    def update_status(
        self, 
        experiment_id: int, 
        new_status: str, 
        timestamp_field: Optional[str] = None
    ) -> bool:
        """
        Update experiment status with optional timestamp.
        
        Args:
            experiment_id: Experiment ID
            new_status: New status value
            timestamp_field: Optional timestamp field to update ('started_at', 'completed_at')
            
        Returns:
            Success status
        """
        updates = {'status': new_status}
        
        if timestamp_field:
            updates[timestamp_field] = datetime.now(timezone.utc).isoformat()
        
        return self.update(experiment_id, updates)
    
    def start_experiment(self, experiment_id: int) -> bool:
        """Mark experiment as started."""
        return self.update_status(
            experiment_id, 
            'running', 
            timestamp_field='started_at'
        )
    
    def complete_experiment(self, experiment_id: int) -> bool:
        """Mark experiment as completed."""
        return self.update_status(
            experiment_id, 
            'completed', 
            timestamp_field='completed_at'
        )
    
    def fail_experiment(self, experiment_id: int) -> bool:
        """Mark experiment as failed."""
        return self.update_status(experiment_id, 'failed')
    
    def add_tag(self, experiment_id: int, tag: str) -> bool:
        """Add a tag to an experiment."""
        try:
            experiment = self.get_by_id(experiment_id)
            if not experiment:
                return False
            
            if tag not in experiment.tags:
                experiment.tags.append(tag)
                return self.update(experiment_id, {
                    'tags': self._serialize_json_field(experiment.tags)
                })
            
            return True  # Tag already exists
            
        except Exception as e:
            self.logger.error(f"Failed to add tag '{tag}' to experiment {experiment_id}: {e}")
            return False
    
    def remove_tag(self, experiment_id: int, tag: str) -> bool:
        """Remove a tag from an experiment."""
        try:
            experiment = self.get_by_id(experiment_id)
            if not experiment:
                return False
            
            if tag in experiment.tags:
                experiment.tags.remove(tag)
                return self.update(experiment_id, {
                    'tags': self._serialize_json_field(experiment.tags)
                })
            
            return True  # Tag doesn't exist
            
        except Exception as e:
            self.logger.error(f"Failed to remove tag '{tag}' from experiment {experiment_id}: {e}")
            return False
    
    def update_metadata(self, experiment_id: int, metadata_updates: Dict[str, Any]) -> bool:
        """Update experiment metadata."""
        try:
            experiment = self.get_by_id(experiment_id)
            if not experiment:
                return False
            
            # Merge metadata
            updated_metadata = experiment.metadata.copy()
            updated_metadata.update(metadata_updates)
            
            return self.update(experiment_id, {
                'metadata_json': self._serialize_json_field(updated_metadata)
            })
            
        except Exception as e:
            self.logger.error(f"Failed to update metadata for experiment {experiment_id}: {e}")
            return False
    
    def find_by_name_pattern(self, pattern: str) -> List[Experiment]:
        """Find experiments by name pattern (case-insensitive)."""
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
            self.logger.error(f"Failed to search experiments by name pattern '{pattern}': {e}")
            raise
    
    def get_experiment_statistics(self) -> Dict[str, Any]:
        """Get experiment statistics summary."""
        try:
            stats_query = f"""
            SELECT 
                status,
                COUNT(*) as count
            FROM {self.table_name}
            GROUP BY status
            """
            
            status_counts = self.db.execute_query(stats_query, fetch="all") or []
            
            # Get total count
            total_count = self.count()
            
            # Get recent activity (last 30 days)
            recent_query = f"""
            SELECT COUNT(*) as count
            FROM {self.table_name}
            WHERE created_at >= date('now', '-30 days')
            """
            
            recent_result = self.db.execute_query(recent_query, fetch="one")
            recent_count = recent_result['count'] if recent_result else 0
            
            return {
                'total_experiments': total_count,
                'recent_experiments_30_days': recent_count,
                'status_breakdown': {
                    row['status']: row['count'] 
                    for row in status_counts
                },
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment statistics: {e}")
            raise
    
    def find_experiments_with_config_value(
        self, 
        config_key: str, 
        config_value: Any
    ) -> List[Experiment]:
        """Find experiments with specific configuration value."""
        try:
            # For JSON field searches in config
            query = f"""
            SELECT * FROM {self.table_name}
            WHERE config_json LIKE ?
            ORDER BY created_at DESC
            """
            
            # Simple pattern matching for JSON (can be enhanced with JSON functions)
            search_pattern = f'%"{config_key}": "{config_value}"%'
            records = self.db.execute_query(query, [search_pattern], fetch="all")
            
            # Filter results more precisely
            experiments = [self._from_dict(record) for record in records or []]
            
            # Additional filtering to ensure exact match
            filtered_experiments = []
            for exp in experiments:
                if config_key in exp.config and exp.config[config_key] == config_value:
                    filtered_experiments.append(exp)
            
            return filtered_experiments
            
        except Exception as e:
            self.logger.error(
                f"Failed to search experiments by config {config_key}={config_value}: {e}"
            )
            raise
    
    def get_experiment_with_models_count(self, experiment_id: int) -> Optional[Tuple[Experiment, int]]:
        """Get experiment with count of associated models."""
        try:
            experiment = self.get_by_id(experiment_id)
            if not experiment:
                return None
            
            # Count associated models
            count_query = """
            SELECT COUNT(*) as count
            FROM models
            WHERE experiment_id = ?
            """
            
            result = self.db.execute_query(count_query, [experiment_id], fetch="one")
            model_count = result['count'] if result else 0
            
            return experiment, model_count
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment {experiment_id} with models count: {e}")
            raise