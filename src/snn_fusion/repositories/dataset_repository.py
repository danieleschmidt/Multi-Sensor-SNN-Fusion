"""
Dataset Repository Implementation

Provides data access operations for dataset management with support
for multi-modal dataset tracking and metadata management.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass

from .base import BaseRepository


@dataclass 
class Dataset:
    """Dataset model class."""
    id: Optional[int] = None
    name: str = ""
    path: str = ""
    modalities: List[str] = None
    n_samples: int = 0
    sample_rate: Optional[float] = None
    sequence_length: Optional[int] = None
    format: str = ""
    size_bytes: int = 0
    checksum: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = []
        if self.metadata is None:
            self.metadata = {}


class DatasetRepository(BaseRepository[Dataset]):
    """Repository for dataset data access operations."""
    
    def _get_table_name(self) -> str:
        return "datasets"
    
    def _to_dict(self, dataset: Dataset) -> Dict[str, Any]:
        """Convert dataset object to dictionary for database storage."""
        return {
            'id': dataset.id,
            'name': dataset.name,
            'path': dataset.path,
            'modalities': self._serialize_json_field(dataset.modalities),
            'n_samples': dataset.n_samples,
            'sample_rate': dataset.sample_rate,
            'sequence_length': dataset.sequence_length,
            'format': dataset.format,
            'size_bytes': dataset.size_bytes,
            'checksum': dataset.checksum,
            'created_at': dataset.created_at,
            'metadata_json': self._serialize_json_field(dataset.metadata)
        }
    
    def _from_dict(self, data: Dict[str, Any]) -> Dataset:
        """Convert dictionary from database to dataset object."""
        return Dataset(
            id=data.get('id'),
            name=data.get('name', ''),
            path=data.get('path', ''),
            modalities=self._deserialize_json_field(data.get('modalities')) or [],
            n_samples=data.get('n_samples', 0),
            sample_rate=data.get('sample_rate'),
            sequence_length=data.get('sequence_length'),
            format=data.get('format', ''),
            size_bytes=data.get('size_bytes', 0),
            checksum=data.get('checksum'),
            created_at=data.get('created_at'),
            metadata=self._deserialize_json_field(data.get('metadata_json')) or {}
        )