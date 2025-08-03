"""
Database Layer for SNN-Fusion

Implements data persistence, model storage, and experimental tracking
for neuromorphic multi-modal datasets and training results.
"""

from .connection import DatabaseManager, get_database
from .models import (
    ExperimentRecord,
    ModelCheckpoint,
    DatasetMetadata,
    TrainingRun,
    HardwareProfile,
)
from .migrations import run_migrations, create_tables
from .schemas import (
    experiment_schema,
    dataset_schema,
    model_schema,
    training_schema,
)

__all__ = [
    # Database management
    "DatabaseManager",
    "get_database",
    # Data models
    "ExperimentRecord",
    "ModelCheckpoint", 
    "DatasetMetadata",
    "TrainingRun",
    "HardwareProfile",
    # Schema management
    "run_migrations",
    "create_tables",
    "experiment_schema",
    "dataset_schema",
    "model_schema",
    "training_schema",
]