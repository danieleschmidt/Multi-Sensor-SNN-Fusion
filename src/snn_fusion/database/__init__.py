"""
Database Layer for SNN-Fusion

Implements data persistence, model storage, and experimental tracking
for neuromorphic multi-modal datasets and training results.
"""

from .connection import DatabaseManager, get_database
from .migrations import run_migrations, create_tables

__all__ = [
    # Database management
    "DatabaseManager",
    "get_database",
    # Schema management
    "run_migrations",
    "create_tables",
]