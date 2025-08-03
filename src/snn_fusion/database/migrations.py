"""
Database Schema Migrations

Defines database schema creation and evolution for SNN-Fusion
experimental tracking and model storage.
"""

from typing import Dict, OrderedDict
from collections import OrderedDict


def get_migrations(db_type: str = "sqlite") -> OrderedDict[str, str]:
    """
    Get database migrations for the specified database type.
    
    Args:
        db_type: Database type ('sqlite' or 'postgresql')
        
    Returns:
        Ordered dictionary of migration name -> SQL
    """
    if db_type == "sqlite":
        return _get_sqlite_migrations()
    elif db_type == "postgresql":
        return _get_postgresql_migrations()
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def _get_sqlite_migrations() -> OrderedDict[str, str]:
    """SQLite-specific migrations."""
    migrations = OrderedDict()
    
    # Initial schema creation
    migrations["001_create_experiments_table"] = """
    CREATE TABLE experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(255) NOT NULL,
        description TEXT,
        config_json TEXT NOT NULL,
        status VARCHAR(50) DEFAULT 'created',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        started_at TIMESTAMP NULL,
        completed_at TIMESTAMP NULL,
        tags TEXT,  -- JSON array of tags
        metadata_json TEXT,  -- Additional metadata
        UNIQUE(name)
    )
    """
    
    migrations["002_create_datasets_table"] = """
    CREATE TABLE datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(255) NOT NULL,
        path VARCHAR(1000) NOT NULL,
        modalities TEXT NOT NULL,  -- JSON array
        n_samples INTEGER NOT NULL,
        sample_rate REAL,
        sequence_length INTEGER,
        format VARCHAR(100),
        size_bytes INTEGER,
        checksum VARCHAR(64),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata_json TEXT,
        UNIQUE(name)
    )
    """
    
    migrations["003_create_models_table"] = """
    CREATE TABLE models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER NOT NULL,
        name VARCHAR(255) NOT NULL,
        architecture VARCHAR(100) NOT NULL,
        parameters_count INTEGER,
        model_config_json TEXT NOT NULL,
        checkpoint_path VARCHAR(1000),
        best_accuracy REAL,
        best_loss REAL,
        training_epochs INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata_json TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
    )
    """
    
    migrations["004_create_training_runs_table"] = """
    CREATE TABLE training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER NOT NULL,
        dataset_id INTEGER NOT NULL,
        status VARCHAR(50) DEFAULT 'started',
        config_json TEXT NOT NULL,
        current_epoch INTEGER DEFAULT 0,
        total_epochs INTEGER NOT NULL,
        best_val_accuracy REAL,
        best_val_loss REAL,
        final_train_accuracy REAL,
        final_train_loss REAL,
        training_time_seconds REAL,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP NULL,
        log_file_path VARCHAR(1000),
        checkpoint_dir VARCHAR(1000),
        metrics_json TEXT,  -- Training metrics history
        FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE,
        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
    )
    """
    
    migrations["005_create_hardware_profiles_table"] = """
    CREATE TABLE hardware_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(255) NOT NULL,
        hardware_type VARCHAR(100) NOT NULL,  -- loihi2, akida, spinnaker, gpu, cpu
        model_id INTEGER,
        deployment_config_json TEXT NOT NULL,
        inference_latency_ms REAL,
        power_consumption_mw REAL,
        accuracy_deployed REAL,
        memory_usage_mb REAL,
        throughput_samples_per_sec REAL,
        optimization_applied BOOLEAN DEFAULT FALSE,
        deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        benchmark_results_json TEXT,
        FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE SET NULL,
        UNIQUE(name)
    )
    """
    
    migrations["006_create_spike_data_table"] = """
    CREATE TABLE spike_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        training_run_id INTEGER NOT NULL,
        epoch INTEGER NOT NULL,
        batch_idx INTEGER NOT NULL,
        modality VARCHAR(100) NOT NULL,
        firing_rate REAL NOT NULL,
        spike_count INTEGER NOT NULL,
        synchrony_index REAL,
        cv_isi REAL,  -- Coefficient of variation of inter-spike intervals
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        spike_patterns_path VARCHAR(1000),  -- Path to detailed spike data
        FOREIGN KEY (training_run_id) REFERENCES training_runs (id) ON DELETE CASCADE
    )
    """
    
    migrations["007_create_performance_metrics_table"] = """
    CREATE TABLE performance_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        training_run_id INTEGER,
        hardware_profile_id INTEGER,
        metric_name VARCHAR(100) NOT NULL,
        metric_value REAL NOT NULL,
        metric_unit VARCHAR(50),
        measurement_context VARCHAR(200),  -- e.g., 'validation_epoch_50', 'inference_batch_32'
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata_json TEXT,
        FOREIGN KEY (training_run_id) REFERENCES training_runs (id) ON DELETE CASCADE,
        FOREIGN KEY (hardware_profile_id) REFERENCES hardware_profiles (id) ON DELETE CASCADE
    )
    """
    
    migrations["008_create_model_artifacts_table"] = """
    CREATE TABLE model_artifacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER NOT NULL,
        artifact_type VARCHAR(100) NOT NULL,  -- checkpoint, weights, config, visualization
        file_path VARCHAR(1000) NOT NULL,
        file_size_bytes INTEGER,
        checksum VARCHAR(64),
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata_json TEXT,
        FOREIGN KEY (model_id) REFERENCES models (id) ON DELETE CASCADE
    )
    """
    
    migrations["009_create_indexes"] = """
    CREATE INDEX idx_experiments_status ON experiments(status);
    CREATE INDEX idx_experiments_created_at ON experiments(created_at);
    CREATE INDEX idx_models_experiment_id ON models(experiment_id);
    CREATE INDEX idx_training_runs_model_id ON training_runs(model_id);
    CREATE INDEX idx_training_runs_status ON training_runs(status);
    CREATE INDEX idx_hardware_profiles_hardware_type ON hardware_profiles(hardware_type);
    CREATE INDEX idx_spike_data_training_run_id ON spike_data(training_run_id);
    CREATE INDEX idx_spike_data_modality ON spike_data(modality);
    CREATE INDEX idx_performance_metrics_training_run_id ON performance_metrics(training_run_id);
    CREATE INDEX idx_performance_metrics_metric_name ON performance_metrics(metric_name);
    CREATE INDEX idx_model_artifacts_model_id ON model_artifacts(model_id);
    CREATE INDEX idx_model_artifacts_artifact_type ON model_artifacts(artifact_type);
    """
    
    return migrations


def _get_postgresql_migrations() -> OrderedDict[str, str]:
    """PostgreSQL-specific migrations."""
    migrations = OrderedDict()
    
    # PostgreSQL uses similar structure but with some differences
    migrations["001_create_experiments_table"] = """
    CREATE TABLE experiments (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL UNIQUE,
        description TEXT,
        config_json JSONB NOT NULL,
        status VARCHAR(50) DEFAULT 'created',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        started_at TIMESTAMP WITH TIME ZONE NULL,
        completed_at TIMESTAMP WITH TIME ZONE NULL,
        tags JSONB,  -- JSON array of tags
        metadata_json JSONB  -- Additional metadata
    )
    """
    
    migrations["002_create_datasets_table"] = """
    CREATE TABLE datasets (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL UNIQUE,
        path VARCHAR(1000) NOT NULL,
        modalities JSONB NOT NULL,  -- JSON array
        n_samples INTEGER NOT NULL,
        sample_rate REAL,
        sequence_length INTEGER,
        format VARCHAR(100),
        size_bytes BIGINT,
        checksum VARCHAR(64),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        metadata_json JSONB
    )
    """
    
    migrations["003_create_models_table"] = """
    CREATE TABLE models (
        id SERIAL PRIMARY KEY,
        experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
        name VARCHAR(255) NOT NULL,
        architecture VARCHAR(100) NOT NULL,
        parameters_count INTEGER,
        model_config_json JSONB NOT NULL,
        checkpoint_path VARCHAR(1000),
        best_accuracy REAL,
        best_loss REAL,
        training_epochs INTEGER DEFAULT 0,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        metadata_json JSONB
    )
    """
    
    migrations["004_create_training_runs_table"] = """
    CREATE TABLE training_runs (
        id SERIAL PRIMARY KEY,
        model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
        dataset_id INTEGER NOT NULL REFERENCES datasets(id),
        status VARCHAR(50) DEFAULT 'started',
        config_json JSONB NOT NULL,
        current_epoch INTEGER DEFAULT 0,
        total_epochs INTEGER NOT NULL,
        best_val_accuracy REAL,
        best_val_loss REAL,
        final_train_accuracy REAL,
        final_train_loss REAL,
        training_time_seconds REAL,
        started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP WITH TIME ZONE NULL,
        log_file_path VARCHAR(1000),
        checkpoint_dir VARCHAR(1000),
        metrics_json JSONB  -- Training metrics history
    )
    """
    
    migrations["005_create_hardware_profiles_table"] = """
    CREATE TABLE hardware_profiles (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255) NOT NULL UNIQUE,
        hardware_type VARCHAR(100) NOT NULL,  -- loihi2, akida, spinnaker, gpu, cpu
        model_id INTEGER REFERENCES models(id) ON DELETE SET NULL,
        deployment_config_json JSONB NOT NULL,
        inference_latency_ms REAL,
        power_consumption_mw REAL,
        accuracy_deployed REAL,
        memory_usage_mb REAL,
        throughput_samples_per_sec REAL,
        optimization_applied BOOLEAN DEFAULT FALSE,
        deployed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        benchmark_results_json JSONB
    )
    """
    
    migrations["006_create_spike_data_table"] = """
    CREATE TABLE spike_data (
        id SERIAL PRIMARY KEY,
        training_run_id INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
        epoch INTEGER NOT NULL,
        batch_idx INTEGER NOT NULL,
        modality VARCHAR(100) NOT NULL,
        firing_rate REAL NOT NULL,
        spike_count INTEGER NOT NULL,
        synchrony_index REAL,
        cv_isi REAL,  -- Coefficient of variation of inter-spike intervals
        recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        spike_patterns_path VARCHAR(1000)  -- Path to detailed spike data
    )
    """
    
    migrations["007_create_performance_metrics_table"] = """
    CREATE TABLE performance_metrics (
        id SERIAL PRIMARY KEY,
        training_run_id INTEGER REFERENCES training_runs(id) ON DELETE CASCADE,
        hardware_profile_id INTEGER REFERENCES hardware_profiles(id) ON DELETE CASCADE,
        metric_name VARCHAR(100) NOT NULL,
        metric_value REAL NOT NULL,
        metric_unit VARCHAR(50),
        measurement_context VARCHAR(200),  -- e.g., 'validation_epoch_50', 'inference_batch_32'
        recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        metadata_json JSONB
    )
    """
    
    migrations["008_create_model_artifacts_table"] = """
    CREATE TABLE model_artifacts (
        id SERIAL PRIMARY KEY,
        model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
        artifact_type VARCHAR(100) NOT NULL,  -- checkpoint, weights, config, visualization
        file_path VARCHAR(1000) NOT NULL,
        file_size_bytes BIGINT,
        checksum VARCHAR(64),
        description TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        metadata_json JSONB
    )
    """
    
    migrations["009_create_indexes"] = """
    CREATE INDEX idx_experiments_status ON experiments(status);
    CREATE INDEX idx_experiments_created_at ON experiments(created_at);
    CREATE INDEX idx_experiments_tags ON experiments USING GIN(tags);
    CREATE INDEX idx_models_experiment_id ON models(experiment_id);
    CREATE INDEX idx_models_config ON models USING GIN(model_config_json);
    CREATE INDEX idx_training_runs_model_id ON training_runs(model_id);
    CREATE INDEX idx_training_runs_status ON training_runs(status);
    CREATE INDEX idx_hardware_profiles_hardware_type ON hardware_profiles(hardware_type);
    CREATE INDEX idx_spike_data_training_run_id ON spike_data(training_run_id);
    CREATE INDEX idx_spike_data_modality ON spike_data(modality);
    CREATE INDEX idx_performance_metrics_training_run_id ON performance_metrics(training_run_id);
    CREATE INDEX idx_performance_metrics_metric_name ON performance_metrics(metric_name);
    CREATE INDEX idx_model_artifacts_model_id ON model_artifacts(model_id);
    CREATE INDEX idx_model_artifacts_artifact_type ON model_artifacts(artifact_type);
    """
    
    return migrations


def create_tables(db_manager) -> None:
    """
    Create all tables using the database manager.
    
    Args:
        db_manager: DatabaseManager instance
    """
    db_manager.run_migrations()