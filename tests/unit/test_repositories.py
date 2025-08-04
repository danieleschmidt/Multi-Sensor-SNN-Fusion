"""
Unit tests for repository implementations

Tests all repository classes with comprehensive CRUD operations,
error handling, and neuromorphic-specific functionality.
"""

import pytest
import json
from datetime import datetime, timezone
from typing import Dict, Any

from snn_fusion.repositories import (
    ExperimentRepository, 
    ModelRepository,
    TrainingRepository,
    DatasetRepository,
    HardwareRepository,
    MetricsRepository
)
from snn_fusion.repositories.experiment_repository import Experiment
from snn_fusion.repositories.model_repository import SNNModel
from snn_fusion.repositories.training_repository import TrainingRun
from snn_fusion.repositories.dataset_repository import Dataset
from snn_fusion.repositories.hardware_repository import HardwareProfile
from snn_fusion.repositories.metrics_repository import PerformanceMetric
from snn_fusion.database import DatabaseManager


class TestExperimentRepository:
    """Test suite for ExperimentRepository."""
    
    @pytest.fixture
    def repo(self, test_database: DatabaseManager):
        """Create experiment repository with test database."""
        return ExperimentRepository(test_database)
    
    @pytest.fixture
    def sample_experiment(self):
        """Sample experiment for testing."""
        return Experiment(
            name="Test Experiment",
            description="Test description",
            config={"modalities": ["audio", "vision"]},
            tags=["test", "multimodal"],
            metadata={"research_goal": "testing"}
        )
    
    def test_create_experiment(self, repo: ExperimentRepository, sample_experiment: Experiment):
        """Test experiment creation."""
        # Create experiment
        experiment_id = repo.create(sample_experiment)
        
        assert experiment_id is not None
        assert experiment_id > 0
        
        # Verify creation
        retrieved = repo.get_by_id(experiment_id)
        assert retrieved is not None
        assert retrieved.name == sample_experiment.name
        assert retrieved.description == sample_experiment.description
        assert retrieved.config == sample_experiment.config
        assert retrieved.tags == sample_experiment.tags
    
    def test_get_experiment_by_id(self, repo: ExperimentRepository, sample_experiment: Experiment):
        """Test getting experiment by ID."""
        # Create experiment
        experiment_id = repo.create(sample_experiment)
        
        # Get by ID
        retrieved = repo.get_by_id(experiment_id)
        
        assert retrieved is not None
        assert retrieved.id == experiment_id
        assert retrieved.name == sample_experiment.name
    
    def test_get_nonexistent_experiment(self, repo: ExperimentRepository):
        """Test getting non-existent experiment returns None."""
        result = repo.get_by_id(99999)
        assert result is None
    
    def test_update_experiment(self, repo: ExperimentRepository, sample_experiment: Experiment):
        """Test updating experiment."""
        # Create experiment
        experiment_id = repo.create(sample_experiment)
        
        # Update experiment
        updates = {
            "description": "Updated description",
            "status": "running"
        }
        success = repo.update(experiment_id, updates)
        
        assert success is True
        
        # Verify update
        updated = repo.get_by_id(experiment_id)
        assert updated.description == "Updated description"
        assert updated.status == "running"
    
    def test_delete_experiment(self, repo: ExperimentRepository, sample_experiment: Experiment):
        """Test deleting experiment."""
        # Create experiment
        experiment_id = repo.create(sample_experiment)
        
        # Verify exists
        assert repo.exists(experiment_id) is True
        
        # Delete experiment
        success = repo.delete(experiment_id)
        
        assert success is True
        assert repo.exists(experiment_id) is False
    
    def test_find_by_status(self, repo: ExperimentRepository, sample_experiment: Experiment):
        """Test finding experiments by status."""
        # Create experiments with different statuses
        sample_experiment.status = "created"
        exp1_id = repo.create(sample_experiment)
        
        sample_experiment.name = "Test Experiment 2"
        sample_experiment.status = "running"
        exp2_id = repo.create(sample_experiment)
        
        # Find by status
        created_experiments = repo.find_by_status("created")
        running_experiments = repo.find_by_status("running")
        
        assert len(created_experiments) == 1
        assert len(running_experiments) == 1
        assert created_experiments[0].id == exp1_id
        assert running_experiments[0].id == exp2_id
    
    def test_find_by_tag(self, repo: ExperimentRepository, sample_experiment: Experiment):
        """Test finding experiments by tag."""
        # Create experiments with different tags
        sample_experiment.tags = ["audio", "test"]
        exp1_id = repo.create(sample_experiment)
        
        sample_experiment.name = "Test Experiment 2"
        sample_experiment.tags = ["vision", "test"]
        exp2_id = repo.create(sample_experiment)
        
        # Find by tag
        audio_experiments = repo.find_by_tag("audio")
        test_experiments = repo.find_by_tag("test")
        
        assert len(audio_experiments) == 1
        assert len(test_experiments) == 2
        assert audio_experiments[0].id == exp1_id
    
    def test_start_experiment(self, repo: ExperimentRepository, sample_experiment: Experiment):
        """Test starting an experiment."""
        # Create experiment
        experiment_id = repo.create(sample_experiment)
        
        # Start experiment
        success = repo.start_experiment(experiment_id)
        
        assert success is True
        
        # Verify status change
        updated = repo.get_by_id(experiment_id)
        assert updated.status == "running"
        assert updated.started_at is not None
    
    def test_complete_experiment(self, repo: ExperimentRepository, sample_experiment: Experiment):
        """Test completing an experiment."""
        # Create and start experiment
        experiment_id = repo.create(sample_experiment)
        repo.start_experiment(experiment_id)
        
        # Complete experiment
        success = repo.complete_experiment(experiment_id)
        
        assert success is True
        
        # Verify status change
        completed = repo.get_by_id(experiment_id)
        assert completed.status == "completed"
        assert completed.completed_at is not None
    
    def test_add_tag(self, repo: ExperimentRepository, sample_experiment: Experiment):
        """Test adding tag to experiment."""
        # Create experiment
        experiment_id = repo.create(sample_experiment)
        
        # Add tag
        success = repo.add_tag(experiment_id, "new_tag")
        
        assert success is True
        
        # Verify tag added
        updated = repo.get_by_id(experiment_id)
        assert "new_tag" in updated.tags
    
    def test_remove_tag(self, repo: ExperimentRepository, sample_experiment: Experiment):
        """Test removing tag from experiment."""
        # Create experiment with tags
        sample_experiment.tags = ["tag1", "tag2", "tag3"]
        experiment_id = repo.create(sample_experiment)
        
        # Remove tag
        success = repo.remove_tag(experiment_id, "tag2")
        
        assert success is True
        
        # Verify tag removed
        updated = repo.get_by_id(experiment_id)
        assert "tag2" not in updated.tags
        assert "tag1" in updated.tags
        assert "tag3" in updated.tags
    
    def test_count_experiments(self, repo: ExperimentRepository, sample_experiment: Experiment):
        """Test counting experiments."""
        # Initially no experiments
        assert repo.count() == 0
        
        # Create experiments
        repo.create(sample_experiment)
        sample_experiment.name = "Test Experiment 2"
        repo.create(sample_experiment)
        
        # Count all
        assert repo.count() == 2
        
        # Count with conditions
        assert repo.count({"status": "created"}) == 2


class TestModelRepository:
    """Test suite for ModelRepository."""
    
    @pytest.fixture
    def repo(self, test_database: DatabaseManager):
        """Create model repository with test database."""
        return ModelRepository(test_database)
    
    @pytest.fixture
    def experiment_repo(self, test_database: DatabaseManager):
        """Create experiment repository for dependencies."""
        return ExperimentRepository(test_database)
    
    @pytest.fixture
    def sample_experiment_id(self, experiment_repo: ExperimentRepository):
        """Create sample experiment for model testing."""
        experiment = Experiment(
            name="Test Experiment",
            description="For model testing"
        )
        return experiment_repo.create(experiment)
    
    @pytest.fixture
    def sample_model(self, sample_experiment_id: int):
        """Sample model for testing."""
        return SNNModel(
            experiment_id=sample_experiment_id,
            name="Test MultiModal LSM",
            architecture="MultiModalLSM",
            parameters_count=100000,
            model_config={
                "n_outputs": 10,
                "modality_configs": {
                    "audio": {"n_inputs": 64, "n_reservoir": 200},
                    "vision": {"n_inputs": 128, "n_reservoir": 300}
                }
            }
        )
    
    def test_create_model(self, repo: ModelRepository, sample_model: SNNModel):
        """Test model creation."""
        model_id = repo.create(sample_model)
        
        assert model_id is not None
        assert model_id > 0
        
        # Verify creation
        retrieved = repo.get_by_id(model_id)
        assert retrieved is not None
        assert retrieved.name == sample_model.name
        assert retrieved.architecture == sample_model.architecture
        assert retrieved.parameters_count == sample_model.parameters_count
    
    def test_find_by_experiment(self, repo: ModelRepository, sample_model: SNNModel):
        """Test finding models by experiment."""
        # Create models
        model1_id = repo.create(sample_model)
        
        sample_model.name = "Test Model 2"
        model2_id = repo.create(sample_model)
        
        # Find by experiment
        models = repo.find_by_experiment(sample_model.experiment_id)
        
        assert len(models) == 2
        model_ids = [m.id for m in models]
        assert model1_id in model_ids
        assert model2_id in model_ids
    
    def test_find_by_architecture(self, repo: ModelRepository, sample_model: SNNModel):
        """Test finding models by architecture."""
        # Create models with different architectures
        sample_model.architecture = "MultiModalLSM"
        lsm_id = repo.create(sample_model)
        
        sample_model.name = "Test Hierarchical"
        sample_model.architecture = "HierarchicalFusionSNN"
        hier_id = repo.create(sample_model)
        
        # Find by architecture
        lsm_models = repo.find_by_architecture("MultiModalLSM")
        hier_models = repo.find_by_architecture("HierarchicalFusionSNN")
        
        assert len(lsm_models) == 1
        assert len(hier_models) == 1
        assert lsm_models[0].id == lsm_id
        assert hier_models[0].id == hier_id
    
    def test_update_performance_metrics(self, repo: ModelRepository, sample_model: SNNModel):
        """Test updating performance metrics."""
        model_id = repo.create(sample_model)
        
        # Update metrics
        success = repo.update_performance_metrics(
            model_id,
            accuracy=0.85,
            loss=0.25,
            epochs=100
        )
        
        assert success is True
        
        # Verify update
        updated = repo.get_by_id(model_id)
        assert updated.best_accuracy == 0.85
        assert updated.best_loss == 0.25
        assert updated.training_epochs == 100
    
    def test_find_best_models_by_accuracy(self, repo: ModelRepository, sample_model: SNNModel):
        """Test finding best models by accuracy."""
        # Create models with different accuracies
        sample_model.best_accuracy = 0.75
        repo.create(sample_model)
        
        sample_model.name = "Test Model 2"
        sample_model.best_accuracy = 0.90
        best_id = repo.create(sample_model)
        
        sample_model.name = "Test Model 3"
        sample_model.best_accuracy = 0.80
        repo.create(sample_model)
        
        # Find best models
        best_models = repo.find_best_models_by_accuracy(limit=2)
        
        assert len(best_models) == 2
        assert best_models[0].id == best_id  # Highest accuracy first
        assert best_models[0].best_accuracy == 0.90


class TestTrainingRepository:
    """Test suite for TrainingRepository."""
    
    @pytest.fixture
    def repo(self, test_database: DatabaseManager):
        """Create training repository with test database."""
        return TrainingRepository(test_database)
    
    @pytest.fixture
    def model_repo(self, test_database: DatabaseManager):
        """Create model repository for dependencies."""
        return ModelRepository(test_database)
    
    @pytest.fixture
    def dataset_repo(self, test_database: DatabaseManager):
        """Create dataset repository for dependencies."""
        return DatasetRepository(test_database)
    
    @pytest.fixture
    def sample_model_id(self, model_repo: ModelRepository):
        """Create sample model for training testing."""
        # First need an experiment
        from snn_fusion.repositories import ExperimentRepository
        exp_repo = ExperimentRepository(model_repo.db)
        exp = Experiment(name="Test Exp")
        exp_id = exp_repo.create(exp)
        
        model = SNNModel(
            experiment_id=exp_id,
            name="Test Model",
            architecture="MultiModalLSM",
            parameters_count=50000
        )
        return model_repo.create(model)
    
    @pytest.fixture
    def sample_dataset_id(self, dataset_repo: DatasetRepository):
        """Create sample dataset for training testing."""
        dataset = Dataset(
            name="Test Dataset",
            path="/test/path",
            modalities=["audio", "vision"],
            n_samples=1000
        )
        return dataset_repo.create(dataset)
    
    @pytest.fixture
    def sample_training_run(self, sample_model_id: int, sample_dataset_id: int):
        """Sample training run for testing."""
        return TrainingRun(
            model_id=sample_model_id,
            dataset_id=sample_dataset_id,
            config={
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            total_epochs=100
        )
    
    def test_create_training_run(self, repo: TrainingRepository, sample_training_run: TrainingRun):
        """Test training run creation."""
        run_id = repo.create(sample_training_run)
        
        assert run_id is not None
        assert run_id > 0
        
        # Verify creation
        retrieved = repo.get_by_id(run_id)
        assert retrieved is not None
        assert retrieved.model_id == sample_training_run.model_id
        assert retrieved.dataset_id == sample_training_run.dataset_id
        assert retrieved.status == "queued"  # Default status
    
    def test_find_by_model(self, repo: TrainingRepository, sample_training_run: TrainingRun):
        """Test finding training runs by model."""
        # Create training runs
        run1_id = repo.create(sample_training_run)
        run2_id = repo.create(sample_training_run)
        
        # Find by model
        runs = repo.find_by_model(sample_training_run.model_id)
        
        assert len(runs) == 2
        run_ids = [r.id for r in runs]
        assert run1_id in run_ids
        assert run2_id in run_ids
    
    def test_start_training(self, repo: TrainingRepository, sample_training_run: TrainingRun):
        """Test starting training run."""
        run_id = repo.create(sample_training_run)
        
        # Start training
        success = repo.start_training(run_id)
        
        assert success is True
        
        # Verify status change
        updated = repo.get_by_id(run_id)
        assert updated.status == "running"
        assert updated.started_at is not None
    
    def test_complete_training(self, repo: TrainingRepository, sample_training_run: TrainingRun):
        """Test completing training run."""
        run_id = repo.create(sample_training_run)
        repo.start_training(run_id)
        
        # Complete training
        final_metrics = {
            "best_val_accuracy": 0.85,
            "final_train_loss": 0.25
        }
        success = repo.complete_training(run_id, final_metrics)
        
        assert success is True
        
        # Verify completion
        completed = repo.get_by_id(run_id)
        assert completed.status == "completed"
        assert completed.completed_at is not None
        assert completed.best_val_accuracy == 0.85
        assert completed.final_train_loss == 0.25
    
    def test_update_progress(self, repo: TrainingRepository, sample_training_run: TrainingRun):
        """Test updating training progress."""
        run_id = repo.create(sample_training_run)
        
        # Update progress
        metrics_update = {"current_loss": 0.5, "current_accuracy": 0.75}
        success = repo.update_progress(run_id, 25, metrics_update)
        
        assert success is True
        
        # Verify update
        updated = repo.get_by_id(run_id)
        assert updated.current_epoch == 25
        assert "current_loss" in updated.metrics
        assert updated.metrics["current_loss"] == 0.5


# Similar comprehensive test classes for other repositories...
class TestDatasetRepository:
    """Test suite for DatasetRepository."""
    
    @pytest.fixture
    def repo(self, test_database: DatabaseManager):
        return DatasetRepository(test_database)
    
    @pytest.fixture
    def sample_dataset(self):
        return Dataset(
            name="MAVEN-Test",
            path="/data/maven_test",
            modalities=["audio", "vision"],
            n_samples=10000,
            format="hdf5"
        )
    
    def test_create_dataset(self, repo: DatasetRepository, sample_dataset: Dataset):
        """Test dataset creation."""
        dataset_id = repo.create(sample_dataset)
        
        assert dataset_id is not None
        retrieved = repo.get_by_id(dataset_id)
        assert retrieved.name == sample_dataset.name
        assert retrieved.modalities == sample_dataset.modalities


class TestHardwareRepository:
    """Test suite for HardwareRepository."""
    
    @pytest.fixture
    def repo(self, test_database: DatabaseManager):
        return HardwareRepository(test_database)
    
    @pytest.fixture
    def sample_profile(self):
        return HardwareProfile(
            name="Test_Loihi_Profile",
            hardware_type="loihi2",
            deployment_config={"cores": 4, "neurons_per_core": 1024}
        )
    
    def test_create_hardware_profile(self, repo: HardwareRepository, sample_profile: HardwareProfile):
        """Test hardware profile creation."""
        profile_id = repo.create(sample_profile)
        
        assert profile_id is not None
        retrieved = repo.get_by_id(profile_id)
        assert retrieved.name == sample_profile.name
        assert retrieved.hardware_type == sample_profile.hardware_type


class TestMetricsRepository:
    """Test suite for MetricsRepository."""
    
    @pytest.fixture
    def repo(self, test_database: DatabaseManager):
        return MetricsRepository(test_database)
    
    @pytest.fixture
    def sample_metric(self):
        return PerformanceMetric(
            metric_name="accuracy",
            metric_value=0.85,
            metric_unit="%",
            measurement_context="validation_epoch_50"
        )
    
    def test_create_metric(self, repo: MetricsRepository, sample_metric: PerformanceMetric):
        """Test performance metric creation."""
        metric_id = repo.create(sample_metric)
        
        assert metric_id is not None
        retrieved = repo.get_by_id(metric_id)
        assert retrieved.metric_name == sample_metric.metric_name
        assert retrieved.metric_value == sample_metric.metric_value


# Integration tests for repository interactions
class TestRepositoryIntegration:
    """Integration tests for repository interactions."""
    
    @pytest.fixture
    def repositories(self, test_database: DatabaseManager):
        """Create all repositories with shared database."""
        return {
            'experiment': ExperimentRepository(test_database),
            'model': ModelRepository(test_database),
            'training': TrainingRepository(test_database),
            'dataset': DatasetRepository(test_database),
            'hardware': HardwareRepository(test_database),
            'metrics': MetricsRepository(test_database)
        }
    
    def test_experiment_model_relationship(self, repositories: Dict[str, Any]):
        """Test relationship between experiments and models."""
        exp_repo = repositories['experiment']
        model_repo = repositories['model']
        
        # Create experiment
        experiment = Experiment(name="Integration Test")
        exp_id = exp_repo.create(experiment)
        
        # Create model for experiment
        model = SNNModel(
            experiment_id=exp_id,
            name="Test Model",
            architecture="MultiModalLSM",
            parameters_count=100000
        )
        model_id = model_repo.create(model)
        
        # Verify relationship
        models = model_repo.find_by_experiment(exp_id)
        assert len(models) == 1
        assert models[0].id == model_id
        assert models[0].experiment_id == exp_id
    
    def test_model_training_hardware_pipeline(self, repositories: Dict[str, Any]):
        """Test complete pipeline from model to training to hardware deployment."""
        exp_repo = repositories['experiment']
        model_repo = repositories['model']
        dataset_repo = repositories['dataset']
        training_repo = repositories['training']
        hardware_repo = repositories['hardware']
        
        # Create experiment
        experiment = Experiment(name="Pipeline Test")
        exp_id = exp_repo.create(experiment)
        
        # Create model
        model = SNNModel(
            experiment_id=exp_id,
            name="Pipeline Model",
            architecture="MultiModalLSM",
            parameters_count=150000
        )
        model_id = model_repo.create(model)
        
        # Create dataset
        dataset = Dataset(
            name="Pipeline Dataset",
            path="/data/pipeline",
            modalities=["audio", "vision"],
            n_samples=5000
        )
        dataset_id = dataset_repo.create(dataset)
        
        # Create training run
        training_run = TrainingRun(
            model_id=model_id,
            dataset_id=dataset_id,
            config={"epochs": 50}
        )
        run_id = training_repo.create(training_run)
        
        # Complete training
        training_repo.start_training(run_id)
        training_repo.complete_training(run_id, {"best_val_accuracy": 0.88})
        
        # Deploy to hardware
        hardware_profile = HardwareProfile(
            name="Pipeline_Deployment",
            hardware_type="loihi2",
            model_id=model_id,
            deployment_config={"optimization": True}
        )
        profile_id = hardware_repo.create(hardware_profile)
        
        # Verify complete pipeline
        assert exp_id is not None
        assert model_id is not None
        assert dataset_id is not None
        assert run_id is not None
        assert profile_id is not None
        
        # Verify relationships
        models = model_repo.find_by_experiment(exp_id)
        assert len(models) == 1
        
        training_runs = training_repo.find_by_model(model_id)
        assert len(training_runs) == 1
        assert training_runs[0].status == "completed"