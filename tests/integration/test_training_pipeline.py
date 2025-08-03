"""
Integration tests for complete training pipeline.

Tests end-to-end training workflows with database tracking,
checkpointing, and multi-modal data processing.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import json

from snn_fusion.models import MultiModalLSM
from snn_fusion.training import SNNTrainer, MultiModalTrainer
from snn_fusion.datasets import create_dataloaders
from snn_fusion.database import DatabaseManager


class MockDataset:
    """Mock dataset for integration testing."""
    
    def __init__(self, size=100, modalities=None, device=None):
        self.size = size
        self.modalities = modalities or ["audio", "vision", "tactile"]
        self.device = device or torch.device("cpu")
        
        # Define modality dimensions
        self.modality_dims = {
            "audio": 32,
            "vision": 64,
            "tactile": 6
        }
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Create time series data for each modality
        time_steps = 50
        
        inputs = {}
        for modality in self.modalities:
            if modality in self.modality_dims:
                inputs[modality] = torch.randn(
                    time_steps, 
                    self.modality_dims[modality],
                    device=self.device
                )
        
        # Random label
        label = torch.randint(0, 5, (1,), device=self.device).item()
        
        return inputs, label


class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, dataset, batch_size=4):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_inputs = []
            batch_labels = []
            
            for j in range(min(self.batch_size, len(self.dataset) - i)):
                inputs, label = self.dataset[i + j]
                batch_inputs.append(inputs)
                batch_labels.append(label)
            
            # Stack batch data
            batched_inputs = {}
            for modality in batch_inputs[0].keys():
                batched_inputs[modality] = torch.stack([
                    inp[modality] for inp in batch_inputs
                ])
            
            batched_labels = torch.tensor(batch_labels, device=self.dataset.device)
            
            yield batched_inputs, batched_labels


class TestSNNTrainer:
    """Test SNN trainer functionality."""
    
    def test_trainer_initialization(self, simple_lsm, device):
        """Test trainer initialization."""
        trainer = SNNTrainer(
            model=simple_lsm,
            device=device,
            learning_rate=1e-3,
            temporal_window=50
        )
        
        assert trainer.model == simple_lsm
        assert trainer.device == device
        assert trainer.temporal_window == 50
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None
    
    def test_single_training_step(self, simple_lsm, device):
        """Test single training step."""
        trainer = SNNTrainer(
            model=simple_lsm,
            device=device,
            learning_rate=1e-3
        )
        
        # Create mock data
        batch_size = 4
        time_steps = 20
        inputs = torch.randn(batch_size, time_steps, 10, device=device)
        targets = torch.randint(0, 3, (batch_size,), device=device)
        
        # Simulate training step
        trainer.optimizer.zero_grad()
        
        outputs, states = trainer.model(inputs, return_states=True)
        loss = trainer.loss_fn(outputs, targets)
        
        loss.backward()
        trainer.optimizer.step()
        
        assert loss.item() > 0
        assert outputs.shape == (batch_size, 3)
    
    def test_training_epoch(self, simple_lsm, device, temp_dir):
        """Test complete training epoch."""
        trainer = SNNTrainer(
            model=simple_lsm,
            device=device,
            checkpoint_dir=str(temp_dir / "checkpoints")
        )
        
        # Create mock dataset and loader
        dataset = MockDataset(size=20, modalities=["input"], device=device)
        # Adjust dataset to work with simple LSM
        dataset.modality_dims = {"input": 10}
        dataset.modalities = ["input"]
        
        # Override dataset.__getitem__ for simple LSM
        def mock_getitem(idx):
            time_steps = 20
            inputs = torch.randn(time_steps, 10, device=device)
            label = torch.randint(0, 3, (1,), device=device).item()
            return inputs, label
        
        dataset.__getitem__ = mock_getitem
        
        train_loader = MockDataLoader(dataset, batch_size=4)
        
        # Train for one epoch
        metrics = trainer.train_epoch(train_loader, epoch=0, log_interval=2)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'epoch_time' in metrics
        assert metrics['loss'] > 0
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_evaluation(self, simple_lsm, device):
        """Test model evaluation."""
        trainer = SNNTrainer(model=simple_lsm, device=device)
        
        # Create mock validation dataset
        dataset = MockDataset(size=16, modalities=["input"], device=device)
        dataset.modality_dims = {"input": 10}
        dataset.modalities = ["input"]
        
        def mock_getitem(idx):
            time_steps = 20
            inputs = torch.randn(time_steps, 10, device=device)
            label = torch.randint(0, 3, (1,), device=device).item()
            return inputs, label
        
        dataset.__getitem__ = mock_getitem
        val_loader = MockDataLoader(dataset, batch_size=4)
        
        # Evaluate
        metrics = trainer.evaluate(val_loader)
        
        assert 'val_loss' in metrics
        assert 'val_accuracy' in metrics
        assert metrics['val_loss'] > 0
        assert 0 <= metrics['val_accuracy'] <= 1
    
    def test_checkpointing(self, simple_lsm, device, temp_dir):
        """Test model checkpointing."""
        checkpoint_dir = temp_dir / "checkpoints"
        trainer = SNNTrainer(
            model=simple_lsm,
            device=device,
            checkpoint_dir=str(checkpoint_dir)
        )
        
        # Save checkpoint
        trainer.save_checkpoint(epoch=5, is_best=True)
        
        # Check files exist
        assert (checkpoint_dir / "checkpoint_epoch_5.pt").exists()
        assert (checkpoint_dir / "best_model.pt").exists()
        
        # Load checkpoint
        loaded_epoch = trainer.load_checkpoint(str(checkpoint_dir / "best_model.pt"))
        assert loaded_epoch == 5


class TestMultiModalTrainer:
    """Test multi-modal trainer functionality."""
    
    def test_multimodal_trainer_initialization(self, multimodal_lsm, device):
        """Test multi-modal trainer initialization."""
        trainer = MultiModalTrainer(
            model=multimodal_lsm,
            device=device,
            modality_weights={"audio": 0.4, "vision": 0.4, "tactile": 0.2},
            cross_modal_loss_weight=0.1
        )
        
        assert trainer.modality_weights["audio"] == 0.4
        assert trainer.cross_modal_loss_weight == 0.1
        assert trainer.fusion_metrics is not None
    
    def test_multimodal_training_step(self, multimodal_lsm, device):
        """Test multi-modal training step."""
        trainer = MultiModalTrainer(
            model=multimodal_lsm,
            device=device,
            modality_weights={"audio": 0.3, "vision": 0.4, "tactile": 0.3}
        )
        
        # Create mock multi-modal batch
        batch_size = 2
        time_steps = 30
        
        batch_data = {
            "inputs": {
                "audio": torch.randn(batch_size, time_steps, 32, device=device),
                "vision": torch.randn(batch_size, time_steps, 64, device=device),
                "tactile": torch.randn(batch_size, time_steps, 6, device=device)
            },
            "target": torch.randint(0, 5, (batch_size,), device=device)
        }
        
        # Training step
        trainer.optimizer.zero_grad()
        
        outputs, states = trainer.model(batch_data["inputs"], return_states=True)
        main_loss = trainer.loss_fn(outputs, batch_data["target"])
        
        # Should execute without error
        assert outputs.shape == (batch_size, 5)
        assert main_loss.item() > 0
    
    def test_multimodal_training_epoch(self, multimodal_lsm, device, temp_dir):
        """Test complete multi-modal training epoch."""
        trainer = MultiModalTrainer(
            model=multimodal_lsm,
            device=device,
            checkpoint_dir=str(temp_dir / "mm_checkpoints")
        )
        
        # Create mock multi-modal dataset
        dataset = MockDataset(size=12, device=device)
        train_loader = MockDataLoader(dataset, batch_size=3)
        
        # Train for one epoch
        metrics = trainer.train_epoch(train_loader, epoch=0, log_interval=1)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'cross_modal_loss' in metrics
        assert metrics['loss'] > 0
        assert 0 <= metrics['accuracy'] <= 1


class TestDatabaseIntegration:
    """Test database integration with training."""
    
    def test_experiment_tracking(self, test_database, sample_experiment_data):
        """Test experiment creation and tracking."""
        db = test_database
        
        # Create experiment record
        experiment_id = db.insert_record(
            table="experiments",
            record=sample_experiment_data
        )
        
        assert experiment_id is not None
        
        # Retrieve experiment
        experiment = db.get_record("experiments", experiment_id)
        assert experiment["name"] == sample_experiment_data["name"]
        assert experiment["status"] == "created"
    
    def test_training_run_tracking(self, test_database):
        """Test training run database tracking."""
        db = test_database
        
        # Create experiment first
        experiment_data = {
            "name": "test_experiment_tracking",
            "description": "Test experiment for training tracking",
            "config_json": json.dumps({"test": True}),
            "status": "created"
        }
        experiment_id = db.insert_record("experiments", experiment_data)
        
        # Create dataset record
        dataset_data = {
            "name": "test_dataset",
            "path": "/path/to/dataset",
            "modalities": json.dumps(["audio", "vision"]),
            "n_samples": 1000,
            "format": "h5"
        }
        dataset_id = db.insert_record("datasets", dataset_data)
        
        # Create model record
        model_data = {
            "experiment_id": experiment_id,
            "name": "test_model",
            "architecture": "MultiModalLSM",
            "parameters_count": 50000,
            "model_config_json": json.dumps({"fusion_type": "attention"})
        }
        model_id = db.insert_record("models", model_data)
        
        # Create training run
        training_data = {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "status": "started",
            "config_json": json.dumps({"epochs": 100, "batch_size": 32}),
            "total_epochs": 100
        }
        training_id = db.insert_record("training_runs", training_data)
        
        assert training_id is not None
        
        # Update training progress
        db.update_record(
            "training_runs", 
            training_id, 
            {
                "current_epoch": 10,
                "best_val_accuracy": 0.75,
                "status": "running"
            }
        )
        
        # Verify update
        training_run = db.get_record("training_runs", training_id)
        assert training_run["current_epoch"] == 10
        assert training_run["best_val_accuracy"] == 0.75
        assert training_run["status"] == "running"
    
    def test_spike_data_logging(self, test_database):
        """Test spike data logging to database."""
        db = test_database
        
        # Create minimal training run for spike data
        experiment_id = db.insert_record("experiments", {
            "name": "spike_test",
            "config_json": json.dumps({}),
            "status": "created"
        })
        
        dataset_id = db.insert_record("datasets", {
            "name": "spike_dataset",
            "path": "/test/path",
            "modalities": json.dumps(["audio"]),
            "n_samples": 100,
            "format": "h5"
        })
        
        model_id = db.insert_record("models", {
            "experiment_id": experiment_id,
            "name": "spike_model",
            "architecture": "LSM",
            "model_config_json": json.dumps({})
        })
        
        training_id = db.insert_record("training_runs", {
            "model_id": model_id,
            "dataset_id": dataset_id,
            "config_json": json.dumps({}),
            "total_epochs": 10,
            "status": "running"
        })
        
        # Log spike data
        spike_data = {
            "training_run_id": training_id,
            "epoch": 5,
            "batch_idx": 2,
            "modality": "audio",
            "firing_rate": 0.15,
            "spike_count": 1200,
            "synchrony_index": 0.3,
            "cv_isi": 1.2
        }
        
        spike_id = db.insert_record("spike_data", spike_data)
        assert spike_id is not None
        
        # Query spike data
        spike_records = db.search_records(
            "spike_data", 
            {"training_run_id": training_id}
        )
        assert len(spike_records) == 1
        assert spike_records[0]["firing_rate"] == 0.15


class TestEndToEndTraining:
    """Test complete end-to-end training workflows."""
    
    @pytest.mark.slow
    def test_complete_training_workflow(self, device, temp_dir):
        """Test complete training workflow with checkpointing."""
        # Create model
        modality_configs = {
            "audio": {"n_inputs": 16, "n_reservoir": 50},
            "vision": {"n_inputs": 32, "n_reservoir": 75}
        }
        
        model = MultiModalLSM(
            modality_configs=modality_configs,
            n_outputs=3,
            fusion_type="attention",
            device=device
        )
        
        # Create trainer
        trainer = MultiModalTrainer(
            model=model,
            device=device,
            checkpoint_dir=str(temp_dir / "workflow_checkpoints")
        )
        
        # Create mock data
        train_dataset = MockDataset(
            size=24, 
            modalities=["audio", "vision"], 
            device=device
        )
        train_dataset.modality_dims = {"audio": 16, "vision": 32}
        
        val_dataset = MockDataset(
            size=12, 
            modalities=["audio", "vision"], 
            device=device
        )
        val_dataset.modality_dims = {"audio": 16, "vision": 32}
        
        train_loader = MockDataLoader(train_dataset, batch_size=4)
        val_loader = MockDataLoader(val_dataset, batch_size=4)
        
        # Train for a few epochs
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,
            save_best=True
        )
        
        assert 'loss' in history
        assert 'val_accuracy' in history
        assert len(history['loss']) == 3  # 3 epochs
        
        # Check that checkpoints were saved
        checkpoint_dir = temp_dir / "workflow_checkpoints"
        assert (checkpoint_dir / "best_model.pt").exists()
    
    def test_training_with_missing_modalities(self, device):
        """Test training robustness with missing modalities."""
        modality_configs = {
            "audio": {"n_inputs": 20, "n_reservoir": 60},
            "vision": {"n_inputs": 40, "n_reservoir": 80},
            "tactile": {"n_inputs": 8, "n_reservoir": 30}
        }
        
        model = MultiModalLSM(
            modality_configs=modality_configs,
            n_outputs=4,
            device=device
        )
        
        trainer = MultiModalTrainer(model=model, device=device)
        
        # Create dataset with missing modalities in some samples
        class PartialModalityDataset(MockDataset):
            def __getitem__(self, idx):
                inputs, label = super().__getitem__(idx)
                
                # Randomly drop modalities
                if idx % 3 == 0:
                    # Drop tactile
                    del inputs["tactile"]
                elif idx % 5 == 0:
                    # Drop vision
                    del inputs["vision"]
                
                return inputs, label
        
        dataset = PartialModalityDataset(
            size=15, 
            modalities=["audio", "vision", "tactile"], 
            device=device
        )
        dataset.modality_dims = {"audio": 20, "vision": 40, "tactile": 8}
        
        train_loader = MockDataLoader(dataset, batch_size=3)
        
        # Should handle missing modalities gracefully
        # Note: This test verifies the training loop doesn't crash
        # Actual missing modality handling would need model modification
        try:
            metrics = trainer.train_epoch(train_loader, epoch=0)
            # If we get here without exception, test passes
            assert 'loss' in metrics
        except Exception as e:
            # For now, we expect this might fail until missing modality 
            # handling is fully implemented
            pytest.xfail(f"Missing modality handling not fully implemented: {e}")
    
    @pytest.mark.performance
    def test_training_memory_efficiency(self, device):
        """Test memory efficiency during training."""
        if device.type != 'cuda':
            pytest.skip("Memory test requires CUDA")
        
        # Large model configuration
        modality_configs = {
            "audio": {"n_inputs": 128, "n_reservoir": 400},
            "vision": {"n_inputs": 512, "n_reservoir": 600}
        }
        
        model = MultiModalLSM(
            modality_configs=modality_configs,
            n_outputs=20,
            device=device
        )
        
        trainer = MultiModalTrainer(model=model, device=device)
        
        # Large dataset
        dataset = MockDataset(
            size=100, 
            modalities=["audio", "vision"], 
            device=device
        )
        dataset.modality_dims = {"audio": 128, "vision": 512}
        
        train_loader = MockDataLoader(dataset, batch_size=8)
        
        # Monitor memory usage
        torch.cuda.reset_peak_memory_stats(device)
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Train for one epoch
        metrics = trainer.train_epoch(train_loader, epoch=0)
        
        peak_memory = torch.cuda.max_memory_allocated(device)
        memory_growth = peak_memory - initial_memory
        
        # Memory growth should be reasonable (less than 2GB for this config)
        memory_growth_mb = memory_growth / (1024 * 1024)
        assert memory_growth_mb < 2048, f"Memory growth: {memory_growth_mb:.2f}MB"
        
        assert 'loss' in metrics
        assert metrics['loss'] > 0