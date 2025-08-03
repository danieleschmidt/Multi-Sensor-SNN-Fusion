"""
Integration tests for SNN-Fusion REST API.

Tests complete API workflows including experiment management,
model training, and neuromorphic hardware deployment.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import patch, Mock

from snn_fusion.api import create_test_app


@pytest.fixture
def client():
    """Create test client for API testing."""
    app = create_test_app()
    with app.test_client() as client:
        with app.app_context():
            # Initialize test database
            app.db.run_migrations()
            yield client


@pytest.fixture
def sample_experiment_data():
    """Sample experiment data for testing."""
    return {
        'name': 'test_api_experiment',
        'description': 'Test experiment for API testing',
        'config': {
            'model_type': 'MultiModalLSM',
            'modalities': ['audio', 'vision'],
            'fusion_type': 'attention'
        },
        'tags': ['test', 'api'],
        'metadata': {
            'created_by': 'test_user',
            'purpose': 'integration_testing'
        }
    }


@pytest.fixture
def sample_model_data():
    """Sample model data for testing."""
    return {
        'name': 'test_multimodal_model',
        'architecture': 'MultiModalLSM',
        'model_config': {
            'modality_configs': {
                'audio': {'n_inputs': 32, 'n_reservoir': 100},
                'vision': {'n_inputs': 64, 'n_reservoir': 150}
            },
            'n_outputs': 5,
            'fusion_type': 'attention'
        },
        'metadata': {
            'test_model': True
        }
    }


class TestHealthCheck:
    """Test health check and basic endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'version' in data
        assert 'database' in data
        assert 'cache' in data
    
    def test_api_info(self, client):
        """Test API info endpoint."""
        response = client.get('/api/v1/info')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['name'] == 'SNN-Fusion API'
        assert 'endpoints' in data
        assert 'experiments' in data['endpoints']
    
    def test_api_docs(self, client):
        """Test API documentation endpoint."""
        response = client.get('/api/v1/docs')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'endpoints_summary' in data


class TestExperimentsAPI:
    """Test experiments API endpoints."""
    
    def test_create_experiment(self, client, sample_experiment_data):
        """Test experiment creation."""
        response = client.post(
            '/api/v1/experiments',
            data=json.dumps(sample_experiment_data),
            content_type='application/json'
        )
        
        assert response.status_code == 201
        data = response.get_json()
        assert data['message'] == 'Experiment created successfully'
        assert 'experiment' in data
        assert data['experiment']['name'] == sample_experiment_data['name']
        
        return data['experiment']['id']
    
    def test_list_experiments(self, client, sample_experiment_data):
        """Test listing experiments."""
        # Create experiment first
        self.test_create_experiment(client, sample_experiment_data)
        
        response = client.get('/api/v1/experiments')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'experiments' in data
        assert data['total'] >= 1
        assert len(data['experiments']) >= 1
    
    def test_get_experiment(self, client, sample_experiment_data):
        """Test getting experiment details."""
        # Create experiment first
        experiment_id = self.test_create_experiment(client, sample_experiment_data)
        
        response = client.get(f'/api/v1/experiments/{experiment_id}')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'experiment' in data
        assert data['experiment']['id'] == experiment_id
        assert 'models' in data['experiment']
    
    def test_update_experiment(self, client, sample_experiment_data):
        """Test updating experiment."""
        # Create experiment first
        experiment_id = self.test_create_experiment(client, sample_experiment_data)
        
        update_data = {
            'description': 'Updated description',
            'status': 'running',
            'tags': ['updated', 'test']
        }
        
        response = client.put(
            f'/api/v1/experiments/{experiment_id}',
            data=json.dumps(update_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['experiment']['description'] == 'Updated description'
        assert data['experiment']['status'] == 'running'
    
    def test_create_experiment_validation(self, client):
        """Test experiment creation validation."""
        # Missing required field
        invalid_data = {
            'description': 'Missing name field'
        }
        
        response = client.post(
            '/api/v1/experiments',
            data=json.dumps(invalid_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_get_nonexistent_experiment(self, client):
        """Test getting non-existent experiment."""
        response = client.get('/api/v1/experiments/99999')
        assert response.status_code == 404
        
        data = response.get_json()
        assert data['error'] == 'Experiment not found'


class TestModelsAPI:
    """Test models API endpoints."""
    
    def setup_experiment(self, client, sample_experiment_data):
        """Helper to create experiment for model tests."""
        response = client.post(
            '/api/v1/experiments',
            data=json.dumps(sample_experiment_data),
            content_type='application/json'
        )
        return response.get_json()['experiment']['id']
    
    def test_create_model(self, client, sample_experiment_data, sample_model_data):
        """Test model creation."""
        # Create experiment first
        experiment_id = self.setup_experiment(client, sample_experiment_data)
        
        # Add experiment_id to model data
        model_data = sample_model_data.copy()
        model_data['experiment_id'] = experiment_id
        
        response = client.post(
            '/api/v1/models',
            data=json.dumps(model_data),
            content_type='application/json'
        )
        
        assert response.status_code == 201
        data = response.get_json()
        assert data['message'] == 'Model created successfully'
        assert 'model' in data
        assert data['model']['name'] == sample_model_data['name']
        assert data['model']['architecture'] == 'MultiModalLSM'
        assert data['model']['parameters_count'] > 0
        
        return data['model']['id']
    
    def test_get_model(self, client, sample_experiment_data, sample_model_data):
        """Test getting model details."""
        # Create model first
        model_id = self.test_create_model(client, sample_experiment_data, sample_model_data)
        
        response = client.get(f'/api/v1/models/{model_id}')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'model' in data
        assert data['model']['id'] == model_id
        assert 'training_runs' in data['model']
    
    def test_create_model_validation(self, client, sample_experiment_data):
        """Test model creation validation."""
        experiment_id = self.setup_experiment(client, sample_experiment_data)
        
        # Invalid model configuration
        invalid_model_data = {
            'experiment_id': experiment_id,
            'name': 'invalid_model',
            'architecture': 'MultiModalLSM',
            'model_config': {
                'invalid_field': 'invalid_value'
            }
        }
        
        response = client.post(
            '/api/v1/models',
            data=json.dumps(invalid_model_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_create_model_unsupported_architecture(self, client, sample_experiment_data):
        """Test creating model with unsupported architecture."""
        experiment_id = self.setup_experiment(client, sample_experiment_data)
        
        unsupported_model = {
            'experiment_id': experiment_id,
            'name': 'unsupported_model',
            'architecture': 'UnsupportedArchitecture',
            'model_config': {}
        }
        
        response = client.post(
            '/api/v1/models',
            data=json.dumps(unsupported_model),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'Unsupported architecture' in data['error']


class TestTrainingAPI:
    """Test training API endpoints."""
    
    def setup_model_and_dataset(self, client, sample_experiment_data, sample_model_data):
        """Helper to create model and dataset for training tests."""
        # Create experiment
        experiment_id = TestModelsAPI().setup_experiment(client, sample_experiment_data)
        
        # Create model
        model_data = sample_model_data.copy()
        model_data['experiment_id'] = experiment_id
        
        model_response = client.post(
            '/api/v1/models',
            data=json.dumps(model_data),
            content_type='application/json'
        )
        model_id = model_response.get_json()['model']['id']
        
        # Create mock dataset (would normally be created through dataset API)
        with client.application.app_context():
            dataset_data = {
                'name': 'test_dataset',
                'path': '/path/to/test/dataset',
                'modalities': json.dumps(['audio', 'vision']),
                'n_samples': 1000,
                'format': 'h5'
            }
            dataset_id = client.application.db.insert_record('datasets', dataset_data)
        
        return model_id, dataset_id
    
    def test_start_training(self, client, sample_experiment_data, sample_model_data):
        """Test starting a training run."""
        model_id, dataset_id = self.setup_model_and_dataset(
            client, sample_experiment_data, sample_model_data
        )
        
        training_config = {
            'epochs': 50,
            'batch_size': 16,
            'learning_rate': 1e-3,
            'temporal_window': 100
        }
        
        training_data = {
            'model_id': model_id,
            'dataset_id': dataset_id,
            'config': training_config
        }
        
        response = client.post(
            '/api/v1/training',
            data=json.dumps(training_data),
            content_type='application/json'
        )
        
        assert response.status_code == 201
        data = response.get_json()
        assert data['message'] == 'Training run queued successfully'
        assert 'training_run' in data
        assert data['training_run']['status'] == 'queued'
        
        return data['training_run']['id']
    
    def test_get_training_status(self, client, sample_experiment_data, sample_model_data):
        """Test getting training status."""
        training_id = self.test_start_training(
            client, sample_experiment_data, sample_model_data
        )
        
        response = client.get(f'/api/v1/training/{training_id}/status')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'training_run' in data
        assert 'metrics' in data
        assert data['training_run']['id'] == training_id
    
    def test_stop_training(self, client, sample_experiment_data, sample_model_data):
        """Test stopping a training run."""
        training_id = self.test_start_training(
            client, sample_experiment_data, sample_model_data
        )
        
        response = client.post(f'/api/v1/training/{training_id}/stop')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['message'] == 'Training run stopped successfully'
    
    def test_start_training_validation(self, client):
        """Test training validation."""
        # Missing required fields
        invalid_training_data = {
            'config': {
                'epochs': 10
            }
        }
        
        response = client.post(
            '/api/v1/training',
            data=json.dumps(invalid_training_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'Missing required fields' in data['error']


class TestInferenceAPI:
    """Test inference API endpoints."""
    
    def test_predict_json_data(self, client, sample_experiment_data, sample_model_data):
        """Test prediction with JSON data."""
        # Create model
        model_id = TestModelsAPI().test_create_model(
            client, sample_experiment_data, sample_model_data
        )
        
        prediction_data = {
            'model_id': model_id,
            'inputs': {
                'audio': [[0.1, 0.2, 0.3]] * 32,
                'vision': [[0.4, 0.5, 0.6]] * 64
            }
        }
        
        response = client.post(
            '/api/v1/inference/predict',
            data=json.dumps(prediction_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'predictions' in data
        assert 'inference_time_ms' in data
        assert data['model_id'] == model_id
    
    def test_predict_missing_model(self, client):
        """Test prediction with missing model."""
        prediction_data = {
            'inputs': {
                'audio': [[0.1, 0.2, 0.3]]
            }
        }
        
        response = client.post(
            '/api/v1/inference/predict',
            data=json.dumps(prediction_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'Missing model_id' in data['error']


class TestHardwareAPI:
    """Test hardware deployment API endpoints."""
    
    def test_list_hardware_profiles(self, client):
        """Test listing hardware profiles."""
        response = client.get('/api/v1/hardware/profiles')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'hardware_profiles' in data
        assert 'total' in data
    
    def test_deploy_to_hardware(self, client, sample_experiment_data, sample_model_data):
        """Test hardware deployment."""
        # Create model
        model_id = TestModelsAPI().test_create_model(
            client, sample_experiment_data, sample_model_data
        )
        
        deployment_data = {
            'model_id': model_id,
            'hardware_type': 'loihi2',
            'deployment_config': {
                'cores_used': 64,
                'optimization_level': 2,
                'quantization_bits': 8
            },
            'optimize': True
        }
        
        response = client.post(
            '/api/v1/hardware/deploy',
            data=json.dumps(deployment_data),
            content_type='application/json'
        )
        
        assert response.status_code == 201
        data = response.get_json()
        assert data['message'] == 'Model deployed successfully'
        assert 'hardware_profile' in data
        assert data['hardware_profile']['hardware_type'] == 'loihi2'
    
    def test_deploy_validation(self, client):
        """Test hardware deployment validation."""
        # Missing required fields
        invalid_deployment = {
            'hardware_type': 'loihi2'
        }
        
        response = client.post(
            '/api/v1/hardware/deploy',
            data=json.dumps(invalid_deployment),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'Missing required fields' in data['error']


class TestMonitoringAPI:
    """Test monitoring API endpoints."""
    
    def test_get_system_status(self, client):
        """Test system status endpoint."""
        response = client.get('/api/v1/monitoring/system')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'cpu_usage' in data
        assert 'memory_usage' in data
        assert 'cache_stats' in data
        assert 'timestamp' in data
    
    def test_get_experiment_metrics(self, client, sample_experiment_data):
        """Test experiment metrics endpoint."""
        # Create experiment
        experiment_id = TestExperimentsAPI().test_create_experiment(
            client, sample_experiment_data
        )
        
        response = client.get(f'/api/v1/monitoring/experiments/{experiment_id}/metrics')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'experiment_id' in data
        assert 'metrics' in data
        assert 'models' in data
        assert data['experiment_id'] == experiment_id
    
    def test_get_live_metrics(self, client):
        """Test live metrics endpoint."""
        response = client.get('/api/v1/monitoring/live-metrics')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'active_training_runs' in data
        assert 'timestamp' in data


class TestAPIWorkflow:
    """Test complete API workflows."""
    
    def test_complete_experiment_workflow(self, client, sample_experiment_data, sample_model_data):
        """Test complete workflow from experiment creation to deployment."""
        
        # 1. Create experiment
        exp_response = client.post(
            '/api/v1/experiments',
            data=json.dumps(sample_experiment_data),
            content_type='application/json'
        )
        assert exp_response.status_code == 201
        experiment_id = exp_response.get_json()['experiment']['id']
        
        # 2. Create model
        model_data = sample_model_data.copy()
        model_data['experiment_id'] = experiment_id
        
        model_response = client.post(
            '/api/v1/models',
            data=json.dumps(model_data),
            content_type='application/json'
        )
        assert model_response.status_code == 201
        model_id = model_response.get_json()['model']['id']
        
        # 3. Create dataset (mock)
        with client.application.app_context():
            dataset_data = {
                'name': 'workflow_dataset',
                'path': '/path/to/dataset',
                'modalities': json.dumps(['audio', 'vision']),
                'n_samples': 500,
                'format': 'h5'
            }
            dataset_id = client.application.db.insert_record('datasets', dataset_data)
        
        # 4. Start training
        training_data = {
            'model_id': model_id,
            'dataset_id': dataset_id,
            'config': {
                'epochs': 10,
                'batch_size': 8,
                'learning_rate': 1e-3
            }
        }
        
        training_response = client.post(
            '/api/v1/training',
            data=json.dumps(training_data),
            content_type='application/json'
        )
        assert training_response.status_code == 201
        training_id = training_response.get_json()['training_run']['id']
        
        # 5. Check training status
        status_response = client.get(f'/api/v1/training/{training_id}/status')
        assert status_response.status_code == 200
        
        # 6. Deploy to hardware
        deployment_data = {
            'model_id': model_id,
            'hardware_type': 'akida',
            'deployment_config': {
                'quantization_bits': 8,
                'sparsity_level': 0.9
            }
        }
        
        deploy_response = client.post(
            '/api/v1/hardware/deploy',
            data=json.dumps(deployment_data),
            content_type='application/json'
        )
        assert deploy_response.status_code == 201
        
        # 7. Run inference
        inference_data = {
            'model_id': model_id,
            'inputs': {
                'audio': [[0.1] * 32] * 10,
                'vision': [[0.2] * 64] * 10
            }
        }
        
        inference_response = client.post(
            '/api/v1/inference/predict',
            data=json.dumps(inference_data),
            content_type='application/json'
        )
        assert inference_response.status_code == 200
        
        # 8. Check experiment metrics
        metrics_response = client.get(f'/api/v1/monitoring/experiments/{experiment_id}/metrics')
        assert metrics_response.status_code == 200
        
        # All steps completed successfully
        assert True  # Workflow completed without errors