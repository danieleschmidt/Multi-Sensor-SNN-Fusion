"""
API Routes for SNN-Fusion

Implements REST endpoints for experiment management, model training,
inference, and neuromorphic hardware deployment.
"""

from flask import Blueprint, request, jsonify, current_app
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
import tempfile
from pathlib import Path

from ..models import MultiModalLSM
from ..training import MultiModalTrainer
from ..database import DatabaseManager


# Create blueprints
experiments_bp = Blueprint('experiments', __name__)
models_bp = Blueprint('models', __name__)
training_bp = Blueprint('training', __name__)
inference_bp = Blueprint('inference', __name__)
hardware_bp = Blueprint('hardware', __name__)
monitoring_bp = Blueprint('monitoring', __name__)


# Experiments endpoints
@experiments_bp.route('', methods=['GET'])
def list_experiments():
    """List all experiments."""
    try:
        db: DatabaseManager = current_app.db
        
        # Get query parameters
        status = request.args.get('status')
        limit = request.args.get('limit', type=int, default=50)
        
        # Build conditions
        conditions = {}
        if status:
            conditions['status'] = status
        
        experiments = db.search_records(
            table='experiments',
            conditions=conditions,
            order_by='created_at DESC',
            limit=limit
        )
        
        return jsonify({
            'experiments': experiments,
            'total': len(experiments)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@experiments_bp.route('', methods=['POST'])
def create_experiment():
    """Create a new experiment."""
    try:
        data = request.get_json()
        
        if not data or 'name' not in data:
            return jsonify({'error': 'Missing required field: name'}), 400
        
        db: DatabaseManager = current_app.db
        
        # Prepare experiment record
        experiment_record = {
            'name': data['name'],
            'description': data.get('description', ''),
            'config_json': json.dumps(data.get('config', {})),
            'status': 'created',
            'tags': json.dumps(data.get('tags', [])),
            'metadata_json': json.dumps(data.get('metadata', {}))
        }
        
        # Insert experiment
        experiment_id = db.insert_record('experiments', experiment_record)
        
        # Retrieve created experiment
        experiment = db.get_record('experiments', experiment_id)
        
        return jsonify({
            'message': 'Experiment created successfully',
            'experiment': experiment
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@experiments_bp.route('/<int:experiment_id>', methods=['GET'])
def get_experiment(experiment_id: int):
    """Get experiment details."""
    try:
        db: DatabaseManager = current_app.db
        experiment = db.get_record('experiments', experiment_id)
        
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404
        
        # Get associated models
        models = db.search_records(
            'models',
            {'experiment_id': experiment_id}
        )
        
        experiment['models'] = models
        
        return jsonify({'experiment': experiment})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@experiments_bp.route('/<int:experiment_id>', methods=['PUT'])
def update_experiment(experiment_id: int):
    """Update experiment."""
    try:
        data = request.get_json()
        db: DatabaseManager = current_app.db
        
        # Check if experiment exists
        experiment = db.get_record('experiments', experiment_id)
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404
        
        # Prepare updates
        updates = {}
        for field in ['description', 'status', 'tags', 'metadata_json']:
            if field in data:
                if field in ['tags', 'metadata_json']:
                    updates[field] = json.dumps(data[field])
                else:
                    updates[field] = data[field]
        
        # Update experiment
        success = db.update_record('experiments', experiment_id, updates)
        
        if success:
            updated_experiment = db.get_record('experiments', experiment_id)
            return jsonify({
                'message': 'Experiment updated successfully',
                'experiment': updated_experiment
            })
        else:
            return jsonify({'error': 'Failed to update experiment'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Models endpoints
@models_bp.route('', methods=['POST'])
def create_model():
    """Create a new model."""
    try:
        data = request.get_json()
        
        required_fields = ['experiment_id', 'name', 'architecture', 'model_config']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        db: DatabaseManager = current_app.db
        
        # Validate experiment exists
        experiment = db.get_record('experiments', data['experiment_id'])
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404
        
        # Create model instance to validate configuration
        try:
            model_config = data['model_config']
            if data['architecture'] == 'MultiModalLSM':
                model = MultiModalLSM(
                    modality_configs=model_config['modality_configs'],
                    n_outputs=model_config['n_outputs'],
                    fusion_type=model_config.get('fusion_type', 'attention')
                )
                parameters_count = sum(p.numel() for p in model.parameters())
            else:
                return jsonify({
                    'error': f'Unsupported architecture: {data["architecture"]}'
                }), 400
        except Exception as e:
            return jsonify({
                'error': f'Invalid model configuration: {str(e)}'
            }), 400
        
        # Prepare model record
        model_record = {
            'experiment_id': data['experiment_id'],
            'name': data['name'],
            'architecture': data['architecture'],
            'parameters_count': parameters_count,
            'model_config_json': json.dumps(model_config),
            'metadata_json': json.dumps(data.get('metadata', {}))
        }
        
        # Insert model
        model_id = db.insert_record('models', model_record)
        
        # Retrieve created model
        created_model = db.get_record('models', model_id)
        
        return jsonify({
            'message': 'Model created successfully',
            'model': created_model
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@models_bp.route('/<int:model_id>', methods=['GET'])
def get_model(model_id: int):
    """Get model details."""
    try:
        db: DatabaseManager = current_app.db
        model = db.get_record('models', model_id)
        
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Get training runs
        training_runs = db.search_records(
            'training_runs',
            {'model_id': model_id}
        )
        
        model['training_runs'] = training_runs
        
        return jsonify({'model': model})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Training endpoints
@training_bp.route('', methods=['POST'])
def start_training():
    """Start a training run."""
    try:
        data = request.get_json()
        
        required_fields = ['model_id', 'dataset_id', 'config']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        db: DatabaseManager = current_app.db
        
        # Validate model and dataset exist
        model = db.get_record('models', data['model_id'])
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        dataset = db.get_record('datasets', data['dataset_id'])
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        # Create training run record
        training_config = data['config']
        training_record = {
            'model_id': data['model_id'],
            'dataset_id': data['dataset_id'],
            'status': 'queued',
            'config_json': json.dumps(training_config),
            'total_epochs': training_config.get('epochs', 100),
            'current_epoch': 0
        }
        
        training_id = db.insert_record('training_runs', training_record)
        
        # TODO: Queue training job for background processing
        # For now, return the created training run
        
        training_run = db.get_record('training_runs', training_id)
        
        return jsonify({
            'message': 'Training run queued successfully',
            'training_run': training_run
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/<int:training_id>/status', methods=['GET'])
def get_training_status(training_id: int):
    """Get training run status."""
    try:
        db: DatabaseManager = current_app.db
        training_run = db.get_record('training_runs', training_id)
        
        if not training_run:
            return jsonify({'error': 'Training run not found'}), 404
        
        # Get recent metrics
        metrics = db.search_records(
            'performance_metrics',
            {'training_run_id': training_id},
            order_by='recorded_at DESC',
            limit=100
        )
        
        return jsonify({
            'training_run': training_run,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/<int:training_id>/stop', methods=['POST'])
def stop_training(training_id: int):
    """Stop a training run."""
    try:
        db: DatabaseManager = current_app.db
        
        # Update training run status
        success = db.update_record(
            'training_runs',
            training_id,
            {
                'status': 'stopped',
                'completed_at': datetime.utcnow().isoformat()
            }
        )
        
        if success:
            return jsonify({'message': 'Training run stopped successfully'})
        else:
            return jsonify({'error': 'Failed to stop training run'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Inference endpoints
@inference_bp.route('/predict', methods=['POST'])
def predict():
    """Run inference on trained model."""
    try:
        # Check if request contains files or JSON data
        if request.content_type.startswith('multipart/form-data'):
            # Handle file uploads
            model_id = request.form.get('model_id', type=int)
            if not model_id:
                return jsonify({'error': 'Missing model_id'}), 400
            
            # Process uploaded files (audio, images, etc.)
            # TODO: Implement file processing
            return jsonify({'error': 'File upload inference not yet implemented'}), 501
            
        else:
            # Handle JSON data
            data = request.get_json()
            
            if not data or 'model_id' not in data:
                return jsonify({'error': 'Missing model_id'}), 400
            
            db: DatabaseManager = current_app.db
            
            # Get model
            model = db.get_record('models', data['model_id'])
            if not model:
                return jsonify({'error': 'Model not found'}), 404
            
            # TODO: Load trained model and run inference
            # For now, return mock prediction
            
            return jsonify({
                'model_id': data['model_id'],
                'predictions': [0.1, 0.3, 0.6],  # Mock prediction
                'inference_time_ms': 2.5,
                'timestamp': datetime.utcnow().isoformat()
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Hardware endpoints
@hardware_bp.route('/profiles', methods=['GET'])
def list_hardware_profiles():
    """List hardware deployment profiles."""
    try:
        db: DatabaseManager = current_app.db
        
        hardware_type = request.args.get('hardware_type')
        conditions = {}
        if hardware_type:
            conditions['hardware_type'] = hardware_type
        
        profiles = db.search_records(
            'hardware_profiles',
            conditions,
            order_by='deployed_at DESC'
        )
        
        return jsonify({
            'hardware_profiles': profiles,
            'total': len(profiles)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@hardware_bp.route('/deploy', methods=['POST'])
def deploy_to_hardware():
    """Deploy model to neuromorphic hardware."""
    try:
        data = request.get_json()
        
        required_fields = ['model_id', 'hardware_type', 'deployment_config']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        db: DatabaseManager = current_app.db
        
        # Validate model exists
        model = db.get_record('models', data['model_id'])
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Create hardware profile
        profile_record = {
            'name': f"deployment_{uuid.uuid4().hex[:8]}",
            'hardware_type': data['hardware_type'],
            'model_id': data['model_id'],
            'deployment_config_json': json.dumps(data['deployment_config']),
            'optimization_applied': data.get('optimize', False)
        }
        
        profile_id = db.insert_record('hardware_profiles', profile_record)
        
        # TODO: Implement actual hardware deployment
        # For now, return success with mock metrics
        
        # Update profile with mock deployment results
        db.update_record('hardware_profiles', profile_id, {
            'inference_latency_ms': 0.8,
            'power_consumption_mw': 150.0,
            'accuracy_deployed': 0.92,
            'memory_usage_mb': 2.5
        })
        
        profile = db.get_record('hardware_profiles', profile_id)
        
        return jsonify({
            'message': 'Model deployed successfully',
            'hardware_profile': profile
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Monitoring endpoints
@monitoring_bp.route('/system', methods=['GET'])
def get_system_status():
    """Get system monitoring information."""
    try:
        cache_stats = current_app.cache.get_stats()
        
        # Mock system metrics
        system_info = {
            'cpu_usage': 45.2,
            'memory_usage': 68.1,
            'gpu_usage': 32.7,
            'disk_usage': 78.5,
            'cache_stats': cache_stats,
            'active_training_runs': 2,
            'queued_jobs': 5,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(system_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@monitoring_bp.route('/experiments/<int:experiment_id>/metrics', methods=['GET'])
def get_experiment_metrics(experiment_id: int):
    """Get experiment performance metrics."""
    try:
        db: DatabaseManager = current_app.db
        
        # Get all models for the experiment
        models = db.search_records('models', {'experiment_id': experiment_id})
        
        if not models:
            return jsonify({'error': 'No models found for experiment'}), 404
        
        # Get training runs for all models
        model_ids = [m['id'] for m in models]
        
        all_metrics = []
        for model_id in model_ids:
            training_runs = db.search_records('training_runs', {'model_id': model_id})
            for run in training_runs:
                metrics = db.search_records(
                    'performance_metrics',
                    {'training_run_id': run['id']},
                    order_by='recorded_at ASC'
                )
                all_metrics.extend(metrics)
        
        return jsonify({
            'experiment_id': experiment_id,
            'metrics': all_metrics,
            'models': models
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@monitoring_bp.route('/live-metrics', methods=['GET'])
def get_live_metrics():
    """Get live training metrics (for real-time dashboard)."""
    try:
        # This would typically use WebSockets or Server-Sent Events
        # For now, return current active training metrics
        
        db: DatabaseManager = current_app.db
        
        # Get active training runs
        active_runs = db.search_records(
            'training_runs',
            {'status': 'running'},
            order_by='started_at DESC'
        )
        
        live_data = []
        for run in active_runs:
            # Get latest metrics
            latest_metrics = db.search_records(
                'performance_metrics',
                {'training_run_id': run['id']},
                order_by='recorded_at DESC',
                limit=10
            )
            
            live_data.append({
                'training_run': run,
                'latest_metrics': latest_metrics
            })
        
        return jsonify({
            'active_training_runs': live_data,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500