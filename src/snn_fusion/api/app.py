"""
Flask Application Factory for SNN-Fusion API

Creates and configures the Flask application with all endpoints,
middleware, and neuromorphic-specific extensions.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import os
from typing import Optional

from .routes import (
    experiments_bp,
    models_bp,
    training_bp,
    inference_bp,
    hardware_bp,
    monitoring_bp,
)
from .middleware import (
    setup_error_handlers,
    setup_request_logging,
    setup_security_headers,
)
from ..database import get_database
from ..cache import get_cache_manager


def create_app(config: Optional[dict] = None) -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-secret-key'),
        DATABASE_TYPE=os.environ.get('DATABASE_TYPE', 'sqlite'),
        DATABASE_PATH=os.environ.get('DATABASE_PATH', 'snn_fusion.db'),
        REDIS_URL=os.environ.get('REDIS_URL'),
        DEBUG=os.environ.get('DEBUG', 'false').lower() == 'true',
        TESTING=False,
        MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100MB max upload
    )
    
    if config:
        app.config.from_mapping(config)
    
    # Setup logging
    if not app.testing:
        logging.basicConfig(
            level=logging.INFO if not app.debug else logging.DEBUG,
            format='%(asctime)s %(levelname)s: %(message)s'
        )
    
    # Enable CORS
    CORS(app, origins=["http://localhost:3000", "http://localhost:8080"])
    
    # Initialize database
    with app.app_context():
        db = get_database(
            db_type=app.config['DATABASE_TYPE'],
            db_path=app.config['DATABASE_PATH']
        )
        app.db = db
    
    # Initialize cache
    cache_manager = get_cache_manager(
        redis_url=app.config.get('REDIS_URL')
    )
    app.cache = cache_manager
    
    # Register blueprints
    app.register_blueprint(experiments_bp, url_prefix='/api/v1/experiments')
    app.register_blueprint(models_bp, url_prefix='/api/v1/models')
    app.register_blueprint(training_bp, url_prefix='/api/v1/training')
    app.register_blueprint(inference_bp, url_prefix='/api/v1/inference')
    app.register_blueprint(hardware_bp, url_prefix='/api/v1/hardware')
    app.register_blueprint(monitoring_bp, url_prefix='/api/v1/monitoring')
    
    # Setup middleware
    setup_error_handlers(app)
    setup_request_logging(app)
    setup_security_headers(app)
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        try:
            # Test database connection
            app.db.execute_query("SELECT 1", fetch="one")
            
            # Test cache
            cache_stats = app.cache.get_stats()
            
            return jsonify({
                'status': 'healthy',
                'version': '0.1.0',
                'database': 'connected',
                'cache': {
                    'status': 'connected',
                    'memory_cache_size': cache_stats['memory_cache']['size'],
                    'redis_connected': cache_stats['redis_cache']['connected'],
                }
            })
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'error': str(e)
            }), 500
    
    # API info endpoint
    @app.route('/api/v1/info')
    def api_info():
        """API information endpoint."""
        return jsonify({
            'name': 'SNN-Fusion API',
            'version': '0.1.0',
            'description': 'REST API for neuromorphic multi-modal sensor fusion',
            'endpoints': {
                'experiments': '/api/v1/experiments',
                'models': '/api/v1/models',
                'training': '/api/v1/training',
                'inference': '/api/v1/inference',
                'hardware': '/api/v1/hardware',
                'monitoring': '/api/v1/monitoring',
            },
            'documentation': '/api/v1/docs',
            'health': '/health',
        })
    
    # API documentation placeholder
    @app.route('/api/v1/docs')
    def api_docs():
        """API documentation endpoint."""
        return jsonify({
            'message': 'API documentation will be available here',
            'swagger_ui': 'Future implementation with OpenAPI/Swagger',
            'endpoints_summary': {
                'experiments': 'Manage neuromorphic experiments',
                'models': 'Create and manage SNN models',
                'training': 'Train models with multi-modal data',
                'inference': 'Run inference on trained models',
                'hardware': 'Deploy to neuromorphic hardware',
                'monitoring': 'Real-time training and system monitoring',
            }
        })
    
    return app


def create_test_app() -> Flask:
    """Create Flask app configured for testing."""
    return create_app({
        'TESTING': True,
        'DATABASE_TYPE': 'memory',
        'DATABASE_PATH': ':memory:',
        'SECRET_KEY': 'test-secret',
        'WTF_CSRF_ENABLED': False,
    })