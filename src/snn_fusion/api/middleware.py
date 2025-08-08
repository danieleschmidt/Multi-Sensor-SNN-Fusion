"""
API Middleware for SNN-Fusion

Implements security, logging, and error handling middleware
for the neuromorphic computing API.
"""

from flask import Flask, request, jsonify, g
import logging
import time
import traceback
from functools import wraps
from typing import Callable, Any
import uuid
from datetime import datetime


def setup_error_handlers(app: Flask) -> None:
    """Setup global error handlers."""
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad Request',
            'message': 'Invalid request data or parameters'
        }), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({
            'error': 'Unauthorized',
            'message': 'Authentication required'
        }), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({
            'error': 'Forbidden',
            'message': 'Insufficient permissions'
        }), 403
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not Found',
            'message': 'Resource not found'
        }), 404
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            'error': 'Request Entity Too Large',
            'message': 'File or request size exceeds maximum allowed'
        }), 413
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        return jsonify({
            'error': 'Rate Limit Exceeded',
            'message': 'Too many requests. Please try again later.'
        }), 429
    
    @app.errorhandler(500)
    def internal_server_error(error):
        app.logger.error(f'Internal server error: {error}')
        if app.debug:
            return jsonify({
                'error': 'Internal Server Error',
                'message': str(error),
                'traceback': traceback.format_exc()
            }), 500
        else:
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred'
            }), 500


def setup_request_logging(app: Flask) -> None:
    """Setup request logging middleware."""
    
    @app.before_request
    def before_request():
        g.start_time = time.time()
        g.request_id = str(uuid.uuid4())
        
        # Log request details
        if not app.testing:
            app.logger.info(
                f'[{g.request_id}] {request.method} {request.path} '
                f'from {request.remote_addr}'
            )
    
    @app.after_request
    def after_request(response):
        # Calculate request duration
        duration = time.time() - g.get('start_time', time.time())
        
        # Log response details
        if not app.testing:
            app.logger.info(
                f'[{g.get("request_id", "unknown")}] '
                f'Response: {response.status_code} '
                f'Duration: {duration:.3f}s'
            )
        
        # Add request ID to response headers
        response.headers['X-Request-ID'] = g.get('request_id', 'unknown')
        
        return response


def setup_security_headers(app: Flask) -> None:
    """Setup security headers middleware."""
    
    @app.after_request
    def set_security_headers(response):
        # Basic security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        response.headers['Content-Security-Policy'] = csp
        
        # Remove server header
        response.headers.pop('Server', None)
        
        return response


def require_api_key(f: Callable) -> Callable:
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({
                'error': 'Missing API Key',
                'message': 'X-API-Key header is required'
            }), 401
        
        # Validate API key against database
        try:
            from ..database.connection import DatabaseManager
            db = DatabaseManager()
            
            # Check if API key exists and is active
            api_key_record = db.search_records(
                'api_keys', 
                {'key_hash': api_key, 'is_active': True},
                limit=1
            )
            
            if not api_key_record:
                return jsonify({
                    'error': 'Invalid API Key',
                    'message': 'Provided API key is invalid or inactive'
                }), 401
            
            # Update last_used timestamp
            db.update_record('api_keys', api_key_record[0]['id'], {
                'last_used_at': datetime.utcnow().isoformat()
            })
            
            # Store API key info in request context
            g.api_key_info = api_key_record[0]
            
        except Exception as e:
            # Fallback to basic validation if database is unavailable
            if not api_key.strip() or len(api_key) < 32:
                return jsonify({
                    'error': 'Invalid API Key',
                    'message': 'Provided API key format is invalid'
                }), 401
        
        return f(*args, **kwargs)
    
    return decorated_function


def rate_limit(requests_per_minute: int = 60):
    """Rate limiting decorator."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Implement rate limiting with Redis/in-memory cache
            try:
                from ..cache import CacheManager
                
                # Get client identifier (API key or IP)
                client_id = getattr(g, 'api_key_info', {}).get('id') or request.remote_addr
                cache_key = f"rate_limit:{client_id}"
                
                # Initialize cache manager
                cache = CacheManager()
                
                # Get current request count and timestamp
                current_data = cache.get(cache_key)
                current_time = int(time.time())
                
                if current_data:
                    request_count, window_start = current_data
                    
                    # Check if we're still in the same minute window
                    if current_time - window_start < 60:
                        if request_count >= requests_per_minute:
                            return jsonify({
                                'error': 'Rate Limit Exceeded',
                                'message': f'Maximum {requests_per_minute} requests per minute exceeded',
                                'retry_after': 60 - (current_time - window_start)
                            }), 429
                        
                        # Increment count
                        cache.put(cache_key, (request_count + 1, window_start), ttl=60)
                    else:
                        # Start new window
                        cache.put(cache_key, (1, current_time), ttl=60)
                else:
                    # First request
                    cache.put(cache_key, (1, current_time), ttl=60)
                
                # Log rate limit check
                if hasattr(g, 'request_id'):
                    app = Flask.current_app
                    app.logger.debug(f'Rate limit check passed for request {g.request_id}')
                    
            except Exception as e:
                # Log error but don't block request if rate limiting fails
                app = Flask.current_app
                app.logger.warning(f'Rate limiting error: {e}')
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def validate_json(required_fields: list = None):
    """Decorator to validate JSON request data."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    'error': 'Invalid Content Type',
                    'message': 'Request must contain valid JSON'
                }), 400
            
            data = request.get_json()
            if not data:
                return jsonify({
                    'error': 'Empty Request',
                    'message': 'Request body cannot be empty'
                }), 400
            
            if required_fields:
                missing_fields = [
                    field for field in required_fields
                    if field not in data
                ]
                if missing_fields:
                    return jsonify({
                        'error': 'Missing Required Fields',
                        'message': f'Required fields: {missing_fields}'
                    }), 400
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def handle_exceptions(f: Callable) -> Callable:
    """Decorator to handle common exceptions."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            return jsonify({
                'error': 'Invalid Value',
                'message': str(e)
            }), 400
        except KeyError as e:
            return jsonify({
                'error': 'Missing Key',
                'message': f'Required key not found: {e}'
            }), 400
        except FileNotFoundError as e:
            return jsonify({
                'error': 'File Not Found',
                'message': str(e)
            }), 404
        except PermissionError as e:
            return jsonify({
                'error': 'Permission Denied',
                'message': str(e)
            }), 403
        except Exception as e:
            # Log unexpected exceptions
            app = Flask.current_app
            app.logger.error(
                f'Unexpected error in {f.__name__}: {e}\n'
                f'Traceback: {traceback.format_exc()}'
            )
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred'
            }), 500
    
    return decorated_function


def cache_response(timeout: int = 300):
    """Decorator to cache API responses."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Create cache key from request details
            cache_key = f"{request.method}:{request.path}:{request.args}"
            
            # Try to get cached response
            try:
                from flask import current_app
                cached_response = current_app.cache.get(cache_key)
                if cached_response:
                    app.logger.debug(f'Cache hit for {cache_key}')
                    return cached_response
            except Exception as e:
                app.logger.warning(f'Cache error: {e}')
            
            # Generate fresh response
            response = f(*args, **kwargs)
            
            # Cache successful responses
            if hasattr(response, 'status_code') and response.status_code == 200:
                try:
                    current_app.cache.put(cache_key, response, ttl=timeout)
                    app.logger.debug(f'Cached response for {cache_key}')
                except Exception as e:
                    app.logger.warning(f'Failed to cache response: {e}')
            
            return response
        
        return decorated_function
    return decorator


class RequestValidator:
    """Request validation utilities."""
    
    @staticmethod
    def validate_model_config(config: dict) -> tuple[bool, str]:
        """Validate model configuration."""
        try:
            required_fields = ['modality_configs', 'n_outputs']
            for field in required_fields:
                if field not in config:
                    return False, f'Missing required field: {field}'
            
            # Validate modality configs
            modality_configs = config['modality_configs']
            if not isinstance(modality_configs, dict):
                return False, 'modality_configs must be a dictionary'
            
            for modality, mod_config in modality_configs.items():
                required_mod_fields = ['n_inputs', 'n_reservoir']
                for field in required_mod_fields:
                    if field not in mod_config:
                        return False, f'Missing {field} in {modality} config'
                    
                    if not isinstance(mod_config[field], int) or mod_config[field] <= 0:
                        return False, f'{field} must be a positive integer in {modality} config'
            
            # Validate n_outputs
            if not isinstance(config['n_outputs'], int) or config['n_outputs'] <= 0:
                return False, 'n_outputs must be a positive integer'
            
            return True, 'Valid configuration'
            
        except Exception as e:
            return False, f'Configuration validation error: {str(e)}'
    
    @staticmethod
    def validate_training_config(config: dict) -> tuple[bool, str]:
        """Validate training configuration."""
        try:
            # Optional fields with default values
            optional_fields = {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 1e-3,
                'temporal_window': 100
            }
            
            for field, default_value in optional_fields.items():
                if field in config:
                    value = config[field]
                    if field in ['epochs', 'batch_size', 'temporal_window']:
                        if not isinstance(value, int) or value <= 0:
                            return False, f'{field} must be a positive integer'
                    elif field == 'learning_rate':
                        if not isinstance(value, (int, float)) or value <= 0:
                            return False, f'{field} must be a positive number'
            
            return True, 'Valid training configuration'
            
        except Exception as e:
            return False, f'Training configuration validation error: {str(e)}'