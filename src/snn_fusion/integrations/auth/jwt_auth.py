"""
JWT Authentication Implementation

Provides JSON Web Token (JWT) authentication for secure API access
and session management in neuromorphic computing applications.
"""

import os
import jwt
import json
import secrets
import hashlib
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app, g


class JWTManager:
    """
    JWT token manager for creating, validating, and managing JWT tokens.
    
    Provides secure token-based authentication with customizable
    claims and expiration policies for neuromorphic APIs.
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = 'HS256',
        access_token_expires: timedelta = timedelta(hours=1),
        refresh_token_expires: timedelta = timedelta(days=30),
        issuer: str = 'snn-fusion',
    ):
        """
        Initialize JWT manager.
        
        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT signing algorithm
            access_token_expires: Access token expiration time
            refresh_token_expires: Refresh token expiration time
            issuer: Token issuer name
        """
        self.secret_key = secret_key or os.getenv('JWT_SECRET_KEY') or secrets.token_urlsafe(64)
        self.algorithm = algorithm
        self.access_token_expires = access_token_expires
        self.refresh_token_expires = refresh_token_expires
        self.issuer = issuer
        self.logger = logging.getLogger(__name__)
        
        if not secret_key and not os.getenv('JWT_SECRET_KEY'):
            self.logger.warning("Using auto-generated JWT secret key. Set JWT_SECRET_KEY environment variable for production.")
    
    def generate_tokens(
        self,
        user_data: Dict[str, Any],
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Generate access and refresh token pair.
        
        Args:
            user_data: User information to include in token
            additional_claims: Additional claims to include
            
        Returns:
            Dictionary containing access_token and refresh_token
        """
        try:
            now = datetime.utcnow()
            
            # Base claims
            base_claims = {
                'iss': self.issuer,
                'iat': now,
                'user_id': user_data.get('id'),
                'username': user_data.get('username'),
                'email': user_data.get('email'),
            }
            
            # Add additional claims
            if additional_claims:
                base_claims.update(additional_claims)
            
            # Generate access token
            access_claims = base_claims.copy()
            access_claims.update({
                'exp': now + self.access_token_expires,
                'type': 'access',
                'jti': secrets.token_urlsafe(16),  # JWT ID
            })
            
            access_token = jwt.encode(
                access_claims,
                self.secret_key,
                algorithm=self.algorithm
            )
            
            # Generate refresh token
            refresh_claims = {
                'iss': self.issuer,
                'iat': now,
                'exp': now + self.refresh_token_expires,
                'type': 'refresh',
                'user_id': user_data.get('id'),
                'jti': secrets.token_urlsafe(16),
            }
            
            refresh_token = jwt.encode(
                refresh_claims,
                self.secret_key,
                algorithm=self.algorithm
            )
            
            self.logger.info(f"Generated tokens for user: {user_data.get('username')}")
            
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'Bearer',
                'expires_in': int(self.access_token_expires.total_seconds()),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate tokens: {e}")
            raise
    
    def validate_token(
        self,
        token: str,
        token_type: str = 'access',
        verify_exp: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Validate and decode JWT token.
        
        Args:
            token: JWT token string
            token_type: Expected token type (access or refresh)
            verify_exp: Whether to verify expiration
            
        Returns:
            Decoded token claims or None if invalid
        """
        try:
            options = {
                'verify_exp': verify_exp,
                'verify_iat': True,
                'verify_signature': True,
                'require': ['iss', 'iat', 'type'],
            }
            
            decoded = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options=options,
                issuer=self.issuer,
            )
            
            # Verify token type
            if decoded.get('type') != token_type:
                self.logger.warning(f"Invalid token type: expected {token_type}, got {decoded.get('type')}")
                return None
            
            return decoded
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """
        Generate new access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New token pair or None if refresh failed
        """
        # Validate refresh token
        refresh_claims = self.validate_token(refresh_token, token_type='refresh')
        if not refresh_claims:
            return None
        
        try:
            # Create user data from refresh token
            user_data = {
                'id': refresh_claims.get('user_id'),
                'username': refresh_claims.get('username'),
                'email': refresh_claims.get('email'),
            }
            
            # Generate new token pair
            return self.generate_tokens(user_data)
            
        except Exception as e:
            self.logger.error(f"Failed to refresh token: {e}")
            return None
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke a JWT token (placeholder implementation).
        
        In production, this would add the token to a blacklist
        stored in Redis or database.
        
        Args:
            token: Token to revoke
            
        Returns:
            Success status
        """
        try:
            # Validate token first
            decoded = self.validate_token(token, verify_exp=False)
            if not decoded:
                return False
            
            # TODO: Add to token blacklist
            jti = decoded.get('jti')
            if jti:
                self.logger.info(f"Token revoked: {jti}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to revoke token: {e}")
            return False
    
    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Get token information without full validation.
        
        Args:
            token: JWT token
            
        Returns:
            Token information or None
        """
        try:
            # Decode without verification for inspection
            decoded = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            
            return {
                'user_id': decoded.get('user_id'),
                'username': decoded.get('username'),
                'email': decoded.get('email'),
                'type': decoded.get('type'),
                'issued_at': decoded.get('iat'),
                'expires_at': decoded.get('exp'),
                'jti': decoded.get('jti'),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get token info: {e}")
            return None


class JWTTokenHandler:
    """
    Flask-specific JWT token handler for API authentication.
    
    Provides decorators and middleware for protecting API endpoints
    with JWT authentication in neuromorphic computing applications.
    """
    
    def __init__(self, jwt_manager: JWTManager):
        """
        Initialize token handler.
        
        Args:
            jwt_manager: JWT manager instance
        """
        self.jwt_manager = jwt_manager
        self.logger = logging.getLogger(__name__)
    
    def jwt_required(
        self,
        optional: bool = False,
        fresh: bool = False,
    ):
        """
        Decorator to require JWT authentication for endpoint.
        
        Args:
            optional: Whether authentication is optional
            fresh: Whether to require fresh token
            
        Returns:
            Decorator function
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                token = self._extract_token()
                
                if not token:
                    if optional:
                        g.current_user = None
                        return f(*args, **kwargs)
                    else:
                        return jsonify({
                            'error': 'Missing Authorization Token',
                            'message': 'Authorization header with Bearer token is required'
                        }), 401
                
                # Validate token
                claims = self.jwt_manager.validate_token(token)
                if not claims:
                    if optional:
                        g.current_user = None
                        return f(*args, **kwargs)
                    else:
                        return jsonify({
                            'error': 'Invalid Token',
                            'message': 'Provided token is invalid or expired'
                        }), 401
                
                # Check for fresh token if required
                if fresh and not claims.get('fresh', False):
                    return jsonify({
                        'error': 'Fresh Token Required',
                        'message': 'This operation requires a fresh authentication token'
                    }), 401
                
                # Store user info in Flask g object
                g.current_user = {
                    'id': claims.get('user_id'),
                    'username': claims.get('username'),
                    'email': claims.get('email'),
                    'claims': claims,
                }
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def admin_required(self):
        """
        Decorator to require admin privileges.
        
        Returns:
            Decorator function
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # First require valid JWT
                jwt_decorator = self.jwt_required()
                wrapped_f = jwt_decorator(f)
                
                # Check admin privileges
                if not hasattr(g, 'current_user') or not g.current_user:
                    return jsonify({
                        'error': 'Authentication Required',
                        'message': 'Authentication is required'
                    }), 401
                
                claims = g.current_user.get('claims', {})
                if not claims.get('admin', False) and not claims.get('is_admin', False):
                    return jsonify({
                        'error': 'Admin Access Required',
                        'message': 'This operation requires administrator privileges'
                    }), 403
                
                return wrapped_f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def scope_required(self, required_scopes: List[str]):
        """
        Decorator to require specific scopes.
        
        Args:
            required_scopes: List of required scopes
            
        Returns:
            Decorator function
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # First require valid JWT
                jwt_decorator = self.jwt_required()
                wrapped_f = jwt_decorator(f)
                
                # Check scopes
                if not hasattr(g, 'current_user') or not g.current_user:
                    return jsonify({
                        'error': 'Authentication Required',
                        'message': 'Authentication is required'
                    }), 401
                
                claims = g.current_user.get('claims', {})
                user_scopes = claims.get('scopes', [])
                
                missing_scopes = [scope for scope in required_scopes if scope not in user_scopes]
                if missing_scopes:
                    return jsonify({
                        'error': 'Insufficient Scope',
                        'message': f'Required scopes: {missing_scopes}'
                    }), 403
                
                return wrapped_f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def rate_limit_by_user(self, max_requests: int, window_seconds: int = 3600):
        """
        Decorator for user-based rate limiting.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
            
        Returns:
            Decorator function
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # First require valid JWT
                jwt_decorator = self.jwt_required()
                wrapped_f = jwt_decorator(f)
                
                # Check rate limit
                if hasattr(g, 'current_user') and g.current_user:
                    user_id = g.current_user.get('id')
                    if user_id:
                        # TODO: Implement actual rate limiting with Redis
                        # For now, just log the request
                        self.logger.debug(f"Rate limit check for user {user_id}: {max_requests}/{window_seconds}s")
                
                return wrapped_f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def _extract_token(self) -> Optional[str]:
        """
        Extract JWT token from request headers.
        
        Returns:
            JWT token string or None
        """
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return None
        
        try:
            # Expected format: "Bearer <token>"
            scheme, token = auth_header.split(' ', 1)
            if scheme.lower() != 'bearer':
                return None
            return token
        except ValueError:
            return None
    
    def create_auth_response(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create authentication response with tokens.
        
        Args:
            user_data: User information
            
        Returns:
            Authentication response
        """
        try:
            # Generate tokens
            tokens = self.jwt_manager.generate_tokens(user_data)
            
            return {
                'message': 'Authentication successful',
                'user': {
                    'id': user_data.get('id'),
                    'username': user_data.get('username'),
                    'email': user_data.get('email'),
                    'name': user_data.get('name'),
                },
                'tokens': tokens,
                'authenticated_at': datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create auth response: {e}")
            raise
    
    def refresh_tokens(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """
        Refresh authentication tokens.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New tokens or None if refresh failed
        """
        try:
            new_tokens = self.jwt_manager.refresh_access_token(refresh_token)
            if not new_tokens:
                return None
            
            return {
                'message': 'Tokens refreshed successfully',
                'tokens': new_tokens,
                'refreshed_at': datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to refresh tokens: {e}")
            return None
    
    def logout_user(self, access_token: str) -> bool:
        """
        Logout user by revoking tokens.
        
        Args:
            access_token: User's access token
            
        Returns:
            Success status
        """
        try:
            return self.jwt_manager.revoke_token(access_token)
            
        except Exception as e:
            self.logger.error(f"Failed to logout user: {e}")
            return False


def setup_jwt_auth(app, secret_key: Optional[str] = None) -> JWTTokenHandler:
    """
    Setup JWT authentication for Flask app.
    
    Args:
        app: Flask application
        secret_key: JWT secret key
        
    Returns:
        JWT token handler instance
    """
    # Create JWT manager
    jwt_manager = JWTManager(secret_key=secret_key)
    
    # Create token handler
    token_handler = JWTTokenHandler(jwt_manager)
    
    # Store in app context
    app.jwt_manager = jwt_manager
    app.jwt_handler = token_handler
    
    return token_handler