"""
Authentication Integration Module

Provides OAuth and JWT authentication implementations for secure
access to neuromorphic computing APIs and services.
"""

from .oauth import OAuthProvider, GitHubOAuth, GoogleOAuth
from .jwt_auth import JWTManager, JWTTokenHandler

__all__ = [
    'OAuthProvider',
    'GitHubOAuth',
    'GoogleOAuth',
    'JWTManager',
    'JWTTokenHandler',
]