"""
OAuth Authentication Implementation

Provides OAuth 2.0 authentication for GitHub, Google, and other providers
to secure access to neuromorphic computing APIs and resources.
"""

import os
import json
import secrets
import hashlib
import base64
import requests
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime, timedelta
from urllib.parse import urlencode, parse_qs
from abc import ABC, abstractmethod


class OAuthProvider(ABC):
    """
    Abstract base class for OAuth 2.0 providers.
    
    Implements common OAuth flow patterns with provider-specific
    customizations for neuromorphic computing authentication.
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scope: Optional[List[str]] = None,
    ):
        """
        Initialize OAuth provider.
        
        Args:
            client_id: OAuth client ID
            client_secret: OAuth client secret
            redirect_uri: Redirect URI for OAuth flow
            scope: OAuth scopes to request
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scope = scope or []
        self.logger = logging.getLogger(__name__)
        
        # OAuth endpoints (to be set by subclasses)
        self.authorization_endpoint = ""
        self.token_endpoint = ""
        self.userinfo_endpoint = ""
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name."""
        pass
    
    def generate_authorization_url(self, state: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate OAuth authorization URL.
        
        Args:
            state: Optional state parameter for CSRF protection
            
        Returns:
            Tuple of (authorization_url, state)
        """
        if not state:
            state = secrets.token_urlsafe(32)
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': ' '.join(self.scope),
            'response_type': 'code',
            'state': state,
        }
        
        # Add provider-specific parameters
        provider_params = self._get_authorization_params()
        params.update(provider_params)
        
        authorization_url = f"{self.authorization_endpoint}?{urlencode(params)}"
        
        self.logger.info(f"Generated authorization URL for {self.get_provider_name()}")
        return authorization_url, state
    
    def exchange_code_for_token(self, code: str, state: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Exchange authorization code for access token.
        
        Args:
            code: Authorization code from OAuth callback
            state: State parameter for verification
            
        Returns:
            Token response or None if failed
        """
        try:
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'code': code,
                'redirect_uri': self.redirect_uri,
                'grant_type': 'authorization_code',
            }
            
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded',
            }
            
            response = requests.post(
                self.token_endpoint,
                data=data,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            token_data = response.json()
            
            if 'access_token' not in token_data:
                self.logger.error(f"No access token in response: {token_data}")
                return None
            
            # Enhance token data with provider info
            token_data['provider'] = self.get_provider_name()
            token_data['obtained_at'] = datetime.now().isoformat()
            
            self.logger.info(f"Successfully obtained access token from {self.get_provider_name()}")
            return token_data
            
        except Exception as e:
            self.logger.error(f"Failed to exchange code for token: {e}")
            return None
    
    def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """
        Get user information using access token.
        
        Args:
            access_token: OAuth access token
            
        Returns:
            User information or None if failed
        """
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json',
            }
            
            response = requests.get(
                self.userinfo_endpoint,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            user_data = response.json()
            
            # Normalize user data
            normalized_data = self._normalize_user_data(user_data)
            normalized_data['provider'] = self.get_provider_name()
            normalized_data['retrieved_at'] = datetime.now().isoformat()
            
            self.logger.info(f"Retrieved user info from {self.get_provider_name()}")
            return normalized_data
            
        except Exception as e:
            self.logger.error(f"Failed to get user info: {e}")
            return None
    
    def refresh_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: OAuth refresh token
            
        Returns:
            New token data or None if failed
        """
        try:
            data = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'refresh_token': refresh_token,
                'grant_type': 'refresh_token',
            }
            
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded',
            }
            
            response = requests.post(
                self.token_endpoint,
                data=data,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            token_data = response.json()
            token_data['provider'] = self.get_provider_name()
            token_data['refreshed_at'] = datetime.now().isoformat()
            
            self.logger.info(f"Successfully refreshed token for {self.get_provider_name()}")
            return token_data
            
        except Exception as e:
            self.logger.error(f"Failed to refresh token: {e}")
            return None
    
    def validate_token(self, access_token: str) -> bool:
        """
        Validate access token by making a test API call.
        
        Args:
            access_token: OAuth access token
            
        Returns:
            True if token is valid
        """
        try:
            user_info = self.get_user_info(access_token)
            return user_info is not None
            
        except Exception as e:
            self.logger.error(f"Token validation failed: {e}")
            return False
    
    def _get_authorization_params(self) -> Dict[str, str]:
        """Get provider-specific authorization parameters."""
        return {}
    
    def _normalize_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize user data to common format."""
        return user_data


class GitHubOAuth(OAuthProvider):
    """
    GitHub OAuth 2.0 provider for neuromorphic computing project access.
    
    Provides GitHub authentication with repository and organization
    access for collaborative neuromorphic development.
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        scope: Optional[List[str]] = None,
    ):
        """
        Initialize GitHub OAuth provider.
        
        Args:
            client_id: GitHub OAuth app client ID
            client_secret: GitHub OAuth app client secret
            redirect_uri: Redirect URI registered with GitHub
            scope: GitHub OAuth scopes
        """
        client_id = client_id or os.getenv('GITHUB_CLIENT_ID')
        client_secret = client_secret or os.getenv('GITHUB_CLIENT_SECRET')
        redirect_uri = redirect_uri or os.getenv('GITHUB_REDIRECT_URI', 'http://localhost:8080/auth/github/callback')
        
        default_scope = ['read:user', 'user:email', 'repo', 'read:org']
        scope = scope or default_scope
        
        super().__init__(client_id, client_secret, redirect_uri, scope)
        
        # GitHub OAuth endpoints
        self.authorization_endpoint = "https://github.com/login/oauth/authorize"
        self.token_endpoint = "https://github.com/login/oauth/access_token"
        self.userinfo_endpoint = "https://api.github.com/user"
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return "github"
    
    def get_user_repositories(self, access_token: str) -> List[Dict[str, Any]]:
        """
        Get user's GitHub repositories.
        
        Args:
            access_token: GitHub access token
            
        Returns:
            List of repository information
        """
        try:
            headers = {
                'Authorization': f'token {access_token}',
                'Accept': 'application/vnd.github.v3+json',
            }
            
            response = requests.get(
                'https://api.github.com/user/repos',
                headers=headers,
                params={'sort': 'updated', 'per_page': 100},
                timeout=10
            )
            response.raise_for_status()
            
            repos = response.json()
            
            # Filter for neuromorphic-related repositories
            neuromorphic_keywords = [
                'neural', 'neuron', 'spike', 'snn', 'neuromorphic',
                'brain', 'loihi', 'akida', 'spinnaker', 'stdp'
            ]
            
            relevant_repos = []
            for repo in repos:
                repo_text = f"{repo.get('name', '')} {repo.get('description', '')}".lower()
                if any(keyword in repo_text for keyword in neuromorphic_keywords):
                    relevant_repos.append({
                        'id': repo['id'],
                        'name': repo['name'],
                        'full_name': repo['full_name'],
                        'description': repo.get('description'),
                        'private': repo['private'],
                        'html_url': repo['html_url'],
                        'language': repo.get('language'),
                        'stars': repo['stargazers_count'],
                        'updated_at': repo['updated_at'],
                    })
            
            self.logger.info(f"Found {len(relevant_repos)} neuromorphic repositories")
            return relevant_repos
            
        except Exception as e:
            self.logger.error(f"Failed to get repositories: {e}")
            return []
    
    def get_user_organizations(self, access_token: str) -> List[Dict[str, Any]]:
        """
        Get user's GitHub organizations.
        
        Args:
            access_token: GitHub access token
            
        Returns:
            List of organization information
        """
        try:
            headers = {
                'Authorization': f'token {access_token}',
                'Accept': 'application/vnd.github.v3+json',
            }
            
            response = requests.get(
                'https://api.github.com/user/orgs',
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            orgs = response.json()
            return [
                {
                    'id': org['id'],
                    'login': org['login'],
                    'description': org.get('description'),
                    'avatar_url': org['avatar_url'],
                    'html_url': org['html_url'],
                }
                for org in orgs
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get organizations: {e}")
            return []
    
    def _normalize_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize GitHub user data."""
        return {
            'id': str(user_data.get('id')),
            'username': user_data.get('login'),
            'name': user_data.get('name'),
            'email': user_data.get('email'),
            'avatar_url': user_data.get('avatar_url'),
            'profile_url': user_data.get('html_url'),
            'bio': user_data.get('bio'),
            'location': user_data.get('location'),
            'company': user_data.get('company'),
            'public_repos': user_data.get('public_repos'),
            'followers': user_data.get('followers'),
            'following': user_data.get('following'),
            'created_at': user_data.get('created_at'),
        }


class GoogleOAuth(OAuthProvider):
    """
    Google OAuth 2.0 provider for Google Workspace integration.
    
    Provides Google authentication with access to Drive, Gmail,
    and other Google services for neuromorphic research collaboration.
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        scope: Optional[List[str]] = None,
    ):
        """
        Initialize Google OAuth provider.
        
        Args:
            client_id: Google OAuth client ID
            client_secret: Google OAuth client secret
            redirect_uri: Redirect URI registered with Google
            scope: Google OAuth scopes
        """
        client_id = client_id or os.getenv('GOOGLE_CLIENT_ID')
        client_secret = client_secret or os.getenv('GOOGLE_CLIENT_SECRET')
        redirect_uri = redirect_uri or os.getenv('GOOGLE_REDIRECT_URI', 'http://localhost:8080/auth/google/callback')
        
        default_scope = [
            'openid',
            'email',
            'profile',
            'https://www.googleapis.com/auth/drive.readonly',
        ]
        scope = scope or default_scope
        
        super().__init__(client_id, client_secret, redirect_uri, scope)
        
        # Google OAuth endpoints
        self.authorization_endpoint = "https://accounts.google.com/o/oauth2/v2/auth"
        self.token_endpoint = "https://oauth2.googleapis.com/token"
        self.userinfo_endpoint = "https://www.googleapis.com/oauth2/v2/userinfo"
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return "google"
    
    def _get_authorization_params(self) -> Dict[str, str]:
        """Get Google-specific authorization parameters."""
        return {
            'access_type': 'offline',
            'prompt': 'consent',
        }
    
    def get_drive_files(self, access_token: str, query: str = "name contains 'neuromorphic' or name contains 'snn'") -> List[Dict[str, Any]]:
        """
        Get Google Drive files related to neuromorphic computing.
        
        Args:
            access_token: Google access token
            query: Drive API query string
            
        Returns:
            List of file information
        """
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json',
            }
            
            params = {
                'q': query,
                'fields': 'files(id,name,mimeType,createdTime,modifiedTime,size,webViewLink)',
                'pageSize': 100,
            }
            
            response = requests.get(
                'https://www.googleapis.com/drive/v3/files',
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            files = data.get('files', [])
            
            self.logger.info(f"Found {len(files)} relevant Drive files")
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to get Drive files: {e}")
            return []
    
    def _normalize_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Google user data."""
        return {
            'id': user_data.get('id'),
            'username': user_data.get('email'),
            'name': user_data.get('name'),
            'email': user_data.get('email'),
            'avatar_url': user_data.get('picture'),
            'profile_url': None,
            'verified_email': user_data.get('verified_email'),
            'given_name': user_data.get('given_name'),
            'family_name': user_data.get('family_name'),
            'locale': user_data.get('locale'),
        }


class OAuthManager:
    """
    OAuth manager for handling multiple OAuth providers and flows.
    """
    
    def __init__(self):
        """Initialize OAuth manager."""
        self.providers = {}
        self.logger = logging.getLogger(__name__)
    
    def register_provider(self, name: str, provider: OAuthProvider) -> None:
        """
        Register OAuth provider.
        
        Args:
            name: Provider name
            provider: OAuth provider instance
        """
        self.providers[name] = provider
        self.logger.info(f"Registered OAuth provider: {name}")
    
    def get_provider(self, name: str) -> Optional[OAuthProvider]:
        """
        Get OAuth provider by name.
        
        Args:
            name: Provider name
            
        Returns:
            OAuth provider instance or None
        """
        return self.providers.get(name)
    
    def get_authorization_url(self, provider_name: str, state: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """
        Get authorization URL for provider.
        
        Args:
            provider_name: OAuth provider name
            state: Optional state parameter
            
        Returns:
            Tuple of (authorization_url, state) or None
        """
        provider = self.get_provider(provider_name)
        if not provider:
            self.logger.error(f"Unknown OAuth provider: {provider_name}")
            return None
        
        return provider.generate_authorization_url(state)
    
    def handle_callback(
        self,
        provider_name: str,
        code: str,
        state: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Handle OAuth callback and complete authentication flow.
        
        Args:
            provider_name: OAuth provider name
            code: Authorization code
            state: State parameter
            
        Returns:
            Complete authentication data or None
        """
        provider = self.get_provider(provider_name)
        if not provider:
            self.logger.error(f"Unknown OAuth provider: {provider_name}")
            return None
        
        # Exchange code for token
        token_data = provider.exchange_code_for_token(code, state)
        if not token_data:
            return None
        
        # Get user information
        user_info = provider.get_user_info(token_data['access_token'])
        if not user_info:
            return None
        
        # Combine token and user data
        auth_data = {
            'provider': provider_name,
            'token_data': token_data,
            'user_info': user_info,
            'authenticated_at': datetime.now().isoformat(),
        }
        
        self.logger.info(f"Successfully authenticated user via {provider_name}")
        return auth_data
    
    def setup_default_providers(self) -> None:
        """Setup default OAuth providers with environment configuration."""
        try:
            # Setup GitHub provider
            if all(key in os.environ for key in ['GITHUB_CLIENT_ID', 'GITHUB_CLIENT_SECRET']):
                github_provider = GitHubOAuth()
                self.register_provider('github', github_provider)
            
            # Setup Google provider
            if all(key in os.environ for key in ['GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET']):
                google_provider = GoogleOAuth()
                self.register_provider('google', google_provider)
            
            self.logger.info(f"Setup {len(self.providers)} OAuth providers")
            
        except Exception as e:
            self.logger.error(f"Failed to setup default providers: {e}")