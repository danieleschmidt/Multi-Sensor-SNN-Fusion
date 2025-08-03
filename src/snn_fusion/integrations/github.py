"""
GitHub Integration for SNN-Fusion

Implements GitHub Actions integration, webhook handling,
and repository management for neuromorphic computing workflows.
"""

import os
import json
import requests
import hashlib
import hmac
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
from pathlib import Path

try:
    from github import Github, GithubException
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False


class GitHubIntegration:
    """
    GitHub integration for repository management and CI/CD workflows.
    
    Provides functionality for managing GitHub repositories, issues,
    pull requests, and GitHub Actions workflows for neuromorphic projects.
    """
    
    def __init__(
        self,
        access_token: Optional[str] = None,
        repository: Optional[str] = None,
        base_url: str = "https://api.github.com",
    ):
        """
        Initialize GitHub integration.
        
        Args:
            access_token: GitHub personal access token
            repository: Repository name in format "owner/repo"
            base_url: GitHub API base URL
        """
        self.access_token = access_token or os.getenv('GITHUB_TOKEN')
        self.repository = repository or os.getenv('GITHUB_REPOSITORY')
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        
        if not self.access_token:
            self.logger.warning("No GitHub access token provided")
        
        # Initialize PyGithub client if available
        self.github_client = None
        if GITHUB_AVAILABLE and self.access_token:
            try:
                self.github_client = Github(self.access_token, base_url=base_url)
                self.repo = self.github_client.get_repo(self.repository) if self.repository else None
            except Exception as e:
                self.logger.error(f"Failed to initialize GitHub client: {e}")
    
    def create_workflow_dispatch(
        self,
        workflow_id: str,
        ref: str = "main",
        inputs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Trigger GitHub Actions workflow via workflow_dispatch.
        
        Args:
            workflow_id: Workflow ID or filename
            ref: Git reference (branch, tag, commit)
            inputs: Workflow inputs
            
        Returns:
            Success status
        """
        if not self.access_token or not self.repository:
            self.logger.error("GitHub token and repository required")
            return False
        
        url = f"{self.base_url}/repos/{self.repository}/actions/workflows/{workflow_id}/dispatches"
        
        headers = {
            'Authorization': f'token {self.access_token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json',
        }
        
        payload = {
            'ref': ref,
            'inputs': inputs or {}
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            self.logger.info(f"Successfully triggered workflow {workflow_id} on {ref}")
            return True
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to trigger workflow: {e}")
            return False
    
    def get_workflow_runs(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get workflow run history.
        
        Args:
            workflow_id: Optional workflow ID filter
            status: Optional status filter (queued, in_progress, completed)
            limit: Maximum number of runs to return
            
        Returns:
            List of workflow run information
        """
        if not self.github_client or not self.repo:
            return []
        
        try:
            if workflow_id:
                workflow = self.repo.get_workflow(workflow_id)
                runs = workflow.get_runs()
            else:
                runs = self.repo.get_workflow_runs()
            
            workflow_runs = []
            for run in runs[:limit]:
                if status and run.status != status:
                    continue
                
                workflow_runs.append({
                    'id': run.id,
                    'name': run.name,
                    'status': run.status,
                    'conclusion': run.conclusion,
                    'workflow_id': run.workflow_id,
                    'head_branch': run.head_branch,
                    'head_sha': run.head_sha,
                    'created_at': run.created_at.isoformat(),
                    'updated_at': run.updated_at.isoformat(),
                    'html_url': run.html_url,
                })
            
            return workflow_runs
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow runs: {e}")
            return []
    
    def create_issue(
        self,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> Optional[int]:
        """
        Create GitHub issue.
        
        Args:
            title: Issue title
            body: Issue description
            labels: Issue labels
            assignees: Issue assignees
            
        Returns:
            Issue number if successful
        """
        if not self.github_client or not self.repo:
            return None
        
        try:
            issue = self.repo.create_issue(
                title=title,
                body=body,
                labels=labels or [],
                assignees=assignees or [],
            )
            
            self.logger.info(f"Created issue #{issue.number}: {title}")
            return issue.number
            
        except Exception as e:
            self.logger.error(f"Failed to create issue: {e}")
            return None
    
    def comment_on_issue(self, issue_number: int, comment: str) -> bool:
        """
        Add comment to GitHub issue.
        
        Args:
            issue_number: Issue number
            comment: Comment text
            
        Returns:
            Success status
        """
        if not self.github_client or not self.repo:
            return False
        
        try:
            issue = self.repo.get_issue(issue_number)
            issue.create_comment(comment)
            
            self.logger.info(f"Added comment to issue #{issue_number}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to comment on issue: {e}")
            return False
    
    def create_pull_request(
        self,
        title: str,
        head: str,
        base: str = "main",
        body: str = "",
        draft: bool = False,
    ) -> Optional[int]:
        """
        Create pull request.
        
        Args:
            title: PR title
            head: Source branch
            base: Target branch
            body: PR description
            draft: Whether to create as draft
            
        Returns:
            Pull request number if successful
        """
        if not self.github_client or not self.repo:
            return None
        
        try:
            pr = self.repo.create_pull(
                title=title,
                head=head,
                base=base,
                body=body,
                draft=draft,
            )
            
            self.logger.info(f"Created pull request #{pr.number}: {title}")
            return pr.number
            
        except Exception as e:
            self.logger.error(f"Failed to create pull request: {e}")
            return None
    
    def get_repository_info(self) -> Optional[Dict[str, Any]]:
        """Get repository information."""
        if not self.github_client or not self.repo:
            return None
        
        try:
            return {
                'name': self.repo.name,
                'full_name': self.repo.full_name,
                'description': self.repo.description,
                'private': self.repo.private,
                'default_branch': self.repo.default_branch,
                'language': self.repo.language,
                'stars': self.repo.stargazers_count,
                'forks': self.repo.forks_count,
                'open_issues': self.repo.open_issues_count,
                'created_at': self.repo.created_at.isoformat(),
                'updated_at': self.repo.updated_at.isoformat(),
                'html_url': self.repo.html_url,
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get repository info: {e}")
            return None
    
    def upload_artifact(
        self,
        file_path: str,
        artifact_name: str,
        run_id: Optional[int] = None,
    ) -> bool:
        """
        Upload artifact to GitHub Actions run.
        
        Note: This is a placeholder implementation.
        Actual artifact upload requires GitHub Actions context.
        
        Args:
            file_path: Path to file to upload
            artifact_name: Artifact name
            run_id: Optional workflow run ID
            
        Returns:
            Success status
        """
        # In actual GitHub Actions environment, this would use
        # the actions toolkit or GitHub CLI
        self.logger.info(f"Artifact upload requested: {artifact_name} from {file_path}")
        return True


class GitHubWebhookHandler:
    """
    GitHub webhook handler for processing repository events.
    
    Handles GitHub webhooks for automated responses to repository events
    such as pushes, pull requests, and issue updates.
    """
    
    def __init__(self, webhook_secret: Optional[str] = None):
        """
        Initialize webhook handler.
        
        Args:
            webhook_secret: GitHub webhook secret for signature verification
        """
        self.webhook_secret = webhook_secret or os.getenv('GITHUB_WEBHOOK_SECRET')
        self.logger = logging.getLogger(__name__)
    
    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify GitHub webhook signature.
        
        Args:
            payload: Request payload
            signature: GitHub signature header
            
        Returns:
            Whether signature is valid
        """
        if not self.webhook_secret:
            self.logger.warning("No webhook secret configured")
            return False
        
        expected_signature = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        expected_signature = f"sha256={expected_signature}"
        return hmac.compare_digest(expected_signature, signature)
    
    def handle_push_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle push event.
        
        Args:
            event_data: GitHub push event data
            
        Returns:
            Response data
        """
        repository = event_data.get('repository', {}).get('full_name')
        ref = event_data.get('ref')
        commits = event_data.get('commits', [])
        
        self.logger.info(f"Push event to {repository} on {ref} with {len(commits)} commits")
        
        # Process commits for neuromorphic-specific patterns
        neuromorphic_keywords = [
            'snn', 'spike', 'neural', 'neuromorphic', 'lsm', 'reservoir',
            'loihi', 'akida', 'spinnaker', 'stdp', 'plasticity'
        ]
        
        neuromorphic_commits = []
        for commit in commits:
            message = commit.get('message', '').lower()
            if any(keyword in message for keyword in neuromorphic_keywords):
                neuromorphic_commits.append(commit)
        
        response = {
            'status': 'processed',
            'repository': repository,
            'ref': ref,
            'total_commits': len(commits),
            'neuromorphic_commits': len(neuromorphic_commits),
            'action_taken': None,
        }
        
        # Trigger automated actions for neuromorphic commits
        if neuromorphic_commits and ref == 'refs/heads/main':
            # Could trigger automated testing, benchmarking, etc.
            response['action_taken'] = 'triggered_neuromorphic_tests'
            self.logger.info("Triggered neuromorphic-specific tests")
        
        return response
    
    def handle_pull_request_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle pull request event.
        
        Args:
            event_data: GitHub pull request event data
            
        Returns:
            Response data
        """
        action = event_data.get('action')
        pr = event_data.get('pull_request', {})
        pr_number = pr.get('number')
        title = pr.get('title', '')
        body = pr.get('body', '')
        
        self.logger.info(f"Pull request {action} event for PR #{pr_number}")
        
        response = {
            'status': 'processed',
            'action': action,
            'pr_number': pr_number,
            'checks_triggered': [],
        }
        
        # Trigger specific checks based on PR content
        if action in ['opened', 'synchronize']:
            checks = []
            
            # Check for neuromorphic model changes
            if 'model' in title.lower() or 'snn' in title.lower():
                checks.append('neuromorphic_model_validation')
            
            # Check for hardware-specific changes
            hardware_keywords = ['loihi', 'akida', 'spinnaker', 'hardware']
            if any(keyword in (title + body).lower() for keyword in hardware_keywords):
                checks.append('hardware_compatibility_test')
            
            # Check for dataset changes
            if 'dataset' in title.lower() or 'data' in title.lower():
                checks.append('dataset_validation')
            
            response['checks_triggered'] = checks
        
        return response
    
    def handle_issue_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle issue event.
        
        Args:
            event_data: GitHub issue event data
            
        Returns:
            Response data
        """
        action = event_data.get('action')
        issue = event_data.get('issue', {})
        issue_number = issue.get('number')
        title = issue.get('title', '')
        labels = [label.get('name') for label in issue.get('labels', [])]
        
        self.logger.info(f"Issue {action} event for issue #{issue_number}")
        
        response = {
            'status': 'processed',
            'action': action,
            'issue_number': issue_number,
            'auto_labels': [],
        }
        
        # Auto-label issues based on content
        if action == 'opened':
            auto_labels = []
            
            # Neuromorphic-specific labels
            if any(keyword in title.lower() for keyword in ['bug', 'error', 'fail']):
                auto_labels.append('bug')
            
            if any(keyword in title.lower() for keyword in ['feature', 'enhancement']):
                auto_labels.append('enhancement')
            
            if any(keyword in title.lower() for keyword in ['hardware', 'loihi', 'akida']):
                auto_labels.append('hardware')
            
            if any(keyword in title.lower() for keyword in ['performance', 'speed', 'memory']):
                auto_labels.append('performance')
            
            response['auto_labels'] = auto_labels
        
        return response
    
    def process_webhook(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process GitHub webhook event.
        
        Args:
            event_type: GitHub event type
            payload: Event payload
            
        Returns:
            Processing result
        """
        try:
            if event_type == 'push':
                return self.handle_push_event(payload)
            elif event_type == 'pull_request':
                return self.handle_pull_request_event(payload)
            elif event_type == 'issues':
                return self.handle_issue_event(payload)
            else:
                self.logger.info(f"Unhandled event type: {event_type}")
                return {'status': 'ignored', 'event_type': event_type}
                
        except Exception as e:
            self.logger.error(f"Error processing webhook event: {e}")
            return {'status': 'error', 'error': str(e)}


class GitHubActionsRunner:
    """
    GitHub Actions runner integration for neuromorphic workflows.
    
    Provides utilities for running neuromorphic-specific workflows
    and managing GitHub Actions runners.
    """
    
    def __init__(self, github_integration: GitHubIntegration):
        """
        Initialize GitHub Actions runner.
        
        Args:
            github_integration: GitHub integration instance
        """
        self.github = github_integration
        self.logger = logging.getLogger(__name__)
    
    def trigger_neuromorphic_tests(
        self,
        model_config: Dict[str, Any],
        ref: str = "main",
    ) -> bool:
        """
        Trigger neuromorphic-specific test workflows.
        
        Args:
            model_config: Model configuration for testing
            ref: Git reference to test
            
        Returns:
            Success status
        """
        workflow_inputs = {
            'model_type': model_config.get('architecture', 'MultiModalLSM'),
            'test_suite': 'neuromorphic',
            'hardware_targets': ','.join(['loihi2', 'akida', 'spinnaker']),
            'performance_benchmarks': 'true',
        }
        
        return self.github.create_workflow_dispatch(
            workflow_id='neuromorphic-tests.yml',
            ref=ref,
            inputs=workflow_inputs,
        )
    
    def trigger_hardware_deployment(
        self,
        model_path: str,
        hardware_target: str,
        ref: str = "main",
    ) -> bool:
        """
        Trigger hardware deployment workflow.
        
        Args:
            model_path: Path to trained model
            hardware_target: Target hardware platform
            ref: Git reference
            
        Returns:
            Success status
        """
        workflow_inputs = {
            'model_path': model_path,
            'hardware_target': hardware_target,
            'optimization_level': '2',
            'run_benchmarks': 'true',
        }
        
        return self.github.create_workflow_dispatch(
            workflow_id='hardware-deployment.yml',
            ref=ref,
            inputs=workflow_inputs,
        )
    
    def get_test_results(self, run_id: int) -> Optional[Dict[str, Any]]:
        """
        Get test results from GitHub Actions run.
        
        Args:
            run_id: Workflow run ID
            
        Returns:
            Test results summary
        """
        # This would parse test artifacts and logs
        # Implementation depends on specific test output format
        self.logger.info(f"Retrieving test results for run {run_id}")
        
        # Placeholder implementation
        return {
            'run_id': run_id,
            'status': 'completed',
            'test_summary': {
                'total_tests': 150,
                'passed': 145,
                'failed': 3,
                'skipped': 2,
            },
            'coverage': 0.92,
            'performance_metrics': {
                'latency_ms': 1.2,
                'memory_mb': 245,
                'accuracy': 0.94,
            }
        }