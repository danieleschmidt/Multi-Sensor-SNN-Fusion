"""
External Service Integrations for SNN-Fusion

Implements integrations with external services including GitHub Actions,
notification systems, experiment tracking, and cloud platforms.
"""

from .github import GitHubIntegration, GitHubWebhookHandler
from .notifications import (
    NotificationManager,
    EmailNotifier,
    SlackNotifier,
    DiscordNotifier,
)
from .tracking import (
    ExperimentTracker,
    WandBTracker,
    MLFlowTracker,
    TensorBoardTracker,
)
from .cloud import (
    CloudStorage,
    AWSIntegration,
    GCPIntegration,
    AzureIntegration,
)

__all__ = [
    # GitHub integration
    "GitHubIntegration",
    "GitHubWebhookHandler",
    # Notifications
    "NotificationManager",
    "EmailNotifier",
    "SlackNotifier", 
    "DiscordNotifier",
    # Experiment tracking
    "ExperimentTracker",
    "WandBTracker",
    "MLFlowTracker",
    "TensorBoardTracker",
    # Cloud platforms
    "CloudStorage",
    "AWSIntegration",
    "GCPIntegration",
    "AzureIntegration",
]