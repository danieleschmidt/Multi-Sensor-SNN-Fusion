"""
SNN-Fusion Production Module

Comprehensive production deployment, monitoring, and operational management
system for neuromorphic multi-modal sensor fusion in enterprise environments.

Production Components:
- Real-time health monitoring and alerting
- Auto-scaling and load balancing
- Performance optimization and tuning
- Disaster recovery and backup management
- Security hardening and compliance
- Operational dashboards and analytics
"""

from .health_monitor import (
    ProductionHealthMonitor,
    SystemMetrics,
    HealthCheck,
    AlertManager,
    AutoScaler,
)

from .deployment_manager import (
    ProductionDeploymentManager,
    DeploymentConfig,
    ScalingPolicy,
    BackupManager,
)

from .operational_dashboard import (
    ProductionDashboard,
    RealTimeMetrics,
    AlertDashboard,
    PerformanceAnalytics,
)

__all__ = [
    'ProductionHealthMonitor',
    'SystemMetrics',
    'HealthCheck',
    'AlertManager',
    'AutoScaler',
    'ProductionDeploymentManager',
    'DeploymentConfig',
    'ScalingPolicy',
    'BackupManager',
    'ProductionDashboard',
    'RealTimeMetrics',
    'AlertDashboard',
    'PerformanceAnalytics',
]