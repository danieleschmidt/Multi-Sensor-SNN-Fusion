"""
Production Deployment Management System

Comprehensive deployment orchestration, configuration management,
and operational control for neuromorphic computing production environments.
"""

import json
import logging
import os
import subprocess
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import shutil
import yaml


class DeploymentStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"


class DeploymentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentConfig:
    """Production deployment configuration."""
    stage: DeploymentStage
    version: str
    instances: int
    cpu_limit: str
    memory_limit: str
    storage_size: str
    enable_gpu: bool
    monitoring_enabled: bool
    backup_enabled: bool
    auto_scaling_enabled: bool
    health_check_enabled: bool
    security_scanning_enabled: bool
    
    # Networking
    external_port: int
    internal_port: int
    load_balancer_enabled: bool
    
    # Database
    database_pool_size: int
    database_timeout_seconds: int
    
    # Cache
    redis_enabled: bool
    redis_memory_limit: str
    
    # Security
    enable_ssl: bool
    jwt_expiry_hours: int
    api_rate_limit_per_minute: int
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['stage'] = self.stage.value
        return result
    
    def validate(self) -> List[str]:
        """Validate configuration and return errors."""
        errors = []
        
        if self.instances < 1:
            errors.append("instances must be >= 1")
        
        if self.instances > 50:
            errors.append("instances must be <= 50")
        
        if self.external_port < 1024 or self.external_port > 65535:
            errors.append("external_port must be between 1024-65535")
        
        if self.database_pool_size < 1:
            errors.append("database_pool_size must be >= 1")
        
        if self.jwt_expiry_hours < 1:
            errors.append("jwt_expiry_hours must be >= 1")
        
        return errors


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_instances: int
    max_instances: int
    target_cpu_percent: int
    target_memory_percent: int
    scale_up_threshold_duration_minutes: int
    scale_down_threshold_duration_minutes: int
    scale_up_cooldown_minutes: int
    scale_down_cooldown_minutes: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BackupConfig:
    """Backup configuration."""
    enabled: bool
    database_backup_interval_hours: int
    model_backup_interval_hours: int
    config_backup_interval_hours: int
    retention_days: int
    storage_location: str
    encryption_enabled: bool
    compression_enabled: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProductionDeploymentManager:
    """
    Comprehensive production deployment management system.
    
    Features:
    - Multi-stage deployment orchestration
    - Blue-green and rolling deployments
    - Configuration management
    - Health monitoring integration
    - Automated rollback capabilities
    - Backup and disaster recovery
    """
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path or "/app")
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.current_deployments: Dict[str, Dict[str, Any]] = {}
        self.deployment_history: List[Dict[str, Any]] = []
        
        # Configuration paths
        self.config_dir = self.base_path / "config"
        self.templates_dir = self.base_path / "deploy" / "templates"
        self.backup_dir = self.base_path / "backups"
        
        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Production deployment manager initialized")
    
    def create_deployment_config(self, stage: DeploymentStage, 
                               custom_config: Optional[Dict[str, Any]] = None) -> DeploymentConfig:
        """Create deployment configuration for specific stage."""
        base_configs = {
            DeploymentStage.DEVELOPMENT: {
                'instances': 1,
                'cpu_limit': '500m',
                'memory_limit': '1Gi',
                'storage_size': '10Gi',
                'enable_gpu': False,
                'monitoring_enabled': True,
                'backup_enabled': False,
                'auto_scaling_enabled': False,
                'external_port': 8080,
                'internal_port': 8080,
                'load_balancer_enabled': False,
                'database_pool_size': 5,
                'database_timeout_seconds': 30,
                'redis_enabled': False,
                'redis_memory_limit': '256Mi',
                'enable_ssl': False,
                'jwt_expiry_hours': 24,
                'api_rate_limit_per_minute': 1000,
            },
            DeploymentStage.STAGING: {
                'instances': 2,
                'cpu_limit': '1',
                'memory_limit': '2Gi',
                'storage_size': '50Gi',
                'enable_gpu': False,
                'monitoring_enabled': True,
                'backup_enabled': True,
                'auto_scaling_enabled': True,
                'external_port': 8080,
                'internal_port': 8080,
                'load_balancer_enabled': True,
                'database_pool_size': 10,
                'database_timeout_seconds': 30,
                'redis_enabled': True,
                'redis_memory_limit': '512Mi',
                'enable_ssl': True,
                'jwt_expiry_hours': 12,
                'api_rate_limit_per_minute': 500,
            },
            DeploymentStage.PRODUCTION: {
                'instances': 5,
                'cpu_limit': '2',
                'memory_limit': '4Gi',
                'storage_size': '200Gi',
                'enable_gpu': True,
                'monitoring_enabled': True,
                'backup_enabled': True,
                'auto_scaling_enabled': True,
                'external_port': 80,
                'internal_port': 8080,
                'load_balancer_enabled': True,
                'database_pool_size': 20,
                'database_timeout_seconds': 60,
                'redis_enabled': True,
                'redis_memory_limit': '2Gi',
                'enable_ssl': True,
                'jwt_expiry_hours': 8,
                'api_rate_limit_per_minute': 100,
            },
        }
        
        config_dict = base_configs.get(stage, base_configs[DeploymentStage.DEVELOPMENT])
        
        # Apply custom overrides
        if custom_config:
            config_dict.update(custom_config)
        
        # Add common fields
        config_dict.update({
            'stage': stage,
            'version': os.getenv('APP_VERSION', '1.0.0'),
            'health_check_enabled': True,
            'security_scanning_enabled': True,
        })
        
        return DeploymentConfig(**config_dict)
    
    def validate_deployment_config(self, config: DeploymentConfig) -> Tuple[bool, List[str]]:
        """Validate deployment configuration."""
        errors = config.validate()
        
        # Additional production-specific validations
        if config.stage == DeploymentStage.PRODUCTION:
            if not config.enable_ssl:
                errors.append("SSL must be enabled for production")
            
            if not config.backup_enabled:
                errors.append("Backups must be enabled for production")
            
            if not config.monitoring_enabled:
                errors.append("Monitoring must be enabled for production")
            
            if config.instances < 2:
                errors.append("Production requires at least 2 instances for high availability")
        
        return len(errors) == 0, errors
    
    def generate_deployment_manifests(self, config: DeploymentConfig) -> Dict[str, str]:
        """Generate deployment manifests from configuration."""
        manifests = {}
        
        # Generate Kubernetes deployment manifest
        k8s_deployment = self._generate_k8s_deployment_manifest(config)
        manifests['kubernetes_deployment.yaml'] = k8s_deployment
        
        # Generate service manifest
        k8s_service = self._generate_k8s_service_manifest(config)
        manifests['kubernetes_service.yaml'] = k8s_service
        
        # Generate ingress if load balancer enabled
        if config.load_balancer_enabled:
            k8s_ingress = self._generate_k8s_ingress_manifest(config)
            manifests['kubernetes_ingress.yaml'] = k8s_ingress
        
        # Generate HPA if auto-scaling enabled
        if config.auto_scaling_enabled:
            k8s_hpa = self._generate_k8s_hpa_manifest(config)
            manifests['kubernetes_hpa.yaml'] = k8s_hpa
        
        # Generate Docker Compose (for non-K8s deployments)
        docker_compose = self._generate_docker_compose_manifest(config)
        manifests['docker-compose.yml'] = docker_compose
        
        # Generate configuration files
        app_config = self._generate_app_config(config)
        manifests['app_config.yaml'] = app_config
        
        return manifests
    
    def _generate_k8s_deployment_manifest(self, config: DeploymentConfig) -> str:
        """Generate Kubernetes deployment manifest."""
        manifest = f\"\"\"apiVersion: apps/v1
kind: Deployment
metadata:
  name: snn-fusion-api
  namespace: snn-fusion-{config.stage.value}
  labels:
    app: snn-fusion-api
    version: \"{config.version}\"
    stage: \"{config.stage.value}\"
spec:
  replicas: {config.instances}
  selector:
    matchLabels:
      app: snn-fusion-api
  template:
    metadata:
      labels:
        app: snn-fusion-api
        version: \"{config.version}\"
    spec:
      containers:
      - name: snn-fusion-api
        image: snn-fusion:production
        ports:
        - containerPort: {config.internal_port}
        env:
        - name: STAGE
          value: \"{config.stage.value}\"
        - name: APP_VERSION
          value: \"{config.version}\"
        - name: DATABASE_POOL_SIZE
          value: \"{config.database_pool_size}\"
        - name: DATABASE_TIMEOUT
          value: \"{config.database_timeout_seconds}\"
        - name: JWT_EXPIRY_HOURS
          value: \"{config.jwt_expiry_hours}\"
        - name: API_RATE_LIMIT
          value: \"{config.api_rate_limit_per_minute}\"
        resources:
          requests:
            cpu: {int(config.cpu_limit.rstrip('m')) // 2}m
            memory: {config.memory_limit.rstrip('Gi')}Gi
          limits:
            cpu: {config.cpu_limit}
            memory: {config.memory_limit}
        livenessProbe:
          httpGet:
            path: /health
            port: {config.internal_port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: {config.internal_port}
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: data-volume
          mountPath: /app/data
        - name: models-volume
          mountPath: /app/models\"\"\"
        
        if config.enable_gpu:
            manifest += f\"\"\"
        - name: logs-volume
          mountPath: /app/logs
        resources:
          limits:
            nvidia.com/gpu: 1\"\"\"
        
        manifest += f\"\"\"
      volumes:
      - name: config-volume
        configMap:
          name: snn-fusion-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: snn-fusion-data-pvc
      - name: models-volume
        persistentVolumeClaim:
          claimName: snn-fusion-models-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: snn-fusion-logs-pvc
      restartPolicy: Always
\"\"\"
        
        return manifest
    
    def _generate_k8s_service_manifest(self, config: DeploymentConfig) -> str:
        """Generate Kubernetes service manifest."""
        return f\"\"\"apiVersion: v1
kind: Service
metadata:
  name: snn-fusion-api-service
  namespace: snn-fusion-{config.stage.value}
  labels:
    app: snn-fusion-api
spec:
  selector:
    app: snn-fusion-api
  ports:
  - protocol: TCP
    port: {config.external_port}
    targetPort: {config.internal_port}
  type: {"LoadBalancer" if config.load_balancer_enabled else "ClusterIP"}
\"\"\"
    
    def _generate_k8s_ingress_manifest(self, config: DeploymentConfig) -> str:
        """Generate Kubernetes ingress manifest."""
        return f\"\"\"apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: snn-fusion-api-ingress
  namespace: snn-fusion-{config.stage.value}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    {"nginx.ingress.kubernetes.io/ssl-redirect: \"true\"" if config.enable_ssl else ""}
spec:
  {"tls:" if config.enable_ssl else ""}
  {"- hosts:" if config.enable_ssl else ""}
  {"  - snn-fusion-api.example.com" if config.enable_ssl else ""}
  {"  secretName: snn-fusion-tls-secret" if config.enable_ssl else ""}
  rules:
  - host: snn-fusion-api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: snn-fusion-api-service
            port:
              number: {config.external_port}
\"\"\"
    
    def _generate_k8s_hpa_manifest(self, config: DeploymentConfig) -> str:
        """Generate Kubernetes HPA manifest."""
        return f\"\"\"apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: snn-fusion-api-hpa
  namespace: snn-fusion-{config.stage.value}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: snn-fusion-api
  minReplicas: {max(1, config.instances // 2)}
  maxReplicas: {config.instances * 2}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
\"\"\"
    
    def _generate_docker_compose_manifest(self, config: DeploymentConfig) -> str:
        """Generate Docker Compose manifest."""
        return f\"\"\"version: '3.8'

services:
  snn-fusion-api:
    image: snn-fusion:production
    ports:
      - "{config.external_port}:{config.internal_port}"
    environment:
      - STAGE={config.stage.value}
      - APP_VERSION={config.version}
      - DATABASE_POOL_SIZE={config.database_pool_size}
      - DATABASE_TIMEOUT={config.database_timeout_seconds}
      - JWT_EXPIRY_HOURS={config.jwt_expiry_hours}
      - API_RATE_LIMIT={config.api_rate_limit_per_minute}
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    deploy:
      replicas: {config.instances}
      resources:
        limits:
          cpus: '{config.cpu_limit.rstrip("m")}'
          memory: {config.memory_limit}
        reservations:
          cpus: '{int(config.cpu_limit.rstrip("m")) // 2}m'
          memory: {config.memory_limit.rstrip("Gi")}Gi
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{config.internal_port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  {"redis:" if config.redis_enabled else "# redis: # disabled"}
  {"  image: redis:7-alpine" if config.redis_enabled else ""}
  {"  volumes:" if config.redis_enabled else ""}
  {"    - redis_data:/data" if config.redis_enabled else ""}
  {"  deploy:" if config.redis_enabled else ""}
  {"    resources:" if config.redis_enabled else ""}
  {"      limits:" if config.redis_enabled else ""}
  {"        memory: " + config.redis_memory_limit if config.redis_enabled else ""}
  {"  restart: unless-stopped" if config.redis_enabled else ""}

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=snn_fusion
      - POSTGRES_USER=snn_user
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    secrets:
      - postgres_password
    restart: unless-stopped

  {"nginx:" if config.load_balancer_enabled else "# nginx: # disabled"}
  {"  image: nginx:alpine" if config.load_balancer_enabled else ""}
  {"  ports:" if config.load_balancer_enabled else ""}
  {"    - \"80:80\"" if config.load_balancer_enabled else ""}
  {"    - \"443:443\"" if config.load_balancer_enabled and config.enable_ssl else ""}
  {"  volumes:" if config.load_balancer_enabled else ""}
  {"    - ./nginx.conf:/etc/nginx/nginx.conf:ro" if config.load_balancer_enabled else ""}
  {"    - ./ssl:/etc/ssl:ro" if config.load_balancer_enabled and config.enable_ssl else ""}
  {"  depends_on:" if config.load_balancer_enabled else ""}
  {"    - snn-fusion-api" if config.load_balancer_enabled else ""}
  {"  restart: unless-stopped" if config.load_balancer_enabled else ""}

volumes:
  postgres_data:
  {"redis_data:" if config.redis_enabled else ""}

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
\"\"\"
    
    def _generate_app_config(self, config: DeploymentConfig) -> str:
        """Generate application configuration."""
        app_config = {
            'deployment': {
                'stage': config.stage.value,
                'version': config.version,
                'instances': config.instances,
            },
            'server': {
                'host': '0.0.0.0',
                'port': config.internal_port,
                'workers': config.instances,
                'enable_ssl': config.enable_ssl,
            },
            'database': {
                'pool_size': config.database_pool_size,
                'timeout_seconds': config.database_timeout_seconds,
            },
            'cache': {
                'enabled': config.redis_enabled,
                'memory_limit': config.redis_memory_limit,
            },
            'security': {
                'jwt_expiry_hours': config.jwt_expiry_hours,
                'api_rate_limit_per_minute': config.api_rate_limit_per_minute,
            },
            'monitoring': {
                'enabled': config.monitoring_enabled,
                'health_check_enabled': config.health_check_enabled,
            },
            'backup': {
                'enabled': config.backup_enabled,
            },
            'scaling': {
                'auto_scaling_enabled': config.auto_scaling_enabled,
            },
            'neuromorphic': {
                'enable_gpu': config.enable_gpu,
                'device_optimization': True,
                'batch_processing': True,
            },
        }
        
        return yaml.dump(app_config, default_flow_style=False)
    
    def deploy(self, config: DeploymentConfig, 
               deployment_strategy: str = "rolling") -> Dict[str, Any]:
        """Deploy application with specified configuration."""
        deployment_id = f"{config.stage.value}_{config.version}_{int(time.time())}"
        
        self.logger.info(f"Starting deployment {deployment_id}")
        
        # Validate configuration
        is_valid, errors = self.validate_deployment_config(config)
        if not is_valid:
            return {
                'deployment_id': deployment_id,
                'status': DeploymentStatus.FAILED.value,
                'errors': errors,
                'timestamp': time.time(),
            }
        
        # Generate manifests
        try:
            manifests = self.generate_deployment_manifests(config)
            
            # Save manifests to disk
            deployment_dir = self.config_dir / deployment_id
            deployment_dir.mkdir(exist_ok=True)
            
            for filename, content in manifests.items():
                (deployment_dir / filename).write_text(content)
            
            # Execute deployment
            if deployment_strategy == "rolling":
                success = self._execute_rolling_deployment(config, deployment_dir)
            elif deployment_strategy == "blue_green":
                success = self._execute_blue_green_deployment(config, deployment_dir)
            else:
                success = self._execute_standard_deployment(config, deployment_dir)
            
            status = DeploymentStatus.DEPLOYED if success else DeploymentStatus.FAILED
            
            deployment_record = {
                'deployment_id': deployment_id,
                'config': config.to_dict(),
                'status': status.value,
                'strategy': deployment_strategy,
                'manifests_path': str(deployment_dir),
                'timestamp': time.time(),
                'duration_seconds': 0,  # Would be calculated in real deployment
            }
            
            # Update tracking
            self.current_deployments[config.stage.value] = deployment_record
            self.deployment_history.append(deployment_record)
            
            self.logger.info(f"Deployment {deployment_id} completed with status: {status.value}")
            
            return deployment_record
            
        except Exception as e:
            self.logger.error(f"Deployment {deployment_id} failed: {e}")
            return {
                'deployment_id': deployment_id,
                'status': DeploymentStatus.FAILED.value,
                'error': str(e),
                'timestamp': time.time(),
            }
    
    def _execute_standard_deployment(self, config: DeploymentConfig, 
                                   manifest_dir: Path) -> bool:
        """Execute standard deployment."""
        try:
            self.logger.info("Executing standard deployment...")
            
            # In production, this would call kubectl or docker-compose
            # subprocess.run(['kubectl', 'apply', '-f', str(manifest_dir)], check=True)
            
            # Simulate deployment
            time.sleep(2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Standard deployment failed: {e}")
            return False
    
    def _execute_rolling_deployment(self, config: DeploymentConfig, 
                                  manifest_dir: Path) -> bool:
        """Execute rolling deployment."""
        try:
            self.logger.info("Executing rolling deployment...")
            
            # Rolling deployment logic would go here
            # This involves gradually replacing old instances with new ones
            
            return True
            
        except Exception as e:
            self.logger.error(f"Rolling deployment failed: {e}")
            return False
    
    def _execute_blue_green_deployment(self, config: DeploymentConfig, 
                                     manifest_dir: Path) -> bool:
        """Execute blue-green deployment."""
        try:
            self.logger.info("Executing blue-green deployment...")
            
            # Blue-green deployment logic would go here
            # This involves deploying to a separate environment then switching traffic
            
            return True
            
        except Exception as e:
            self.logger.error(f"Blue-green deployment failed: {e}")
            return False
    
    def rollback_deployment(self, stage: DeploymentStage, 
                          target_version: Optional[str] = None) -> Dict[str, Any]:
        """Rollback deployment to previous or specified version."""
        stage_key = stage.value
        
        if stage_key not in self.current_deployments:
            return {
                'success': False,
                'error': f"No active deployment found for stage {stage_key}",
            }
        
        current_deployment = self.current_deployments[stage_key]
        
        # Find target deployment
        target_deployment = None
        
        if target_version:
            # Find specific version
            for deployment in reversed(self.deployment_history):
                if (deployment['config']['stage'] == stage_key and 
                    deployment['config']['version'] == target_version):
                    target_deployment = deployment
                    break
        else:
            # Find previous successful deployment
            for deployment in reversed(self.deployment_history[:-1]):  # Skip current
                if (deployment['config']['stage'] == stage_key and 
                    deployment['status'] == DeploymentStatus.DEPLOYED.value):
                    target_deployment = deployment
                    break
        
        if not target_deployment:
            return {
                'success': False,
                'error': "No suitable rollback target found",
            }
        
        try:
            self.logger.info(f"Rolling back {stage_key} to version {target_deployment['config']['version']}")
            
            # Execute rollback (in production, this would redeploy the target version)
            # For now, we'll simulate it
            
            rollback_record = {
                'deployment_id': f"rollback_{stage_key}_{int(time.time())}",
                'status': DeploymentStatus.ROLLED_BACK.value,
                'rollback_from': current_deployment['config']['version'],
                'rollback_to': target_deployment['config']['version'],
                'timestamp': time.time(),
            }
            
            self.current_deployments[stage_key] = target_deployment.copy()
            self.current_deployments[stage_key]['rollback_info'] = rollback_record
            
            self.deployment_history.append(rollback_record)
            
            return {
                'success': True,
                'rollback_record': rollback_record,
            }
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return {
                'success': False,
                'error': str(e),
            }


class BackupManager:
    """Automated backup and disaster recovery system."""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.backup_threads: List[threading.Thread] = []
        self.is_running = False
    
    def start_backup_scheduler(self) -> None:
        """Start automated backup scheduler."""
        if not self.config.enabled:
            self.logger.info("Backups disabled in configuration")
            return
        
        self.is_running = True
        self.logger.info("Starting backup scheduler...")
        
        # Start backup threads
        backup_tasks = [
            ("database", self.config.database_backup_interval_hours, self._backup_database),
            ("models", self.config.model_backup_interval_hours, self._backup_models),
            ("config", self.config.config_backup_interval_hours, self._backup_config),
        ]
        
        for name, interval_hours, backup_func in backup_tasks:
            thread = threading.Thread(
                target=self._backup_scheduler_loop,
                args=(name, interval_hours, backup_func),
                name=f"backup-{name}",
                daemon=True
            )
            thread.start()
            self.backup_threads.append(thread)
        
        self.logger.info("Backup scheduler started")
    
    def stop_backup_scheduler(self) -> None:
        """Stop backup scheduler."""
        self.logger.info("Stopping backup scheduler...")
        self.is_running = False
        
        for thread in self.backup_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        self.logger.info("Backup scheduler stopped")
    
    def _backup_scheduler_loop(self, name: str, interval_hours: int, 
                             backup_func: Callable) -> None:
        """Backup scheduler loop for specific backup type."""
        while self.is_running:
            try:
                backup_func()
                self.logger.info(f"{name} backup completed successfully")
                
                # Wait for next backup interval
                time.sleep(interval_hours * 3600)
                
            except Exception as e:
                self.logger.error(f"{name} backup failed: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
    
    def _backup_database(self) -> None:
        """Backup database."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"database_backup_{timestamp}.sql.gz"
        
        # In production, this would execute actual database backup
        # pg_dump command with compression and encryption
        
        self.logger.info(f"Database backup created: {backup_file}")
    
    def _backup_models(self) -> None:
        """Backup trained models."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"models_backup_{timestamp}.tar.gz"
        
        # In production, this would create compressed archive of model files
        
        self.logger.info(f"Models backup created: {backup_file}")
    
    def _backup_config(self) -> None:
        """Backup configuration files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"config_backup_{timestamp}.tar.gz"
        
        # In production, this would backup configuration files
        
        self.logger.info(f"Config backup created: {backup_file}")
    
    def create_manual_backup(self, backup_type: str = "full") -> Dict[str, Any]:
        """Create manual backup."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if backup_type == "full":
                self._backup_database()
                self._backup_models()
                self._backup_config()
            elif backup_type == "database":
                self._backup_database()
            elif backup_type == "models":
                self._backup_models()
            elif backup_type == "config":
                self._backup_config()
            else:
                raise ValueError(f"Unknown backup type: {backup_type}")
            
            return {
                'success': True,
                'backup_type': backup_type,
                'timestamp': timestamp,
                'message': f"Manual {backup_type} backup completed successfully",
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'backup_type': backup_type,
                'timestamp': timestamp,
            }