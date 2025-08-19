# SNN Fusion Production Deployment Guide

## Overview

This document provides comprehensive instructions for deploying the SNN Fusion neuromorphic processing system to production environments.

## Architecture

The SNN Fusion system is deployed as a microservices architecture with the following components:

- **API Service**: Main neuromorphic processing API
- **Monitoring**: Prometheus + Grafana stack
- **Caching**: Redis for performance optimization
- **Load Balancing**: Kubernetes ingress with auto-scaling

## Prerequisites

### Required Tools
- kubectl (Kubernetes CLI)
- docker (Container runtime)
- helm (Package manager for Kubernetes)

### Cluster Requirements
- Kubernetes 1.20+
- 4 CPU cores minimum
- 8GB RAM minimum
- 50GB storage
- Ingress controller (nginx recommended)

## Deployment Steps

### 1. Build and Push Container Image

```bash
# Build production image
docker build -f deployment_production/Dockerfile -t snn-fusion:latest .

# Tag and push to registry
docker tag snn-fusion:latest your-registry.com/snn-fusion:latest
docker push your-registry.com/snn-fusion:latest
```

### 2. Deploy to Kubernetes

```bash
# Execute deployment script
cd deployment_production/scripts
./deploy.sh
```

### 3. Verify Deployment

```bash
# Check deployment status
./health_check.sh

# Monitor logs
kubectl logs -f deployment/snn-fusion-api -n snn-fusion
```

## Configuration

### Environment Variables

- `SNN_FUSION_ENV`: Environment (production/staging/development)
- `SNN_FUSION_LOG_LEVEL`: Logging level (INFO/DEBUG/WARNING/ERROR)
- `SNN_FUSION_WORKERS`: Number of worker processes
- `SNN_FUSION_PORT`: Application port (default: 8080)

### Resource Limits

Current configuration:
- CPU Request: 500m
- Memory Request: 1Gi
- CPU Limit: 1000m
- Memory Limit: 2Gi

### Auto-scaling

- Minimum Replicas: 3
- Maximum Replicas: 10
- CPU Target: 70%

## Monitoring

### Prometheus Metrics

The system exposes the following metrics:

- `snn_fusion_requests_total`: Total API requests
- `snn_fusion_request_duration_seconds`: Request latency
- `snn_fusion_spikes_processed_total`: Neuromorphic spikes processed
- `snn_fusion_fusion_quality_score`: Fusion quality metrics

### Grafana Dashboards

Access Grafana at: http://localhost:3000
- Username: admin
- Password: admin

### Alerts

Critical alerts are configured for:
- High latency (>10ms p95)
- High error rate (>5%)
- Pod failures
- High memory usage (>80%)

## Security

### Pod Security

- Non-root containers
- Read-only filesystem
- Dropped capabilities
- Network policies

### RBAC

Service accounts with minimal required permissions.

### Secrets Management

Sensitive configuration stored in Kubernetes secrets.

## Operations

### Scaling

```bash
# Manual scaling
./scaling.sh scale 5

# Update HPA
./scaling.sh hpa 3 10 70
```

### Updates

```bash
# Rolling update
kubectl set image deployment/snn-fusion-api snn-fusion-api=snn-fusion:v2.0 -n snn-fusion
```

### Backup

```bash
# Backup configurations
kubectl get all -n snn-fusion -o yaml > backup.yaml
```

## Troubleshooting

### Common Issues

1. **Pods not starting**: Check resource limits and node capacity
2. **High latency**: Review HPA settings and resource allocation
3. **Memory leaks**: Monitor memory usage and restart pods if needed

### Debugging Commands

```bash
# Check pod status
kubectl get pods -n snn-fusion

# View logs
kubectl logs -f deployment/snn-fusion-api -n snn-fusion

# Describe problematic resources
kubectl describe pod <pod-name> -n snn-fusion

# Check resource usage
kubectl top pods -n snn-fusion
```

## Performance Tuning

### Optimization Guidelines

1. **CPU**: Monitor CPU usage and adjust requests/limits
2. **Memory**: Set appropriate JVM heap size
3. **Network**: Tune ingress and service mesh settings
4. **Storage**: Use SSD storage for better I/O performance

## Support

For production support:
- Slack: #snn-fusion-ops
- Email: ops@terragonlabs.com
- On-call: Pager duty integration

## Changelog

- v1.0: Initial production deployment
- v1.1: Added auto-scaling and monitoring
- v1.2: Enhanced security and compliance
