# SNN-Fusion Production Deployment Guide

This guide provides comprehensive instructions for deploying SNN-Fusion in production environments across multiple platforms and configurations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Neuromorphic Hardware Integration](#neuromorphic-hardware-integration)
6. [Monitoring & Observability](#monitoring--observability)
7. [Security Configuration](#security-configuration)
8. [Multi-Region Setup](#multi-region-setup)
9. [Troubleshooting](#troubleshooting)
10. [Performance Optimization](#performance-optimization)

## Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Python 3.11+
- At least 8GB RAM and 4 CPU cores
- SSL certificates for production domains
- Database (PostgreSQL 15+)
- Redis 7+

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/snn-fusion.git
cd snn-fusion
```

2. Copy and configure environment variables:
```bash
cp .env.example .env.production
# Edit .env.production with your production values
```

3. Generate SSL certificates:
```bash
# Using Let's Encrypt
certbot certonly --standalone -d your-domain.com
```

## Docker Deployment

### Single Node Deployment

1. **Configure environment variables** in `.env.production`:
```bash
# Database Configuration
DATABASE_URL=postgresql://snn_user:secure_password@postgres:5432/snn_fusion
POSTGRES_PASSWORD=secure_password

# Redis Configuration
REDIS_URL=redis://:redis_password@redis:6379/0
REDIS_PASSWORD=redis_password

# Security
SECRET_KEY=your-very-secure-secret-key-here
JWT_SECRET_KEY=another-secure-jwt-key

# Monitoring
ENABLE_MONITORING=true
ENABLE_TRACING=true

# Grafana
GRAFANA_PASSWORD=admin_password

# RabbitMQ
RABBITMQ_PASSWORD=rabbitmq_password
```

2. **Deploy the stack**:
```bash
cd deployment/production
docker-compose --env-file .env.production up -d
```

3. **Verify deployment**:
```bash
# Check all services are running
docker-compose ps

# Check API health
curl -f http://localhost/health

# Check logs
docker-compose logs -f snn-fusion-api
```

### Multi-Node Deployment with Docker Swarm

1. **Initialize Docker Swarm**:
```bash
docker swarm init --advertise-addr <manager-ip>
```

2. **Deploy as a stack**:
```bash
docker stack deploy -c docker-compose.yml -c docker-compose.swarm.yml snn-fusion
```

3. **Scale services**:
```bash
docker service scale snn-fusion_snn-fusion-api=5
docker service scale snn-fusion_snn-fusion-worker=3
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes 1.25+
- kubectl configured
- Ingress controller (NGINX recommended)
- cert-manager for SSL certificates
- Persistent volumes for data storage

### Step-by-Step Deployment

1. **Create namespace**:
```bash
kubectl create namespace snn-fusion
```

2. **Deploy secrets**:
```bash
kubectl create secret generic snn-fusion-secrets \
  --from-literal=database-url="postgresql://user:pass@postgres:5432/snn_fusion" \
  --from-literal=redis-url="redis://:password@redis:6379/0" \
  --from-literal=secret-key="your-secret-key" \
  -n snn-fusion
```

3. **Deploy ConfigMap**:
```bash
kubectl apply -f deployment/kubernetes/configmap.yaml
```

4. **Deploy persistent volumes**:
```bash
kubectl apply -f deployment/kubernetes/volumes.yaml
```

5. **Deploy PostgreSQL**:
```bash
kubectl apply -f deployment/kubernetes/postgres.yaml
```

6. **Deploy Redis**:
```bash
kubectl apply -f deployment/kubernetes/redis.yaml
```

7. **Deploy SNN-Fusion API**:
```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
```

8. **Verify deployment**:
```bash
kubectl get pods -n snn-fusion
kubectl get services -n snn-fusion
kubectl get ingress -n snn-fusion

# Check pod logs
kubectl logs -f deployment/snn-fusion-api -n snn-fusion
```

### Auto-Scaling Configuration

The Kubernetes deployment includes Horizontal Pod Autoscaler (HPA):
- Minimum replicas: 3
- Maximum replicas: 20
- CPU target: 70%
- Memory target: 80%

Monitor scaling events:
```bash
kubectl get hpa -n snn-fusion -w
kubectl describe hpa snn-fusion-api-hpa -n snn-fusion
```

## Cloud Deployment

### AWS Deployment

#### Using EKS (Elastic Kubernetes Service)

1. **Create EKS cluster**:
```bash
eksctl create cluster --name snn-fusion-cluster \
  --version 1.25 \
  --nodegroup-name standard-workers \
  --node-type m5.2xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed
```

2. **Configure kubectl**:
```bash
aws eks update-kubeconfig --region us-east-1 --name snn-fusion-cluster
```

3. **Deploy using Helm**:
```bash
helm repo add snn-fusion https://charts.snn-fusion.com
helm install snn-fusion snn-fusion/snn-fusion \
  --namespace snn-fusion \
  --create-namespace \
  --values deployment/helm/aws-values.yaml
```

#### Using ECS (Elastic Container Service)

1. **Create ECS cluster**:
```bash
aws ecs create-cluster --cluster-name snn-fusion-cluster
```

2. **Deploy using CDK**:
```bash
cd deployment/aws-cdk
npm install
cdk deploy SnnFusionStack
```

### Azure Deployment

#### Using AKS (Azure Kubernetes Service)

1. **Create resource group**:
```bash
az group create --name snn-fusion-rg --location eastus
```

2. **Create AKS cluster**:
```bash
az aks create \
  --resource-group snn-fusion-rg \
  --name snn-fusion-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys
```

3. **Get credentials**:
```bash
az aks get-credentials --resource-group snn-fusion-rg --name snn-fusion-cluster
```

### Google Cloud Platform

#### Using GKE (Google Kubernetes Engine)

1. **Create cluster**:
```bash
gcloud container clusters create snn-fusion-cluster \
  --num-nodes=3 \
  --machine-type=e2-standard-4 \
  --enable-autorepair \
  --enable-autoupgrade \
  --zone=us-central1-a
```

2. **Get credentials**:
```bash
gcloud container clusters get-credentials snn-fusion-cluster --zone=us-central1-a
```

## Neuromorphic Hardware Integration

### Intel Loihi 2 Integration

1. **Install Loihi drivers** (requires Intel NRC access):
```bash
# This requires special access and NDA with Intel
# Contact Intel Neuromorphic Research Community
```

2. **Configure hardware access**:
```yaml
# In docker-compose.yml
neuromorphic-interface:
  environment:
    - NEUROMORPHIC_HARDWARE_TYPE=loihi2
    - LOIHI_DEVICE_ADDRESS=/dev/loihi0
  devices:
    - "/dev/loihi0:/dev/loihi0"
  profiles:
    - neuromorphic
```

3. **Deploy with neuromorphic support**:
```bash
docker-compose --profile neuromorphic up -d
```

### BrainChip Akida Integration

1. **Install Akida runtime**:
```bash
pip install akida
```

2. **Configure Akida device**:
```yaml
neuromorphic-interface:
  environment:
    - NEUROMORPHIC_HARDWARE_TYPE=akida
    - AKIDA_DEVICE_ID=0
```

### SpiNNaker 2 Integration

1. **Install SpiNNaker tools**:
```bash
pip install spinnaker2
```

2. **Configure SpiNNaker board**:
```yaml
neuromorphic-interface:
  environment:
    - NEUROMORPHIC_HARDWARE_TYPE=spinnaker2
    - SPINNAKER_BOARD_ADDRESS=192.168.1.100
```

## Monitoring & Observability

### Prometheus Configuration

The deployment includes comprehensive monitoring:
- **Application metrics**: Request rates, response times, error rates
- **System metrics**: CPU, memory, disk, network
- **SNN-specific metrics**: Spike rates, model accuracy, hardware utilization
- **Business metrics**: Multi-modal processing rates, regional performance

### Grafana Dashboards

Access Grafana at `http://your-domain:3000`:
- Username: `admin`
- Password: Set in `GRAFANA_PASSWORD`

Pre-configured dashboards:
- **SNN-Fusion Overview**: Main production dashboard
- **Neural Network Performance**: Model training and inference metrics
- **Multi-Modal Processing**: Audio, visual, tactile processing rates
- **Hardware Utilization**: Neuromorphic hardware monitoring
- **Regional Performance**: Multi-region deployment status

### Log Management

Logs are collected using the ELK stack:
- **Elasticsearch**: Log storage and indexing
- **Logstash**: Log processing pipeline
- **Kibana**: Log visualization and analysis

Access Kibana at `http://your-domain:5601`.

### Distributed Tracing

Jaeger provides distributed tracing:
- Access at `http://your-domain:16686`
- Traces request flow across microservices
- Performance bottleneck identification

## Security Configuration

### SSL/TLS Setup

1. **Generate certificates**:
```bash
# Let's Encrypt (recommended)
certbot certonly --standalone -d api.your-domain.com

# Or use your corporate CA
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt
```

2. **Configure NGINX**:
```nginx
server {
    listen 443 ssl http2;
    server_name api.your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/tls.crt;
    ssl_certificate_key /etc/nginx/ssl/tls.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
}
```

### API Authentication

Configure JWT authentication:
```bash
# Generate secure keys
JWT_SECRET_KEY=$(openssl rand -base64 64)
REFRESH_TOKEN_SECRET=$(openssl rand -base64 64)
```

### Network Security

1. **Firewall rules**:
```bash
# Allow only necessary ports
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 22/tcp
ufw deny 5432/tcp  # Postgres (internal only)
ufw deny 6379/tcp  # Redis (internal only)
```

2. **VPC/Network isolation** (Cloud):
- Place databases in private subnets
- Use security groups/network policies
- Enable VPC flow logs

## Multi-Region Setup

### Regional Deployment

1. **Configure regions** in `regional_config.yaml`:
```yaml
regions:
  us-east-1:
    primary: true
    endpoint: https://us-east-1.api.your-domain.com
    compliance: ["CCPA"]
  
  eu-west-1:
    endpoint: https://eu-west-1.api.your-domain.com  
    compliance: ["GDPR"]
    
  ap-southeast-1:
    endpoint: https://ap-southeast-1.api.your-domain.com
    compliance: ["PDPA"]
```

2. **Deploy regional instances**:
```bash
# Deploy to each region
for region in us-east-1 eu-west-1 ap-southeast-1; do
  export AWS_DEFAULT_REGION=$region
  helm install snn-fusion-$region snn-fusion/snn-fusion \
    --namespace snn-fusion \
    --values deployment/helm/region-$region-values.yaml
done
```

3. **Configure global load balancer**:
```yaml
# Global load balancer (e.g., AWS Route 53, Cloudflare)
routing_policy: latency_based
health_checks: enabled
failover: automatic
```

### Data Sovereignty Compliance

Configure data residency rules:
```python
# In application configuration
DATA_SOVEREIGNTY_RULES = {
    "GDPR": {
        "allowed_regions": ["eu-west-1", "eu-central-1"],
        "prohibited_regions": ["us-east-1"]
    },
    "CCPA": {
        "allowed_regions": ["us-east-1", "us-west-2"]
    }
}
```

## Troubleshooting

### Common Issues

#### 1. Application Won't Start

**Symptoms**: Container exits immediately
**Diagnosis**:
```bash
docker-compose logs snn-fusion-api
kubectl logs deployment/snn-fusion-api -n snn-fusion
```

**Common causes**:
- Database connection failure
- Missing environment variables
- Port conflicts
- Insufficient resources

#### 2. High Memory Usage

**Symptoms**: OOM kills, slow performance
**Diagnosis**:
```bash
# Check memory usage
docker stats
kubectl top pods -n snn-fusion

# Check for memory leaks
curl http://localhost:8000/debug/memory
```

**Solutions**:
- Increase memory limits
- Enable memory optimization
- Check for data leaks in model training

#### 3. Database Connection Issues

**Symptoms**: Connection timeouts, authentication failures
**Diagnosis**:
```bash
# Test database connectivity
docker-compose exec snn-fusion-api python -c "
import psycopg2
conn = psycopg2.connect('$DATABASE_URL')
print('Connection successful')
"
```

#### 4. Neuromorphic Hardware Not Detected

**Symptoms**: Hardware initialization failures
**Diagnosis**:
```bash
# Check device permissions
ls -la /dev/loihi*
ls -la /dev/akida*

# Check driver loading
lsmod | grep loihi
lsmod | grep akida
```

### Performance Issues

#### High Latency

1. **Check database queries**:
```bash
# Enable slow query logging
docker-compose exec postgres psql -U snn_user -d snn_fusion
postgres=# SET log_min_duration_statement = 100;
```

2. **Monitor cache hit rates**:
```bash
# Redis cache stats
docker-compose exec redis redis-cli info stats
```

3. **Profile application**:
```bash
# Enable profiling
curl -X POST http://localhost:8000/debug/profiling/start
# Run workload
curl -X POST http://localhost:8000/debug/profiling/stop
curl http://localhost:8000/debug/profiling/results
```

## Performance Optimization

### Resource Tuning

#### Database Optimization

1. **PostgreSQL tuning**:
```sql
-- Connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET work_mem = '256MB';

-- For SNN workloads
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
```

2. **Create indexes for SNN queries**:
```sql
CREATE INDEX idx_spike_data_timestamp ON spike_data(timestamp);
CREATE INDEX idx_neural_patterns_hash ON neural_patterns USING hash(pattern_hash);
```

#### Application Tuning

1. **Memory optimization**:
```yaml
environment:
  - SNN_MEMORY_OPTIMIZATION=aggressive
  - TORCH_MEMORY_POOL_SIZE=2GB
  - CUDA_MEMORY_FRACTION=0.8
```

2. **Concurrency settings**:
```yaml
environment:
  - WORKER_PROCESSES=4
  - WORKER_THREADS=8
  - ASYNC_POOL_SIZE=100
```

### Neuromorphic Hardware Optimization

#### Intel Loihi 2

```python
# Optimize for Loihi 2 deployment
loihi_config = {
    'compartment_sharing': True,
    'axon_sharing': True,
    'synapse_compression': True,
    'learning_rule_optimization': 'stdp_optimized'
}
```

#### BrainChip Akida

```python
# Optimize for Akida deployment
akida_config = {
    'quantization': 'int8',
    'sparsity_optimization': True,
    'batch_processing': True,
    'hardware_acceleration': 'maximum'
}
```

### Multi-Modal Optimization

```yaml
multi_modal_config:
  audio:
    preprocessing: 'optimized'
    encoding: 'sparse_temporal'
    buffer_size: '1MB'
  
  visual:
    preprocessing: 'gpu_accelerated'
    encoding: 'event_based'
    compression: 'lossless'
  
  tactile:
    preprocessing: 'realtime'
    encoding: 'population_vector'
    sampling_rate: '1000Hz'
```

## Support and Maintenance

### Health Monitoring

Set up automated health checks:
```bash
# API health endpoint
curl -f http://localhost:8000/health

# Database health
docker-compose exec postgres pg_isready

# Neuromorphic hardware health
curl http://localhost:8000/hardware/status
```

### Backup Strategy

1. **Database backups**:
```bash
# Daily automated backups
0 2 * * * docker-compose exec postgres pg_dump -U snn_user snn_fusion > backup_$(date +%Y%m%d).sql
```

2. **Model backups**:
```bash
# Backup trained models
rsync -av /app/models/ s3://your-bucket/models/$(date +%Y%m%d)/
```

### Updates and Rollbacks

1. **Rolling updates**:
```bash
# Kubernetes rolling update
kubectl set image deployment/snn-fusion-api snn-fusion-api=snn-fusion:v1.1.0 -n snn-fusion

# Docker Compose update
docker-compose pull
docker-compose up -d --no-deps snn-fusion-api
```

2. **Rollback procedure**:
```bash
# Kubernetes rollback
kubectl rollout undo deployment/snn-fusion-api -n snn-fusion

# Docker Compose rollback
docker-compose down
docker tag snn-fusion:v1.0.0 snn-fusion:latest
docker-compose up -d
```

---

## Additional Resources

- [SNN-Fusion Architecture Documentation](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Performance Benchmarks](BENCHMARKS.md)

For support, please contact the SNN-Fusion team or create an issue in the repository.