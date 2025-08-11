#!/bin/bash
#
# SNN Fusion Framework - Production Deployment Script
# Automates the deployment process for production environments
#

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="/tmp/snn_fusion_deploy_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${1}" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} ${1}"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} ${1}"
}

log_warning() {
    log "${YELLOW}[WARNING]${NC} ${1}"
}

log_error() {
    log "${RED}[ERROR]${NC} ${1}"
}

# Help function
show_help() {
    cat << EOF
SNN Fusion Framework - Production Deployment Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -t, --type TYPE         Deployment type: docker|kubernetes|aws|gcp|azure
    -e, --env ENV          Environment: staging|production
    -g, --gpu              Enable GPU support
    -m, --monitoring       Enable full monitoring stack
    -b, --backup           Configure backup services
    -s, --ssl              Configure SSL/TLS
    --dry-run              Show what would be done without executing
    --force                Skip confirmation prompts

Examples:
    $0 --type docker --env production --gpu --monitoring
    $0 --type kubernetes --env staging --ssl
    $0 --type aws --env production --backup --monitoring
EOF
}

# Default configuration
DEPLOYMENT_TYPE=""
ENVIRONMENT="production"
ENABLE_GPU=false
ENABLE_MONITORING=false
ENABLE_BACKUP=false
ENABLE_SSL=false
DRY_RUN=false
FORCE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -g|--gpu)
            ENABLE_GPU=true
            shift
            ;;
        -m|--monitoring)
            ENABLE_MONITORING=true
            shift
            ;;
        -b|--backup)
            ENABLE_BACKUP=true
            shift
            ;;
        -s|--ssl)
            ENABLE_SSL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$DEPLOYMENT_TYPE" ]]; then
    log_error "Deployment type is required. Use -t or --type"
    show_help
    exit 1
fi

if [[ ! "$DEPLOYMENT_TYPE" =~ ^(docker|kubernetes|aws|gcp|azure)$ ]]; then
    log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
    show_help
    exit 1
fi

# Header
log_info "🚀 SNN Fusion Framework - Production Deployment"
log_info "================================================="
log_info "Deployment Type: $DEPLOYMENT_TYPE"
log_info "Environment: $ENVIRONMENT"
log_info "GPU Support: $ENABLE_GPU"
log_info "Monitoring: $ENABLE_MONITORING"
log_info "Backup: $ENABLE_BACKUP"
log_info "SSL/TLS: $ENABLE_SSL"
log_info "Dry Run: $DRY_RUN"
log_info "Log File: $LOG_FILE"
log_info ""

# Change to project root
cd "$PROJECT_ROOT"

# Pre-deployment checks
check_prerequisites() {
    log_info "📋 Checking prerequisites..."
    
    # Check if running as root (not recommended)
    if [[ $EUID -eq 0 ]] && [[ "$FORCE" != true ]]; then
        log_warning "Running as root is not recommended for production deployments"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Check required commands based on deployment type
    case "$DEPLOYMENT_TYPE" in
        docker)
            command -v docker >/dev/null 2>&1 || { log_error "Docker is required but not installed"; exit 1; }
            command -v docker-compose >/dev/null 2>&1 || command -v docker compose >/dev/null 2>&1 || { log_error "Docker Compose is required but not installed"; exit 1; }
            ;;
        kubernetes)
            command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required but not installed"; exit 1; }
            ;;
        aws)
            command -v aws >/dev/null 2>&1 || { log_error "AWS CLI is required but not installed"; exit 1; }
            command -v eksctl >/dev/null 2>&1 || { log_error "eksctl is required but not installed"; exit 1; }
            ;;
        gcp)
            command -v gcloud >/dev/null 2>&1 || { log_error "Google Cloud SDK is required but not installed"; exit 1; }
            ;;
        azure)
            command -v az >/dev/null 2>&1 || { log_error "Azure CLI is required but not installed"; exit 1; }
            ;;
    esac
    
    # Check GPU support
    if [[ "$ENABLE_GPU" == true ]]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            log_info "✅ NVIDIA GPU detected"
        else
            log_warning "⚠️  NVIDIA drivers not found - GPU support may not work"
        fi
    fi
    
    log_success "Prerequisites check completed"
}

# Setup environment configuration
setup_environment() {
    log_info "🔧 Setting up environment configuration..."
    
    # Check if .env already exists
    if [[ -f ".env" ]] && [[ "$FORCE" != true ]]; then
        log_warning "Environment file .env already exists"
        read -p "Overwrite existing configuration? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Using existing .env file"
            return
        fi
    fi
    
    # Copy environment template
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would copy .env.production to .env"
    else
        cp .env.production .env
        log_success "Environment template copied to .env"
    fi
    
    # Generate secure secrets
    if [[ "$DRY_RUN" != true ]]; then
        SECRET_KEY=$(openssl rand -hex 32)
        JWT_SECRET=$(openssl rand -hex 32)
        REDIS_PASSWORD=$(openssl rand -hex 16)
        POSTGRES_PASSWORD=$(openssl rand -hex 16)
        GRAFANA_PASSWORD=$(openssl rand -hex 16)
        
        # Update .env file with generated secrets
        sed -i "s/your-super-secret-key-change-this-in-production/$SECRET_KEY/" .env
        sed -i "s/your-jwt-secret-key-change-this-too/$JWT_SECRET/" .env
        sed -i "s/your-redis-password/$REDIS_PASSWORD/" .env
        sed -i "s/your-postgres-password/$POSTGRES_PASSWORD/" .env
        sed -i "s/your-grafana-admin-password/$GRAFANA_PASSWORD/" .env
        
        log_success "Secure secrets generated and configured"
        log_warning "IMPORTANT: Save these credentials securely!"
        echo "  Grafana Admin Password: $GRAFANA_PASSWORD"
    fi
    
    # Configure GPU support
    if [[ "$ENABLE_GPU" == true ]]; then
        if [[ "$DRY_RUN" != true ]]; then
            sed -i "s/ENABLE_GPU=false/ENABLE_GPU=true/" .env
        fi
        log_info "GPU support enabled in configuration"
    fi
    
    # Configure monitoring
    if [[ "$ENABLE_MONITORING" == true ]]; then
        if [[ "$DRY_RUN" != true ]]; then
            sed -i "s/ENABLE_MONITORING=false/ENABLE_MONITORING=true/" .env
        fi
        log_info "Monitoring enabled in configuration"
    fi
}

# Create required directories
create_directories() {
    log_info "📁 Creating required directories..."
    
    DIRS=(
        "volumes/postgres"
        "volumes/redis" 
        "volumes/prometheus"
        "volumes/grafana"
        "volumes/elasticsearch"
        "logs"
        "data"
        "models"
        "config"
        "backups"
    )
    
    for dir in "${DIRS[@]}"; do
        if [[ "$DRY_RUN" == true ]]; then
            log_info "[DRY RUN] Would create directory: $dir"
        else
            mkdir -p "$dir"
            log_info "✅ Created directory: $dir"
        fi
    done
    
    log_success "Directory structure created"
}

# Build Docker images
build_images() {
    if [[ "$DEPLOYMENT_TYPE" != "docker" && "$DEPLOYMENT_TYPE" != "kubernetes" ]]; then
        return
    fi
    
    log_info "🐳 Building Docker images..."
    
    if [[ "$ENABLE_GPU" == true ]]; then
        BUILD_TARGET="gpu"
        BUILD_ARGS="--build-arg ENABLE_GPU=true"
    else
        BUILD_TARGET="production"
        BUILD_ARGS=""
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would build image: snn-fusion:$BUILD_TARGET"
    else
        docker build -f deploy/production/Dockerfile -t "snn-fusion:$BUILD_TARGET" --target "$BUILD_TARGET" $BUILD_ARGS . \
            || { log_error "Failed to build Docker image"; exit 1; }
        log_success "Docker image built successfully"
    fi
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "🐳 Deploying with Docker Compose..."
    
    COMPOSE_FILE="deploy/production/docker-compose.production.yml"
    
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would start services with: docker-compose -f $COMPOSE_FILE up -d"
    else
        # Start services
        docker-compose -f "$COMPOSE_FILE" up -d \
            || { log_error "Failed to start Docker services"; exit 1; }
        
        # Wait for services to be ready
        log_info "⏳ Waiting for services to start..."
        sleep 30
        
        # Health check
        if python3 scripts/health_check.py --quiet; then
            log_success "Docker deployment completed successfully"
        else
            log_error "Health check failed after deployment"
            docker-compose -f "$COMPOSE_FILE" logs --tail=50
            exit 1
        fi
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "☸️  Deploying to Kubernetes..."
    
    NAMESPACE="snn-fusion-prod"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would deploy to Kubernetes namespace: $NAMESPACE"
        return
    fi
    
    # Create namespace
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets from .env file
    kubectl create secret generic snn-fusion-secrets \
        --from-env-file=.env \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy application
    kubectl apply -f deploy/kubernetes/deployment.yaml -n "$NAMESPACE" \
        || { log_error "Failed to deploy to Kubernetes"; exit 1; }
    
    # Wait for deployment
    kubectl rollout status deployment/snn-fusion-api -n "$NAMESPACE" --timeout=300s \
        || { log_error "Kubernetes deployment timeout"; exit 1; }
    
    log_success "Kubernetes deployment completed successfully"
}

# Deploy to cloud providers
deploy_cloud() {
    case "$DEPLOYMENT_TYPE" in
        aws)
            log_info "☁️  Deploying to AWS..."
            # AWS-specific deployment logic would go here
            log_warning "AWS deployment not implemented in this script"
            ;;
        gcp)
            log_info "☁️  Deploying to Google Cloud Platform..."
            # GCP-specific deployment logic would go here
            log_warning "GCP deployment not implemented in this script"
            ;;
        azure)
            log_info "☁️  Deploying to Microsoft Azure..."
            # Azure-specific deployment logic would go here
            log_warning "Azure deployment not implemented in this script"
            ;;
    esac
}

# Setup SSL/TLS
setup_ssl() {
    if [[ "$ENABLE_SSL" != true ]]; then
        return
    fi
    
    log_info "🔒 Setting up SSL/TLS..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would configure SSL/TLS certificates"
        return
    fi
    
    # This is a simplified example - in production, you'd use proper certificate management
    log_warning "SSL setup requires manual configuration of certificates"
    log_info "Please refer to the Production Deployment Guide for detailed SSL setup instructions"
}

# Configure monitoring
setup_monitoring() {
    if [[ "$ENABLE_MONITORING" != true ]]; then
        return
    fi
    
    log_info "📊 Setting up monitoring..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would configure monitoring services"
        return
    fi
    
    case "$DEPLOYMENT_TYPE" in
        docker)
            # Monitoring is included in docker-compose.yml
            log_info "Monitoring services started with Docker Compose"
            log_info "Grafana Dashboard: http://localhost:3000"
            log_info "Prometheus Metrics: http://localhost:9090"
            ;;
        kubernetes)
            # Deploy monitoring stack to Kubernetes
            if [[ -d "deploy/kubernetes/monitoring" ]]; then
                kubectl apply -f deploy/kubernetes/monitoring/ -n "$NAMESPACE"
                log_success "Monitoring stack deployed to Kubernetes"
            fi
            ;;
    esac
}

# Configure backups
setup_backup() {
    if [[ "$ENABLE_BACKUP" != true ]]; then
        return
    fi
    
    log_info "💾 Setting up backup services..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would configure backup services"
        return
    fi
    
    # Create backup scripts
    mkdir -p scripts/backup
    
    # Database backup script
    cat > scripts/backup/database_backup.sh << 'EOF'
#!/bin/bash
# Automated database backup script
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="/app/backups"
docker-compose exec postgres pg_dump -U $POSTGRES_USER $POSTGRES_DB | gzip > "$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz"
find "$BACKUP_DIR" -name "db_backup_*.sql.gz" -mtime +7 -delete
EOF
    
    chmod +x scripts/backup/database_backup.sh
    log_success "Backup scripts created"
}

# Post-deployment verification
post_deployment_check() {
    log_info "🔍 Running post-deployment verification..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would run post-deployment checks"
        return
    fi
    
    # Run health check
    if python3 scripts/health_check.py; then
        log_success "All health checks passed"
    else
        log_error "Some health checks failed - please review the logs"
    fi
    
    # Display access information
    log_info ""
    log_info "🎉 Deployment completed successfully!"
    log_info "================================================"
    
    case "$DEPLOYMENT_TYPE" in
        docker)
            log_info "API Endpoint: http://localhost"
            log_info "API Documentation: http://localhost/docs"
            if [[ "$ENABLE_MONITORING" == true ]]; then
                log_info "Grafana Dashboard: http://localhost:3000"
                log_info "Prometheus Metrics: http://localhost:9090"
            fi
            ;;
        kubernetes)
            EXTERNAL_IP=$(kubectl get service snn-fusion-api-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
            log_info "API Endpoint: http://$EXTERNAL_IP"
            log_info "Use 'kubectl port-forward' for local access"
            ;;
    esac
    
    log_info ""
    log_info "📋 Next Steps:"
    log_info "1. Review and customize the configuration in .env"
    log_info "2. Configure your domain name and SSL certificates"
    log_info "3. Set up monitoring alerts and notifications"
    log_info "4. Test the backup and recovery procedures"
    log_info "5. Review the Production Deployment Guide for detailed configuration"
    log_info ""
    log_info "📄 Full deployment log: $LOG_FILE"
}

# Cleanup function
cleanup() {
    log_info "🧹 Cleaning up temporary files..."
    # Clean up any temporary files created during deployment
}

# Error handler
error_handler() {
    log_error "Deployment failed at line $1"
    log_error "Check the log file for details: $LOG_FILE"
    cleanup
    exit 1
}

# Set error handler
trap 'error_handler $LINENO' ERR

# Confirmation prompt
if [[ "$FORCE" != true && "$DRY_RUN" != true ]]; then
    log_warning "This will deploy SNN Fusion Framework to $ENVIRONMENT environment"
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deployment cancelled by user"
        exit 0
    fi
fi

# Main deployment flow
main() {
    log_info "Starting deployment process..."
    
    check_prerequisites
    setup_environment
    create_directories
    build_images
    
    case "$DEPLOYMENT_TYPE" in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        aws|gcp|azure)
            deploy_cloud
            ;;
    esac
    
    setup_ssl
    setup_monitoring
    setup_backup
    post_deployment_check
    cleanup
    
    log_success "🚀 SNN Fusion Framework deployment completed!"
}

# Run main function
main