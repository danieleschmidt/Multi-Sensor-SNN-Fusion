#!/bin/bash
set -euo pipefail

# SNN Fusion Production Deployment Script
echo "🚀 Starting SNN Fusion Production Deployment"

# Configuration
NAMESPACE="snn-fusion"
DEPLOYMENT="snn-fusion-api"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo "❌ kubectl is required but not installed"
        exit 1
    fi
    
    # Check docker
    if ! command -v docker &> /dev/null; then
        echo "❌ docker is required but not installed"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        echo "❌ Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log "✅ Prerequisites check passed"
}

build_image() {
    log "Building Docker image..."
    
    cd ..
    docker build -f deployment_production/Dockerfile -t snn-fusion:${IMAGE_TAG} .
    
    log "✅ Docker image built successfully"
}

deploy_to_kubernetes() {
    log "Deploying to Kubernetes..."
    
    # Apply namespace first
    kubectl apply -f ../kubernetes/namespace.yaml
    
    # Apply all other manifests
    kubectl apply -f ../kubernetes/
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/${DEPLOYMENT} -n ${NAMESPACE}
    
    log "✅ Deployment completed successfully"
}

verify_deployment() {
    log "Verifying deployment..."
    
    # Check pod status
    kubectl get pods -n ${NAMESPACE}
    
    # Check service endpoints
    kubectl get svc -n ${NAMESPACE}
    
    # Check HPA status
    kubectl get hpa -n ${NAMESPACE}
    
    # Check ingress
    kubectl get ingress -n ${NAMESPACE}
    
    log "✅ Deployment verification completed"
}

run_health_check() {
    log "Running health checks..."
    
    # Port forward for health check
    kubectl port-forward svc/snn-fusion-service 8080:80 -n ${NAMESPACE} &
    PF_PID=$!
    
    sleep 5
    
    # Check health endpoint
    if curl -f http://localhost:8080/health; then
        log "✅ Health check passed"
    else
        log "❌ Health check failed"
        kill $PF_PID
        exit 1
    fi
    
    kill $PF_PID
}

# Main execution
main() {
    log "Starting deployment process..."
    
    check_prerequisites
    build_image
    deploy_to_kubernetes
    verify_deployment
    run_health_check
    
    log "🎉 Deployment completed successfully!"
    log "Access your application at: https://snn-fusion.terragonlabs.com"
}

# Execute main function
main "$@"
