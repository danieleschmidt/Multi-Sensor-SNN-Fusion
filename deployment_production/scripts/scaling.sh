#!/bin/bash
set -euo pipefail

# SNN Fusion Scaling Management Script
NAMESPACE="snn-fusion"
DEPLOYMENT="snn-fusion-api"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

scale_deployment() {
    local replicas=$1
    
    log "Scaling deployment to $replicas replicas..."
    
    kubectl scale deployment ${DEPLOYMENT} --replicas=$replicas -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/${DEPLOYMENT} -n ${NAMESPACE}
    
    log "✅ Scaling completed"
}

update_hpa() {
    local min_replicas=$1
    local max_replicas=$2
    local cpu_target=$3
    
    log "Updating HPA: min=$min_replicas, max=$max_replicas, cpu_target=$cpu_target%"
    
    kubectl patch hpa snn-fusion-hpa -n ${NAMESPACE} --patch "{\"spec\": {\"minReplicas\": $min_replicas, \"maxReplicas\": $max_replicas, \"metrics\": [{\"type\": \"Resource\", \"resource\": {\"name\": \"cpu\", \"target\": {\"type\": \"Utilization\", \"averageUtilization\": $cpu_target}}}]}}"
    
    log "✅ HPA updated"
}

show_status() {
    log "Current deployment status:"
    kubectl get deployment ${DEPLOYMENT} -n ${NAMESPACE}
    
    log "Current HPA status:"
    kubectl get hpa snn-fusion-hpa -n ${NAMESPACE}
    
    log "Current pods:"
    kubectl get pods -l app=snn-fusion-api -n ${NAMESPACE}
}

case "${1:-status}" in
    scale)
        scale_deployment ${2:-3}
        ;;
    hpa)
        update_hpa ${2:-3} ${3:-10} ${4:-70}
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {scale|hpa|status} [args...]"
        echo "  scale <replicas>              Scale to specific number of replicas"
        echo "  hpa <min> <max> <cpu_target>  Update HPA configuration"
        echo "  status                        Show current status"
        exit 1
        ;;
esac
