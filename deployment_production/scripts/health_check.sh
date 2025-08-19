#!/bin/bash
set -euo pipefail

# SNN Fusion Health Check Script
NAMESPACE="snn-fusion"
SERVICE="snn-fusion-service"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

check_pods() {
    log "Checking pod health..."
    
    PODS=$(kubectl get pods -n ${NAMESPACE} -l app=snn-fusion-api -o jsonpath='{.items[*].metadata.name}')
    
    for pod in $PODS; do
        STATUS=$(kubectl get pod $pod -n ${NAMESPACE} -o jsonpath='{.status.phase}')
        READY=$(kubectl get pod $pod -n ${NAMESPACE} -o jsonpath='{.status.containerStatuses[0].ready}')
        
        if [[ "$STATUS" == "Running" && "$READY" == "true" ]]; then
            log "✅ Pod $pod is healthy"
        else
            log "❌ Pod $pod is unhealthy (Status: $STATUS, Ready: $READY)"
        fi
    done
}

check_service_endpoints() {
    log "Checking service endpoints..."
    
    ENDPOINTS=$(kubectl get endpoints ${SERVICE} -n ${NAMESPACE} -o jsonpath='{.subsets[*].addresses[*].ip}')
    
    if [[ -n "$ENDPOINTS" ]]; then
        log "✅ Service has endpoints: $ENDPOINTS"
    else
        log "❌ Service has no endpoints"
    fi
}

check_hpa() {
    log "Checking HPA status..."
    
    HPA_STATUS=$(kubectl get hpa snn-fusion-hpa -n ${NAMESPACE} -o jsonpath='{.status.currentReplicas}')
    TARGET_REPLICAS=$(kubectl get hpa snn-fusion-hpa -n ${NAMESPACE} -o jsonpath='{.status.desiredReplicas}')
    
    log "📊 Current replicas: $HPA_STATUS, Target: $TARGET_REPLICAS"
}

check_metrics() {
    log "Checking metrics availability..."
    
    # Port forward to check metrics
    kubectl port-forward svc/${SERVICE} 8080:80 -n ${NAMESPACE} &
    PF_PID=$!
    
    sleep 3
    
    if curl -s http://localhost:8080/metrics | grep -q "snn_fusion"; then
        log "✅ Metrics endpoint is working"
    else
        log "❌ Metrics endpoint is not working"
    fi
    
    kill $PF_PID
}

main() {
    log "Starting health check..."
    
    check_pods
    check_service_endpoints
    check_hpa
    check_metrics
    
    log "✅ Health check completed"
}

main "$@"
