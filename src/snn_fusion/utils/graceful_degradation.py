"""
Graceful Degradation and Adaptive Resilience

This module implements graceful degradation strategies that allow the
SNN-Fusion system to continue operating with reduced functionality
when components fail or resources become limited.
"""

import time
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import psutil


class ServiceLevel(Enum):
    """Service level definitions."""
    FULL = "full"              # All features available
    DEGRADED = "degraded"      # Reduced functionality
    MINIMAL = "minimal"        # Core functionality only
    EMERGENCY = "emergency"    # Survival mode


class ComponentType(Enum):
    """Types of system components."""
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    MODEL = "model"
    DATA = "data"
    HARDWARE = "hardware"


class ComponentStatus(Enum):
    """Component operational status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass
class ComponentState:
    """State information for a system component."""
    component_id: str
    component_type: ComponentType
    status: ComponentStatus
    health_score: float  # 0.0 to 1.0
    last_check: datetime
    failure_count: int = 0
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    is_critical: bool = False
    fallback_available: bool = False
    fallback_component: Optional[str] = None
    
    def is_operational(self) -> bool:
        """Check if component is operational."""
        return self.status in [ComponentStatus.HEALTHY, ComponentStatus.DEGRADED]
    
    def needs_recovery(self) -> bool:
        """Check if component needs recovery attempt."""
        return (self.status == ComponentStatus.FAILED and 
                self.recovery_attempts < self.max_recovery_attempts)


class GracefulDegradationManager:
    """
    Manager for graceful degradation and adaptive resilience.
    
    Monitors system components, detects failures, and implements
    fallback strategies to maintain service availability.
    """
    
    def __init__(
        self,
        min_service_level: ServiceLevel = ServiceLevel.MINIMAL,
        health_check_interval: int = 30,
        recovery_timeout: int = 300,
        enable_auto_recovery: bool = True
    ):
        """
        Initialize graceful degradation manager.
        
        Args:
            min_service_level: Minimum acceptable service level
            health_check_interval: Health check interval in seconds
            recovery_timeout: Recovery attempt timeout in seconds
            enable_auto_recovery: Enable automatic recovery attempts
        """
        self.min_service_level = min_service_level
        self.health_check_interval = health_check_interval
        self.recovery_timeout = recovery_timeout
        self.enable_auto_recovery = enable_auto_recovery
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Component tracking
        self.components: Dict[str, ComponentState] = {}
        self.service_level = ServiceLevel.FULL
        self.fallback_strategies: Dict[ComponentType, List[Callable]] = {}
        self.recovery_strategies: Dict[ComponentType, List[Callable]] = {}
        
        # Feature availability
        self.available_features: Set[str] = set()
        self.disabled_features: Set[str] = set()
        
        # Performance thresholds
        self.performance_thresholds = {
            ComponentType.COMPUTE: {'cpu_percent': 90, 'load_avg': 8.0},
            ComponentType.MEMORY: {'memory_percent': 85, 'swap_percent': 50},
            ComponentType.STORAGE: {'disk_percent': 90, 'io_wait': 50},
        }
        
        # Monitoring
        self.monitoring_enabled = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Register default strategies
        self._register_default_strategies()
        
        self.logger.info("GracefulDegradationManager initialized")
    
    def register_component(
        self,
        component_id: str,
        component_type: ComponentType,
        is_critical: bool = False,
        fallback_component: Optional[str] = None,
        health_check_func: Optional[Callable[[], float]] = None
    ):
        """
        Register a system component for monitoring.
        
        Args:
            component_id: Unique identifier for the component
            component_type: Type of component
            is_critical: Whether component is critical for operation
            fallback_component: ID of fallback component if available
            health_check_func: Custom health check function
        """
        component = ComponentState(
            component_id=component_id,
            component_type=component_type,
            status=ComponentStatus.HEALTHY,
            health_score=1.0,
            last_check=datetime.now(),
            is_critical=is_critical,
            fallback_available=fallback_component is not None,
            fallback_component=fallback_component
        )
        
        self.components[component_id] = component
        
        # Register custom health check if provided
        if health_check_func:
            self._register_health_check(component_id, health_check_func)
        
        self.logger.info(f"Registered component: {component_id} ({component_type.value})")
    
    def register_fallback_strategy(
        self,
        component_type: ComponentType,
        strategy: Callable[[ComponentState], bool]
    ):
        """Register a fallback strategy for component type."""
        if component_type not in self.fallback_strategies:
            self.fallback_strategies[component_type] = []
        
        self.fallback_strategies[component_type].append(strategy)
        self.logger.debug(f"Registered fallback strategy for {component_type.value}")
    
    def register_recovery_strategy(
        self,
        component_type: ComponentType,
        strategy: Callable[[ComponentState], bool]
    ):
        """Register a recovery strategy for component type."""
        if component_type not in self.recovery_strategies:
            self.recovery_strategies[component_type] = []
        
        self.recovery_strategies[component_type].append(strategy)
        self.logger.debug(f"Registered recovery strategy for {component_type.value}")
    
    def start_monitoring(self):
        """Start continuous component monitoring."""
        if self.monitoring_enabled:
            self.logger.warning("Monitoring already running")
            return
        
        self.monitoring_enabled = True
        self.stop_monitoring.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Started graceful degradation monitoring")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self.monitoring_enabled:
            return
        
        self.monitoring_enabled = False
        self.stop_monitoring.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped graceful degradation monitoring")
    
    def check_all_components(self) -> Dict[str, ComponentState]:
        """
        Check health of all registered components.
        
        Returns:
            Dictionary of component states
        """
        for component_id, component in self.components.items():
            self._check_component_health(component_id)
        
        # Update service level based on component states
        self._update_service_level()
        
        return self.components.copy()
    
    def get_service_level(self) -> ServiceLevel:
        """Get current service level."""
        return self.service_level
    
    def get_available_features(self) -> Set[str]:
        """Get currently available features."""
        return self.available_features.copy()
    
    def is_feature_available(self, feature: str) -> bool:
        """Check if a specific feature is available."""
        return feature in self.available_features
    
    def force_degradation(
        self,
        target_level: ServiceLevel,
        reason: str = "Manual degradation"
    ):
        """
        Manually force system to degrade to specified service level.
        
        Args:
            target_level: Target service level
            reason: Reason for forced degradation
        """
        self.logger.warning(f"Forcing degradation to {target_level.value}: {reason}")
        
        self.service_level = target_level
        self._apply_service_level_changes()
    
    def attempt_recovery(self, component_id: Optional[str] = None) -> bool:
        """
        Attempt recovery of failed components.
        
        Args:
            component_id: Specific component to recover, or None for all
            
        Returns:
            True if any recovery was successful
        """
        recovery_success = False
        
        components_to_recover = []
        if component_id:
            if component_id in self.components:
                components_to_recover.append(self.components[component_id])
        else:
            components_to_recover = [c for c in self.components.values() if c.needs_recovery()]
        
        for component in components_to_recover:
            if self._attempt_component_recovery(component):
                recovery_success = True
        
        if recovery_success:
            self._update_service_level()
        
        return recovery_success
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get comprehensive degradation status."""
        total_components = len(self.components)
        healthy_components = len([c for c in self.components.values() if c.status == ComponentStatus.HEALTHY])
        failed_components = len([c for c in self.components.values() if c.status == ComponentStatus.FAILED])
        
        critical_failures = [
            c.component_id for c in self.components.values()
            if c.is_critical and c.status == ComponentStatus.FAILED
        ]
        
        return {
            'service_level': self.service_level.value,
            'total_components': total_components,
            'healthy_components': healthy_components,
            'failed_components': failed_components,
            'health_percentage': healthy_components / max(total_components, 1) * 100,
            'critical_failures': critical_failures,
            'available_features': list(self.available_features),
            'disabled_features': list(self.disabled_features),
            'monitoring_enabled': self.monitoring_enabled,
            'last_check': max([c.last_check for c in self.components.values()]) if self.components else None
        }
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled and not self.stop_monitoring.is_set():
            try:
                # Check all components
                self.check_all_components()
                
                # Attempt recovery if enabled
                if self.enable_auto_recovery:
                    self.attempt_recovery()
                
                # Wait for next check
                if self.stop_monitoring.wait(self.health_check_interval):
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                if self.stop_monitoring.wait(5):
                    break
    
    def _check_component_health(self, component_id: str):
        """Check health of a specific component."""
        if component_id not in self.components:
            return
        
        component = self.components[component_id]
        previous_status = component.status
        
        try:
            # Get health score
            health_score = self._get_component_health_score(component)
            component.health_score = health_score
            component.last_check = datetime.now()
            
            # Determine status based on health score
            if health_score >= 0.8:
                component.status = ComponentStatus.HEALTHY
                component.failure_count = 0  # Reset failure count on recovery
            elif health_score >= 0.5:
                component.status = ComponentStatus.DEGRADED
            else:
                component.status = ComponentStatus.FAILED
                component.failure_count += 1
            
            # Log status changes
            if component.status != previous_status:
                self.logger.info(
                    f"Component {component_id} status changed: "
                    f"{previous_status.value} -> {component.status.value} "
                    f"(health: {health_score:.2f})"
                )
                
                # Apply fallback strategies if component failed
                if component.status == ComponentStatus.FAILED:
                    self._apply_fallback_strategies(component)
            
        except Exception as e:
            self.logger.error(f"Failed to check health of component {component_id}: {e}")
            component.status = ComponentStatus.UNAVAILABLE
            component.health_score = 0.0
    
    def _get_component_health_score(self, component: ComponentState) -> float:
        """Get health score for a component."""
        # Check if custom health check is registered
        health_check_func = getattr(self, f'_health_check_{component.component_id}', None)
        if health_check_func:
            return health_check_func()
        
        # Default health checks based on component type
        if component.component_type == ComponentType.COMPUTE:
            return self._check_compute_health()
        elif component.component_type == ComponentType.MEMORY:
            return self._check_memory_health()
        elif component.component_type == ComponentType.STORAGE:
            return self._check_storage_health()
        elif component.component_type == ComponentType.NETWORK:
            return self._check_network_health()
        else:
            # Default: assume healthy if no specific check
            return 1.0
    
    def _check_compute_health(self) -> float:
        """Check compute resource health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            
            thresholds = self.performance_thresholds[ComponentType.COMPUTE]
            
            # Calculate health based on thresholds
            cpu_health = max(0, 1.0 - (cpu_percent / 100.0))
            load_health = max(0, 1.0 - (load_avg / thresholds['load_avg']))
            
            return (cpu_health + load_health) / 2
            
        except Exception:
            return 0.5  # Unknown state
    
    def _check_memory_health(self) -> float:
        """Check memory resource health."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            thresholds = self.performance_thresholds[ComponentType.MEMORY]
            
            memory_health = max(0, 1.0 - (memory.percent / 100.0))
            swap_health = max(0, 1.0 - (swap.percent / 100.0))
            
            return (memory_health + swap_health) / 2
            
        except Exception:
            return 0.5
    
    def _check_storage_health(self) -> float:
        """Check storage resource health."""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Simple health based on disk usage
            return max(0, 1.0 - (disk_percent / 100.0))
            
        except Exception:
            return 0.5
    
    def _check_network_health(self) -> float:
        """Check network connectivity health."""
        try:
            import socket
            
            # Simple connectivity test
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return 1.0
            
        except Exception:
            return 0.0
    
    def _update_service_level(self):
        """Update service level based on component states."""
        critical_components = [c for c in self.components.values() if c.is_critical]
        healthy_critical = [c for c in critical_components if c.is_operational()]
        
        total_components = len(self.components)
        healthy_components = len([c for c in self.components.values() if c.is_operational()])
        
        # Calculate health percentage
        if total_components > 0:
            health_percentage = healthy_components / total_components
        else:
            health_percentage = 1.0
        
        # Determine service level
        previous_level = self.service_level
        
        if len(critical_components) > 0 and len(healthy_critical) == 0:
            # Critical components failed
            self.service_level = ServiceLevel.EMERGENCY
        elif health_percentage >= 0.8:
            self.service_level = ServiceLevel.FULL
        elif health_percentage >= 0.6:
            self.service_level = ServiceLevel.DEGRADED
        elif health_percentage >= 0.3:
            self.service_level = ServiceLevel.MINIMAL
        else:
            self.service_level = ServiceLevel.EMERGENCY
        
        # Ensure we don't go below minimum service level
        service_levels = [ServiceLevel.EMERGENCY, ServiceLevel.MINIMAL, ServiceLevel.DEGRADED, ServiceLevel.FULL]
        min_index = service_levels.index(self.min_service_level)
        current_index = service_levels.index(self.service_level)
        
        if current_index < min_index:
            self.service_level = self.min_service_level
        
        # Apply changes if service level changed
        if self.service_level != previous_level:
            self.logger.warning(f"Service level changed: {previous_level.value} -> {self.service_level.value}")
            self._apply_service_level_changes()
    
    def _apply_service_level_changes(self):
        """Apply feature availability changes based on service level."""
        # Reset feature sets
        self.available_features.clear()
        self.disabled_features.clear()
        
        # Define feature availability by service level
        if self.service_level == ServiceLevel.FULL:
            self.available_features.update([
                'full_training', 'multi_modal_fusion', 'advanced_analytics',
                'visualization', 'auto_tuning', 'hardware_acceleration',
                'distributed_processing', 'real_time_monitoring'
            ])
            
        elif self.service_level == ServiceLevel.DEGRADED:
            self.available_features.update([
                'basic_training', 'single_modal', 'basic_analytics',
                'limited_visualization', 'manual_tuning'
            ])
            self.disabled_features.update([
                'hardware_acceleration', 'distributed_processing',
                'advanced_analytics', 'auto_tuning'
            ])
            
        elif self.service_level == ServiceLevel.MINIMAL:
            self.available_features.update([
                'basic_inference', 'data_loading', 'simple_logging'
            ])
            self.disabled_features.update([
                'training', 'multi_modal_fusion', 'analytics',
                'visualization', 'tuning', 'hardware_acceleration'
            ])
            
        else:  # EMERGENCY
            self.available_features.update([
                'emergency_logging', 'basic_status'
            ])
            self.disabled_features.update([
                'training', 'inference', 'multi_modal_fusion',
                'analytics', 'visualization', 'hardware_acceleration'
            ])
        
        self.logger.info(f"Applied service level changes: {len(self.available_features)} features available")
    
    def _apply_fallback_strategies(self, component: ComponentState):
        """Apply fallback strategies for a failed component."""
        if component.component_type not in self.fallback_strategies:
            return
        
        self.logger.info(f"Applying fallback strategies for {component.component_id}")
        
        for strategy in self.fallback_strategies[component.component_type]:
            try:
                if strategy(component):
                    self.logger.info(f"Fallback strategy successful for {component.component_id}")
                    break
            except Exception as e:
                self.logger.error(f"Fallback strategy failed for {component.component_id}: {e}")
    
    def _attempt_component_recovery(self, component: ComponentState) -> bool:
        """Attempt to recover a failed component."""
        if not component.needs_recovery():
            return False
        
        self.logger.info(f"Attempting recovery for component {component.component_id}")
        component.recovery_attempts += 1
        
        # Try recovery strategies
        if component.component_type in self.recovery_strategies:
            for strategy in self.recovery_strategies[component.component_type]:
                try:
                    if strategy(component):
                        self.logger.info(f"Recovery successful for {component.component_id}")
                        component.status = ComponentStatus.HEALTHY
                        component.health_score = 1.0
                        component.failure_count = 0
                        return True
                except Exception as e:
                    self.logger.error(f"Recovery strategy failed for {component.component_id}: {e}")
        
        # If max attempts reached, mark as permanently failed
        if component.recovery_attempts >= component.max_recovery_attempts:
            self.logger.warning(f"Max recovery attempts reached for {component.component_id}")
        
        return False
    
    def _register_health_check(self, component_id: str, health_check_func: Callable[[], float]):
        """Register custom health check function for component."""
        setattr(self, f'_health_check_{component_id}', health_check_func)
    
    def _register_default_strategies(self):
        """Register default fallback and recovery strategies."""
        # Memory fallback: reduce batch size
        def memory_fallback(component: ComponentState) -> bool:
            self.logger.info("Applying memory fallback: reducing batch sizes")
            # In practice, this would modify global batch size settings
            return True
        
        # Compute fallback: disable parallel processing
        def compute_fallback(component: ComponentState) -> bool:
            self.logger.info("Applying compute fallback: disabling parallel processing")
            return True
        
        # Storage fallback: use temporary storage
        def storage_fallback(component: ComponentState) -> bool:
            self.logger.info("Applying storage fallback: switching to temporary storage")
            return True
        
        # Register fallback strategies
        self.register_fallback_strategy(ComponentType.MEMORY, memory_fallback)
        self.register_fallback_strategy(ComponentType.COMPUTE, compute_fallback)
        self.register_fallback_strategy(ComponentType.STORAGE, storage_fallback)
        
        # Recovery strategies
        def memory_recovery(component: ComponentState) -> bool:
            # Simulate memory cleanup
            return True
        
        def compute_recovery(component: ComponentState) -> bool:
            # Simulate process restart
            return True
        
        self.register_recovery_strategy(ComponentType.MEMORY, memory_recovery)
        self.register_recovery_strategy(ComponentType.COMPUTE, compute_recovery)


# Decorators for feature availability

def requires_service_level(min_level: ServiceLevel):
    """Decorator to enforce minimum service level for function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get degradation manager from global context or create one
            manager = getattr(wrapper, '_degradation_manager', None)
            if manager is None:
                # Function can execute if no manager is available
                return func(*args, **kwargs)
            
            current_level = manager.get_service_level()
            service_levels = [ServiceLevel.EMERGENCY, ServiceLevel.MINIMAL, ServiceLevel.DEGRADED, ServiceLevel.FULL]
            
            current_index = service_levels.index(current_level)
            required_index = service_levels.index(min_level)
            
            if current_index < required_index:
                raise RuntimeError(
                    f"Function requires {min_level.value} service level, "
                    f"but current level is {current_level.value}"
                )
            
            return func(*args, **kwargs)
        
        wrapper._min_service_level = min_level
        return wrapper
    return decorator


def feature_available(feature_name: str):
    """Decorator to check feature availability before execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = getattr(wrapper, '_degradation_manager', None)
            if manager is None:
                # Function can execute if no manager is available
                return func(*args, **kwargs)
            
            if not manager.is_feature_available(feature_name):
                raise RuntimeError(f"Feature '{feature_name}' is not available at current service level")
            
            return func(*args, **kwargs)
        
        wrapper._required_feature = feature_name
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    print("Testing Graceful Degradation System...")
    
    # Create degradation manager
    manager = GracefulDegradationManager(
        min_service_level=ServiceLevel.MINIMAL,
        health_check_interval=5
    )
    
    # Register some components
    manager.register_component("cpu", ComponentType.COMPUTE, is_critical=True)
    manager.register_component("memory", ComponentType.MEMORY, is_critical=True)
    manager.register_component("storage", ComponentType.STORAGE, is_critical=False)
    manager.register_component("network", ComponentType.NETWORK, is_critical=False)
    
    # Test decorator
    @requires_service_level(ServiceLevel.DEGRADED)
    @feature_available("basic_training")
    def test_training_function():
        return "Training executed successfully"
    
    # Attach manager to function
    test_training_function._degradation_manager = manager
    
    # Check initial state
    print(f"Initial service level: {manager.get_service_level().value}")
    print(f"Available features: {manager.get_available_features()}")
    
    # Check component health
    components = manager.check_all_components()
    print(f"\nComponent health check:")
    for comp_id, comp in components.items():
        print(f"  {comp_id}: {comp.status.value} (health: {comp.health_score:.2f})")
    
    # Test function execution
    try:
        result = test_training_function()
        print(f"Function result: {result}")
    except RuntimeError as e:
        print(f"Function blocked: {e}")
    
    # Force degradation
    manager.force_degradation(ServiceLevel.MINIMAL, "Testing degradation")
    print(f"\nAfter degradation:")
    print(f"  Service level: {manager.get_service_level().value}")
    print(f"  Available features: {manager.get_available_features()}")
    
    # Test function execution after degradation
    try:
        result = test_training_function()
        print(f"Function result: {result}")
    except RuntimeError as e:
        print(f"Function blocked: {e}")
    
    # Get status report
    status = manager.get_degradation_status()
    print(f"\nSystem status:")
    print(f"  Service level: {status['service_level']}")
    print(f"  Health percentage: {status['health_percentage']:.1f}%")
    print(f"  Available features: {len(status['available_features'])}")
    print(f"  Disabled features: {len(status['disabled_features'])}")
    
    print("✓ Graceful degradation test completed!")