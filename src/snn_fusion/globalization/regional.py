"""
Regional Deployment Management for SNN-Fusion

Multi-region deployment orchestration, latency optimization,
data sovereignty, and global infrastructure management.
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import socket
import urllib.request
import urllib.error


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALIBABA = "alibaba"
    TENCENT = "tencent"
    BAIDU = "baidu"
    ON_PREMISE = "on_premise"


class DeploymentTier(Enum):
    """Deployment tiers by performance and cost."""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class RegionStatus(Enum):
    """Status of regional deployment."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    FAILED = "failed"


@dataclass
class RegionInfo:
    """Information about a deployment region."""
    region_id: str
    region_name: str
    country_code: str
    continent: str
    cloud_provider: CloudProvider
    deployment_tier: DeploymentTier
    status: RegionStatus = RegionStatus.INACTIVE
    endpoint_url: Optional[str] = None
    latency_ms: Optional[float] = None
    capacity: int = 100
    current_load: int = 0
    data_sovereignty_compliant: bool = True
    supported_regulations: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_health_check: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def load_percentage(self) -> float:
        """Calculate current load as percentage of capacity."""
        return (self.current_load / max(self.capacity, 1)) * 100
    
    @property
    def is_healthy(self) -> bool:
        """Check if region is healthy."""
        return self.status == RegionStatus.ACTIVE and self.load_percentage < 90


@dataclass
class RegionalConfig:
    """Configuration for regional deployment."""
    primary_region: str
    secondary_regions: List[str] = field(default_factory=list)
    enable_auto_failover: bool = True
    enable_load_balancing: bool = True
    enable_data_replication: bool = True
    latency_threshold_ms: float = 500.0
    health_check_interval: int = 30
    failover_timeout: int = 300
    data_sync_interval: int = 3600
    enable_edge_caching: bool = True
    compliance_requirements: Set[str] = field(default_factory=set)


class LatencyOptimizer:
    """Optimize request routing based on latency."""
    
    def __init__(self, regions: Dict[str, RegionInfo]):
        """Initialize latency optimizer."""
        self.regions = regions
        self.latency_matrix: Dict[Tuple[str, str], float] = {}
        self.logger = logging.getLogger(__name__)
        
    def measure_latency(self, source_region: str, target_region: str) -> Optional[float]:
        """Measure latency between regions."""
        if target_region not in self.regions:
            return None
        
        target_region_info = self.regions[target_region]
        if not target_region_info.endpoint_url:
            return None
        
        try:
            start_time = time.time()
            
            # Simple HTTP ping to measure latency
            url = f"{target_region_info.endpoint_url}/health"
            request = urllib.request.Request(url, method='HEAD')
            response = urllib.request.urlopen(request, timeout=5)
            
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Cache the measurement
            self.latency_matrix[(source_region, target_region)] = latency
            target_region_info.latency_ms = latency
            
            return latency
            
        except (urllib.error.URLError, socket.timeout) as e:
            self.logger.warning(f"Failed to measure latency to {target_region}: {e}")
            return None
    
    def get_optimal_region(
        self, 
        client_region: str, 
        exclude_regions: Optional[Set[str]] = None
    ) -> Optional[str]:
        """Get optimal region for client based on latency and load."""
        exclude_regions = exclude_regions or set()
        
        # Filter available regions
        available_regions = {
            region_id: region_info
            for region_id, region_info in self.regions.items()
            if (region_id not in exclude_regions and 
                region_info.is_healthy and
                region_info.status == RegionStatus.ACTIVE)
        }
        
        if not available_regions:
            return None
        
        # Score regions based on latency and load
        region_scores = {}
        
        for region_id, region_info in available_regions.items():
            # Get or measure latency
            latency_key = (client_region, region_id)
            if latency_key in self.latency_matrix:
                latency = self.latency_matrix[latency_key]
            else:
                latency = self.measure_latency(client_region, region_id)
                if latency is None:
                    continue
            
            # Calculate composite score (lower is better)
            latency_score = latency / 1000.0  # Normalize to seconds
            load_score = region_info.load_percentage / 100.0
            
            # Weighted combination (latency: 70%, load: 30%)
            composite_score = latency_score * 0.7 + load_score * 0.3
            region_scores[region_id] = composite_score
        
        # Return region with best (lowest) score
        if region_scores:
            optimal_region = min(region_scores.keys(), key=lambda r: region_scores[r])
            return optimal_region
        
        return None
    
    def update_latency_matrix(self):
        """Update latency measurements between all regions."""
        active_regions = [
            region_id for region_id, region_info in self.regions.items()
            if region_info.status == RegionStatus.ACTIVE
        ]
        
        for source_region in active_regions:
            for target_region in active_regions:
                if source_region != target_region:
                    self.measure_latency(source_region, target_region)


class DataSovereigntyManager:
    """Manage data sovereignty and compliance requirements."""
    
    def __init__(self):
        """Initialize data sovereignty manager."""
        self.logger = logging.getLogger(__name__)
        
        # Data residency requirements by regulation
        self.residency_requirements = {
            "GDPR": {"allowed_countries": ["EU"], "prohibited_countries": []},
            "CCPA": {"allowed_countries": ["US"], "prohibited_countries": []},
            "PDPA_SG": {"allowed_countries": ["SG"], "prohibited_countries": []},
            "RUSSIA_DATA": {"allowed_countries": ["RU"], "prohibited_countries": []},
            "CHINA_DATA": {"allowed_countries": ["CN"], "prohibited_countries": []}
        }
        
        # Country to continent mapping
        self.country_continent_map = {
            "US": "North America",
            "CA": "North America",
            "GB": "Europe",
            "DE": "Europe",
            "FR": "Europe",
            "IT": "Europe",
            "ES": "Europe",
            "NL": "Europe",
            "SG": "Asia",
            "JP": "Asia",
            "KR": "Asia",
            "AU": "Oceania",
            "CN": "Asia",
            "RU": "Europe/Asia",
            "BR": "South America",
            "IN": "Asia"
        }
    
    def validate_data_placement(
        self, 
        data_regulations: Set[str], 
        target_region: RegionInfo
    ) -> Tuple[bool, List[str]]:
        """
        Validate if data can be placed in target region.
        
        Args:
            data_regulations: Set of applicable regulations
            target_region: Target region information
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        for regulation in data_regulations:
            if regulation in self.residency_requirements:
                requirements = self.residency_requirements[regulation]
                
                # Check allowed countries
                allowed_countries = requirements.get("allowed_countries", [])
                if allowed_countries and target_region.country_code not in allowed_countries:
                    violations.append(
                        f"{regulation} requires data in {allowed_countries}, "
                        f"but target region is in {target_region.country_code}"
                    )
                
                # Check prohibited countries
                prohibited_countries = requirements.get("prohibited_countries", [])
                if target_region.country_code in prohibited_countries:
                    violations.append(
                        f"{regulation} prohibits data in {target_region.country_code}"
                    )
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def get_compliant_regions(
        self, 
        regions: Dict[str, RegionInfo], 
        data_regulations: Set[str]
    ) -> Dict[str, RegionInfo]:
        """Get regions that are compliant with data regulations."""
        compliant_regions = {}
        
        for region_id, region_info in regions.items():
            is_valid, violations = self.validate_data_placement(data_regulations, region_info)
            if is_valid:
                compliant_regions[region_id] = region_info
            else:
                self.logger.debug(f"Region {region_id} not compliant: {violations}")
        
        return compliant_regions


class RegionalDeploymentManager:
    """
    Comprehensive regional deployment management system.
    
    Handles multi-region deployments, failover, load balancing,
    data sovereignty, and latency optimization.
    """
    
    def __init__(self, config: RegionalConfig):
        """Initialize regional deployment manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Regional infrastructure
        self.regions: Dict[str, RegionInfo] = {}
        self.active_regions: Set[str] = set()
        
        # Optimization components
        self.latency_optimizer = LatencyOptimizer(self.regions)
        self.data_sovereignty_manager = DataSovereigntyManager()
        
        # Monitoring and health checks
        self.health_check_active = False
        self.health_check_thread: Optional[threading.Thread] = None
        self.failover_in_progress = False
        
        # Traffic routing
        self.request_routing_table: Dict[str, str] = {}
        self.region_weights: Dict[str, float] = {}
        
        # Metrics and monitoring
        self.region_metrics: Dict[str, Dict[str, Any]] = {}
        self.deployment_history: List[Dict[str, Any]] = []
        
        self.logger.info("RegionalDeploymentManager initialized")
    
    def register_region(self, region_info: RegionInfo):
        """Register a new deployment region."""
        self.regions[region_info.region_id] = region_info
        
        # Initialize metrics
        self.region_metrics[region_info.region_id] = {
            "requests_per_second": 0.0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "uptime_percentage": 100.0,
            "last_updated": datetime.now(timezone.utc)
        }
        
        # Initialize routing weight
        if region_info.deployment_tier == DeploymentTier.ENTERPRISE:
            self.region_weights[region_info.region_id] = 2.0
        elif region_info.deployment_tier == DeploymentTier.PREMIUM:
            self.region_weights[region_info.region_id] = 1.5
        else:
            self.region_weights[region_info.region_id] = 1.0
        
        self.logger.info(f"Registered region: {region_info.region_id} ({region_info.region_name})")
    
    def activate_region(self, region_id: str) -> bool:
        """Activate a region for serving traffic."""
        if region_id not in self.regions:
            self.logger.error(f"Region {region_id} not found")
            return False
        
        region = self.regions[region_id]
        
        # Perform health check before activation
        if not self._perform_region_health_check(region_id):
            self.logger.error(f"Health check failed for region {region_id}")
            return False
        
        # Activate region
        region.status = RegionStatus.ACTIVE
        self.active_regions.add(region_id)
        
        # Update latency optimizer
        self.latency_optimizer.measure_latency(self.config.primary_region, region_id)
        
        self.logger.info(f"Activated region: {region_id}")
        return True
    
    def deactivate_region(self, region_id: str, reason: str = "Manual deactivation") -> bool:
        """Deactivate a region."""
        if region_id not in self.regions:
            return False
        
        region = self.regions[region_id]
        region.status = RegionStatus.INACTIVE
        self.active_regions.discard(region_id)
        
        # Remove from routing table
        routing_updates = []
        for client_region, target_region in self.request_routing_table.items():
            if target_region == region_id:
                routing_updates.append(client_region)
        
        # Reroute traffic from deactivated region
        for client_region in routing_updates:
            new_target = self.latency_optimizer.get_optimal_region(
                client_region,
                exclude_regions={region_id}
            )
            if new_target:
                self.request_routing_table[client_region] = new_target
        
        self.logger.warning(f"Deactivated region {region_id}: {reason}")
        return True
    
    def route_request(
        self, 
        client_region: str, 
        data_regulations: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Route request to optimal region based on latency, load, and compliance.
        
        Args:
            client_region: Region where client request originates
            data_regulations: Set of data regulations that apply
            
        Returns:
            Target region ID for handling the request
        """
        # Check cached routing decision
        cache_key = f"{client_region}:{hash(frozenset(data_regulations or set()))}"
        if cache_key in self.request_routing_table:
            cached_target = self.request_routing_table[cache_key]
            if cached_target in self.active_regions:
                region = self.regions[cached_target]
                if region.is_healthy:
                    return cached_target
        
        # Find compliant regions if data regulations apply
        if data_regulations:
            compliant_regions = self.data_sovereignty_manager.get_compliant_regions(
                self.regions, data_regulations
            )
            # Filter to only active compliant regions
            available_regions = {
                region_id: region_info
                for region_id, region_info in compliant_regions.items()
                if region_id in self.active_regions and region_info.is_healthy
            }
        else:
            # Use all active healthy regions
            available_regions = {
                region_id: region_info
                for region_id, region_info in self.regions.items()
                if region_id in self.active_regions and region_info.is_healthy
            }
        
        if not available_regions:
            self.logger.error("No available regions for request routing")
            return None
        
        # Use latency optimizer to find best region
        optimal_region = self.latency_optimizer.get_optimal_region(
            client_region,
            exclude_regions=set(self.regions.keys()) - set(available_regions.keys())
        )
        
        # Cache the routing decision
        if optimal_region:
            self.request_routing_table[cache_key] = optimal_region
        
        return optimal_region
    
    def start_health_monitoring(self):
        """Start continuous health monitoring of all regions."""
        if self.health_check_active:
            return
        
        self.health_check_active = True
        self.health_check_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True,
            name="RegionalHealthMonitor"
        )
        self.health_check_thread.start()
        
        self.logger.info("Started regional health monitoring")
    
    def stop_health_monitoring(self):
        """Stop health monitoring."""
        self.health_check_active = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        
        self.logger.info("Stopped regional health monitoring")
    
    def trigger_failover(self, failed_region: str) -> bool:
        """Trigger failover from a failed region."""
        if self.failover_in_progress:
            self.logger.warning("Failover already in progress")
            return False
        
        if failed_region not in self.regions:
            return False
        
        self.failover_in_progress = True
        
        try:
            self.logger.warning(f"Triggering failover from region: {failed_region}")
            
            # Mark region as failed
            self.regions[failed_region].status = RegionStatus.FAILED
            self.active_regions.discard(failed_region)
            
            # Reroute all traffic from failed region
            affected_clients = []
            for client_region, target_region in self.request_routing_table.items():
                if target_region == failed_region:
                    affected_clients.append(client_region)
            
            successful_reroutes = 0
            for client_region in affected_clients:
                new_target = self.latency_optimizer.get_optimal_region(
                    client_region,
                    exclude_regions={failed_region}
                )
                if new_target:
                    self.request_routing_table[client_region] = new_target
                    successful_reroutes += 1
                    self.logger.info(f"Rerouted {client_region} from {failed_region} to {new_target}")
            
            # Record failover event
            failover_event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "failover",
                "failed_region": failed_region,
                "affected_clients": len(affected_clients),
                "successful_reroutes": successful_reroutes,
                "duration_seconds": time.time()  # Will be updated when complete
            }
            
            # Update latency matrix
            self.latency_optimizer.update_latency_matrix()
            
            failover_event["duration_seconds"] = time.time() - failover_event["duration_seconds"]
            self.deployment_history.append(failover_event)
            
            self.logger.info(f"Failover completed: {successful_reroutes}/{len(affected_clients)} clients rerouted")
            return True
            
        except Exception as e:
            self.logger.error(f"Failover failed: {e}")
            return False
        finally:
            self.failover_in_progress = False
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status."""
        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_regions": len(self.regions),
            "active_regions": len(self.active_regions),
            "primary_region": self.config.primary_region,
            "health_monitoring": self.health_check_active,
            "failover_in_progress": self.failover_in_progress,
            "regions": {},
            "routing_summary": {
                "total_routes": len(self.request_routing_table),
                "route_distribution": {}
            },
            "compliance": {
                "data_sovereignty_enabled": True,
                "supported_regulations": list(self.config.compliance_requirements)
            }
        }
        
        # Regional status
        for region_id, region_info in self.regions.items():
            region_status = {
                "status": region_info.status.value,
                "load_percentage": region_info.load_percentage,
                "latency_ms": region_info.latency_ms,
                "last_health_check": region_info.last_health_check.isoformat() if region_info.last_health_check else None,
                "deployment_tier": region_info.deployment_tier.value,
                "cloud_provider": region_info.cloud_provider.value,
                "country_code": region_info.country_code,
                "data_sovereignty_compliant": region_info.data_sovereignty_compliant
            }
            
            # Add metrics if available
            if region_id in self.region_metrics:
                region_status["metrics"] = self.region_metrics[region_id]
            
            status["regions"][region_id] = region_status
        
        # Routing distribution
        target_counts = {}
        for target_region in self.request_routing_table.values():
            target_counts[target_region] = target_counts.get(target_region, 0) + 1
        
        status["routing_summary"]["route_distribution"] = target_counts
        
        return status
    
    def get_optimal_deployment_plan(
        self, 
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimal deployment plan based on requirements."""
        plan = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requirements": requirements,
            "recommendations": {
                "primary_region": None,
                "secondary_regions": [],
                "deployment_tiers": {},
                "compliance_considerations": []
            },
            "estimated_costs": {},
            "performance_projections": {}
        }
        
        # Analyze requirements
        target_latency = requirements.get("max_latency_ms", 200)
        data_regulations = set(requirements.get("data_regulations", []))
        expected_load = requirements.get("expected_rps", 1000)
        availability_target = requirements.get("availability_percentage", 99.9)
        
        # Get compliant regions
        compliant_regions = self.data_sovereignty_manager.get_compliant_regions(
            self.regions, data_regulations
        )
        
        if not compliant_regions:
            plan["recommendations"]["compliance_considerations"].append(
                "No regions available that satisfy all data sovereignty requirements"
            )
            return plan
        
        # Recommend primary region (lowest latency, highest tier)
        region_scores = {}
        for region_id, region_info in compliant_regions.items():
            latency_score = (region_info.latency_ms or 100) / target_latency
            tier_score = {
                DeploymentTier.BASIC: 1.0,
                DeploymentTier.STANDARD: 1.2,
                DeploymentTier.PREMIUM: 1.5,
                DeploymentTier.ENTERPRISE: 2.0
            }[region_info.deployment_tier]
            
            composite_score = tier_score / latency_score
            region_scores[region_id] = composite_score
        
        if region_scores:
            primary_region = max(region_scores.keys(), key=lambda r: region_scores[r])
            plan["recommendations"]["primary_region"] = primary_region
            
            # Recommend secondary regions
            remaining_regions = [r for r in region_scores.keys() if r != primary_region]
            remaining_regions.sort(key=lambda r: region_scores[r], reverse=True)
            plan["recommendations"]["secondary_regions"] = remaining_regions[:3]
        
        # Deployment tier recommendations
        for region_id in compliant_regions:
            if expected_load > 10000:
                plan["recommendations"]["deployment_tiers"][region_id] = DeploymentTier.ENTERPRISE.value
            elif expected_load > 5000:
                plan["recommendations"]["deployment_tiers"][region_id] = DeploymentTier.PREMIUM.value
            elif expected_load > 1000:
                plan["recommendations"]["deployment_tiers"][region_id] = DeploymentTier.STANDARD.value
            else:
                plan["recommendations"]["deployment_tiers"][region_id] = DeploymentTier.BASIC.value
        
        return plan
    
    def _health_monitoring_loop(self):
        """Main health monitoring loop."""
        while self.health_check_active:
            try:
                # Check health of all active regions
                failed_regions = []
                
                for region_id in list(self.active_regions):
                    if not self._perform_region_health_check(region_id):
                        failed_regions.append(region_id)
                
                # Trigger failover for failed regions
                for failed_region in failed_regions:
                    if self.config.enable_auto_failover:
                        self.trigger_failover(failed_region)
                
                # Update latency matrix
                self.latency_optimizer.update_latency_matrix()
                
                # Wait for next check
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(5)
    
    def _perform_region_health_check(self, region_id: str) -> bool:
        """Perform health check on a specific region."""
        if region_id not in self.regions:
            return False
        
        region = self.regions[region_id]
        
        try:
            # Simple health check - measure latency
            latency = self.latency_optimizer.measure_latency(
                self.config.primary_region,
                region_id
            )
            
            if latency is None:
                # Failed to reach region
                region.status = RegionStatus.FAILED
                region.last_health_check = datetime.now(timezone.utc)
                return False
            
            # Check if latency exceeds threshold
            if latency > self.config.latency_threshold_ms:
                region.status = RegionStatus.DEGRADED
                self.logger.warning(f"Region {region_id} degraded: latency {latency:.1f}ms")
            else:
                region.status = RegionStatus.ACTIVE
            
            region.latency_ms = latency
            region.last_health_check = datetime.now(timezone.utc)
            
            # Update metrics
            self.region_metrics[region_id]["last_updated"] = datetime.now(timezone.utc)
            self.region_metrics[region_id]["average_response_time"] = latency / 1000.0
            
            return region.status in [RegionStatus.ACTIVE, RegionStatus.DEGRADED]
            
        except Exception as e:
            self.logger.error(f"Health check failed for region {region_id}: {e}")
            region.status = RegionStatus.FAILED
            region.last_health_check = datetime.now(timezone.utc)
            return False


# Example usage and testing
if __name__ == "__main__":
    print("Testing Regional Deployment Management...")
    
    # Create regional config
    config = RegionalConfig(
        primary_region="us-east-1",
        secondary_regions=["eu-west-1", "ap-southeast-1"],
        enable_auto_failover=True,
        compliance_requirements={"GDPR", "CCPA"}
    )
    
    # Initialize deployment manager
    deployment_manager = RegionalDeploymentManager(config)
    
    print("\n1. Testing region registration:")
    
    # Register regions
    regions_to_register = [
        RegionInfo(
            region_id="us-east-1",
            region_name="US East (N. Virginia)",
            country_code="US",
            continent="North America",
            cloud_provider=CloudProvider.AWS,
            deployment_tier=DeploymentTier.ENTERPRISE,
            endpoint_url="https://us-east-1.example.com",
            capacity=1000
        ),
        RegionInfo(
            region_id="eu-west-1",
            region_name="EU West (Ireland)",
            country_code="GB",
            continent="Europe",
            cloud_provider=CloudProvider.AWS,
            deployment_tier=DeploymentTier.PREMIUM,
            endpoint_url="https://eu-west-1.example.com",
            capacity=800,
            supported_regulations={"GDPR"}
        ),
        RegionInfo(
            region_id="ap-southeast-1",
            region_name="Asia Pacific (Singapore)",
            country_code="SG",
            continent="Asia",
            cloud_provider=CloudProvider.AWS,
            deployment_tier=DeploymentTier.STANDARD,
            endpoint_url="https://ap-southeast-1.example.com",
            capacity=600,
            supported_regulations={"PDPA_SG"}
        ),
        RegionInfo(
            region_id="cn-north-1",
            region_name="China (Beijing)",
            country_code="CN",
            continent="Asia",
            cloud_provider=CloudProvider.ALIBABA,
            deployment_tier=DeploymentTier.STANDARD,
            endpoint_url="https://cn-north-1.example.com",
            capacity=400,
            supported_regulations={"CHINA_DATA"}
        )
    ]
    
    for region in regions_to_register:
        deployment_manager.register_region(region)
        print(f"  Registered {region.region_id} ({region.country_code})")
    
    print("\n2. Testing region activation:")
    
    # Activate regions (skipping actual health checks for demo)
    for region_info in regions_to_register:
        # Mock successful activation
        deployment_manager.regions[region_info.region_id].status = RegionStatus.ACTIVE
        deployment_manager.active_regions.add(region_info.region_id)
        print(f"  Activated {region_info.region_id}")
    
    print("\n3. Testing request routing:")
    
    # Test routing for different scenarios
    test_scenarios = [
        ("us-west-2", None, "US client, no regulations"),
        ("eu-central-1", {"GDPR"}, "EU client with GDPR"),
        ("ap-northeast-1", {"PDPA_SG"}, "Asia client with PDPA"),
        ("cn-east-1", {"CHINA_DATA"}, "China client with data sovereignty")
    ]
    
    for client_region, regulations, description in test_scenarios:
        target_region = deployment_manager.route_request(client_region, regulations)
        print(f"  {description}: {client_region} -> {target_region}")
    
    print("\n4. Testing data sovereignty compliance:")
    
    sovereignty_manager = deployment_manager.data_sovereignty_manager
    
    # Test compliance validation
    test_regulations = {"GDPR", "CCPA"}
    compliant_regions = sovereignty_manager.get_compliant_regions(
        deployment_manager.regions,
        test_regulations
    )
    
    print(f"  Regions compliant with {test_regulations}:")
    for region_id in compliant_regions:
        region_info = compliant_regions[region_id]
        print(f"    {region_id}: {region_info.country_code} ({region_info.region_name})")
    
    print("\n5. Testing deployment planning:")
    
    # Generate optimal deployment plan
    requirements = {
        "max_latency_ms": 150,
        "data_regulations": ["GDPR"],
        "expected_rps": 5000,
        "availability_percentage": 99.9
    }
    
    plan = deployment_manager.get_optimal_deployment_plan(requirements)
    
    print(f"  Optimal deployment plan:")
    print(f"    Primary region: {plan['recommendations']['primary_region']}")
    print(f"    Secondary regions: {plan['recommendations']['secondary_regions']}")
    
    print("\n6. Testing deployment status:")
    
    status = deployment_manager.get_deployment_status()
    
    print(f"  Deployment status:")
    print(f"    Total regions: {status['total_regions']}")
    print(f"    Active regions: {status['active_regions']}")
    print(f"    Primary region: {status['primary_region']}")
    print(f"    Health monitoring: {status['health_monitoring']}")
    
    print(f"  Regional details:")
    for region_id, region_status in status['regions'].items():
        print(f"    {region_id}: {region_status['status']} "
              f"({region_status['country_code']}, "
              f"{region_status['deployment_tier']})")
    
    print("\n✓ Regional deployment management test completed!")