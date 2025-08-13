"""
Multi-Region Deployment Manager for Neuromorphic System

Provides comprehensive multi-region deployment capabilities including
region-specific configurations, data residency compliance, latency optimization,
and disaster recovery across global infrastructure.
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Mock cloud provider APIs
try:
    # This would be replaced with actual cloud provider SDKs
    class MockCloudProvider:
        def get_regions(self):
            return ["us-east-1", "eu-west-1", "ap-southeast-1"]
        
        def deploy_to_region(self, region, config):
            return {"status": "success", "region": region}
        
        def get_region_latency(self, region):
            latencies = {"us-east-1": 50, "eu-west-1": 120, "ap-southeast-1": 180}
            return latencies.get(region, 100)
    
    cloud_provider = MockCloudProvider()
except ImportError:
    cloud_provider = None


class DeploymentRegion(Enum):
    """Supported deployment regions."""
    US_EAST_1 = "us-east-1"          # US East (Virginia)
    US_WEST_2 = "us-west-2"          # US West (Oregon)
    EU_WEST_1 = "eu-west-1"          # Europe (Ireland)
    EU_CENTRAL_1 = "eu-central-1"    # Europe (Frankfurt)
    AP_SOUTHEAST_1 = "ap-southeast-1" # Asia Pacific (Singapore)
    AP_NORTHEAST_1 = "ap-northeast-1" # Asia Pacific (Tokyo)
    AP_SOUTH_1 = "ap-south-1"        # Asia Pacific (Mumbai)
    CA_CENTRAL_1 = "ca-central-1"    # Canada (Central)
    SA_EAST_1 = "sa-east-1"          # South America (São Paulo)


class DataResidencyRequirement(Enum):
    """Data residency requirements."""
    GDPR_EU = "gdpr_eu"              # GDPR - EU data must stay in EU
    CCPA_US = "ccpa_us"              # CCPA - US data preferences
    PIPL_CHINA = "pipl_china"        # PIPL - China data localization
    LGPD_BRAZIL = "lgpd_brazil"      # LGPD - Brazil data protection
    PIPEDA_CANADA = "pipeda_canada"  # PIPEDA - Canada privacy
    NONE = "none"                    # No specific requirements


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    SINGLE_REGION = "single_region"          # Deploy to one region only
    MULTI_REGION_ACTIVE = "multi_region_active"  # Active in multiple regions
    MULTI_REGION_STANDBY = "multi_region_standby"  # Active-passive setup
    GLOBAL_DISTRIBUTION = "global_distribution"    # Globally distributed
    EDGE_DEPLOYMENT = "edge_deployment"      # Edge computing deployment


@dataclass
class RegionConfig:
    """Configuration for a deployment region."""
    region: DeploymentRegion
    display_name: str
    continent: str
    country: str
    timezone: str
    data_residency_rules: List[DataResidencyRequirement]
    compliance_certifications: List[str]
    available_instance_types: List[str]
    storage_encryption_required: bool = True
    network_isolation_required: bool = False
    backup_retention_days: int = 30
    disaster_recovery_rto_hours: int = 4  # Recovery Time Objective
    disaster_recovery_rpo_hours: int = 1  # Recovery Point Objective
    cost_multiplier: float = 1.0  # Cost relative to base region
    latency_characteristics: Dict[str, int] = field(default_factory=dict)


@dataclass
class DeploymentProfile:
    """Deployment profile for specific use cases."""
    name: str
    description: str
    strategy: DeploymentStrategy
    primary_regions: List[DeploymentRegion]
    secondary_regions: List[DeploymentRegion] = field(default_factory=list)
    data_residency_requirements: List[DataResidencyRequirement] = field(default_factory=list)
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    compliance_requirements: List[str] = field(default_factory=list)
    disaster_recovery_enabled: bool = True
    auto_scaling_enabled: bool = True
    cost_optimization_enabled: bool = True


@dataclass
class RegionHealthStatus:
    """Health status for a region."""
    region: DeploymentRegion
    status: str  # "healthy", "degraded", "unhealthy"
    latency_ms: float
    availability_percent: float
    error_rate_percent: float
    last_check: float
    issues: List[str] = field(default_factory=list)


class MultiRegionManager:
    """Comprehensive multi-region deployment manager."""
    
    def __init__(self):
        self.region_configs: Dict[DeploymentRegion, RegionConfig] = {}
        self.deployment_profiles: Dict[str, DeploymentProfile] = {}
        self.active_deployments: Dict[DeploymentRegion, Dict[str, Any]] = {}
        self.region_health: Dict[DeploymentRegion, RegionHealthStatus] = {}
        
        # Monitoring and management
        self.health_check_interval = 60  # seconds
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
        # Initialize configurations
        self._initialize_region_configs()
        self._initialize_deployment_profiles()
        
        # Start health monitoring
        self.start_health_monitoring()
    
    def _initialize_region_configs(self):
        """Initialize region configurations."""
        regions_data = {
            DeploymentRegion.US_EAST_1: RegionConfig(
                region=DeploymentRegion.US_EAST_1,
                display_name="US East (Virginia)",
                continent="North America",
                country="United States",
                timezone="UTC-5",
                data_residency_rules=[DataResidencyRequirement.CCPA_US],
                compliance_certifications=["SOC2", "HIPAA", "FedRAMP"],
                available_instance_types=["standard", "compute_optimized", "memory_optimized", "gpu"],
                disaster_recovery_rto_hours=2,
                disaster_recovery_rpo_hours=0.5,
                cost_multiplier=1.0,
                latency_characteristics={
                    "us-west-2": 70,
                    "eu-west-1": 120,
                    "ap-southeast-1": 180
                }
            ),
            DeploymentRegion.US_WEST_2: RegionConfig(
                region=DeploymentRegion.US_WEST_2,
                display_name="US West (Oregon)",
                continent="North America",
                country="United States",
                timezone="UTC-8",
                data_residency_rules=[DataResidencyRequirement.CCPA_US],
                compliance_certifications=["SOC2", "HIPAA"],
                available_instance_types=["standard", "compute_optimized", "memory_optimized"],
                cost_multiplier=1.05,
                latency_characteristics={
                    "us-east-1": 70,
                    "ap-northeast-1": 130
                }
            ),
            DeploymentRegion.EU_WEST_1: RegionConfig(
                region=DeploymentRegion.EU_WEST_1,
                display_name="Europe (Ireland)",
                continent="Europe",
                country="Ireland",
                timezone="UTC+0",
                data_residency_rules=[DataResidencyRequirement.GDPR_EU],
                compliance_certifications=["GDPR", "ISO27001", "SOC2"],
                available_instance_types=["standard", "compute_optimized", "memory_optimized"],
                storage_encryption_required=True,
                cost_multiplier=1.15,
                latency_characteristics={
                    "eu-central-1": 25,
                    "us-east-1": 120
                }
            ),
            DeploymentRegion.EU_CENTRAL_1: RegionConfig(
                region=DeploymentRegion.EU_CENTRAL_1,
                display_name="Europe (Frankfurt)",
                continent="Europe",
                country="Germany",
                timezone="UTC+1",
                data_residency_rules=[DataResidencyRequirement.GDPR_EU],
                compliance_certifications=["GDPR", "ISO27001", "SOC2"],
                available_instance_types=["standard", "compute_optimized", "memory_optimized", "gpu"],
                storage_encryption_required=True,
                network_isolation_required=True,
                cost_multiplier=1.20,
            ),
            DeploymentRegion.AP_SOUTHEAST_1: RegionConfig(
                region=DeploymentRegion.AP_SOUTHEAST_1,
                display_name="Asia Pacific (Singapore)",
                continent="Asia",
                country="Singapore",
                timezone="UTC+8",
                data_residency_rules=[DataResidencyRequirement.NONE],
                compliance_certifications=["SOC2", "ISO27001"],
                available_instance_types=["standard", "compute_optimized"],
                cost_multiplier=1.25,
                latency_characteristics={
                    "ap-northeast-1": 80,
                    "ap-south-1": 70
                }
            ),
            DeploymentRegion.AP_NORTHEAST_1: RegionConfig(
                region=DeploymentRegion.AP_NORTHEAST_1,
                display_name="Asia Pacific (Tokyo)",
                continent="Asia",
                country="Japan",
                timezone="UTC+9",
                data_residency_rules=[DataResidencyRequirement.NONE],
                compliance_certifications=["SOC2", "ISO27001"],
                available_instance_types=["standard", "compute_optimized", "memory_optimized", "gpu"],
                cost_multiplier=1.30,
            ),
            DeploymentRegion.AP_SOUTH_1: RegionConfig(
                region=DeploymentRegion.AP_SOUTH_1,
                display_name="Asia Pacific (Mumbai)",
                continent="Asia",
                country="India",
                timezone="UTC+5:30",
                data_residency_rules=[DataResidencyRequirement.NONE],
                compliance_certifications=["SOC2"],
                available_instance_types=["standard", "compute_optimized"],
                cost_multiplier=0.85,
            ),
            DeploymentRegion.CA_CENTRAL_1: RegionConfig(
                region=DeploymentRegion.CA_CENTRAL_1,
                display_name="Canada (Central)",
                continent="North America",
                country="Canada",
                timezone="UTC-5",
                data_residency_rules=[DataResidencyRequirement.PIPEDA_CANADA],
                compliance_certifications=["SOC2", "PIPEDA"],
                available_instance_types=["standard", "compute_optimized"],
                cost_multiplier=1.10,
            ),
            DeploymentRegion.SA_EAST_1: RegionConfig(
                region=DeploymentRegion.SA_EAST_1,
                display_name="South America (São Paulo)",
                continent="South America",
                country="Brazil",
                timezone="UTC-3",
                data_residency_rules=[DataResidencyRequirement.LGPD_BRAZIL],
                compliance_certifications=["LGPD", "SOC2"],
                available_instance_types=["standard"],
                cost_multiplier=1.35,
            )
        }
        
        self.region_configs.update(regions_data)
    
    def _initialize_deployment_profiles(self):
        """Initialize common deployment profiles."""
        profiles = {
            "global_enterprise": DeploymentProfile(
                name="Global Enterprise",
                description="Global deployment for enterprise customers with strict compliance",
                strategy=DeploymentStrategy.MULTI_REGION_ACTIVE,
                primary_regions=[
                    DeploymentRegion.US_EAST_1,
                    DeploymentRegion.EU_WEST_1,
                    DeploymentRegion.AP_SOUTHEAST_1
                ],
                secondary_regions=[
                    DeploymentRegion.US_WEST_2,
                    DeploymentRegion.EU_CENTRAL_1,
                    DeploymentRegion.AP_NORTHEAST_1
                ],
                data_residency_requirements=[
                    DataResidencyRequirement.GDPR_EU,
                    DataResidencyRequirement.CCPA_US
                ],
                performance_requirements={
                    "max_latency_ms": 100,
                    "min_availability_percent": 99.9,
                    "max_error_rate_percent": 0.1
                },
                compliance_requirements=["GDPR", "CCPA", "SOC2", "ISO27001"]
            ),
            
            "regional_deployment": DeploymentProfile(
                name="Regional Deployment",
                description="Single-region deployment with local disaster recovery",
                strategy=DeploymentStrategy.MULTI_REGION_STANDBY,
                primary_regions=[DeploymentRegion.US_EAST_1],
                secondary_regions=[DeploymentRegion.US_WEST_2],
                performance_requirements={
                    "max_latency_ms": 50,
                    "min_availability_percent": 99.5,
                    "max_error_rate_percent": 0.5
                },
                compliance_requirements=["SOC2"]
            ),
            
            "gdpr_compliant": DeploymentProfile(
                name="GDPR Compliant",
                description="EU-only deployment for GDPR compliance",
                strategy=DeploymentStrategy.MULTI_REGION_ACTIVE,
                primary_regions=[DeploymentRegion.EU_WEST_1],
                secondary_regions=[DeploymentRegion.EU_CENTRAL_1],
                data_residency_requirements=[DataResidencyRequirement.GDPR_EU],
                performance_requirements={
                    "max_latency_ms": 80,
                    "min_availability_percent": 99.8,
                    "max_error_rate_percent": 0.2
                },
                compliance_requirements=["GDPR", "ISO27001"]
            ),
            
            "cost_optimized": DeploymentProfile(
                name="Cost Optimized",
                description="Cost-optimized deployment in low-cost regions",
                strategy=DeploymentStrategy.SINGLE_REGION,
                primary_regions=[DeploymentRegion.AP_SOUTH_1],
                performance_requirements={
                    "max_latency_ms": 200,
                    "min_availability_percent": 99.0,
                    "max_error_rate_percent": 1.0
                },
                cost_optimization_enabled=True,
                auto_scaling_enabled=True
            ),
            
            "edge_computing": DeploymentProfile(
                name="Edge Computing",
                description="Edge deployment for low-latency applications",
                strategy=DeploymentStrategy.EDGE_DEPLOYMENT,
                primary_regions=[
                    DeploymentRegion.US_EAST_1,
                    DeploymentRegion.US_WEST_2,
                    DeploymentRegion.EU_WEST_1,
                    DeploymentRegion.AP_SOUTHEAST_1,
                    DeploymentRegion.AP_NORTHEAST_1
                ],
                performance_requirements={
                    "max_latency_ms": 30,
                    "min_availability_percent": 99.5,
                    "max_error_rate_percent": 0.3
                },
                disaster_recovery_enabled=False
            )
        }
        
        self.deployment_profiles.update(profiles)
    
    def get_available_regions(self) -> List[DeploymentRegion]:
        """Get list of available deployment regions."""
        return list(self.region_configs.keys())
    
    def get_region_config(self, region: DeploymentRegion) -> RegionConfig:
        """Get configuration for a specific region."""
        return self.region_configs.get(region)
    
    def get_deployment_profiles(self) -> Dict[str, DeploymentProfile]:
        """Get all available deployment profiles."""
        return self.deployment_profiles.copy()
    
    def recommend_regions(self, user_location: str, 
                         data_residency_requirements: List[DataResidencyRequirement] = None,
                         performance_requirements: Dict[str, Any] = None) -> List[Tuple[DeploymentRegion, float]]:
        """
        Recommend regions based on user location and requirements.
        
        Args:
            user_location: User's location (country code or region)
            data_residency_requirements: Data residency requirements
            performance_requirements: Performance requirements
            
        Returns:
            List of tuples (region, score) sorted by recommendation score
        """
        if data_residency_requirements is None:
            data_residency_requirements = []
        if performance_requirements is None:
            performance_requirements = {}
        
        recommendations = []
        
        for region, config in self.region_configs.items():
            score = 0.0
            
            # Geographic proximity score (simplified)
            if user_location.lower() in config.country.lower():
                score += 50  # Same country
            elif user_location.lower() in config.continent.lower():
                score += 30  # Same continent
            else:
                score += 10  # Different continent
            
            # Data residency compliance score
            residency_compliance = 0
            for requirement in data_residency_requirements:
                if requirement in config.data_residency_rules or requirement == DataResidencyRequirement.NONE:
                    residency_compliance += 20
                else:
                    residency_compliance -= 30  # Penalty for non-compliance
            score += residency_compliance
            
            # Performance score
            if performance_requirements:
                max_latency = performance_requirements.get("max_latency_ms", 200)
                estimated_latency = self._estimate_latency(region, user_location)
                if estimated_latency <= max_latency:
                    score += 20
                else:
                    score -= (estimated_latency - max_latency) * 0.1
            
            # Cost score (lower cost is better)
            cost_score = (2.0 - config.cost_multiplier) * 10
            score += cost_score
            
            # Certification score
            cert_score = len(config.compliance_certifications) * 5
            score += cert_score
            
            recommendations.append((region, max(0, score)))
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations
    
    def _estimate_latency(self, region: DeploymentRegion, user_location: str) -> float:
        """Estimate latency from user location to region."""
        config = self.region_configs[region]
        
        # Simplified latency estimation based on geographic distance
        latency_map = {
            "us": {"us-east-1": 20, "us-west-2": 25, "eu-west-1": 120, "ap-southeast-1": 180},
            "eu": {"eu-west-1": 20, "eu-central-1": 25, "us-east-1": 120, "ap-southeast-1": 200},
            "ap": {"ap-southeast-1": 20, "ap-northeast-1": 80, "us-east-1": 180, "eu-west-1": 200}
        }
        
        # Extract region from user location
        user_region = "us"  # Default
        if any(eu_country in user_location.lower() for eu_country in ["eu", "europe", "germany", "france", "ireland"]):
            user_region = "eu"
        elif any(ap_country in user_location.lower() for ap_country in ["ap", "asia", "japan", "singapore", "india"]):
            user_region = "ap"
        
        return latency_map.get(user_region, {}).get(region.value, 150)
    
    def deploy_to_regions(self, profile_name: str, 
                         configuration: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Deploy neuromorphic system to regions based on deployment profile.
        
        Args:
            profile_name: Name of the deployment profile to use
            configuration: Additional configuration parameters
            
        Returns:
            Deployment results
        """
        if profile_name not in self.deployment_profiles:
            raise ValueError(f"Unknown deployment profile: {profile_name}")
        
        profile = self.deployment_profiles[profile_name]
        if configuration is None:
            configuration = {}
        
        deployment_results = {
            "profile": profile_name,
            "status": "in_progress",
            "deployments": {},
            "errors": [],
            "start_time": time.time()
        }
        
        # Deploy to primary regions
        for region in profile.primary_regions:
            try:
                result = self._deploy_to_region(region, profile, configuration, is_primary=True)
                deployment_results["deployments"][region.value] = result
            except Exception as e:
                error_msg = f"Failed to deploy to primary region {region.value}: {e}"
                deployment_results["errors"].append(error_msg)
        
        # Deploy to secondary regions if specified
        for region in profile.secondary_regions:
            try:
                result = self._deploy_to_region(region, profile, configuration, is_primary=False)
                deployment_results["deployments"][region.value] = result
            except Exception as e:
                error_msg = f"Failed to deploy to secondary region {region.value}: {e}"
                deployment_results["errors"].append(error_msg)
        
        # Update deployment status
        successful_deployments = [d for d in deployment_results["deployments"].values() if d["status"] == "success"]
        if len(successful_deployments) == len(profile.primary_regions):
            deployment_results["status"] = "success"
        elif len(successful_deployments) > 0:
            deployment_results["status"] = "partial_success"
        else:
            deployment_results["status"] = "failed"
        
        deployment_results["end_time"] = time.time()
        deployment_results["duration"] = deployment_results["end_time"] - deployment_results["start_time"]
        
        return deployment_results
    
    def _deploy_to_region(self, region: DeploymentRegion, profile: DeploymentProfile,
                         configuration: Dict[str, Any], is_primary: bool = True) -> Dict[str, Any]:
        """Deploy to a specific region."""
        region_config = self.region_configs[region]
        
        # Build deployment configuration
        deploy_config = {
            "region": region.value,
            "is_primary": is_primary,
            "instance_type": configuration.get("instance_type", "standard"),
            "storage_encryption": region_config.storage_encryption_required,
            "network_isolation": region_config.network_isolation_required,
            "backup_retention_days": region_config.backup_retention_days,
            "disaster_recovery": profile.disaster_recovery_enabled,
            "auto_scaling": profile.auto_scaling_enabled,
            "compliance_requirements": profile.compliance_requirements,
            **configuration
        }
        
        # Simulate deployment (in real implementation, this would call cloud APIs)
        try:
            # Mock deployment call
            if cloud_provider:
                result = cloud_provider.deploy_to_region(region.value, deploy_config)
            else:
                result = {"status": "success", "region": region.value}
            
            # Store active deployment
            self.active_deployments[region] = {
                "profile": profile.name,
                "config": deploy_config,
                "deploy_time": time.time(),
                "status": "active"
            }
            
            return {
                "status": "success",
                "region": region.value,
                "is_primary": is_primary,
                "deployment_id": f"{region.value}-{int(time.time())}",
                "configuration": deploy_config
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "region": region.value,
                "error": str(e)
            }
    
    def start_health_monitoring(self):
        """Start health monitoring for all regions."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_health_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _health_monitoring_loop(self):
        """Health monitoring loop."""
        while self.monitoring_active:
            try:
                for region in self.region_configs.keys():
                    if region in self.active_deployments:
                        health_status = self._check_region_health(region)
                        self.region_health[region] = health_status
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                print(f"Health monitoring error: {e}")
                time.sleep(10)  # Brief pause on error
    
    def _check_region_health(self, region: DeploymentRegion) -> RegionHealthStatus:
        """Check health of a specific region."""
        try:
            # Mock health check (in real implementation, this would call monitoring APIs)
            if cloud_provider:
                latency = cloud_provider.get_region_latency(region.value)
            else:
                latency = 50.0  # Mock latency
            
            # Determine status based on latency and other factors
            if latency < 100:
                status = "healthy"
                availability = 99.9
                error_rate = 0.1
            elif latency < 200:
                status = "degraded"
                availability = 99.5
                error_rate = 0.5
            else:
                status = "unhealthy"
                availability = 98.0
                error_rate = 2.0
            
            issues = []
            if latency > 150:
                issues.append("High latency detected")
            if availability < 99.0:
                issues.append("Low availability")
            
            return RegionHealthStatus(
                region=region,
                status=status,
                latency_ms=latency,
                availability_percent=availability,
                error_rate_percent=error_rate,
                last_check=time.time(),
                issues=issues
            )
            
        except Exception as e:
            return RegionHealthStatus(
                region=region,
                status="unhealthy",
                latency_ms=999.0,
                availability_percent=0.0,
                error_rate_percent=100.0,
                last_check=time.time(),
                issues=[f"Health check failed: {e}"]
            )
    
    def get_region_health(self) -> Dict[DeploymentRegion, RegionHealthStatus]:
        """Get current health status for all regions."""
        return self.region_health.copy()
    
    def failover_to_region(self, failed_region: DeploymentRegion, 
                          target_region: DeploymentRegion) -> Dict[str, Any]:
        """Failover from one region to another."""
        failover_result = {
            "operation": "failover",
            "from_region": failed_region.value,
            "to_region": target_region.value,
            "status": "in_progress",
            "start_time": time.time()
        }
        
        try:
            # Check if target region is available
            if target_region not in self.active_deployments:
                # Need to deploy to target region first
                if failed_region in self.active_deployments:
                    failed_deployment = self.active_deployments[failed_region]
                    profile = self.deployment_profiles.get(failed_deployment["profile"])
                    
                    if profile:
                        # Deploy to target region
                        deploy_result = self._deploy_to_region(
                            target_region, profile, failed_deployment["config"]
                        )
                        
                        if deploy_result["status"] != "success":
                            failover_result["status"] = "failed"
                            failover_result["error"] = "Failed to deploy to target region"
                            return failover_result
            
            # Update traffic routing (mock)
            failover_result["traffic_routing_updated"] = True
            
            # Mark failed region as inactive
            if failed_region in self.active_deployments:
                self.active_deployments[failed_region]["status"] = "failed_over"
            
            failover_result["status"] = "success"
            failover_result["end_time"] = time.time()
            failover_result["duration"] = failover_result["end_time"] - failover_result["start_time"]
            
        except Exception as e:
            failover_result["status"] = "failed"
            failover_result["error"] = str(e)
            failover_result["end_time"] = time.time()
        
        return failover_result
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status across all regions."""
        status = {
            "total_regions": len(self.active_deployments),
            "healthy_regions": 0,
            "degraded_regions": 0,
            "unhealthy_regions": 0,
            "regions": {},
            "last_updated": time.time()
        }
        
        for region, deployment in self.active_deployments.items():
            region_info = {
                "deployment": deployment,
                "health": None
            }
            
            if region in self.region_health:
                health = self.region_health[region]
                region_info["health"] = {
                    "status": health.status,
                    "latency_ms": health.latency_ms,
                    "availability_percent": health.availability_percent,
                    "error_rate_percent": health.error_rate_percent,
                    "last_check": health.last_check,
                    "issues": health.issues
                }
                
                # Count by health status
                if health.status == "healthy":
                    status["healthy_regions"] += 1
                elif health.status == "degraded":
                    status["degraded_regions"] += 1
                else:
                    status["unhealthy_regions"] += 1
            
            status["regions"][region.value] = region_info
        
        return status
    
    def estimate_costs(self, profile_name: str, 
                      usage_hours_per_month: int = 720) -> Dict[str, Any]:
        """Estimate monthly costs for a deployment profile."""
        if profile_name not in self.deployment_profiles:
            raise ValueError(f"Unknown deployment profile: {profile_name}")
        
        profile = self.deployment_profiles[profile_name]
        
        # Base costs (simplified calculation)
        base_cost_per_hour = {
            "standard": 0.10,
            "compute_optimized": 0.15,
            "memory_optimized": 0.20,
            "gpu": 0.50
        }
        
        total_cost = 0.0
        cost_breakdown = {}
        
        # Calculate costs for primary regions
        for region in profile.primary_regions:
            config = self.region_configs[region]
            instance_cost = base_cost_per_hour["standard"] * config.cost_multiplier * usage_hours_per_month
            
            # Additional costs for compliance and features
            if profile.disaster_recovery_enabled:
                instance_cost *= 1.2  # 20% premium for DR
            if profile.auto_scaling_enabled:
                instance_cost *= 1.1  # 10% premium for auto-scaling
            
            cost_breakdown[f"{region.value}_primary"] = instance_cost
            total_cost += instance_cost
        
        # Calculate costs for secondary regions
        for region in profile.secondary_regions:
            config = self.region_configs[region]
            # Secondary regions typically cost 50% of primary
            instance_cost = base_cost_per_hour["standard"] * config.cost_multiplier * usage_hours_per_month * 0.5
            
            cost_breakdown[f"{region.value}_secondary"] = instance_cost
            total_cost += instance_cost
        
        return {
            "profile": profile_name,
            "monthly_cost_usd": total_cost,
            "usage_hours": usage_hours_per_month,
            "cost_breakdown": cost_breakdown,
            "cost_per_hour": total_cost / usage_hours_per_month,
            "currency": "USD"
        }
    
    def validate_compliance(self, region: DeploymentRegion, 
                           requirements: List[str]) -> Dict[str, Any]:
        """Validate compliance requirements for a region."""
        config = self.region_configs[region]
        
        validation_result = {
            "region": region.value,
            "requirements_checked": requirements,
            "compliant_requirements": [],
            "non_compliant_requirements": [],
            "warnings": [],
            "overall_compliant": True
        }
        
        for requirement in requirements:
            if requirement in config.compliance_certifications:
                validation_result["compliant_requirements"].append(requirement)
            else:
                validation_result["non_compliant_requirements"].append(requirement)
                validation_result["overall_compliant"] = False
        
        # Additional compliance checks
        if "GDPR" in requirements:
            if config.region not in [DeploymentRegion.EU_WEST_1, DeploymentRegion.EU_CENTRAL_1]:
                validation_result["warnings"].append("GDPR compliance requires EU regions")
        
        if "HIPAA" in requirements:
            if not config.storage_encryption_required:
                validation_result["warnings"].append("HIPAA compliance requires storage encryption")
        
        return validation_result


# Example usage and testing
if __name__ == "__main__":
    print("🌍 Testing Multi-Region Deployment Manager")
    print("=" * 60)
    
    # Create multi-region manager
    manager = MultiRegionManager()
    
    # Test region recommendations
    print("\n1. Testing Region Recommendations:")
    recommendations = manager.recommend_regions(
        user_location="Germany",
        data_residency_requirements=[DataResidencyRequirement.GDPR_EU],
        performance_requirements={"max_latency_ms": 100}
    )
    
    for region, score in recommendations[:3]:
        config = manager.get_region_config(region)
        print(f"  {region.value}: {config.display_name} (Score: {score:.1f})")
    
    # Test deployment profiles
    print("\n2. Available Deployment Profiles:")
    profiles = manager.get_deployment_profiles()
    for name, profile in profiles.items():
        print(f"  {name}: {profile.description}")
        print(f"    Strategy: {profile.strategy.value}")
        print(f"    Primary Regions: {[r.value for r in profile.primary_regions]}")
    
    # Test deployment
    print("\n3. Testing Deployment:")
    deployment_result = manager.deploy_to_regions(
        profile_name="regional_deployment",
        configuration={"instance_type": "standard"}
    )
    
    print(f"  Deployment Status: {deployment_result['status']}")
    print(f"  Duration: {deployment_result['duration']:.2f}s")
    print(f"  Deployed Regions: {len(deployment_result['deployments'])}")
    
    # Test health monitoring
    print("\n4. Testing Health Monitoring:")
    time.sleep(2)  # Let monitoring run briefly
    health_status = manager.get_region_health()
    
    for region, health in health_status.items():
        print(f"  {region.value}: {health.status} (Latency: {health.latency_ms}ms)")
    
    # Test cost estimation
    print("\n5. Cost Estimation:")
    cost_estimate = manager.estimate_costs("global_enterprise", usage_hours_per_month=720)
    print(f"  Profile: {cost_estimate['profile']}")
    print(f"  Monthly Cost: ${cost_estimate['monthly_cost_usd']:.2f}")
    print(f"  Cost per Hour: ${cost_estimate['cost_per_hour']:.3f}")
    
    # Test compliance validation
    print("\n6. Compliance Validation:")
    compliance_result = manager.validate_compliance(
        DeploymentRegion.EU_WEST_1,
        ["GDPR", "SOC2", "HIPAA"]
    )
    
    print(f"  Region: {compliance_result['region']}")
    print(f"  Overall Compliant: {compliance_result['overall_compliant']}")
    print(f"  Compliant: {compliance_result['compliant_requirements']}")
    print(f"  Non-Compliant: {compliance_result['non_compliant_requirements']}")
    
    # Test deployment status
    print("\n7. Deployment Status:")
    status = manager.get_deployment_status()
    print(f"  Total Regions: {status['total_regions']}")
    print(f"  Healthy Regions: {status['healthy_regions']}")
    print(f"  Degraded Regions: {status['degraded_regions']}")
    print(f"  Unhealthy Regions: {status['unhealthy_regions']}")
    
    # Stop monitoring
    manager.stop_health_monitoring()
    
    print("\n✅ Multi-region deployment testing completed!")