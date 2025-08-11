#!/usr/bin/env python3
"""
Health Check Script for SNN Fusion Framework Production Deployment

Comprehensive health checking for Docker containers, Kubernetes pods,
and standalone deployments.
"""

import os
import sys
import json
import time
import psutil
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from snn_fusion.utils.health_monitoring import HealthMonitor, HealthStatus
    HEALTH_MONITOR_AVAILABLE = True
except ImportError:
    HEALTH_MONITOR_AVAILABLE = False
    # Define minimal health status enum
    class HealthStatus:
        HEALTHY = "healthy"
        DEGRADED = "degraded"
        UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Health check result structure."""
    service: str
    status: str
    message: str
    timestamp: float
    details: Optional[Dict[str, Any]] = None


class ProductionHealthChecker:
    """Comprehensive health checker for production deployments."""
    
    def __init__(self):
        self.api_host = os.getenv('SNN_FUSION_API_HOST', 'localhost')
        self.api_port = int(os.getenv('SNN_FUSION_API_PORT', '8000'))
        self.metrics_port = int(os.getenv('SNN_FUSION_METRICS_PORT', '8080'))
        self.timeout = float(os.getenv('HEALTH_CHECK_TIMEOUT', '10.0'))
        
        # Health thresholds
        self.cpu_threshold = float(os.getenv('CPU_THRESHOLD', '80.0'))
        self.memory_threshold = float(os.getenv('MEMORY_THRESHOLD', '85.0'))
        self.disk_threshold = float(os.getenv('DISK_THRESHOLD', '90.0'))
        
        self.results: List[HealthCheckResult] = []
    
    def check_all(self) -> bool:
        """Run all health checks and return overall status."""
        print("🏥 SNN Fusion Framework - Production Health Check")
        print("=" * 55)
        
        # Core service checks
        self.check_api_service()
        self.check_metrics_endpoint()
        self.check_system_resources()
        
        # Optional service checks
        self.check_database_connection()
        self.check_redis_connection()
        self.check_file_system()
        
        # Application-specific checks
        self.check_model_files()
        self.check_application_health()
        
        # Print results
        self.print_results()
        
        # Return overall health status
        return all(result.status in [HealthStatus.HEALTHY, 'healthy'] 
                  for result in self.results if result.service in ['api', 'system'])
    
    def check_api_service(self):
        """Check main API service health."""
        try:
            url = f"http://{self.api_host}:{self.api_port}/health"
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                self.results.append(HealthCheckResult(
                    service="api",
                    status=HealthStatus.HEALTHY,
                    message="API service is healthy",
                    timestamp=time.time(),
                    details=data
                ))
            else:
                self.results.append(HealthCheckResult(
                    service="api",
                    status=HealthStatus.UNHEALTHY,
                    message=f"API returned status {response.status_code}",
                    timestamp=time.time()
                ))
                
        except requests.exceptions.ConnectRefused:
            self.results.append(HealthCheckResult(
                service="api",
                status=HealthStatus.UNHEALTHY,
                message="API service not accessible - connection refused",
                timestamp=time.time()
            ))
        except requests.exceptions.Timeout:
            self.results.append(HealthCheckResult(
                service="api",
                status=HealthStatus.DEGRADED,
                message=f"API service timeout (>{self.timeout}s)",
                timestamp=time.time()
            ))
        except Exception as e:
            self.results.append(HealthCheckResult(
                service="api",
                status=HealthStatus.UNHEALTHY,
                message=f"API check failed: {str(e)}",
                timestamp=time.time()
            ))
    
    def check_metrics_endpoint(self):
        """Check metrics endpoint health."""
        try:
            url = f"http://{self.api_host}:{self.metrics_port}/metrics"
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                # Simple check for Prometheus metrics format
                content = response.text
                if "# HELP" in content and "# TYPE" in content:
                    self.results.append(HealthCheckResult(
                        service="metrics",
                        status=HealthStatus.HEALTHY,
                        message="Metrics endpoint is healthy",
                        timestamp=time.time()
                    ))
                else:
                    self.results.append(HealthCheckResult(
                        service="metrics",
                        status=HealthStatus.DEGRADED,
                        message="Metrics endpoint returns non-standard format",
                        timestamp=time.time()
                    ))
            else:
                self.results.append(HealthCheckResult(
                    service="metrics",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Metrics endpoint returned {response.status_code}",
                    timestamp=time.time()
                ))
                
        except Exception as e:
            self.results.append(HealthCheckResult(
                service="metrics",
                status=HealthStatus.DEGRADED,
                message=f"Metrics check failed: {str(e)}",
                timestamp=time.time()
            ))
    
    def check_system_resources(self):
        """Check system resource utilization."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Determine overall status
            issues = []
            status = HealthStatus.HEALTHY
            
            if cpu_percent > self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
                status = HealthStatus.DEGRADED
            
            if memory_percent > self.memory_threshold:
                issues.append(f"High memory usage: {memory_percent:.1f}%")
                status = HealthStatus.DEGRADED
            
            if disk_percent > self.disk_threshold:
                issues.append(f"High disk usage: {disk_percent:.1f}%")
                status = HealthStatus.UNHEALTHY if disk_percent > 95 else HealthStatus.DEGRADED
            
            message = "System resources are healthy" if not issues else "; ".join(issues)
            
            self.results.append(HealthCheckResult(
                service="system",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3)
                }
            ))
            
        except Exception as e:
            self.results.append(HealthCheckResult(
                service="system",
                status=HealthStatus.UNHEALTHY,
                message=f"System check failed: {str(e)}",
                timestamp=time.time()
            ))
    
    def check_database_connection(self):
        """Check database connection if configured."""
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            return
        
        try:
            import sqlalchemy
            from sqlalchemy import create_engine, text
            
            engine = create_engine(database_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            self.results.append(HealthCheckResult(
                service="database",
                status=HealthStatus.HEALTHY,
                message="Database connection is healthy",
                timestamp=time.time()
            ))
            
        except ImportError:
            # SQLAlchemy not available - skip check
            pass
        except Exception as e:
            self.results.append(HealthCheckResult(
                service="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                timestamp=time.time()
            ))
    
    def check_redis_connection(self):
        """Check Redis connection if configured."""
        redis_url = os.getenv('REDIS_URL')
        if not redis_url:
            return
        
        try:
            import redis
            
            r = redis.from_url(redis_url)
            r.ping()
            
            self.results.append(HealthCheckResult(
                service="redis",
                status=HealthStatus.HEALTHY,
                message="Redis connection is healthy",
                timestamp=time.time()
            ))
            
        except ImportError:
            # Redis not available - skip check
            pass
        except Exception as e:
            self.results.append(HealthCheckResult(
                service="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}",
                timestamp=time.time()
            ))
    
    def check_file_system(self):
        """Check file system permissions and required directories."""
        try:
            required_dirs = [
                '/app/data',
                '/app/logs', 
                '/app/models',
                '/app/config'
            ]
            
            issues = []
            
            for dir_path in required_dirs:
                path = Path(dir_path)
                
                if not path.exists():
                    issues.append(f"Missing directory: {dir_path}")
                elif not path.is_dir():
                    issues.append(f"Not a directory: {dir_path}")
                elif not os.access(path, os.R_OK | os.W_OK):
                    issues.append(f"Insufficient permissions: {dir_path}")
            
            # Check writable temp directory
            temp_file = Path('/tmp/snn_fusion_health_check')
            try:
                temp_file.write_text("test")
                temp_file.unlink()
            except Exception:
                issues.append("Cannot write to temp directory")
            
            status = HealthStatus.HEALTHY if not issues else HealthStatus.DEGRADED
            message = "File system is healthy" if not issues else "; ".join(issues)
            
            self.results.append(HealthCheckResult(
                service="filesystem",
                status=status,
                message=message,
                timestamp=time.time()
            ))
            
        except Exception as e:
            self.results.append(HealthCheckResult(
                service="filesystem",
                status=HealthStatus.UNHEALTHY,
                message=f"File system check failed: {str(e)}",
                timestamp=time.time()
            ))
    
    def check_model_files(self):
        """Check if required model files exist."""
        try:
            models_dir = Path('/app/models')
            if not models_dir.exists():
                self.results.append(HealthCheckResult(
                    service="models",
                    status=HealthStatus.DEGRADED,
                    message="Models directory does not exist",
                    timestamp=time.time()
                ))
                return
            
            # Check for common model files
            model_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth"))
            
            if model_files:
                self.results.append(HealthCheckResult(
                    service="models",
                    status=HealthStatus.HEALTHY,
                    message=f"Found {len(model_files)} model files",
                    timestamp=time.time(),
                    details={"model_count": len(model_files)}
                ))
            else:
                self.results.append(HealthCheckResult(
                    service="models",
                    status=HealthStatus.DEGRADED,
                    message="No model files found (may be loaded dynamically)",
                    timestamp=time.time()
                ))
                
        except Exception as e:
            self.results.append(HealthCheckResult(
                service="models",
                status=HealthStatus.DEGRADED,
                message=f"Model check failed: {str(e)}",
                timestamp=time.time()
            ))
    
    def check_application_health(self):
        """Check application-specific health using internal health monitor."""
        if not HEALTH_MONITOR_AVAILABLE:
            return
        
        try:
            health_monitor = HealthMonitor(monitoring_interval=1.0, enable_alerts=False)
            snapshot = health_monitor.take_snapshot()
            
            self.results.append(HealthCheckResult(
                service="application",
                status=snapshot.overall_status.value if hasattr(snapshot.overall_status, 'value') else str(snapshot.overall_status),
                message=f"Application health: {len(snapshot.metrics)} metrics collected",
                timestamp=time.time(),
                details={"metrics_count": len(snapshot.metrics)}
            ))
            
        except Exception as e:
            self.results.append(HealthCheckResult(
                service="application",
                status=HealthStatus.DEGRADED,
                message=f"Application health check failed: {str(e)}",
                timestamp=time.time()
            ))
    
    def print_results(self):
        """Print health check results in a readable format."""
        print("\n📊 Health Check Results:")
        print("-" * 55)
        
        overall_healthy = True
        
        for result in self.results:
            # Status icon
            if result.status in [HealthStatus.HEALTHY, 'healthy']:
                icon = "✅"
            elif result.status in [HealthStatus.DEGRADED, 'degraded']:
                icon = "⚠️"
                overall_healthy = False
            else:
                icon = "❌"
                overall_healthy = False
            
            print(f"{icon} {result.service.upper()}: {result.message}")
            
            if result.details:
                for key, value in result.details.items():
                    if isinstance(value, float):
                        print(f"   └─ {key}: {value:.2f}")
                    else:
                        print(f"   └─ {key}: {value}")
        
        print("\n" + "=" * 55)
        if overall_healthy:
            print("🎉 Overall Status: HEALTHY")
        else:
            print("⚠️  Overall Status: ISSUES DETECTED")
        
        return overall_healthy
    
    def export_results(self, output_file: str = None):
        """Export results to JSON file."""
        if not output_file:
            output_file = f"/tmp/health_check_{int(time.time())}.json"
        
        try:
            results_dict = [asdict(result) for result in self.results]
            
            with open(output_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'overall_healthy': all(r.status in [HealthStatus.HEALTHY, 'healthy'] 
                                         for r in self.results if r.service in ['api', 'system']),
                    'results': results_dict
                }, f, indent=2)
            
            print(f"📄 Results exported to: {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to export results: {e}")


def main():
    """Main entry point for health check."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SNN Fusion Framework Health Check')
    parser.add_argument('--export', help='Export results to JSON file')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    health_checker = ProductionHealthChecker()
    
    if not args.quiet:
        is_healthy = health_checker.check_all()
    else:
        # Quiet mode - just run checks
        health_checker.check_api_service()
        health_checker.check_system_resources()
        is_healthy = all(result.status in [HealthStatus.HEALTHY, 'healthy'] 
                        for result in health_checker.results)
        print("HEALTHY" if is_healthy else "UNHEALTHY")
    
    if args.export:
        health_checker.export_results(args.export)
    
    # Exit with appropriate code
    sys.exit(0 if is_healthy else 1)


if __name__ == "__main__":
    main()