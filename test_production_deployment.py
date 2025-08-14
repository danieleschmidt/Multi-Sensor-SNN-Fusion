#!/usr/bin/env python3
"""
Comprehensive Production Deployment Validation

Validates all production deployment components including health monitoring,
deployment management, operational dashboards, and system integration.
"""

import sys
import json
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import production modules
from snn_fusion.production.health_monitor import (
    ProductionHealthMonitor,
    SystemMetrics,
    AlertManager,
    AutoScaler,
    HealthStatus,
    AlertSeverity,
)
from snn_fusion.production.deployment_manager import (
    ProductionDeploymentManager,
    DeploymentConfig,
    DeploymentStage,
    ScalingPolicy,
    BackupManager,
    BackupConfig,
)
from snn_fusion.production.operational_dashboard import (
    ProductionDashboard,
    RealTimeMetrics,
    AlertDashboard,
    PerformanceAnalytics,
    DashboardType,
)


class ProductionValidationSuite:
    """
    Comprehensive validation suite for production deployment components.
    
    Validates:
    - Health monitoring system
    - Deployment management
    - Operational dashboards
    - System integration
    - Performance characteristics
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.validation_results = []
        self.test_start_time = time.time()
        
        self.logger.info("Production validation suite initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('production_validation.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive production validation."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING COMPREHENSIVE PRODUCTION VALIDATION")
        self.logger.info("=" * 60)
        
        validation_sections = [
            ("Health Monitor Validation", self._validate_health_monitor),
            ("Deployment Manager Validation", self._validate_deployment_manager),
            ("Operational Dashboard Validation", self._validate_operational_dashboard),
            ("System Integration Validation", self._validate_system_integration),
            ("Performance Validation", self._validate_performance_characteristics),
            ("Backup and Recovery Validation", self._validate_backup_recovery),
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for section_name, validation_func in validation_sections:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"VALIDATING: {section_name}")
            self.logger.info(f"{'='*50}")
            
            try:
                section_results = validation_func()
                self.validation_results.append({
                    'section': section_name,
                    'results': section_results,
                    'timestamp': time.time(),
                })
                
                section_passed = sum(1 for r in section_results if r.get('passed', False))
                section_total = len(section_results)
                
                total_tests += section_total
                passed_tests += section_passed
                
                self.logger.info(f"{section_name}: {section_passed}/{section_total} tests passed")
                
            except Exception as e:
                self.logger.error(f"Error in {section_name}: {e}")
                self.validation_results.append({
                    'section': section_name,
                    'error': str(e),
                    'timestamp': time.time(),
                })
        
        # Generate final report
        duration = time.time() - self.test_start_time
        
        final_report = {
            'validation_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                'duration_seconds': duration,
                'timestamp': time.time(),
            },
            'section_results': self.validation_results,
        }
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("PRODUCTION VALIDATION SUMMARY")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests}")
        self.logger.info(f"Failed: {total_tests - passed_tests}")
        self.logger.info(f"Success Rate: {final_report['validation_summary']['success_rate']:.1f}%")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        
        return final_report
    
    def _validate_health_monitor(self) -> List[Dict[str, Any]]:
        """Validate health monitoring system."""
        results = []
        
        # Test 1: Health monitor initialization
        try:
            monitor = ProductionHealthMonitor()
            results.append({
                'test': 'health_monitor_initialization',
                'passed': monitor is not None,
                'message': 'Health monitor initialized successfully',
            })
        except Exception as e:
            results.append({
                'test': 'health_monitor_initialization',
                'passed': False,
                'error': str(e),
            })
            return results
        
        # Test 2: System metrics collection
        try:
            # Start monitoring briefly
            monitor.start_monitoring()
            time.sleep(2)  # Allow metrics collection
            
            status = monitor.get_current_status()
            has_metrics = bool(status.get('system_metrics'))
            
            results.append({
                'test': 'system_metrics_collection',
                'passed': has_metrics,
                'message': f"System metrics collected: {has_metrics}",
                'metrics_count': len(status.get('system_metrics', {})),
            })
            
            monitor.stop_monitoring()
            
        except Exception as e:
            results.append({
                'test': 'system_metrics_collection',
                'passed': False,
                'error': str(e),
            })
        
        # Test 3: Health check execution
        try:
            monitor = ProductionHealthMonitor()
            
            # Get health checks (they run automatically when started)
            monitor.start_monitoring()
            time.sleep(3)  # Allow health checks to run
            
            status = monitor.get_current_status()
            health_checks = status.get('health_checks', {})
            
            results.append({
                'test': 'health_check_execution',
                'passed': len(health_checks) > 0,
                'message': f"Health checks executed: {len(health_checks)}",
                'health_checks': list(health_checks.keys()),
            })
            
            monitor.stop_monitoring()
            
        except Exception as e:
            results.append({
                'test': 'health_check_execution',
                'passed': False,
                'error': str(e),
            })
        
        # Test 4: Alert system
        try:
            alert_manager = AlertManager()
            
            results.append({
                'test': 'alert_manager_initialization',
                'passed': True,
                'message': 'Alert manager initialized successfully',
            })
            
        except Exception as e:
            results.append({
                'test': 'alert_manager_initialization',
                'passed': False,
                'error': str(e),
            })
        
        # Test 5: Auto-scaler
        try:
            scaler = AutoScaler()
            
            # Create mock metrics for scaling decision
            mock_metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=85.0,
                memory_percent=75.0,
                memory_available_gb=2.0,
                disk_usage_percent=60.0,
                disk_free_gb=10.0,
                network_io_mb=150.0,
                load_average=4.0,
                active_connections=100,
                process_count=50,
            )
            
            should_scale_up = scaler.should_scale_up(mock_metrics)
            
            results.append({
                'test': 'auto_scaler_functionality',
                'passed': True,  # Test passes if no exceptions
                'message': f"Auto-scaler decisions: scale_up={should_scale_up}",
                'scale_up_decision': should_scale_up,
            })
            
        except Exception as e:
            results.append({
                'test': 'auto_scaler_functionality',
                'passed': False,
                'error': str(e),
            })
        
        return results
    
    def _validate_deployment_manager(self) -> List[Dict[str, Any]]:
        """Validate deployment management system."""
        results = []
        
        # Test 1: Deployment manager initialization
        try:
            deployment_manager = ProductionDeploymentManager()
            results.append({
                'test': 'deployment_manager_initialization',
                'passed': deployment_manager is not None,
                'message': 'Deployment manager initialized successfully',
            })
        except Exception as e:
            results.append({
                'test': 'deployment_manager_initialization',
                'passed': False,
                'error': str(e),
            })
            return results
        
        # Test 2: Deployment configuration creation
        try:
            config = deployment_manager.create_deployment_config(DeploymentStage.PRODUCTION)
            
            is_valid, errors = deployment_manager.validate_deployment_config(config)
            
            results.append({
                'test': 'deployment_config_creation',
                'passed': is_valid,
                'message': f"Deployment config valid: {is_valid}",
                'config_stage': config.stage.value,
                'config_instances': config.instances,
                'validation_errors': errors,
            })
            
        except Exception as e:
            results.append({
                'test': 'deployment_config_creation',
                'passed': False,
                'error': str(e),
            })
        
        # Test 3: Manifest generation
        try:
            config = deployment_manager.create_deployment_config(DeploymentStage.STAGING)
            manifests = deployment_manager.generate_deployment_manifests(config)
            
            expected_manifests = [
                'kubernetes_deployment.yaml',
                'kubernetes_service.yaml',
                'docker-compose.yml',
                'app_config.yaml',
            ]
            
            has_expected_manifests = all(
                manifest in manifests for manifest in expected_manifests
            )
            
            results.append({
                'test': 'manifest_generation',
                'passed': has_expected_manifests,
                'message': f"Generated {len(manifests)} manifests",
                'manifests': list(manifests.keys()),
                'expected_manifests': expected_manifests,
            })
            
        except Exception as e:
            results.append({
                'test': 'manifest_generation',
                'passed': False,
                'error': str(e),
            })
        
        # Test 4: Deployment execution (mock)
        try:
            config = deployment_manager.create_deployment_config(DeploymentStage.DEVELOPMENT)
            deployment_result = deployment_manager.deploy(config, "standard")
            
            deployment_success = deployment_result.get('status') == 'deployed'
            
            results.append({
                'test': 'deployment_execution',
                'passed': deployment_success,
                'message': f"Deployment status: {deployment_result.get('status')}",
                'deployment_id': deployment_result.get('deployment_id'),
            })
            
        except Exception as e:
            results.append({
                'test': 'deployment_execution',
                'passed': False,
                'error': str(e),
            })
        
        # Test 5: Backup manager
        try:
            backup_config = BackupConfig(
                enabled=True,
                database_backup_interval_hours=24,
                model_backup_interval_hours=24,
                config_backup_interval_hours=12,
                retention_days=7,
                storage_location="/tmp/backups",
                encryption_enabled=True,
                compression_enabled=True,
            )
            
            backup_manager = BackupManager(backup_config)
            
            # Test manual backup
            backup_result = backup_manager.create_manual_backup("database")
            
            results.append({
                'test': 'backup_manager_functionality',
                'passed': backup_result.get('success', False),
                'message': backup_result.get('message', 'No message'),
                'backup_type': backup_result.get('backup_type'),
            })
            
        except Exception as e:
            results.append({
                'test': 'backup_manager_functionality',
                'passed': False,
                'error': str(e),
            })
        
        return results
    
    def _validate_operational_dashboard(self) -> List[Dict[str, Any]]:
        """Validate operational dashboard system."""
        results = []
        
        # Test 1: Dashboard initialization
        try:
            dashboard = ProductionDashboard()
            results.append({
                'test': 'dashboard_initialization',
                'passed': dashboard is not None,
                'message': 'Production dashboard initialized successfully',
            })
        except Exception as e:
            results.append({
                'test': 'dashboard_initialization',
                'passed': False,
                'error': str(e),
            })
            return results
        
        # Test 2: Real-time metrics system
        try:
            metrics_system = RealTimeMetrics()
            metrics_system.start_collection()
            time.sleep(2)  # Allow metrics collection
            
            current_metrics = metrics_system.get_current_metrics()
            
            results.append({
                'test': 'realtime_metrics_collection',
                'passed': len(current_metrics) > 0,
                'message': f"Collected {len(current_metrics)} metric types",
                'metric_types': list(current_metrics.keys()),
            })
            
            metrics_system.stop_collection()
            
        except Exception as e:
            results.append({
                'test': 'realtime_metrics_collection',
                'passed': False,
                'error': str(e),
            })
        
        # Test 3: Performance analytics
        try:
            metrics_system = RealTimeMetrics()
            performance_analytics = PerformanceAnalytics(metrics_system)
            
            # Start metrics briefly to generate data
            metrics_system.start_collection()
            time.sleep(2)
            
            # Generate some performance history
            for _ in range(5):
                metrics_system._generate_performance_summary()
                time.sleep(0.1)
            
            analysis = performance_analytics.analyze_performance_trends(1)
            
            results.append({
                'test': 'performance_analytics',
                'passed': 'data_points' in analysis,
                'message': f"Performance analysis completed",
                'data_points': analysis.get('data_points', 0),
                'recommendations': len(analysis.get('recommendations', [])),
            })
            
            metrics_system.stop_collection()
            
        except Exception as e:
            results.append({
                'test': 'performance_analytics',
                'passed': False,
                'error': str(e),
            })
        
        # Test 4: Alert dashboard
        try:
            alert_dashboard = AlertDashboard(None)  # No health monitor for testing
            
            # Test alert rule evaluation
            mock_metrics = {
                'cpu_percent': {'value': 90},
                'memory_percent': {'value': 70},
                'response_time_ms': {'value': 2000},
                'error_rate_percent': {'value': 1},
            }
            
            triggered_alerts = alert_dashboard.evaluate_alert_rules(mock_metrics)
            
            results.append({
                'test': 'alert_dashboard_functionality',
                'passed': True,  # Test passes if no exceptions
                'message': f"Alert evaluation completed",
                'triggered_alerts': len(triggered_alerts),
                'alert_rules': len(alert_dashboard.alert_rules),
            })
            
        except Exception as e:
            results.append({
                'test': 'alert_dashboard_functionality',
                'passed': False,
                'error': str(e),
            })
        
        # Test 5: Dashboard data generation
        try:
            dashboard = ProductionDashboard()
            dashboard.start_dashboard()
            
            # Wait for data collection
            time.sleep(2)
            
            # Test different dashboard types
            dashboard_types = [
                DashboardType.SYSTEM_OVERVIEW,
                DashboardType.PERFORMANCE_METRICS,
                DashboardType.ALERT_MANAGEMENT,
                DashboardType.NEUROMORPHIC_METRICS,
            ]
            
            dashboard_data_results = []
            for dashboard_type in dashboard_types:
                try:
                    data = dashboard.get_dashboard_data(dashboard_type)
                    dashboard_data_results.append({
                        'type': dashboard_type.value,
                        'success': True,
                        'data_keys': list(data.keys()),
                    })
                except Exception as e:
                    dashboard_data_results.append({
                        'type': dashboard_type.value,
                        'success': False,
                        'error': str(e),
                    })
            
            successful_dashboards = sum(1 for r in dashboard_data_results if r['success'])
            
            results.append({
                'test': 'dashboard_data_generation',
                'passed': successful_dashboards == len(dashboard_types),
                'message': f"Dashboard data generation: {successful_dashboards}/{len(dashboard_types)}",
                'dashboard_results': dashboard_data_results,
            })
            
            dashboard.stop_dashboard()
            
        except Exception as e:
            results.append({
                'test': 'dashboard_data_generation',
                'passed': False,
                'error': str(e),
            })
        
        return results
    
    def _validate_system_integration(self) -> List[Dict[str, Any]]:
        """Validate system integration between components."""
        results = []
        
        # Test 1: Health monitor + Dashboard integration
        try:
            monitor = ProductionHealthMonitor()
            dashboard = ProductionDashboard(monitor)
            
            # Start both systems
            monitor.start_monitoring()
            dashboard.start_dashboard()
            
            # Allow systems to collect data
            time.sleep(3)
            
            # Get integrated data
            system_overview = dashboard.get_dashboard_data(DashboardType.SYSTEM_OVERVIEW)
            
            has_health_data = bool(system_overview.get('alert_summary'))
            has_metrics_data = bool(system_overview.get('current_metrics'))
            
            integration_success = has_health_data and has_metrics_data
            
            results.append({
                'test': 'health_monitor_dashboard_integration',
                'passed': integration_success,
                'message': f"Integration success: {integration_success}",
                'has_health_data': has_health_data,
                'has_metrics_data': has_metrics_data,
            })
            
            # Cleanup
            monitor.stop_monitoring()
            dashboard.stop_dashboard()
            
        except Exception as e:
            results.append({
                'test': 'health_monitor_dashboard_integration',
                'passed': False,
                'error': str(e),
            })
        
        # Test 2: Deployment manager + Health monitor integration
        try:
            deployment_manager = ProductionDeploymentManager()
            monitor = ProductionHealthMonitor()
            
            # Create deployment configuration
            config = deployment_manager.create_deployment_config(DeploymentStage.DEVELOPMENT)
            
            # Validate configuration is compatible with monitoring
            is_valid, errors = deployment_manager.validate_deployment_config(config)
            
            # Check if health monitoring is enabled in config
            health_monitoring_enabled = config.health_check_enabled and config.monitoring_enabled
            
            integration_compatible = is_valid and health_monitoring_enabled
            
            results.append({
                'test': 'deployment_health_integration',
                'passed': integration_compatible,
                'message': f"Deployment-health integration compatible: {integration_compatible}",
                'config_valid': is_valid,
                'health_enabled': health_monitoring_enabled,
            })
            
        except Exception as e:
            results.append({
                'test': 'deployment_health_integration',
                'passed': False,
                'error': str(e),
            })
        
        # Test 3: Multi-component data flow
        try:
            # Initialize all components
            monitor = ProductionHealthMonitor()
            deployment_manager = ProductionDeploymentManager()
            dashboard = ProductionDashboard(monitor)
            
            # Start monitoring and dashboard
            monitor.start_monitoring()
            dashboard.start_dashboard()
            
            # Allow data collection
            time.sleep(3)
            
            # Check data flow through all components
            monitor_status = monitor.get_current_status()
            dashboard_data = dashboard.get_dashboard_data(DashboardType.SYSTEM_OVERVIEW)
            health_score = dashboard.get_system_health_score()
            
            data_flow_success = (
                bool(monitor_status.get('system_metrics')) and
                bool(dashboard_data.get('current_metrics')) and
                bool(health_score.get('overall_score'))
            )
            
            results.append({
                'test': 'multi_component_data_flow',
                'passed': data_flow_success,
                'message': f"Multi-component data flow: {data_flow_success}",
                'health_score': health_score.get('overall_score'),
                'health_status': health_score.get('status'),
            })
            
            # Cleanup
            monitor.stop_monitoring()
            dashboard.stop_dashboard()
            
        except Exception as e:
            results.append({
                'test': 'multi_component_data_flow',
                'passed': False,
                'error': str(e),
            })
        
        return results
    
    def _validate_performance_characteristics(self) -> List[Dict[str, Any]]:
        """Validate performance characteristics of production components."""
        results = []
        
        # Test 1: Health monitor performance
        try:
            monitor = ProductionHealthMonitor()
            
            # Measure initialization time
            start_time = time.time()
            monitor.start_monitoring()
            init_time = time.time() - start_time
            
            # Measure metrics collection performance
            start_time = time.time()
            for _ in range(10):  # Collect metrics 10 times
                status = monitor.get_current_status()
            collection_time = (time.time() - start_time) / 10
            
            performance_acceptable = init_time < 2.0 and collection_time < 0.5
            
            results.append({
                'test': 'health_monitor_performance',
                'passed': performance_acceptable,
                'message': f"Health monitor performance acceptable: {performance_acceptable}",
                'init_time_seconds': round(init_time, 3),
                'collection_time_seconds': round(collection_time, 3),
            })
            
            monitor.stop_monitoring()
            
        except Exception as e:
            results.append({
                'test': 'health_monitor_performance',
                'passed': False,
                'error': str(e),
            })
        
        # Test 2: Dashboard performance
        try:
            dashboard = ProductionDashboard()
            
            # Measure dashboard startup time
            start_time = time.time()
            dashboard.start_dashboard()
            startup_time = time.time() - start_time
            
            time.sleep(1)  # Allow data collection
            
            # Measure dashboard data retrieval performance
            start_time = time.time()
            for dashboard_type in [DashboardType.SYSTEM_OVERVIEW, DashboardType.PERFORMANCE_METRICS]:
                dashboard.get_dashboard_data(dashboard_type)
            retrieval_time = (time.time() - start_time) / 2
            
            performance_acceptable = startup_time < 3.0 and retrieval_time < 1.0
            
            results.append({
                'test': 'dashboard_performance',
                'passed': performance_acceptable,
                'message': f"Dashboard performance acceptable: {performance_acceptable}",
                'startup_time_seconds': round(startup_time, 3),
                'retrieval_time_seconds': round(retrieval_time, 3),
            })
            
            dashboard.stop_dashboard()
            
        except Exception as e:
            results.append({
                'test': 'dashboard_performance',
                'passed': False,
                'error': str(e),
            })
        
        # Test 3: Memory usage validation
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Start all components
            monitor = ProductionHealthMonitor()
            dashboard = ProductionDashboard(monitor)
            
            monitor.start_monitoring()
            dashboard.start_dashboard()
            
            time.sleep(3)  # Allow full operation
            
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = final_memory - initial_memory
            
            # Check if memory usage is reasonable (< 100MB increase)
            memory_acceptable = memory_increase < 100
            
            results.append({
                'test': 'memory_usage_validation',
                'passed': memory_acceptable,
                'message': f"Memory usage acceptable: {memory_acceptable}",
                'initial_memory_mb': round(initial_memory, 1),
                'final_memory_mb': round(final_memory, 1),
                'memory_increase_mb': round(memory_increase, 1),
            })
            
            # Cleanup
            monitor.stop_monitoring()
            dashboard.stop_dashboard()
            
        except Exception as e:
            results.append({
                'test': 'memory_usage_validation',
                'passed': False,
                'error': str(e),
            })
        
        return results
    
    def _validate_backup_recovery(self) -> List[Dict[str, Any]]:
        """Validate backup and recovery functionality."""
        results = []
        
        # Test 1: Backup configuration validation
        try:
            backup_config = BackupConfig(
                enabled=True,
                database_backup_interval_hours=24,
                model_backup_interval_hours=24,
                config_backup_interval_hours=12,
                retention_days=7,
                storage_location="/tmp/test_backups",
                encryption_enabled=True,
                compression_enabled=True,
            )
            
            # Validate configuration
            config_dict = backup_config.to_dict()
            config_valid = all(
                key in config_dict for key in [
                    'enabled', 'database_backup_interval_hours',
                    'storage_location', 'retention_days'
                ]
            )
            
            results.append({
                'test': 'backup_configuration_validation',
                'passed': config_valid,
                'message': f"Backup configuration valid: {config_valid}",
                'config_keys': list(config_dict.keys()),
            })
            
        except Exception as e:
            results.append({
                'test': 'backup_configuration_validation',
                'passed': False,
                'error': str(e),
            })
        
        # Test 2: Manual backup execution
        try:
            backup_config = BackupConfig(
                enabled=True,
                database_backup_interval_hours=24,
                model_backup_interval_hours=24,
                config_backup_interval_hours=12,
                retention_days=7,
                storage_location="/tmp/test_backups",
                encryption_enabled=False,  # Disable for testing
                compression_enabled=False,
            )
            
            backup_manager = BackupManager(backup_config)
            
            # Test different backup types
            backup_types = ['database', 'models', 'config', 'full']
            backup_results = []
            
            for backup_type in backup_types:
                result = backup_manager.create_manual_backup(backup_type)
                backup_results.append({
                    'type': backup_type,
                    'success': result.get('success', False),
                })
            
            successful_backups = sum(1 for r in backup_results if r['success'])
            
            results.append({
                'test': 'manual_backup_execution',
                'passed': successful_backups == len(backup_types),
                'message': f"Manual backups: {successful_backups}/{len(backup_types)} successful",
                'backup_results': backup_results,
            })
            
        except Exception as e:
            results.append({
                'test': 'manual_backup_execution',
                'passed': False,
                'error': str(e),
            })
        
        return results
    
    def save_validation_report(self, report: Dict[str, Any], filepath: str = "production_validation_report.json") -> None:
        """Save validation report to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Validation report saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")


def main():
    """Main validation function."""
    print("🚀 Starting Production Deployment Validation Suite")
    print("=" * 60)
    
    try:
        # Initialize validation suite
        suite = ProductionValidationSuite()
        
        # Run comprehensive validation
        report = suite.run_comprehensive_validation()
        
        # Save report
        suite.save_validation_report(report)
        
        # Print summary
        summary = report['validation_summary']
        print(f"\n🎯 VALIDATION COMPLETE")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Duration: {summary['duration_seconds']:.2f} seconds")
        
        if summary['success_rate'] >= 90:
            print("✅ Production deployment system is ready for use!")
            return 0
        elif summary['success_rate'] >= 75:
            print("⚠️ Production deployment system is mostly ready with minor issues")
            return 0
        else:
            print("❌ Production deployment system needs attention")
            return 1
            
    except Exception as e:
        print(f"❌ Validation suite failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())