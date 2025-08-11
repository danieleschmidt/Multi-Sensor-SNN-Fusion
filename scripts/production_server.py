#!/usr/bin/env python3
"""
Production Server Entry Point for SNN Fusion Framework

Handles graceful startup, shutdown, and production-ready configuration
for the neuromorphic computing framework.
"""

import os
import sys
import signal
import logging
import asyncio
import uvloop
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn
from snn_fusion.utils.config import load_config, create_default_config
from snn_fusion.utils.logging import setup_logging
from snn_fusion.utils.health_monitoring import HealthMonitor
from snn_fusion.optimization.performance_optimizer import PerformanceOptimizer
from snn_fusion.deployment.distributed_deployment import (
    DistributedDeploymentManager, create_deployment_config
)

class ProductionServer:
    """Production server manager with health monitoring and graceful shutdown."""
    
    def __init__(self):
        self.config = None
        self.health_monitor = None
        self.performance_optimizer = None
        self.deployment_manager = None
        self.logger = None
        self._shutdown_event = asyncio.Event()
        self._tasks = []
        
    async def initialize(self):
        """Initialize all production components."""
        # Load configuration
        self.config = self._load_production_config()
        
        # Setup logging
        self.logger = setup_logging(
            log_level=self.config.get('log_level', 'INFO'),
            enable_console=True,
            structured_format=True
        )
        
        self.logger.info("Starting SNN Fusion Framework in production mode")
        
        # Initialize performance optimizer
        self.performance_optimizer = PerformanceOptimizer(
            optimization_level=self.config.get('optimization_level', 'BALANCED'),
            enable_profiling=False  # Disabled in production for performance
        )
        
        # Initialize health monitoring
        self.health_monitor = HealthMonitor(
            monitoring_interval=30.0,
            enable_alerts=True
        )
        
        # Initialize distributed deployment if configured
        if self.config.get('enable_distributed', False):
            deployment_config = create_deployment_config(
                deployment_id="snn-fusion-production",
                min_nodes=self.config.get('min_nodes', 1),
                max_nodes=self.config.get('max_nodes', 10)
            )
            self.deployment_manager = DistributedDeploymentManager(deployment_config)
        
        self.logger.info("Production components initialized successfully")
    
    def _load_production_config(self) -> Dict[str, Any]:
        """Load production configuration from environment and files."""
        config = {
            'host': os.getenv('SNN_FUSION_API_HOST', '0.0.0.0'),
            'port': int(os.getenv('SNN_FUSION_API_PORT', '8000')),
            'workers': int(os.getenv('SNN_FUSION_WORKERS', '4')),
            'log_level': os.getenv('SNN_FUSION_LOG_LEVEL', 'INFO'),
            'reload': False,  # Never reload in production
            'access_log': True,
            'enable_distributed': os.getenv('ENABLE_DISTRIBUTED', 'false').lower() == 'true',
            'min_nodes': int(os.getenv('MIN_NODES', '1')),
            'max_nodes': int(os.getenv('MAX_NODES', '10')),
            'optimization_level': os.getenv('OPTIMIZATION_LEVEL', 'BALANCED'),
        }
        
        # Load additional config from file if exists
        config_file = Path('config/production.yaml')
        if config_file.exists():
            try:
                file_config = load_config(str(config_file))
                config.update(file_config)
            except Exception as e:
                # Use defaults if config file is invalid
                logging.warning(f"Failed to load config file: {e}")
        
        return config
    
    async def start_background_tasks(self):
        """Start background monitoring and optimization tasks."""
        # Start health monitoring
        if self.health_monitor:
            await self._start_task(
                self.health_monitor.start_monitoring(),
                "HealthMonitor"
            )
        
        # Start distributed deployment if configured
        if self.deployment_manager:
            await self._start_task(
                self.deployment_manager.start_deployment(),
                "DistributedDeployment"
            )
        
        # Start performance monitoring
        await self._start_task(
            self._performance_monitoring_loop(),
            "PerformanceMonitor"
        )
    
    async def _start_task(self, coro, name: str):
        """Start a background task with error handling."""
        try:
            task = asyncio.create_task(coro, name=name)
            self._tasks.append(task)
            self.logger.info(f"Started background task: {name}")
        except Exception as e:
            self.logger.error(f"Failed to start background task {name}: {e}")
    
    async def _performance_monitoring_loop(self):
        """Monitor performance metrics continuously."""
        while not self._shutdown_event.is_set():
            try:
                # Get performance summary
                if self.performance_optimizer:
                    summary = self.performance_optimizer.get_optimization_summary()
                    
                    # Log key metrics
                    hardware_info = summary.get('hardware_info', {})
                    self.logger.info(
                        "Performance metrics",
                        extra={
                            'cpu_count': hardware_info.get('cpu_count'),
                            'memory_gb': hardware_info.get('memory_gb'),
                            'gpu_available': hardware_info.get('gpu_available', False)
                        }
                    )
                
                # Wait before next check
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(10)
    
    async def shutdown(self):
        """Graceful shutdown of all components."""
        self.logger.info("Starting graceful shutdown...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.error(f"Error during task shutdown: {e}")
        
        # Stop components
        if self.deployment_manager:
            try:
                self.deployment_manager.stop_deployment()
            except Exception as e:
                self.logger.error(f"Error stopping deployment manager: {e}")
        
        if self.health_monitor:
            try:
                await self.health_monitor.stop_monitoring()
            except Exception as e:
                self.logger.error(f"Error stopping health monitor: {e}")
        
        self.logger.info("Graceful shutdown completed")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def create_app():
    """Create and configure the FastAPI application."""
    try:
        from snn_fusion.api.app import app
        return app
    except ImportError as e:
        logging.error(f"Failed to import application: {e}")
        # Create a minimal health check app as fallback
        from fastapi import FastAPI
        
        fallback_app = FastAPI(title="SNN Fusion Framework - Limited Mode")
        
        @fallback_app.get("/health")
        async def health_check():
            return {"status": "limited", "error": "Main application failed to load"}
        
        return fallback_app


def main():
    """Main entry point for production server."""
    # Use uvloop for better performance
    if sys.platform != 'win32':
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    # Create server instance
    server = ProductionServer()
    
    async def startup():
        """Startup handler."""
        try:
            await server.initialize()
            await server.start_background_tasks()
            server.setup_signal_handlers()
        except Exception as e:
            logging.error(f"Failed to initialize server: {e}")
            sys.exit(1)
    
    async def shutdown_handler():
        """Shutdown handler."""
        await server.shutdown()
    
    # Configure uvicorn
    config = server._load_production_config() if hasattr(server, '_load_production_config') else {
        'host': '0.0.0.0',
        'port': 8000,
        'workers': 1,  # Will be overridden if using gunicorn
        'log_level': 'info',
        'access_log': True
    }
    
    # Create application
    async def app_factory():
        await startup()
        return await create_app()
    
    # Run server
    try:
        uvicorn.run(
            app_factory,
            host=config['host'],
            port=config['port'],
            log_level=config['log_level'].lower(),
            access_log=config['access_log'],
            loop='uvloop' if sys.platform != 'win32' else 'asyncio',
            lifespan='on'
        )
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.run_until_complete(shutdown_handler())


if __name__ == "__main__":
    main()