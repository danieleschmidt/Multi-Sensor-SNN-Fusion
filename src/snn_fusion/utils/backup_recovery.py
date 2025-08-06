"""
Backup and Recovery System

This module provides comprehensive backup and recovery capabilities
for models, configurations, data, and system state in the SNN-Fusion framework.
"""

import os
import shutil
import json
import pickle
import gzip
import tarfile
import time
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from contextlib import contextmanager


class BackupType(Enum):
    """Types of backup operations."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"


@dataclass
class BackupMetadata:
    """Backup metadata information."""
    backup_id: str
    timestamp: datetime
    backup_type: BackupType
    status: BackupStatus
    size_bytes: int
    file_count: int
    checksum: str
    source_paths: List[str]
    backup_path: str
    compression_used: bool = True
    encryption_used: bool = False
    retention_days: int = 30
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['backup_type'] = self.backup_type.value
        data['status'] = self.status.value
        return data


class BackupManager:
    """
    Comprehensive backup and recovery manager.
    
    Provides automated backup scheduling, incremental backups,
    data integrity verification, and recovery operations.
    """
    
    def __init__(
        self,
        backup_root: str = "./backups",
        max_backup_size_gb: float = 10.0,
        retention_days: int = 30,
        compression_level: int = 6,
        enable_encryption: bool = False,
        auto_backup_interval: int = 3600,  # seconds
        max_concurrent_backups: int = 2
    ):
        """
        Initialize backup manager.
        
        Args:
            backup_root: Root directory for backups
            max_backup_size_gb: Maximum backup size in GB
            retention_days: Default retention period in days
            compression_level: Compression level (0-9)
            enable_encryption: Whether to encrypt backups
            auto_backup_interval: Automatic backup interval in seconds
            max_concurrent_backups: Maximum concurrent backup operations
        """
        self.backup_root = Path(backup_root)
        self.max_backup_size_bytes = int(max_backup_size_gb * 1024**3)
        self.retention_days = retention_days
        self.compression_level = compression_level
        self.enable_encryption = enable_encryption
        self.auto_backup_interval = auto_backup_interval
        self.max_concurrent_backups = max_concurrent_backups
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create backup directories
        self._setup_backup_directories()
        
        # Backup tracking
        self.backups: Dict[str, BackupMetadata] = {}
        self.load_backup_registry()
        
        # Automatic backup scheduling
        self.auto_backup_enabled = False
        self.auto_backup_thread: Optional[threading.Thread] = None
        self.backup_stop_event = threading.Event()
        
        # Concurrent backup control
        self.backup_semaphore = threading.Semaphore(max_concurrent_backups)
        self.active_backups: Dict[str, threading.Thread] = {}
        
        self.logger.info("BackupManager initialized")
    
    def _setup_backup_directories(self):
        """Setup backup directory structure."""
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['full', 'incremental', 'metadata', 'temp', 'archive']
        for subdir in subdirs:
            (self.backup_root / subdir).mkdir(exist_ok=True)
        
        # Backup registry file
        self.registry_file = self.backup_root / "backup_registry.json"
    
    def load_backup_registry(self):
        """Load backup registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                
                for backup_id, backup_data in registry_data.items():
                    # Convert timestamp back to datetime
                    backup_data['timestamp'] = datetime.fromisoformat(backup_data['timestamp'])
                    backup_data['backup_type'] = BackupType(backup_data['backup_type'])
                    backup_data['status'] = BackupStatus(backup_data['status'])
                    
                    self.backups[backup_id] = BackupMetadata(**backup_data)
                
                self.logger.info(f"Loaded {len(self.backups)} backup records")
                
            except Exception as e:
                self.logger.error(f"Failed to load backup registry: {e}")
    
    def save_backup_registry(self):
        """Save backup registry to disk."""
        try:
            registry_data = {backup_id: backup.to_dict() for backup_id, backup in self.backups.items()}
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save backup registry: {e}")
    
    def create_backup(
        self,
        source_paths: List[Union[str, Path]],
        backup_type: BackupType = BackupType.FULL,
        description: str = "",
        retention_days: Optional[int] = None
    ) -> str:
        """
        Create a backup of specified paths.
        
        Args:
            source_paths: List of paths to backup
            backup_type: Type of backup to create
            description: Optional description
            retention_days: Custom retention period
            
        Returns:
            Backup ID
        """
        backup_id = self._generate_backup_id(backup_type)
        
        # Validate source paths
        validated_paths = []
        for path in source_paths:
            path_obj = Path(path)
            if path_obj.exists():
                validated_paths.append(str(path_obj.resolve()))
            else:
                self.logger.warning(f"Source path does not exist: {path}")
        
        if not validated_paths:
            raise ValueError("No valid source paths provided")
        
        # Create backup metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            timestamp=datetime.now(),
            backup_type=backup_type,
            status=BackupStatus.PENDING,
            size_bytes=0,
            file_count=0,
            checksum="",
            source_paths=validated_paths,
            backup_path="",
            retention_days=retention_days or self.retention_days,
            description=description
        )
        
        # Register backup
        self.backups[backup_id] = metadata
        self.save_backup_registry()
        
        # Start backup in separate thread
        backup_thread = threading.Thread(
            target=self._execute_backup,
            args=(backup_id,),
            daemon=True
        )
        
        self.active_backups[backup_id] = backup_thread
        backup_thread.start()
        
        self.logger.info(f"Started backup {backup_id} for {len(validated_paths)} paths")
        
        return backup_id
    
    def _execute_backup(self, backup_id: str):
        """Execute backup operation."""
        metadata = self.backups[backup_id]
        
        with self.backup_semaphore:
            try:
                metadata.status = BackupStatus.IN_PROGRESS
                self.save_backup_registry()
                
                self.logger.info(f"Executing backup {backup_id}")
                
                # Create backup file
                backup_filename = f"{backup_id}.tar.gz"
                backup_subdir = metadata.backup_type.value
                backup_path = self.backup_root / backup_subdir / backup_filename
                
                # Create compressed archive
                total_size = 0
                file_count = 0
                
                with tarfile.open(backup_path, 'w:gz', compresslevel=self.compression_level) as tar:
                    for source_path in metadata.source_paths:
                        source_path_obj = Path(source_path)
                        
                        if source_path_obj.is_file():
                            # Add single file
                            tar.add(source_path_obj, arcname=source_path_obj.name)
                            total_size += source_path_obj.stat().st_size
                            file_count += 1
                        elif source_path_obj.is_dir():
                            # Add directory recursively
                            for file_path in source_path_obj.rglob('*'):
                                if file_path.is_file():
                                    relative_path = file_path.relative_to(source_path_obj.parent)
                                    tar.add(file_path, arcname=str(relative_path))
                                    total_size += file_path.stat().st_size
                                    file_count += 1
                
                # Check backup size
                backup_size = backup_path.stat().st_size
                if backup_size > self.max_backup_size_bytes:
                    raise ValueError(f"Backup size ({backup_size} bytes) exceeds limit ({self.max_backup_size_bytes} bytes)")
                
                # Calculate checksum
                checksum = self._calculate_file_checksum(backup_path)
                
                # Update metadata
                metadata.status = BackupStatus.COMPLETED
                metadata.size_bytes = backup_size
                metadata.file_count = file_count
                metadata.checksum = checksum
                metadata.backup_path = str(backup_path)
                
                self.save_backup_registry()
                
                self.logger.info(f"Backup {backup_id} completed successfully ({backup_size} bytes, {file_count} files)")
                
            except Exception as e:
                self.logger.error(f"Backup {backup_id} failed: {e}")
                metadata.status = BackupStatus.FAILED
                self.save_backup_registry()
                
            finally:
                # Remove from active backups
                if backup_id in self.active_backups:
                    del self.active_backups[backup_id]
    
    def restore_backup(
        self,
        backup_id: str,
        restore_path: Union[str, Path],
        overwrite_existing: bool = False,
        verify_integrity: bool = True
    ) -> bool:
        """
        Restore a backup to specified location.
        
        Args:
            backup_id: ID of backup to restore
            restore_path: Path where to restore files
            overwrite_existing: Whether to overwrite existing files
            verify_integrity: Whether to verify backup integrity first
            
        Returns:
            True if restore was successful
        """
        if backup_id not in self.backups:
            raise ValueError(f"Backup {backup_id} not found")
        
        metadata = self.backups[backup_id]
        
        if metadata.status != BackupStatus.COMPLETED:
            raise ValueError(f"Backup {backup_id} is not in completed state: {metadata.status}")
        
        backup_path = Path(metadata.backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        restore_path_obj = Path(restore_path)
        
        try:
            self.logger.info(f"Restoring backup {backup_id} to {restore_path}")
            
            # Verify integrity if requested
            if verify_integrity:
                if not self.verify_backup_integrity(backup_id):
                    raise ValueError(f"Backup {backup_id} failed integrity check")
            
            # Create restore directory
            restore_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Extract archive
            with tarfile.open(backup_path, 'r:gz') as tar:
                # Check for path traversal attacks
                for member in tar.getmembers():
                    if os.path.isabs(member.name) or ".." in member.name:
                        self.logger.warning(f"Skipping potentially dangerous path: {member.name}")
                        continue
                    
                    # Check if file exists
                    target_path = restore_path_obj / member.name
                    if target_path.exists() and not overwrite_existing:
                        self.logger.warning(f"Skipping existing file: {target_path}")
                        continue
                    
                    # Extract file
                    tar.extract(member, restore_path_obj)
            
            self.logger.info(f"Backup {backup_id} restored successfully to {restore_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore backup {backup_id}: {e}")
            return False
    
    def verify_backup_integrity(self, backup_id: str) -> bool:
        """
        Verify backup integrity using checksum.
        
        Args:
            backup_id: ID of backup to verify
            
        Returns:
            True if backup integrity is valid
        """
        if backup_id not in self.backups:
            return False
        
        metadata = self.backups[backup_id]
        backup_path = Path(metadata.backup_path)
        
        if not backup_path.exists():
            self.logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            current_checksum = self._calculate_file_checksum(backup_path)
            
            if current_checksum != metadata.checksum:
                self.logger.error(f"Backup {backup_id} checksum mismatch: {current_checksum} != {metadata.checksum}")
                metadata.status = BackupStatus.CORRUPTED
                self.save_backup_registry()
                return False
            
            self.logger.debug(f"Backup {backup_id} integrity verified")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to verify backup {backup_id}: {e}")
            return False
    
    def list_backups(self, backup_type: Optional[BackupType] = None) -> List[BackupMetadata]:
        """
        List available backups.
        
        Args:
            backup_type: Optional filter by backup type
            
        Returns:
            List of backup metadata
        """
        backups = list(self.backups.values())
        
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.timestamp, reverse=True)
        
        return backups
    
    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_id: ID of backup to delete
            
        Returns:
            True if deletion was successful
        """
        if backup_id not in self.backups:
            return False
        
        metadata = self.backups[backup_id]
        
        try:
            # Delete backup file
            backup_path = Path(metadata.backup_path)
            if backup_path.exists():
                backup_path.unlink()
            
            # Remove from registry
            del self.backups[backup_id]
            self.save_backup_registry()
            
            self.logger.info(f"Deleted backup {backup_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    def cleanup_expired_backups(self) -> int:
        """
        Clean up expired backups based on retention policy.
        
        Returns:
            Number of backups deleted
        """
        deleted_count = 0
        current_time = datetime.now()
        
        for backup_id, metadata in list(self.backups.items()):
            age = current_time - metadata.timestamp
            
            if age.days > metadata.retention_days:
                if self.delete_backup(backup_id):
                    deleted_count += 1
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} expired backups")
        
        return deleted_count
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup statistics."""
        total_backups = len(self.backups)
        
        if total_backups == 0:
            return {"total_backups": 0, "message": "No backups found"}
        
        # Count by status
        status_counts = {}
        type_counts = {}
        total_size = 0
        
        for backup in self.backups.values():
            status = backup.status.value
            backup_type = backup.backup_type.value
            
            status_counts[status] = status_counts.get(status, 0) + 1
            type_counts[backup_type] = type_counts.get(backup_type, 0) + 1
            
            if backup.status == BackupStatus.COMPLETED:
                total_size += backup.size_bytes
        
        # Latest backup
        latest_backup = max(self.backups.values(), key=lambda x: x.timestamp)
        
        return {
            "total_backups": total_backups,
            "status_breakdown": status_counts,
            "type_breakdown": type_counts,
            "total_size_gb": total_size / (1024**3),
            "latest_backup": {
                "id": latest_backup.backup_id,
                "timestamp": latest_backup.timestamp.isoformat(),
                "type": latest_backup.backup_type.value,
                "size_mb": latest_backup.size_bytes / (1024**2)
            }
        }
    
    def start_auto_backup(self, source_paths: List[str], backup_type: BackupType = BackupType.INCREMENTAL):
        """Start automatic backup scheduling."""
        if self.auto_backup_enabled:
            self.logger.warning("Auto backup already running")
            return
        
        self.auto_backup_enabled = True
        self.backup_stop_event.clear()
        
        self.auto_backup_thread = threading.Thread(
            target=self._auto_backup_loop,
            args=(source_paths, backup_type),
            daemon=True
        )
        self.auto_backup_thread.start()
        
        self.logger.info(f"Started automatic backup every {self.auto_backup_interval} seconds")
    
    def stop_auto_backup(self):
        """Stop automatic backup scheduling."""
        if not self.auto_backup_enabled:
            return
        
        self.auto_backup_enabled = False
        self.backup_stop_event.set()
        
        if self.auto_backup_thread:
            self.auto_backup_thread.join(timeout=10)
        
        self.logger.info("Stopped automatic backup")
    
    def _auto_backup_loop(self, source_paths: List[str], backup_type: BackupType):
        """Automatic backup loop."""
        while self.auto_backup_enabled and not self.backup_stop_event.is_set():
            try:
                # Create backup
                backup_id = self.create_backup(
                    source_paths,
                    backup_type,
                    description=f"Automatic {backup_type.value} backup"
                )
                
                # Wait for backup to complete or timeout
                start_time = time.time()
                while backup_id in self.active_backups and time.time() - start_time < 1800:  # 30 min timeout
                    if self.backup_stop_event.wait(10):
                        break
                
                # Cleanup expired backups
                self.cleanup_expired_backups()
                
                # Wait for next backup
                if not self.backup_stop_event.wait(self.auto_backup_interval):
                    continue
                else:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in auto backup loop: {e}")
                # Wait before retrying
                if self.backup_stop_event.wait(60):
                    break
    
    def _generate_backup_id(self, backup_type: BackupType) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{backup_type.value}_{timestamp}_{int(time.time()) % 10000:04d}"
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()


# Context manager for automatic backup
@contextmanager
def backup_context(
    backup_manager: BackupManager,
    source_paths: List[str],
    description: str = "Context backup"
):
    """Context manager for automatic backup on entry and exit."""
    # Create backup on entry
    entry_backup_id = backup_manager.create_backup(
        source_paths,
        BackupType.SNAPSHOT,
        description=f"{description} (entry)"
    )
    
    try:
        yield backup_manager
    finally:
        # Create backup on exit
        exit_backup_id = backup_manager.create_backup(
            source_paths,
            BackupType.SNAPSHOT,
            description=f"{description} (exit)"
        )


# Model-specific backup utilities
class ModelBackupManager(BackupManager):
    """Specialized backup manager for SNN models."""
    
    def backup_model(
        self,
        model,
        model_name: str,
        include_optimizer_state: bool = True,
        include_training_history: bool = True
    ) -> str:
        """
        Backup SNN model with metadata.
        
        Args:
            model: SNN model to backup
            model_name: Name for the model backup
            include_optimizer_state: Whether to include optimizer state
            include_training_history: Whether to include training history
            
        Returns:
            Backup ID
        """
        # Create temporary directory for model files
        temp_dir = self.backup_root / "temp" / f"model_{int(time.time())}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model state
            model_file = temp_dir / f"{model_name}_model.pkl"
            
            backup_data = {
                'model_state': getattr(model, 'state_dict', lambda: {})(),
                'model_config': getattr(model, 'config', {}),
                'model_type': type(model).__name__,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add optimizer state if available
            if include_optimizer_state and hasattr(model, 'optimizer'):
                backup_data['optimizer_state'] = model.optimizer.state_dict()
            
            # Add training history if available
            if include_training_history and hasattr(model, 'training_history'):
                backup_data['training_history'] = model.training_history
            
            # Save to file
            with open(model_file, 'wb') as f:
                pickle.dump(backup_data, f)
            
            # Create backup
            backup_id = self.create_backup(
                [str(temp_dir)],
                BackupType.SNAPSHOT,
                description=f"Model backup: {model_name}"
            )
            
            return backup_id
            
        finally:
            # Cleanup temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def restore_model(self, backup_id: str, restore_path: Optional[str] = None):
        """
        Restore SNN model from backup.
        
        Args:
            backup_id: Backup ID to restore
            restore_path: Optional path to restore to
            
        Returns:
            Model data dictionary
        """
        if not restore_path:
            restore_path = self.backup_root / "temp" / f"restore_{int(time.time())}"
        
        restore_path_obj = Path(restore_path)
        
        try:
            # Restore backup
            if not self.restore_backup(backup_id, restore_path_obj, overwrite_existing=True):
                raise RuntimeError(f"Failed to restore backup {backup_id}")
            
            # Find model file
            model_files = list(restore_path_obj.rglob("*_model.pkl"))
            if not model_files:
                raise FileNotFoundError("No model file found in backup")
            
            # Load model data
            with open(model_files[0], 'rb') as f:
                model_data = pickle.load(f)
            
            return model_data
            
        finally:
            # Cleanup
            if restore_path_obj.exists():
                shutil.rmtree(restore_path_obj, ignore_errors=True)


# Convenience functions

def create_backup_manager(config: Dict[str, Any]) -> BackupManager:
    """Create backup manager from configuration."""
    return BackupManager(
        backup_root=config.get('backup_root', './backups'),
        max_backup_size_gb=config.get('max_backup_size_gb', 10.0),
        retention_days=config.get('retention_days', 30),
        compression_level=config.get('compression_level', 6),
        enable_encryption=config.get('enable_encryption', False),
        auto_backup_interval=config.get('auto_backup_interval', 3600)
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Backup and Recovery System...")
    
    # Create backup manager
    backup_manager = BackupManager(
        backup_root="./test_backups",
        retention_days=7,
        max_backup_size_gb=1.0
    )
    
    # Create test files
    test_dir = Path("./test_data_backup")
    test_dir.mkdir(exist_ok=True)
    
    # Create some test files
    (test_dir / "test1.txt").write_text("Test file 1 content")
    (test_dir / "test2.txt").write_text("Test file 2 content")
    
    subdir = test_dir / "subdir"
    subdir.mkdir(exist_ok=True)
    (subdir / "test3.txt").write_text("Test file 3 content")
    
    print(f"Created test files in {test_dir}")
    
    # Create backup
    backup_id = backup_manager.create_backup(
        [str(test_dir)],
        BackupType.FULL,
        description="Test backup"
    )
    
    print(f"Started backup: {backup_id}")
    
    # Wait for backup to complete
    time.sleep(2)
    while backup_id in backup_manager.active_backups:
        time.sleep(1)
    
    # Check backup status
    if backup_id in backup_manager.backups:
        metadata = backup_manager.backups[backup_id]
        print(f"Backup status: {metadata.status.value}")
        print(f"Backup size: {metadata.size_bytes} bytes")
        print(f"Files backed up: {metadata.file_count}")
    
    # Verify integrity
    integrity_ok = backup_manager.verify_backup_integrity(backup_id)
    print(f"Backup integrity: {'OK' if integrity_ok else 'FAILED'}")
    
    # List backups
    backups = backup_manager.list_backups()
    print(f"\nAvailable backups: {len(backups)}")
    for backup in backups:
        print(f"  {backup.backup_id}: {backup.backup_type.value} ({backup.status.value})")
    
    # Test restore
    restore_dir = Path("./test_restore")
    if restore_dir.exists():
        shutil.rmtree(restore_dir)
    
    restore_success = backup_manager.restore_backup(backup_id, restore_dir)
    print(f"Restore result: {'SUCCESS' if restore_success else 'FAILED'}")
    
    if restore_success:
        restored_files = list(restore_dir.rglob("*"))
        print(f"Restored files: {len([f for f in restored_files if f.is_file()])}")
    
    # Get statistics
    stats = backup_manager.get_backup_statistics()
    print(f"\nBackup statistics:")
    print(f"  Total backups: {stats['total_backups']}")
    print(f"  Total size: {stats['total_size_gb']:.2f} GB")
    
    # Cleanup test files
    shutil.rmtree(test_dir, ignore_errors=True)
    shutil.rmtree(restore_dir, ignore_errors=True)
    shutil.rmtree("./test_backups", ignore_errors=True)
    
    print("✓ Backup and recovery test completed!")