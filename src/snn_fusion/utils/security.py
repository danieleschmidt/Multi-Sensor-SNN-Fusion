"""
Security Utilities for SNN-Fusion

This module provides security functions for input sanitization, file validation,
and resource management to protect against malicious inputs and resource exhaustion.
"""

import os
import re
import hashlib
import tempfile
import resource
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import warnings
import json


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    
    def __init__(self, message: str, error_code: str = None, severity: str = "HIGH"):
        super().__init__(message)
        self.error_code = error_code
        self.severity = severity


class ResourceLimits:
    """Resource limits for security."""
    
    # Memory limits (bytes)
    MAX_MEMORY_USAGE = 8 * 1024 * 1024 * 1024  # 8GB
    MAX_TENSOR_SIZE = 1024 * 1024 * 1024  # 1GB per tensor
    
    # File limits
    MAX_FILE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB
    MAX_FILES_PER_DIRECTORY = 10000
    
    # Computation limits
    MAX_SEQUENCE_LENGTH = 100000
    MAX_BATCH_SIZE = 1024
    MAX_MODEL_PARAMETERS = 1000000000  # 1B parameters
    
    # Time limits (seconds)
    MAX_PROCESSING_TIME = 3600  # 1 hour
    MAX_TRAINING_TIME = 86400  # 24 hours


def sanitize_input(
    input_value: Any,
    input_type: type,
    max_length: Optional[int] = None,
    allowed_chars: Optional[str] = None
) -> Any:
    """
    Sanitize input value to prevent injection attacks.
    
    Args:
        input_value: Value to sanitize
        input_type: Expected type
        max_length: Maximum length for strings
        allowed_chars: Regex pattern for allowed characters
        
    Returns:
        Sanitized value
        
    Raises:
        SecurityError: If input is invalid or potentially malicious
    """
    # Type validation
    if not isinstance(input_value, input_type):
        raise SecurityError(
            f"Input type mismatch: expected {input_type}, got {type(input_value)}",
            error_code="TYPE_MISMATCH"
        )
    
    # String sanitization
    if isinstance(input_value, str):
        # Check length
        if max_length and len(input_value) > max_length:
            raise SecurityError(
                f"Input string too long: {len(input_value)} > {max_length}",
                error_code="STRING_TOO_LONG"
            )
        
        # Check for potentially malicious patterns
        dangerous_patterns = [
            r'<script[^>]*>',  # Script tags
            r'javascript:',    # JavaScript URLs
            r'on\w+\s*=',     # Event handlers
            r'eval\s*\(',     # Eval calls
            r'exec\s*\(',     # Exec calls
            r'import\s+os',   # OS imports
            r'__import__',    # Dynamic imports
            r'\.\./',         # Directory traversal
            r'[;&|`$]',       # Command injection
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, input_value, re.IGNORECASE):
                raise SecurityError(
                    f"Potentially malicious pattern detected: {pattern}",
                    error_code="MALICIOUS_PATTERN",
                    severity="CRITICAL"
                )
        
        # Character whitelist
        if allowed_chars:
            if not re.match(f"^[{allowed_chars}]*$", input_value):
                raise SecurityError(
                    f"Input contains disallowed characters",
                    error_code="INVALID_CHARACTERS"
                )
    
    # Numeric sanitization
    elif isinstance(input_value, (int, float)):
        # Check for extreme values
        if abs(input_value) > 1e10:
            warnings.warn("Extremely large numeric value detected")
        
        # Check for NaN/Inf
        if isinstance(input_value, float):
            if not (input_value == input_value):  # NaN check
                raise SecurityError(
                    "NaN value not allowed",
                    error_code="NAN_VALUE"
                )
            if abs(input_value) == float('inf'):
                raise SecurityError(
                    "Infinite value not allowed",
                    error_code="INF_VALUE"
                )
    
    # List/dict sanitization
    elif isinstance(input_value, (list, dict)):
        max_container_size = 10000
        if len(input_value) > max_container_size:
            raise SecurityError(
                f"Container too large: {len(input_value)} > {max_container_size}",
                error_code="CONTAINER_TOO_LARGE"
            )
    
    return input_value


def validate_file_path(
    file_path: Union[str, Path],
    allowed_extensions: Optional[List[str]] = None,
    check_exists: bool = True,
    allow_create: bool = False
) -> Path:
    """
    Validate and sanitize file path to prevent directory traversal and other attacks.
    
    Args:
        file_path: Path to validate
        allowed_extensions: List of allowed file extensions
        check_exists: Whether file must exist
        allow_create: Whether to allow creation of new files
        
    Returns:
        Validated Path object
        
    Raises:
        SecurityError: If path is invalid or potentially malicious
    """
    if isinstance(file_path, str):
        # Sanitize string input
        file_path = sanitize_input(file_path, str, max_length=4096, 
                                 allowed_chars=r'a-zA-Z0-9._/\-\\:')
    
    path = Path(file_path).resolve()  # Resolve to absolute path
    
    # Check for directory traversal attempts
    if '..' in str(path) or str(path).startswith('/'):
        # Allow absolute paths but be cautious
        if not str(path).startswith(('/tmp', '/var/tmp', str(Path.cwd()))):
            warnings.warn(f"Potentially unsafe absolute path: {path}")
    
    # Check file extension
    if allowed_extensions:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise SecurityError(
                f"File extension '{path.suffix}' not allowed. "
                f"Allowed: {allowed_extensions}",
                error_code="INVALID_EXTENSION"
            )
    
    # Check if file exists
    if check_exists and not path.exists():
        if not allow_create:
            raise SecurityError(
                f"File does not exist: {path}",
                error_code="FILE_NOT_FOUND"
            )
    
    # Check file size if exists
    if path.exists() and path.is_file():
        file_size = path.stat().st_size
        if file_size > ResourceLimits.MAX_FILE_SIZE:
            raise SecurityError(
                f"File too large: {file_size} bytes > {ResourceLimits.MAX_FILE_SIZE}",
                error_code="FILE_TOO_LARGE"
            )
    
    # Check parent directory permissions
    parent_dir = path.parent
    if parent_dir.exists():
        if not os.access(parent_dir, os.R_OK):
            raise SecurityError(
                f"No read permission for directory: {parent_dir}",
                error_code="NO_READ_PERMISSION"
            )
        
        if allow_create and not os.access(parent_dir, os.W_OK):
            raise SecurityError(
                f"No write permission for directory: {parent_dir}",
                error_code="NO_WRITE_PERMISSION"
            )
    
    return path


def check_resource_limits(
    memory_usage: Optional[int] = None,
    tensor_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    model_parameters: Optional[int] = None
) -> bool:
    """
    Check if resource usage is within safe limits.
    
    Args:
        memory_usage: Current memory usage in bytes
        tensor_size: Size of tensor in bytes
        sequence_length: Length of temporal sequence
        batch_size: Batch size
        model_parameters: Number of model parameters
        
    Returns:
        True if within limits
        
    Raises:
        SecurityError: If limits are exceeded
    """
    # Memory usage check
    if memory_usage is None:
        try:
            process = psutil.Process()
            memory_usage = process.memory_info().rss
        except:
            memory_usage = 0
    
    if memory_usage > ResourceLimits.MAX_MEMORY_USAGE:
        raise SecurityError(
            f"Memory usage exceeds limit: {memory_usage} > {ResourceLimits.MAX_MEMORY_USAGE}",
            error_code="MEMORY_LIMIT_EXCEEDED",
            severity="CRITICAL"
        )
    
    # Tensor size check
    if tensor_size and tensor_size > ResourceLimits.MAX_TENSOR_SIZE:
        raise SecurityError(
            f"Tensor size exceeds limit: {tensor_size} > {ResourceLimits.MAX_TENSOR_SIZE}",
            error_code="TENSOR_TOO_LARGE"
        )
    
    # Sequence length check
    if sequence_length and sequence_length > ResourceLimits.MAX_SEQUENCE_LENGTH:
        raise SecurityError(
            f"Sequence length exceeds limit: {sequence_length} > {ResourceLimits.MAX_SEQUENCE_LENGTH}",
            error_code="SEQUENCE_TOO_LONG"
        )
    
    # Batch size check
    if batch_size and batch_size > ResourceLimits.MAX_BATCH_SIZE:
        raise SecurityError(
            f"Batch size exceeds limit: {batch_size} > {ResourceLimits.MAX_BATCH_SIZE}",
            error_code="BATCH_TOO_LARGE"
        )
    
    # Model parameters check
    if model_parameters and model_parameters > ResourceLimits.MAX_MODEL_PARAMETERS:
        raise SecurityError(
            f"Model parameter count exceeds limit: {model_parameters} > {ResourceLimits.MAX_MODEL_PARAMETERS}",
            error_code="MODEL_TOO_LARGE"
        )
    
    return True


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize configuration dictionary to prevent injection attacks.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Sanitized configuration
        
    Raises:
        SecurityError: If configuration contains malicious content
    """
    if not isinstance(config, dict):
        raise SecurityError(
            f"Configuration must be a dictionary, got {type(config)}",
            error_code="INVALID_CONFIG_TYPE"
        )
    
    sanitized_config = {}
    
    for key, value in config.items():
        # Sanitize key
        sanitized_key = sanitize_input(key, str, max_length=256, 
                                     allowed_chars=r'a-zA-Z0-9._\-')
        
        # Sanitize value based on type
        if isinstance(value, str):
            sanitized_value = sanitize_input(value, str, max_length=4096)
        elif isinstance(value, (int, float)):
            sanitized_value = sanitize_input(value, type(value))
        elif isinstance(value, bool):
            sanitized_value = value
        elif isinstance(value, list):
            # Recursively sanitize list elements
            sanitized_value = []
            for item in value:
                if isinstance(item, str):
                    sanitized_value.append(sanitize_input(item, str, max_length=1024))
                else:
                    sanitized_value.append(item)
        elif isinstance(value, dict):
            # Recursively sanitize nested dictionary
            sanitized_value = sanitize_config(value)
        elif value is None:
            sanitized_value = None
        else:
            warnings.warn(f"Unknown value type in config: {type(value)}")
            sanitized_value = value
        
        sanitized_config[sanitized_key] = sanitized_value
    
    return sanitized_config


def create_secure_temp_file(
    suffix: str = "",
    prefix: str = "snn_fusion_",
    dir: Optional[str] = None
) -> Path:
    """
    Create a secure temporary file.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        dir: Directory to create file in (None for system temp)
        
    Returns:
        Path to secure temporary file
    """
    # Sanitize inputs
    suffix = sanitize_input(suffix, str, max_length=10, allowed_chars=r'a-zA-Z0-9._\-')
    prefix = sanitize_input(prefix, str, max_length=50, allowed_chars=r'a-zA-Z0-9._\-')
    
    if dir:
        dir = validate_file_path(dir, check_exists=True, allow_create=False)
    
    # Create secure temporary file
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
    os.close(fd)  # Close file descriptor
    
    # Set restrictive permissions (owner read/write only)
    os.chmod(temp_path, 0o600)
    
    return Path(temp_path)


def compute_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Compute cryptographic hash of file for integrity checking.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('sha256', 'sha512', 'md5')
        
    Returns:
        Hexadecimal hash string
        
    Raises:
        SecurityError: If file cannot be hashed
    """
    path = validate_file_path(file_path, check_exists=True)
    
    # Validate algorithm
    if algorithm not in ['sha256', 'sha512', 'md5']:
        raise SecurityError(
            f"Unsupported hash algorithm: {algorithm}",
            error_code="INVALID_ALGORITHM"
        )
    
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    except IOError as e:
        raise SecurityError(
            f"Cannot compute hash for file {path}: {e}",
            error_code="HASH_COMPUTATION_FAILED"
        )


def validate_json_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely load and validate JSON configuration file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        SecurityError: If configuration is invalid or malicious
    """
    path = validate_file_path(
        config_path, 
        allowed_extensions=['.json', '.jsonc'],
        check_exists=True
    )
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # Limit file size during read
            content = f.read(1024 * 1024)  # Max 1MB config file
            if len(content) >= 1024 * 1024:
                raise SecurityError(
                    "Configuration file too large (>1MB)",
                    error_code="CONFIG_TOO_LARGE"
                )
        
        # Parse JSON
        config = json.loads(content)
        
        # Sanitize configuration
        sanitized_config = sanitize_config(config)
        
        return sanitized_config
    
    except json.JSONDecodeError as e:
        raise SecurityError(
            f"Invalid JSON in configuration file: {e}",
            error_code="INVALID_JSON"
        )
    except IOError as e:
        raise SecurityError(
            f"Cannot read configuration file {path}: {e}",
            error_code="CONFIG_READ_FAILED"
        )


# Security audit functions

def audit_model_security(model_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Audit model state for security issues.
    
    Args:
        model_state: Model state dictionary
        
    Returns:
        List of security findings
    """
    findings = []
    
    # Check for suspicious parameter names
    suspicious_patterns = [
        r'eval',
        r'exec',
        r'import',
        r'subprocess',
        r'os\.',
        r'sys\.',
    ]
    
    for key in model_state.keys():
        for pattern in suspicious_patterns:
            if re.search(pattern, key, re.IGNORECASE):
                findings.append({
                    'severity': 'HIGH',
                    'type': 'SUSPICIOUS_PARAMETER_NAME',
                    'details': f"Parameter name '{key}' matches suspicious pattern '{pattern}'"
                })
    
    # Check parameter sizes
    total_params = 0
    for key, value in model_state.items():
        if hasattr(value, 'numel'):
            param_count = value.numel()
            total_params += param_count
            
            if param_count > 100_000_000:  # 100M parameters in single tensor
                findings.append({
                    'severity': 'MEDIUM',
                    'type': 'LARGE_PARAMETER_TENSOR',
                    'details': f"Parameter '{key}' has {param_count} elements"
                })
    
    if total_params > ResourceLimits.MAX_MODEL_PARAMETERS:
        findings.append({
            'severity': 'HIGH',
            'type': 'MODEL_TOO_LARGE',
            'details': f"Total parameters {total_params} exceeds limit {ResourceLimits.MAX_MODEL_PARAMETERS}"
        })
    
    return findings


# Example usage and testing
if __name__ == "__main__":
    print("Testing security functions...")
    
    # Test input sanitization
    try:
        safe_string = sanitize_input("normal_string", str, max_length=100)
        print("✓ String sanitization passed")
    except SecurityError as e:
        print(f"✗ String sanitization failed: {e}")
    
    try:
        sanitize_input("<script>alert('xss')</script>", str)
        print("✗ XSS detection failed")
    except SecurityError:
        print("✓ XSS detection passed")
    
    # Test file path validation
    try:
        validate_file_path("../../../etc/passwd", check_exists=False)
        print("✗ Directory traversal detection failed")
    except SecurityError:
        print("✓ Directory traversal detection passed")
    
    # Test resource limits
    try:
        check_resource_limits(
            sequence_length=50,
            batch_size=32,
            model_parameters=1000000
        )
        print("✓ Resource limit check passed")
    except SecurityError as e:
        print(f"✗ Resource limit check failed: {e}")
    
    print("Security tests completed!")