"""
Security-Focused Input Validation

Implements security validation to prevent malicious inputs, injection attacks,
and data corruption in neuromorphic systems.
"""

import re
import hashlib
import hmac
import secrets
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import warnings
import logging
from enum import Enum


class SecurityLevel(Enum):
    """Security validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    PARANOID = "paranoid"


class SecurityThreat(Enum):
    """Types of security threats."""
    INJECTION = "injection"
    OVERFLOW = "overflow"
    MALFORMED_DATA = "malformed_data"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_CORRUPTION = "data_corruption"


class SecurityValidationError(Exception):
    """Exception for security validation failures."""
    
    def __init__(self, message: str, threat_type: SecurityThreat, 
                 severity: str = "high", context: Optional[Dict] = None):
        super().__init__(message)
        self.threat_type = threat_type
        self.severity = severity
        self.context = context or {}


class SecureInputValidator:
    """
    Comprehensive security-focused input validator.
    
    Validates inputs against various attack vectors and malicious patterns.
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.logger = logging.getLogger(__name__)
        
        # Compile security patterns
        self._compile_security_patterns()
        
        # Security limits
        self.limits = self._get_security_limits()
        
        # Track validation attempts
        self.validation_attempts = 0
        self.failed_validations = 0
        self.threat_counts = {threat: 0 for threat in SecurityThreat}
    
    def _compile_security_patterns(self):
        """Compile regex patterns for security threats."""
        # SQL injection patterns
        self.sql_injection_patterns = [
            re.compile(r"('|\"|;|--|\*|%|\\|\/\*|\*\/)", re.IGNORECASE),
            re.compile(r"\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b", re.IGNORECASE),
            re.compile(r"\b(script|javascript|vbscript|onload|onerror|alert)\b", re.IGNORECASE),
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            re.compile(r"(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c)", re.IGNORECASE),
            re.compile(r"(\/etc\/|\/var\/|\/proc\/|\/sys\/|c:\\|%windir%)", re.IGNORECASE),
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            re.compile(r"(\||&|;|\$\(|\`|>|<|\{|\})", re.IGNORECASE),
            re.compile(r"\b(rm|mv|cp|cat|grep|awk|sed|bash|sh|cmd|powershell)\b", re.IGNORECASE),
        ]
        
        # Code injection patterns
        self.code_injection_patterns = [
            re.compile(r"(eval\(|exec\(|import\s|__import__|getattr|setattr|delattr)", re.IGNORECASE),
            re.compile(r"(pickle|marshal|subprocess|os\.system|open\()", re.IGNORECASE),
        ]
        
        # Buffer overflow indicators
        self.overflow_patterns = [
            re.compile(r"A{100,}|0{100,}|1{100,}"),  # Repeated characters
            re.compile(r"(%n|%s|%x|%d){10,}"),  # Format string attacks
        ]
    
    def _get_security_limits(self) -> Dict[str, int]:
        """Get security limits based on security level."""
        base_limits = {
            'max_string_length': 10000,
            'max_list_length': 1000,
            'max_dict_depth': 10,
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'max_tensor_elements': 1000000,
            'max_recursion_depth': 50
        }
        
        if self.security_level == SecurityLevel.BASIC:
            multiplier = 2.0
        elif self.security_level == SecurityLevel.STANDARD:
            multiplier = 1.0
        elif self.security_level == SecurityLevel.HIGH:
            multiplier = 0.5
        else:  # PARANOID
            multiplier = 0.1
        
        return {k: int(v * multiplier) for k, v in base_limits.items()}
    
    def validate_string(self, data: str, context: str = "input") -> bool:
        """
        Validate string data for security threats.
        
        Args:
            data: String to validate
            context: Context description for logging
            
        Returns:
            True if validation passes
            
        Raises:
            SecurityValidationError: If security threat detected
        """
        self.validation_attempts += 1
        
        if not isinstance(data, str):
            raise SecurityValidationError(
                f"Expected string, got {type(data).__name__}",
                SecurityThreat.MALFORMED_DATA,
                context={'input_type': type(data).__name__}
            )
        
        # Length check
        if len(data) > self.limits['max_string_length']:
            self.threat_counts[SecurityThreat.RESOURCE_EXHAUSTION] += 1
            raise SecurityValidationError(
                f"String too long: {len(data)} > {self.limits['max_string_length']}",
                SecurityThreat.RESOURCE_EXHAUSTION,
                context={'length': len(data), 'context': context}
            )
        
        # Check for injection patterns
        self._check_injection_patterns(data, context)
        
        # Check for null bytes and control characters
        if '\x00' in data or any(ord(c) < 32 and c not in '\t\n\r' for c in data):
            self.threat_counts[SecurityThreat.MALFORMED_DATA] += 1
            raise SecurityValidationError(
                "String contains null bytes or invalid control characters",
                SecurityThreat.MALFORMED_DATA,
                context={'context': context}
            )
        
        return True
    
    def _check_injection_patterns(self, data: str, context: str):
        """Check for various injection attack patterns."""
        # SQL injection
        for pattern in self.sql_injection_patterns:
            if pattern.search(data):
                self.threat_counts[SecurityThreat.INJECTION] += 1
                raise SecurityValidationError(
                    f"Potential SQL injection detected in {context}",
                    SecurityThreat.INJECTION,
                    context={'pattern': pattern.pattern, 'context': context}
                )
        
        # Path traversal
        for pattern in self.path_traversal_patterns:
            if pattern.search(data):
                self.threat_counts[SecurityThreat.INJECTION] += 1
                raise SecurityValidationError(
                    f"Potential path traversal attack detected in {context}",
                    SecurityThreat.INJECTION,
                    context={'pattern': pattern.pattern, 'context': context}
                )
        
        # Command injection
        for pattern in self.command_injection_patterns:
            if pattern.search(data):
                self.threat_counts[SecurityThreat.INJECTION] += 1
                raise SecurityValidationError(
                    f"Potential command injection detected in {context}",
                    SecurityThreat.INJECTION,
                    context={'pattern': pattern.pattern, 'context': context}
                )
        
        # Code injection
        for pattern in self.code_injection_patterns:
            if pattern.search(data):
                self.threat_counts[SecurityThreat.INJECTION] += 1
                raise SecurityValidationError(
                    f"Potential code injection detected in {context}",
                    SecurityThreat.INJECTION,
                    context={'pattern': pattern.pattern, 'context': context}
                )
        
        # Buffer overflow patterns
        for pattern in self.overflow_patterns:
            if pattern.search(data):
                self.threat_counts[SecurityThreat.OVERFLOW] += 1
                raise SecurityValidationError(
                    f"Potential buffer overflow pattern detected in {context}",
                    SecurityThreat.OVERFLOW,
                    context={'pattern': pattern.pattern, 'context': context}
                )
    
    def validate_path(self, path: Union[str, Path], allow_create: bool = False) -> bool:
        """
        Validate file/directory paths for security.
        
        Args:
            path: Path to validate
            allow_create: Whether to allow non-existent paths
            
        Returns:
            True if validation passes
            
        Raises:
            SecurityValidationError: If security threat detected
        """
        self.validation_attempts += 1
        
        path_str = str(path)
        
        # Basic string validation
        self.validate_string(path_str, "file_path")
        
        # Convert to Path object for normalization
        path_obj = Path(path_str).resolve()
        
        # Check for path traversal
        try:
            path_obj.relative_to(Path.cwd())
        except ValueError:
            # Path is outside current working directory
            if self.security_level in [SecurityLevel.HIGH, SecurityLevel.PARANOID]:
                self.threat_counts[SecurityThreat.PRIVILEGE_ESCALATION] += 1
                raise SecurityValidationError(
                    f"Path outside working directory: {path_obj}",
                    SecurityThreat.PRIVILEGE_ESCALATION,
                    context={'path': str(path_obj)}
                )
        
        # Check for restricted directories
        restricted_dirs = ['/etc', '/var', '/proc', '/sys', '/root', '/boot']
        if any(str(path_obj).startswith(restricted) for restricted in restricted_dirs):
            self.threat_counts[SecurityThreat.PRIVILEGE_ESCALATION] += 1
            raise SecurityValidationError(
                f"Access to restricted directory: {path_obj}",
                SecurityThreat.PRIVILEGE_ESCALATION,
                context={'path': str(path_obj)}
            )
        
        # Check if path exists (if required)
        if not allow_create and not path_obj.exists():
            self.threat_counts[SecurityThreat.MALFORMED_DATA] += 1
            raise SecurityValidationError(
                f"Path does not exist: {path_obj}",
                SecurityThreat.MALFORMED_DATA,
                context={'path': str(path_obj)}
            )
        
        return True
    
    def validate_tensor_data(self, data: Any, context: str = "tensor") -> bool:
        """
        Validate tensor-like data for security threats.
        
        Args:
            data: Tensor data to validate
            context: Context description
            
        Returns:
            True if validation passes
            
        Raises:
            SecurityValidationError: If security threat detected
        """
        self.validation_attempts += 1
        
        # Check tensor size limits
        if hasattr(data, 'numel'):
            num_elements = data.numel()
        elif hasattr(data, 'size'):
            num_elements = data.size
        elif hasattr(data, '__len__'):
            num_elements = len(data)
        else:
            num_elements = 1
        
        if num_elements > self.limits['max_tensor_elements']:
            self.threat_counts[SecurityThreat.RESOURCE_EXHAUSTION] += 1
            raise SecurityValidationError(
                f"Tensor too large: {num_elements} > {self.limits['max_tensor_elements']}",
                SecurityThreat.RESOURCE_EXHAUSTION,
                context={'elements': num_elements, 'context': context}
            )
        
        # Check for suspicious patterns in data
        if hasattr(data, 'isnan') and data.isnan().any():
            warnings.warn(f"NaN values detected in {context}", RuntimeWarning)
        
        if hasattr(data, 'isinf') and data.isinf().any():
            warnings.warn(f"Infinite values detected in {context}", RuntimeWarning)
        
        # Check for data corruption patterns
        if hasattr(data, 'std') and hasattr(data, 'mean'):
            try:
                std_val = float(data.std())
                mean_val = float(data.mean())
                
                # Extremely large standard deviation might indicate corruption
                if std_val > 1e6 or abs(mean_val) > 1e6:
                    self.threat_counts[SecurityThreat.DATA_CORRUPTION] += 1
                    raise SecurityValidationError(
                        f"Suspicious data statistics in {context}: mean={mean_val}, std={std_val}",
                        SecurityThreat.DATA_CORRUPTION,
                        context={'mean': mean_val, 'std': std_val, 'context': context}
                    )
            except Exception:
                # Can't compute statistics, skip this check
                pass
        
        return True
    
    def validate_config(self, config: Dict[str, Any], context: str = "config") -> bool:
        """
        Validate configuration dictionary for security threats.
        
        Args:
            config: Configuration dictionary
            context: Context description
            
        Returns:
            True if validation passes
            
        Raises:
            SecurityValidationError: If security threat detected
        """
        self.validation_attempts += 1
        
        if not isinstance(config, dict):
            raise SecurityValidationError(
                f"Expected dictionary, got {type(config).__name__}",
                SecurityThreat.MALFORMED_DATA,
                context={'input_type': type(config).__name__}
            )
        
        # Check dictionary depth to prevent stack overflow
        depth = self._get_dict_depth(config)
        if depth > self.limits['max_dict_depth']:
            self.threat_counts[SecurityThreat.RESOURCE_EXHAUSTION] += 1
            raise SecurityValidationError(
                f"Dictionary too deep: {depth} > {self.limits['max_dict_depth']}",
                SecurityThreat.RESOURCE_EXHAUSTION,
                context={'depth': depth, 'context': context}
            )
        
        # Validate all string values
        self._validate_dict_strings(config, context)
        
        # Check for dangerous configuration keys
        dangerous_keys = ['__import__', '__class__', '__globals__', '__builtins__', 
                         'eval', 'exec', 'compile', 'open', 'file']
        
        for key in self._get_all_keys(config):
            if any(dangerous in str(key).lower() for dangerous in dangerous_keys):
                self.threat_counts[SecurityThreat.INJECTION] += 1
                raise SecurityValidationError(
                    f"Dangerous configuration key detected: {key}",
                    SecurityThreat.INJECTION,
                    context={'key': str(key), 'context': context}
                )
        
        return True
    
    def _get_dict_depth(self, d: Dict, current_depth: int = 0) -> int:
        """Get maximum depth of nested dictionary."""
        if not isinstance(d, dict):
            return current_depth
        
        if not d:
            return current_depth + 1
        
        return max(self._get_dict_depth(v, current_depth + 1) 
                  for v in d.values() if isinstance(v, dict)) or current_depth + 1
    
    def _validate_dict_strings(self, d: Dict, context: str, current_path: str = ""):
        """Recursively validate all string values in dictionary."""
        for key, value in d.items():
            current_key_path = f"{current_path}.{key}" if current_path else str(key)
            
            if isinstance(value, str):
                self.validate_string(value, f"{context}.{current_key_path}")
            elif isinstance(value, dict):
                self._validate_dict_strings(value, context, current_key_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        self.validate_string(item, f"{context}.{current_key_path}[{i}]")
                    elif isinstance(item, dict):
                        self._validate_dict_strings(item, context, f"{current_key_path}[{i}]")
    
    def _get_all_keys(self, d: Dict, keys: Optional[List] = None) -> List:
        """Get all keys from nested dictionary."""
        if keys is None:
            keys = []
        
        for key, value in d.items():
            keys.append(key)
            if isinstance(value, dict):
                self._get_all_keys(value, keys)
        
        return keys
    
    def validate_model_input(self, inputs: Any, context: str = "model_input") -> bool:
        """
        Validate model inputs comprehensively.
        
        Args:
            inputs: Model inputs to validate
            context: Context description
            
        Returns:
            True if validation passes
            
        Raises:
            SecurityValidationError: If security threat detected
        """
        self.validation_attempts += 1
        
        if isinstance(inputs, dict):
            # Multi-modal inputs
            for modality, data in inputs.items():
                self.validate_string(modality, f"{context}.modality_name")
                self.validate_tensor_data(data, f"{context}.{modality}")
        elif hasattr(inputs, 'shape'):
            # Single tensor input
            self.validate_tensor_data(inputs, context)
        elif isinstance(inputs, (list, tuple)):
            # Multiple tensor inputs
            for i, data in enumerate(inputs):
                self.validate_tensor_data(data, f"{context}[{i}]")
        else:
            raise SecurityValidationError(
                f"Unsupported input type: {type(inputs).__name__}",
                SecurityThreat.MALFORMED_DATA,
                context={'input_type': type(inputs).__name__}
            )
        
        return True
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security validation report."""
        total_attempts = max(self.validation_attempts, 1)
        
        return {
            'security_level': self.security_level.value,
            'total_validations': self.validation_attempts,
            'failed_validations': self.failed_validations,
            'success_rate': (total_attempts - self.failed_validations) / total_attempts,
            'threat_counts': {threat.value: count for threat, count in self.threat_counts.items()},
            'total_threats_detected': sum(self.threat_counts.values()),
            'security_limits': self.limits,
            'status': 'healthy' if self.failed_validations == 0 else 'threats_detected'
        }
    
    def reset_statistics(self):
        """Reset validation statistics."""
        self.validation_attempts = 0
        self.failed_validations = 0
        self.threat_counts = {threat: 0 for threat in SecurityThreat}


class SecurityTokenValidator:
    """
    Validates security tokens and API keys.
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.logger = logging.getLogger(__name__)
    
    def generate_token(self, payload: Dict[str, Any], expiry_hours: int = 24) -> str:
        """
        Generate secure token with payload and expiry.
        
        Args:
            payload: Data to include in token
            expiry_hours: Token expiry in hours
            
        Returns:
            Secure token string
        """
        import base64
        import json
        from datetime import datetime, timedelta
        
        # Add expiry to payload
        payload['exp'] = (datetime.utcnow() + timedelta(hours=expiry_hours)).isoformat()
        
        # Create token
        payload_json = json.dumps(payload, sort_keys=True)
        payload_b64 = base64.b64encode(payload_json.encode()).decode()
        
        # Create signature
        signature = hmac.new(
            self.secret_key.encode(),
            payload_b64.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{payload_b64}.{signature}"
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate and decode secure token.
        
        Args:
            token: Token to validate
            
        Returns:
            Decoded payload if valid
            
        Raises:
            SecurityValidationError: If token is invalid
        """
        import base64
        import json
        from datetime import datetime
        
        try:
            payload_b64, signature = token.split('.')
        except ValueError:
            raise SecurityValidationError(
                "Invalid token format",
                SecurityThreat.MALFORMED_DATA
            )
        
        # Verify signature
        expected_signature = hmac.new(
            self.secret_key.encode(),
            payload_b64.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(signature, expected_signature):
            raise SecurityValidationError(
                "Invalid token signature",
                SecurityThreat.PRIVILEGE_ESCALATION
            )
        
        # Decode payload
        try:
            payload_json = base64.b64decode(payload_b64).decode()
            payload = json.loads(payload_json)
        except Exception as e:
            raise SecurityValidationError(
                f"Cannot decode token payload: {e}",
                SecurityThreat.MALFORMED_DATA
            )
        
        # Check expiry
        if 'exp' in payload:
            try:
                expiry = datetime.fromisoformat(payload['exp'])
                if datetime.utcnow() > expiry:
                    raise SecurityValidationError(
                        "Token has expired",
                        SecurityThreat.PRIVILEGE_ESCALATION,
                        context={'expiry': payload['exp']}
                    )
            except ValueError:
                raise SecurityValidationError(
                    "Invalid expiry format in token",
                    SecurityThreat.MALFORMED_DATA
                )
        
        return payload


# Global security validator instance
default_security_validator = SecureInputValidator()


def secure_validate(data: Any, validation_type: str = "auto", 
                   context: str = "input") -> bool:
    """
    Convenience function for security validation.
    
    Args:
        data: Data to validate
        validation_type: Type of validation (auto, string, path, tensor, config, model_input)
        context: Context description
        
    Returns:
        True if validation passes
        
    Raises:
        SecurityValidationError: If security threat detected
    """
    validator = default_security_validator
    
    if validation_type == "auto":
        if isinstance(data, str):
            validation_type = "string"
        elif isinstance(data, (Path, str)) and ('/' in str(data) or '\\' in str(data)):
            validation_type = "path"
        elif hasattr(data, 'shape') or hasattr(data, 'numel'):
            validation_type = "tensor"
        elif isinstance(data, dict):
            validation_type = "config"
        else:
            validation_type = "model_input"
    
    if validation_type == "string":
        return validator.validate_string(data, context)
    elif validation_type == "path":
        return validator.validate_path(data)
    elif validation_type == "tensor":
        return validator.validate_tensor_data(data, context)
    elif validation_type == "config":
        return validator.validate_config(data, context)
    elif validation_type == "model_input":
        return validator.validate_model_input(data, context)
    else:
        raise ValueError(f"Unknown validation type: {validation_type}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Security Input Validation...")
    
    # Create validator
    validator = SecureInputValidator(SecurityLevel.STANDARD)
    
    # Test string validation
    print("\n1. Testing String Validation:")
    try:
        validator.validate_string("normal string", "test")
        print("  ✓ Normal string: passed")
    except SecurityValidationError as e:
        print(f"  ✗ Normal string: {e}")
    
    try:
        validator.validate_string("SELECT * FROM users; DROP TABLE users;", "test")
        print("  ✗ SQL injection: should have failed")
    except SecurityValidationError as e:
        print(f"  ✓ SQL injection: blocked ({e.threat_type.value})")
    
    # Test path validation
    print("\n2. Testing Path Validation:")
    try:
        validator.validate_path("./safe_file.txt", allow_create=True)
        print("  ✓ Safe path: passed")
    except SecurityValidationError as e:
        print(f"  ✗ Safe path: {e}")
    
    try:
        validator.validate_path("../../../etc/passwd", allow_create=True)
        print("  ✗ Path traversal: should have failed")
    except SecurityValidationError as e:
        print(f"  ✓ Path traversal: blocked ({e.threat_type.value})")
    
    # Test config validation
    print("\n3. Testing Config Validation:")
    safe_config = {'learning_rate': 0.001, 'batch_size': 32}
    try:
        validator.validate_config(safe_config, "test")
        print("  ✓ Safe config: passed")
    except SecurityValidationError as e:
        print(f"  ✗ Safe config: {e}")
    
    dangerous_config = {'__import__': 'os', 'command': 'rm -rf /'}
    try:
        validator.validate_config(dangerous_config, "test")
        print("  ✗ Dangerous config: should have failed")
    except SecurityValidationError as e:
        print(f"  ✓ Dangerous config: blocked ({e.threat_type.value})")
    
    # Test token validation
    print("\n4. Testing Token Validation:")
    token_validator = SecurityTokenValidator()
    
    payload = {'user_id': 123, 'role': 'user'}
    token = token_validator.generate_token(payload, expiry_hours=1)
    print(f"  Generated token: {token[:50]}...")
    
    try:
        decoded = token_validator.validate_token(token)
        print(f"  ✓ Token validation: passed, user_id={decoded['user_id']}")
    except SecurityValidationError as e:
        print(f"  ✗ Token validation: {e}")
    
    # Get security report
    print("\n5. Security Report:")
    report = validator.get_security_report()
    print(f"  Total validations: {report['total_validations']}")
    print(f"  Success rate: {report['success_rate']:.2%}")
    print(f"  Threats detected: {report['total_threats_detected']}")
    print(f"  Status: {report['status']}")
    
    print("\n✓ Security validation test completed!")