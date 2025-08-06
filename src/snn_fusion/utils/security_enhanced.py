"""
Enhanced Security and Input Sanitization

This module provides comprehensive security measures, input validation,
and sanitization for the SNN-Fusion framework to prevent security vulnerabilities
and ensure data integrity.
"""

import re
import os
import hashlib
import hmac
import secrets
import logging
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum
import base64


class SecurityLevel(Enum):
    """Security validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ThreatType(Enum):
    """Types of security threats."""
    INJECTION = "injection"
    PATH_TRAVERSAL = "path_traversal"
    MALICIOUS_DATA = "malicious_data"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"


@dataclass
class SecurityViolation:
    """Security violation report."""
    timestamp: datetime
    violation_type: ThreatType
    severity: str
    message: str
    blocked: bool
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class InputSanitizer:
    """
    Comprehensive input sanitization and validation.
    
    Provides multiple layers of security validation to protect against
    various attack vectors and malicious inputs.
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        """
        Initialize input sanitizer.
        
        Args:
            security_level: Level of security validation to apply
        """
        self.security_level = security_level
        self.logger = logging.getLogger(__name__)
        
        # Compile regex patterns for performance
        self._compile_patterns()
        
        # Security violation tracking
        self.violations: List[SecurityViolation] = []
        self.max_violations_history = 1000
        
        self.logger.info(f"InputSanitizer initialized with {security_level.value} security level")
    
    def _compile_patterns(self):
        """Compile regex patterns for security checks."""
        # SQL injection patterns
        self.sql_injection_patterns = [
            re.compile(r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)", re.IGNORECASE),
            re.compile(r"(\b(or|and)\b\s*\d+\s*=\s*\d+)", re.IGNORECASE),
            re.compile(r"(--|#|/\*|\*/)", re.IGNORECASE),
            re.compile(r"(\'\s*(or|and)\s*\'\w*\'\s*=\s*\'\w*)", re.IGNORECASE),
        ]
        
        # XSS patterns
        self.xss_patterns = [
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),
            re.compile(r"<\s*iframe", re.IGNORECASE),
            re.compile(r"<\s*object", re.IGNORECASE),
            re.compile(r"<\s*embed", re.IGNORECASE),
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            re.compile(r"\.\.[/\\]"),
            re.compile(r"[/\\]\.\."),
            re.compile(r"\.(bat|cmd|exe|sh|ps1|vbs|js)$", re.IGNORECASE),
            re.compile(r"[<>:\"|?*]"),  # Windows forbidden characters
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            re.compile(r"[;&|`$]"),
            re.compile(r"\$\([^)]+\)"),  # Command substitution
            re.compile(r"`[^`]+`"),       # Backtick execution
            re.compile(r">\s*/dev/"),     # Redirect to devices
        ]
        
        # Malicious payload patterns
        self.malicious_patterns = [
            re.compile(r"\x00"),  # Null bytes
            re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]"),  # Control characters
            re.compile(r"(rm|del|format|fdisk|mkfs)\s", re.IGNORECASE),
            re.compile(r"(wget|curl|nc|netcat)\s", re.IGNORECASE),
        ]
    
    def sanitize_string(
        self,
        input_string: str,
        max_length: Optional[int] = None,
        allowed_chars: Optional[str] = None,
        strip_html: bool = True
    ) -> str:
        """
        Sanitize string input with comprehensive validation.
        
        Args:
            input_string: String to sanitize
            max_length: Maximum allowed length
            allowed_chars: Regex pattern of allowed characters
            strip_html: Whether to strip HTML tags
            
        Returns:
            Sanitized string
            
        Raises:
            ValueError: If input fails security validation
        """
        if not isinstance(input_string, str):
            raise ValueError("Input must be a string")
        
        # Check for null bytes
        if '\x00' in input_string:
            self._log_violation(ThreatType.MALICIOUS_DATA, "Null byte detected in input", True)
            raise ValueError("Null bytes not allowed in input")
        
        # Length validation
        if max_length and len(input_string) > max_length:
            self._log_violation(ThreatType.RESOURCE_EXHAUSTION, f"Input too long: {len(input_string)} > {max_length}", True)
            raise ValueError(f"Input exceeds maximum length of {max_length}")
        
        # Check for malicious patterns
        self._check_malicious_patterns(input_string)
        
        # Strip HTML if requested
        if strip_html:
            input_string = self._strip_html_tags(input_string)
        
        # Character whitelist validation
        if allowed_chars:
            pattern = re.compile(f"^[{allowed_chars}]*$")
            if not pattern.match(input_string):
                self._log_violation(ThreatType.MALICIOUS_DATA, "Invalid characters in input", True)
                raise ValueError("Input contains invalid characters")
        
        # Additional sanitization based on security level
        if self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            input_string = self._strict_sanitization(input_string)
        
        return input_string.strip()
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal attacks.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Safe filename
        """
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string")
        
        # Check for path traversal attempts
        for pattern in self.path_traversal_patterns:
            if pattern.search(filename):
                self._log_violation(ThreatType.PATH_TRAVERSAL, f"Path traversal attempt in filename: {filename}", True)
                raise ValueError("Invalid filename: path traversal attempt detected")
        
        # Remove/replace dangerous characters
        safe_filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        
        # Ensure filename is not empty or only dots
        if not safe_filename.strip('.'):
            safe_filename = 'unnamed_file'
        
        # Truncate if too long
        if len(safe_filename) > 255:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:250] + ext
        
        return safe_filename
    
    def validate_file_path(self, file_path: Union[str, Path], allowed_dirs: Optional[List[str]] = None) -> Path:
        """
        Validate and sanitize file path.
        
        Args:
            file_path: Path to validate
            allowed_dirs: List of allowed base directories
            
        Returns:
            Validated Path object
        """
        path = Path(str(file_path)).resolve()
        
        # Check for path traversal
        path_str = str(path)
        for pattern in self.path_traversal_patterns[:2]:  # Only check .. patterns for paths
            if pattern.search(path_str):
                self._log_violation(ThreatType.PATH_TRAVERSAL, f"Path traversal in file path: {path_str}", True)
                raise ValueError("Invalid file path: path traversal detected")
        
        # Check against allowed directories
        if allowed_dirs:
            allowed = False
            for allowed_dir in allowed_dirs:
                try:
                    path.relative_to(Path(allowed_dir).resolve())
                    allowed = True
                    break
                except ValueError:
                    continue
            
            if not allowed:
                self._log_violation(ThreatType.PATH_TRAVERSAL, f"Path outside allowed directories: {path_str}", True)
                raise ValueError("File path outside allowed directories")
        
        return path
    
    def sanitize_numeric_input(
        self,
        value: Union[str, int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        data_type: type = int
    ) -> Union[int, float]:
        """
        Sanitize and validate numeric input.
        
        Args:
            value: Numeric value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            data_type: Expected data type (int or float)
            
        Returns:
            Validated numeric value
        """
        try:
            if isinstance(value, str):
                # Check for injection attempts in string numbers
                if any(pattern.search(value) for pattern in self.command_injection_patterns):
                    self._log_violation(ThreatType.INJECTION, f"Injection attempt in numeric input: {value}", True)
                    raise ValueError("Invalid numeric input")
                
                # Convert to numeric
                numeric_value = data_type(value)
            else:
                numeric_value = data_type(value)
            
            # Range validation
            if min_value is not None and numeric_value < min_value:
                raise ValueError(f"Value {numeric_value} below minimum {min_value}")
            
            if max_value is not None and numeric_value > max_value:
                # Could be a DoS attempt
                self._log_violation(ThreatType.RESOURCE_EXHAUSTION, f"Value {numeric_value} exceeds maximum {max_value}", True)
                raise ValueError(f"Value {numeric_value} exceeds maximum {max_value}")
            
            return numeric_value
            
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric input: {e}")
    
    def validate_json_input(self, json_data: Union[str, dict], max_size: int = 1024*1024) -> dict:
        """
        Validate and sanitize JSON input.
        
        Args:
            json_data: JSON data to validate
            max_size: Maximum size in bytes
            
        Returns:
            Validated dictionary
        """
        if isinstance(json_data, str):
            # Size check
            if len(json_data.encode('utf-8')) > max_size:
                self._log_violation(ThreatType.RESOURCE_EXHAUSTION, f"JSON too large: {len(json_data)} bytes", True)
                raise ValueError("JSON payload too large")
            
            # Parse JSON
            try:
                parsed_data = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}")
        else:
            parsed_data = json_data
        
        # Validate dictionary
        if not isinstance(parsed_data, dict):
            raise ValueError("JSON must be an object")
        
        # Check for deeply nested structures (DoS protection)
        self._check_json_depth(parsed_data, max_depth=20)
        
        # Sanitize string values in JSON
        return self._sanitize_json_values(parsed_data)
    
    def _check_malicious_patterns(self, input_string: str):
        """Check input against malicious patterns."""
        # Check SQL injection
        for pattern in self.sql_injection_patterns:
            if pattern.search(input_string):
                self._log_violation(ThreatType.INJECTION, "SQL injection pattern detected", True)
                raise ValueError("Potentially malicious input detected")
        
        # Check XSS
        for pattern in self.xss_patterns:
            if pattern.search(input_string):
                self._log_violation(ThreatType.INJECTION, "XSS pattern detected", True)
                raise ValueError("Potentially malicious input detected")
        
        # Check command injection
        for pattern in self.command_injection_patterns:
            if pattern.search(input_string):
                self._log_violation(ThreatType.INJECTION, "Command injection pattern detected", True)
                raise ValueError("Potentially malicious input detected")
        
        # Check malicious payloads
        for pattern in self.malicious_patterns:
            if pattern.search(input_string):
                self._log_violation(ThreatType.MALICIOUS_DATA, "Malicious payload detected", True)
                raise ValueError("Potentially malicious input detected")
    
    def _strip_html_tags(self, text: str) -> str:
        """Strip HTML tags from text."""
        # Simple HTML tag removal
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        html_entities = {
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#x27;': "'",
            '&#x2F;': '/'
        }
        
        for entity, char in html_entities.items():
            clean_text = clean_text.replace(entity, char)
        
        return clean_text
    
    def _strict_sanitization(self, input_string: str) -> str:
        """Apply strict sanitization rules."""
        # Remove control characters
        sanitized = ''.join(char for char in input_string if ord(char) >= 32 or char in '\t\n\r')
        
        # Limit special characters
        if self.security_level == SecurityLevel.PARANOID:
            # Only allow alphanumeric, spaces, and basic punctuation
            sanitized = re.sub(r'[^a-zA-Z0-9\s\.,!?()-]', '', sanitized)
        
        return sanitized
    
    def _check_json_depth(self, obj: Any, current_depth: int = 0, max_depth: int = 20):
        """Check JSON nesting depth to prevent DoS attacks."""
        if current_depth > max_depth:
            self._log_violation(ThreatType.RESOURCE_EXHAUSTION, f"JSON too deeply nested: {current_depth}", True)
            raise ValueError("JSON structure too deeply nested")
        
        if isinstance(obj, dict):
            for value in obj.values():
                self._check_json_depth(value, current_depth + 1, max_depth)
        elif isinstance(obj, list):
            for item in obj:
                self._check_json_depth(item, current_depth + 1, max_depth)
    
    def _sanitize_json_values(self, obj: Any) -> Any:
        """Recursively sanitize values in JSON structure."""
        if isinstance(obj, dict):
            return {key: self._sanitize_json_values(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_json_values(item) for item in obj]
        elif isinstance(obj, str):
            try:
                return self.sanitize_string(obj, max_length=10000, strip_html=True)
            except ValueError:
                return "[SANITIZED]"
        else:
            return obj
    
    def _log_violation(self, threat_type: ThreatType, message: str, blocked: bool):
        """Log security violation."""
        violation = SecurityViolation(
            timestamp=datetime.now(),
            violation_type=threat_type,
            severity="HIGH",
            message=message,
            blocked=blocked
        )
        
        self.violations.append(violation)
        
        # Trim violations history
        if len(self.violations) > self.max_violations_history:
            self.violations = self.violations[-self.max_violations_history:]
        
        # Log the violation
        log_level = logging.ERROR if blocked else logging.WARNING
        self.logger.log(log_level, f"Security violation: {threat_type.value} - {message}")
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of security violations."""
        if not self.violations:
            return {"total_violations": 0, "message": "No security violations recorded"}
        
        # Count by type
        type_counts = {}
        for violation in self.violations:
            violation_type = violation.violation_type.value
            type_counts[violation_type] = type_counts.get(violation_type, 0) + 1
        
        # Recent violations (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_violations = [v for v in self.violations if v.timestamp > recent_cutoff]
        
        return {
            "total_violations": len(self.violations),
            "recent_violations_1h": len(recent_violations),
            "violations_by_type": type_counts,
            "blocked_violations": len([v for v in self.violations if v.blocked]),
            "last_violation": self.violations[-1].timestamp.isoformat() if self.violations else None
        }


class SecureConfig:
    """Secure configuration management with encryption and validation."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initialize secure config manager.
        
        Args:
            encryption_key: Key for encrypting sensitive configuration values
        """
        self.encryption_key = encryption_key or self._generate_key()
        self.logger = logging.getLogger(__name__)
        self.sanitizer = InputSanitizer(SecurityLevel.STRICT)
    
    def _generate_key(self) -> bytes:
        """Generate a new encryption key."""
        return secrets.token_bytes(32)
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a configuration value."""
        try:
            # Simple encryption using Fernet-like approach
            from cryptography.fernet import Fernet
            f = Fernet(base64.urlsafe_b64encode(self.encryption_key[:32]))
            encrypted = f.encrypt(value.encode('utf-8'))
            return base64.urlsafe_b64encode(encrypted).decode('utf-8')
        except ImportError:
            # Fallback: Base64 encoding (not secure, just obfuscation)
            self.logger.warning("cryptography library not available, using base64 obfuscation")
            return base64.urlsafe_b64encode(value.encode('utf-8')).decode('utf-8')
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a configuration value."""
        try:
            from cryptography.fernet import Fernet
            f = Fernet(base64.urlsafe_b64encode(self.encryption_key[:32]))
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode('utf-8'))
            decrypted = f.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except ImportError:
            # Fallback: Base64 decoding
            return base64.urlsafe_b64decode(encrypted_value.encode('utf-8')).decode('utf-8')
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration dictionary."""
        validated_config = {}
        
        for key, value in config.items():
            # Validate key
            safe_key = self.sanitizer.sanitize_string(key, max_length=100, allowed_chars=r'a-zA-Z0-9_-')
            
            # Validate value based on type
            if isinstance(value, str):
                if key.lower() in ['password', 'secret', 'key', 'token']:
                    # Sensitive values - encrypt
                    validated_config[safe_key] = self.encrypt_value(value)
                else:
                    # Regular string values - sanitize
                    validated_config[safe_key] = self.sanitizer.sanitize_string(value, max_length=10000)
            elif isinstance(value, (int, float)):
                validated_config[safe_key] = self.sanitizer.sanitize_numeric_input(value, -1e9, 1e9, type(value))
            elif isinstance(value, dict):
                validated_config[safe_key] = self.validate_config(value)
            elif isinstance(value, list):
                validated_config[safe_key] = [self.validate_config({'item': item})['item'] if isinstance(item, dict) else item for item in value[:100]]  # Limit list size
            else:
                validated_config[safe_key] = value
        
        return validated_config


# Convenience functions and decorators

def secure_input(security_level: SecurityLevel = SecurityLevel.STANDARD):
    """Decorator for automatic input sanitization."""
    def decorator(func):
        sanitizer = InputSanitizer(security_level)
        
        def wrapper(*args, **kwargs):
            # Sanitize string arguments
            sanitized_args = []
            for arg in args:
                if isinstance(arg, str):
                    try:
                        sanitized_args.append(sanitizer.sanitize_string(arg))
                    except ValueError:
                        sanitized_args.append("[SANITIZED]")
                else:
                    sanitized_args.append(arg)
            
            # Sanitize string keyword arguments
            sanitized_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, str):
                    try:
                        sanitized_kwargs[key] = sanitizer.sanitize_string(value)
                    except ValueError:
                        sanitized_kwargs[key] = "[SANITIZED]"
                else:
                    sanitized_kwargs[key] = value
            
            return func(*sanitized_args, **sanitized_kwargs)
        
        return wrapper
    return decorator


def create_security_report(sanitizer: InputSanitizer) -> Dict[str, Any]:
    """Create comprehensive security report."""
    violation_summary = sanitizer.get_violation_summary()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "security_level": sanitizer.security_level.value,
        "violation_summary": violation_summary,
        "recommendations": _get_security_recommendations(violation_summary)
    }


def _get_security_recommendations(violation_summary: Dict[str, Any]) -> List[str]:
    """Generate security recommendations based on violations."""
    recommendations = []
    
    if violation_summary["total_violations"] > 100:
        recommendations.append("Consider implementing rate limiting")
    
    if "injection" in violation_summary.get("violations_by_type", {}):
        recommendations.append("Review input validation for injection vulnerabilities")
    
    if "path_traversal" in violation_summary.get("violations_by_type", {}):
        recommendations.append("Implement stricter file path validation")
    
    if violation_summary.get("recent_violations_1h", 0) > 10:
        recommendations.append("Investigate recent spike in security violations")
    
    return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Test input sanitization
    print("Testing Enhanced Security...")
    
    # Create sanitizer
    sanitizer = InputSanitizer(SecurityLevel.STRICT)
    
    # Test cases
    test_inputs = [
        "Hello World",  # Safe input
        "<script>alert('xss')</script>",  # XSS attempt
        "../../../etc/passwd",  # Path traversal
        "'; DROP TABLE users; --",  # SQL injection
        "rm -rf /",  # Command injection
        "Hello\x00World",  # Null byte
    ]
    
    print("Testing string sanitization:")
    for i, test_input in enumerate(test_inputs):
        try:
            result = sanitizer.sanitize_string(test_input)
            print(f"  Test {i+1}: '{test_input[:30]}...' -> OK")
        except ValueError as e:
            print(f"  Test {i+1}: '{test_input[:30]}...' -> BLOCKED ({e})")
    
    # Test filename sanitization
    print("\nTesting filename sanitization:")
    test_filenames = [
        "document.pdf",
        "../../../secret.txt",
        "file<>with*bad?chars.txt",
        "normal_file-2023.doc"
    ]
    
    for filename in test_filenames:
        try:
            safe_filename = sanitizer.sanitize_filename(filename)
            print(f"  '{filename}' -> '{safe_filename}'")
        except ValueError as e:
            print(f"  '{filename}' -> BLOCKED ({e})")
    
    # Test JSON validation
    print("\nTesting JSON validation:")
    test_json = {
        "name": "Test User",
        "data": "<script>alert('xss')</script>",
        "nested": {"level": 1}
    }
    
    try:
        sanitized_json = sanitizer.validate_json_input(test_json)
        print(f"  JSON sanitized successfully")
        print(f"  Before: {test_json}")
        print(f"  After: {sanitized_json}")
    except ValueError as e:
        print(f"  JSON validation failed: {e}")
    
    # Test secure config
    print("\nTesting secure configuration:")
    config = SecureConfig()
    
    test_config = {
        "api_key": "secret-api-key-12345",
        "database_url": "postgresql://user:pass@localhost/db",
        "debug": True,
        "max_connections": 100
    }
    
    validated_config = config.validate_config(test_config)
    print("  Configuration validated and sensitive values encrypted")
    
    # Get security report
    report = create_security_report(sanitizer)
    print(f"\nSecurity Report:")
    print(f"  Total violations: {report['violation_summary']['total_violations']}")
    print(f"  Security level: {report['security_level']}")
    
    if report['recommendations']:
        print("  Recommendations:")
        for rec in report['recommendations']:
            print(f"    - {rec}")
    
    print("✓ Enhanced security test completed!")