#!/usr/bin/env python3
"""
Security Test Suite for SNN-Fusion

Tests security aspects of the codebase including input validation,
file access controls, and potential vulnerabilities.
"""

import sys
import os
import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def scan_for_security_issues(file_path: Path) -> List[Dict[str, Any]]:
    """Scan a Python file for potential security issues."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return [{'type': 'syntax_error', 'file': str(file_path), 'severity': 'HIGH'}]
        
        # Dangerous function calls
        dangerous_functions = [
            'eval', 'exec', 'compile', '__import__',
            'open', 'input', 'raw_input'
        ]
        
        # SQL injection patterns
        sql_patterns = [
            r'SELECT.*\+.*',
            r'INSERT.*\+.*', 
            r'UPDATE.*\+.*',
            r'DELETE.*\+.*'
        ]
        
        # Command injection patterns
        cmd_patterns = [
            r'os\.system\(',
            r'subprocess\.',
            r'commands\.',
            r'shell=True'
        ]
        
        # Check for dangerous patterns in source
        for pattern in sql_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    'type': 'potential_sql_injection',
                    'file': str(file_path),
                    'pattern': pattern,
                    'severity': 'HIGH'
                })
        
        for pattern in cmd_patterns:
            if re.search(pattern, content):
                issues.append({
                    'type': 'command_injection_risk',
                    'file': str(file_path),
                    'pattern': pattern,
                    'severity': 'MEDIUM'
                })
        
        # AST-based checks
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in dangerous_functions:
                    issues.append({
                        'type': 'dangerous_function',
                        'file': str(file_path),
                        'function': node.func.id,
                        'line': getattr(node, 'lineno', 0),
                        'severity': 'HIGH' if node.func.id in ['eval', 'exec'] else 'MEDIUM'
                    })
            
            # Check for hardcoded secrets (simple patterns)
            if isinstance(node, ast.Str):
                value = node.s
                if len(value) > 20 and any(c in value for c in 'abcdef0123456789'):
                    # Potential API key or secret
                    if re.match(r'^[a-f0-9]{32,}$', value.lower()):
                        issues.append({
                            'type': 'potential_hardcoded_secret',
                            'file': str(file_path),
                            'line': getattr(node, 'lineno', 0),
                            'severity': 'HIGH'
                        })
    
    except Exception as e:
        issues.append({
            'type': 'scan_error',
            'file': str(file_path),
            'error': str(e),
            'severity': 'LOW'
        })
    
    return issues


def check_file_permissions() -> List[Dict[str, Any]]:
    """Check for insecure file permissions."""
    issues = []
    
    # Check key files for appropriate permissions
    key_files = [
        'setup.py',
        'pyproject.toml',
        'requirements.txt'
    ]
    
    for file_name in key_files:
        file_path = Path(file_name)
        if file_path.exists():
            # Check if file is world-writable (security risk)
            stat = file_path.stat()
            mode = stat.st_mode
            
            if mode & 0o002:  # World writable
                issues.append({
                    'type': 'world_writable_file',
                    'file': str(file_path),
                    'permissions': oct(mode),
                    'severity': 'MEDIUM'
                })
    
    return issues


def check_dependencies() -> List[Dict[str, Any]]:
    """Check for known vulnerable dependencies."""
    issues = []
    
    # This would normally check against a vulnerability database
    # For now, we'll check for obviously old or risky packages
    
    vulnerable_patterns = [
        r'pillow<8\.0\.0',  # Old Pillow versions had vulnerabilities
        r'requests<2\.20\.0',  # Old requests versions
        r'urllib3<1\.24\.0',   # Old urllib3 versions
    ]
    
    requirements_file = Path('requirements.txt')
    if requirements_file.exists():
        try:
            with open(requirements_file, 'r') as f:
                content = f.read()
            
            for pattern in vulnerable_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append({
                        'type': 'vulnerable_dependency',
                        'pattern': pattern,
                        'severity': 'HIGH'
                    })
        
        except Exception as e:
            issues.append({
                'type': 'dependency_check_error',
                'error': str(e),
                'severity': 'LOW'
            })
    
    return issues


def test_input_validation() -> List[Dict[str, Any]]:
    """Test input validation in key modules."""
    issues = []
    
    # Test validation module if it exists
    validation_file = Path('src/snn_fusion/utils/validation.py')
    if validation_file.exists():
        try:
            # Try to import and test validation functions
            sys.path.insert(0, 'src')
            
            # Test basic validation (this would be more comprehensive in practice)
            # For now, just verify the module can be imported
            import importlib.util
            spec = importlib.util.spec_from_file_location("validation", validation_file)
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                
                # Check if key functions exist
                required_functions = ['validate_tensor_shape', 'validate_modality_data', 'sanitize_input']
                for func_name in required_functions:
                    if not hasattr(validation_module, func_name):
                        issues.append({
                            'type': 'missing_validation_function',
                            'function': func_name,
                            'severity': 'MEDIUM'
                        })
        
        except Exception as e:
            issues.append({
                'type': 'validation_test_error',
                'error': str(e),
                'severity': 'LOW'
            })
    else:
        issues.append({
            'type': 'missing_validation_module',
            'severity': 'HIGH'
        })
    
    return issues


def run_security_scan() -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Run comprehensive security scan."""
    all_issues = []
    
    # Scan all Python files
    src_dir = Path('src')
    if src_dir.exists():
        for py_file in src_dir.rglob('*.py'):
            file_issues = scan_for_security_issues(py_file)
            all_issues.extend(file_issues)
    
    # Check file permissions
    perm_issues = check_file_permissions()
    all_issues.extend(perm_issues)
    
    # Check dependencies
    dep_issues = check_dependencies()
    all_issues.extend(dep_issues)
    
    # Test input validation
    validation_issues = test_input_validation()
    all_issues.extend(validation_issues)
    
    # Count issues by severity
    severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for issue in all_issues:
        severity = issue.get('severity', 'LOW')
        severity_counts[severity] += 1
    
    return all_issues, severity_counts


def main():
    """Run security tests and report results."""
    print("🔒 Starting SNN-Fusion Security Scan\n")
    
    issues, severity_counts = run_security_scan()
    
    # Group issues by type
    issues_by_type = {}
    for issue in issues:
        issue_type = issue['type']
        if issue_type not in issues_by_type:
            issues_by_type[issue_type] = []
        issues_by_type[issue_type].append(issue)
    
    # Report results
    print("🔍 Security Scan Results:")
    print("=" * 60)
    
    if not issues:
        print("✅ No security issues found!")
    else:
        for issue_type, type_issues in issues_by_type.items():
            print(f"\n📋 {issue_type.replace('_', ' ').title()}:")
            for issue in type_issues:
                severity_icon = {
                    'HIGH': '🔴',
                    'MEDIUM': '🟡', 
                    'LOW': '🟢'
                }.get(issue['severity'], '⚪')
                
                print(f"  {severity_icon} {issue['severity']}: {issue.get('file', 'N/A')}")
                if 'function' in issue:
                    print(f"    Function: {issue['function']}")
                if 'pattern' in issue:
                    print(f"    Pattern: {issue['pattern']}")
                if 'line' in issue:
                    print(f"    Line: {issue['line']}")
    
    print("\n📊 Security Summary:")
    print("=" * 60)
    total_issues = len(issues)
    print(f"Total Issues: {total_issues}")
    print(f"🔴 High Severity: {severity_counts['HIGH']}")
    print(f"🟡 Medium Severity: {severity_counts['MEDIUM']}")
    print(f"🟢 Low Severity: {severity_counts['LOW']}")
    
    # Security score
    if total_issues == 0:
        score = 100
    else:
        # Weight by severity
        weighted_issues = (severity_counts['HIGH'] * 3 + 
                          severity_counts['MEDIUM'] * 2 + 
                          severity_counts['LOW'] * 1)
        score = max(0, 100 - weighted_issues * 2)
    
    print(f"\n🏆 Security Score: {score}/100")
    
    if score >= 90:
        print("🎉 Excellent security posture!")
        return True
    elif score >= 70:
        print("✅ Good security posture with minor issues")
        return True
    else:
        print("⚠️  Security issues need attention")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)