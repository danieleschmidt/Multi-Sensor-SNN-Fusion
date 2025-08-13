"""
Security Scanning and Compliance Framework

Implements comprehensive security scanning, vulnerability detection,
and compliance checking for neuromorphic systems.
"""

import ast
import re
import hashlib
import json
import sys
import os
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import warnings
import subprocess
import time


class SecuritySeverity(Enum):
    """Security issue severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStandard(Enum):
    """Compliance standards to check against."""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    NIST = "nist"


@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning."""
    issue_id: str
    severity: SecuritySeverity
    category: str
    description: str
    file_path: str
    line_number: int = 0
    column_number: int = 0
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'issue_id': self.issue_id,
            'severity': self.severity.value,
            'category': self.category,
            'description': self.description,
            'file_path': str(self.file_path),
            'line_number': self.line_number,
            'column_number': self.column_number,
            'cwe_id': self.cwe_id,
            'recommendation': self.recommendation,
            'metadata': self.metadata
        }


@dataclass
class ComplianceIssue:
    """Represents a compliance issue."""
    standard: ComplianceStandard
    requirement: str
    description: str
    severity: SecuritySeverity
    file_path: str
    recommendation: str = ""


class StaticAnalysisScanner:
    """
    Static analysis security scanner for Python code.
    """
    
    def __init__(self):
        self.issues: List[SecurityIssue] = []
        self.scanned_files: Set[str] = set()
        
        # Define security patterns
        self.security_patterns = {
            'hardcoded_secrets': [
                (r'password\s*=\s*["\'][^"\']{8,}["\']', 'Hardcoded password detected'),
                (r'api_key\s*=\s*["\'][^"\']{20,}["\']', 'Hardcoded API key detected'),
                (r'secret\s*=\s*["\'][^"\']{16,}["\']', 'Hardcoded secret detected'),
                (r'token\s*=\s*["\'][^"\']{20,}["\']', 'Hardcoded token detected'),
            ],
            'sql_injection': [
                (r'execute\s*\(\s*["\'].*%.*["\']', 'Potential SQL injection vulnerability'),
                (r'query\s*\(\s*["\'].*\+.*["\']', 'Potential SQL injection vulnerability'),
                (r'cursor\.execute\s*\([^)]*%', 'Potential SQL injection vulnerability'),
            ],
            'command_injection': [
                (r'os\.system\s*\(', 'Command injection risk with os.system'),
                (r'subprocess\.(call|run|Popen)\s*\(.*shell\s*=\s*True', 'Command injection risk with shell=True'),
                (r'eval\s*\(', 'Code injection risk with eval()'),
                (r'exec\s*\(', 'Code injection risk with exec()'),
            ],
            'path_traversal': [
                (r'open\s*\(\s*.*\+.*["\']', 'Potential path traversal vulnerability'),
                (r'file\s*=.*\+.*["\']', 'Potential path traversal vulnerability'),
            ],
            'crypto_issues': [
                (r'md5\s*\(', 'Weak cryptographic hash MD5 used'),
                (r'sha1\s*\(', 'Weak cryptographic hash SHA1 used'),
                (r'random\.random\s*\(', 'Cryptographically weak random number generator'),
            ],
            'debug_code': [
                (r'print\s*\(.*password', 'Password potentially logged in debug code'),
                (r'print\s*\(.*secret', 'Secret potentially logged in debug code'),
                (r'DEBUG\s*=\s*True', 'Debug mode enabled in production code'),
            ]
        }
        
        # Dangerous functions and their recommendations
        self.dangerous_functions = {
            'eval': 'Use ast.literal_eval() for safe evaluation of literals',
            'exec': 'Avoid dynamic code execution; use safer alternatives',
            'pickle.loads': 'Use json or safer serialization methods',
            'yaml.load': 'Use yaml.safe_load() instead',
            '__import__': 'Use importlib.import_module() with validation'
        }
    
    def scan_file(self, file_path: Path) -> List[SecurityIssue]:
        """Scan a single Python file for security issues."""
        if str(file_path) in self.scanned_files:
            return []
        
        self.scanned_files.add(str(file_path))
        file_issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Pattern-based scanning
            file_issues.extend(self._scan_patterns(file_path, content))
            
            # AST-based scanning
            try:
                tree = ast.parse(content)
                file_issues.extend(self._scan_ast(file_path, tree, content))
            except SyntaxError as e:
                file_issues.append(SecurityIssue(
                    issue_id=self._generate_issue_id(file_path, 'syntax_error', e.lineno or 0),
                    severity=SecuritySeverity.LOW,
                    category='syntax',
                    description=f'Syntax error: {e.msg}',
                    file_path=str(file_path),
                    line_number=e.lineno or 0,
                    recommendation='Fix syntax error'
                ))
            
        except Exception as e:
            logging.error(f"Error scanning file {file_path}: {e}")
        
        self.issues.extend(file_issues)
        return file_issues
    
    def scan_directory(self, directory: Path, recursive: bool = True) -> List[SecurityIssue]:
        """Scan a directory for security issues."""
        all_issues = []
        
        if recursive:
            python_files = directory.rglob('*.py')
        else:
            python_files = directory.glob('*.py')
        
        for file_path in python_files:
            if file_path.is_file():
                issues = self.scan_file(file_path)
                all_issues.extend(issues)
        
        return all_issues
    
    def _scan_patterns(self, file_path: Path, content: str) -> List[SecurityIssue]:
        """Scan content using regex patterns."""
        issues = []
        lines = content.split('\n')
        
        for category, patterns in self.security_patterns.items():
            for pattern, description in patterns:
                regex = re.compile(pattern, re.IGNORECASE)
                
                for line_num, line in enumerate(lines, 1):
                    matches = regex.finditer(line)
                    for match in matches:
                        severity = self._determine_severity(category)
                        
                        issue = SecurityIssue(
                            issue_id=self._generate_issue_id(file_path, category, line_num),
                            severity=severity,
                            category=category,
                            description=description,
                            file_path=str(file_path),
                            line_number=line_num,
                            column_number=match.start(),
                            cwe_id=self._get_cwe_id(category),
                            recommendation=self._get_recommendation(category),
                            metadata={'pattern': pattern, 'matched_text': match.group()}
                        )
                        issues.append(issue)
        
        return issues
    
    def _scan_ast(self, file_path: Path, tree: ast.AST, content: str) -> List[SecurityIssue]:
        """Scan AST for security issues."""
        issues = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, scanner):
                self.scanner = scanner
                self.issues = []
            
            def visit_Call(self, node):
                # Check for dangerous function calls
                func_name = self._get_function_name(node)
                
                if func_name in self.scanner.dangerous_functions:
                    issue = SecurityIssue(
                        issue_id=self.scanner._generate_issue_id(
                            file_path, 'dangerous_function', node.lineno
                        ),
                        severity=SecuritySeverity.HIGH,
                        category='dangerous_function',
                        description=f'Dangerous function call: {func_name}()',
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column_number=getattr(node, 'col_offset', 0),
                        cwe_id='CWE-676',
                        recommendation=self.scanner.dangerous_functions[func_name],
                        metadata={'function_name': func_name}
                    )
                    self.issues.append(issue)
                
                # Check for subprocess calls with shell=True
                if func_name in ['subprocess.call', 'subprocess.run', 'subprocess.Popen']:
                    for keyword in node.keywords:
                        if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                            if keyword.value.value is True:
                                issue = SecurityIssue(
                                    issue_id=self.scanner._generate_issue_id(
                                        file_path, 'shell_injection', node.lineno
                                    ),
                                    severity=SecuritySeverity.HIGH,
                                    category='command_injection',
                                    description='subprocess call with shell=True is dangerous',
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    column_number=getattr(node, 'col_offset', 0),
                                    cwe_id='CWE-78',
                                    recommendation='Use shell=False and pass arguments as list'
                                )
                                self.issues.append(issue)
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for imports of dangerous modules
                dangerous_imports = {
                    'pickle': 'Consider using json or safer serialization',
                    'marshal': 'Consider using json or safer serialization', 
                    'subprocess': 'Be careful with subprocess, avoid shell=True'
                }
                
                for alias in node.names:
                    if alias.name in dangerous_imports:
                        issue = SecurityIssue(
                            issue_id=self.scanner._generate_issue_id(
                                file_path, 'dangerous_import', node.lineno
                            ),
                            severity=SecuritySeverity.LOW,
                            category='dangerous_import',
                            description=f'Potentially dangerous import: {alias.name}',
                            file_path=str(file_path),
                            line_number=node.lineno,
                            column_number=getattr(node, 'col_offset', 0),
                            recommendation=dangerous_imports[alias.name]
                        )
                        self.issues.append(issue)
                
                self.generic_visit(node)
            
            def _get_function_name(self, node):
                """Extract function name from Call node."""
                if isinstance(node.func, ast.Name):
                    return node.func.id
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        return f"{node.func.value.id}.{node.func.attr}"
                    else:
                        return node.func.attr
                return "unknown"
        
        visitor = SecurityVisitor(self)
        visitor.visit(tree)
        issues.extend(visitor.issues)
        
        return issues
    
    def _generate_issue_id(self, file_path: Path, category: str, line_number: int) -> str:
        """Generate unique issue ID."""
        content = f"{file_path}:{category}:{line_number}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _determine_severity(self, category: str) -> SecuritySeverity:
        """Determine severity based on category."""
        severity_map = {
            'hardcoded_secrets': SecuritySeverity.HIGH,
            'sql_injection': SecuritySeverity.CRITICAL,
            'command_injection': SecuritySeverity.CRITICAL,
            'path_traversal': SecuritySeverity.HIGH,
            'crypto_issues': SecuritySeverity.MEDIUM,
            'debug_code': SecuritySeverity.LOW,
            'dangerous_function': SecuritySeverity.HIGH,
            'dangerous_import': SecuritySeverity.LOW
        }
        return severity_map.get(category, SecuritySeverity.MEDIUM)
    
    def _get_cwe_id(self, category: str) -> Optional[str]:
        """Get CWE ID for category."""
        cwe_map = {
            'hardcoded_secrets': 'CWE-798',
            'sql_injection': 'CWE-89',
            'command_injection': 'CWE-78',
            'path_traversal': 'CWE-22',
            'crypto_issues': 'CWE-327',
            'dangerous_function': 'CWE-676'
        }
        return cwe_map.get(category)
    
    def _get_recommendation(self, category: str) -> str:
        """Get recommendation for category."""
        recommendations = {
            'hardcoded_secrets': 'Use environment variables or secure secret management',
            'sql_injection': 'Use parameterized queries or ORM',
            'command_injection': 'Validate input and use safe APIs',
            'path_traversal': 'Validate file paths and use safe path operations',
            'crypto_issues': 'Use strong cryptographic algorithms (SHA-256+)',
            'debug_code': 'Remove debug code from production builds'
        }
        return recommendations.get(category, 'Review and fix security issue')
    
    def get_summary(self) -> Dict[str, Any]:
        """Get scan summary."""
        severity_counts = {}
        category_counts = {}
        
        for issue in self.issues:
            severity = issue.severity.value
            category = issue.category
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total_issues': len(self.issues),
            'files_scanned': len(self.scanned_files),
            'severity_breakdown': severity_counts,
            'category_breakdown': category_counts,
            'critical_issues': severity_counts.get('critical', 0),
            'high_issues': severity_counts.get('high', 0)
        }


class ComplianceChecker:
    """
    Checks code and configuration for compliance with standards.
    """
    
    def __init__(self):
        self.compliance_rules = {
            ComplianceStandard.GDPR: self._gdpr_rules(),
            ComplianceStandard.CCPA: self._ccpa_rules(),
            ComplianceStandard.HIPAA: self._hipaa_rules(),
            ComplianceStandard.SOC2: self._soc2_rules(),
        }
        
        self.issues: List[ComplianceIssue] = []
    
    def check_compliance(self, directory: Path, standards: List[ComplianceStandard]) -> List[ComplianceIssue]:
        """Check compliance against specified standards."""
        all_issues = []
        
        for standard in standards:
            if standard in self.compliance_rules:
                rules = self.compliance_rules[standard]
                issues = self._apply_rules(directory, standard, rules)
                all_issues.extend(issues)
        
        self.issues.extend(all_issues)
        return all_issues
    
    def _gdpr_rules(self) -> List[Dict[str, Any]]:
        """Define GDPR compliance rules."""
        return [
            {
                'name': 'data_encryption',
                'description': 'Personal data must be encrypted at rest and in transit',
                'check_function': self._check_encryption_usage,
                'severity': SecuritySeverity.HIGH
            },
            {
                'name': 'data_retention',
                'description': 'Data retention policies must be documented',
                'check_function': self._check_data_retention,
                'severity': SecuritySeverity.MEDIUM
            },
            {
                'name': 'consent_tracking',
                'description': 'User consent must be tracked and manageable',
                'check_function': self._check_consent_mechanism,
                'severity': SecuritySeverity.HIGH
            },
            {
                'name': 'privacy_by_design',
                'description': 'Privacy controls must be built into the system',
                'check_function': self._check_privacy_controls,
                'severity': SecuritySeverity.MEDIUM
            }
        ]
    
    def _ccpa_rules(self) -> List[Dict[str, Any]]:
        """Define CCPA compliance rules."""
        return [
            {
                'name': 'data_deletion',
                'description': 'Users must be able to request data deletion',
                'check_function': self._check_deletion_capability,
                'severity': SecuritySeverity.HIGH
            },
            {
                'name': 'data_portability',
                'description': 'Users must be able to export their data',
                'check_function': self._check_data_export,
                'severity': SecuritySeverity.MEDIUM
            }
        ]
    
    def _hipaa_rules(self) -> List[Dict[str, Any]]:
        """Define HIPAA compliance rules."""
        return [
            {
                'name': 'phi_encryption',
                'description': 'Protected Health Information must be encrypted',
                'check_function': self._check_phi_encryption,
                'severity': SecuritySeverity.CRITICAL
            },
            {
                'name': 'access_logging',
                'description': 'All PHI access must be logged',
                'check_function': self._check_access_logging,
                'severity': SecuritySeverity.HIGH
            }
        ]
    
    def _soc2_rules(self) -> List[Dict[str, Any]]:
        """Define SOC2 compliance rules."""
        return [
            {
                'name': 'access_controls',
                'description': 'Proper access controls must be implemented',
                'check_function': self._check_access_controls,
                'severity': SecuritySeverity.HIGH
            },
            {
                'name': 'monitoring',
                'description': 'System monitoring and alerting must be in place',
                'check_function': self._check_monitoring,
                'severity': SecuritySeverity.MEDIUM
            }
        ]
    
    def _apply_rules(self, directory: Path, standard: ComplianceStandard, 
                    rules: List[Dict[str, Any]]) -> List[ComplianceIssue]:
        """Apply compliance rules to directory."""
        issues = []
        
        for rule in rules:
            try:
                rule_issues = rule['check_function'](directory, rule)
                issues.extend(rule_issues)
            except Exception as e:
                logging.error(f"Error applying rule {rule['name']}: {e}")
        
        return issues
    
    def _check_encryption_usage(self, directory: Path, rule: Dict) -> List[ComplianceIssue]:
        """Check for encryption usage."""
        issues = []
        
        # Look for encryption libraries and usage
        encryption_patterns = [
            r'from\s+cryptography',
            r'import\s+cryptography',
            r'AES\.',
            r'encrypt\(',
            r'decrypt\('
        ]
        
        has_encryption = False
        
        for py_file in directory.rglob('*.py'):
            try:
                content = py_file.read_text()
                for pattern in encryption_patterns:
                    if re.search(pattern, content):
                        has_encryption = True
                        break
                
                if has_encryption:
                    break
            except Exception:
                continue
        
        if not has_encryption:
            issues.append(ComplianceIssue(
                standard=ComplianceStandard.GDPR,
                requirement=rule['name'],
                description='No encryption usage found in codebase',
                severity=rule['severity'],
                file_path=str(directory),
                recommendation='Implement encryption for sensitive data'
            ))
        
        return issues
    
    def _check_data_retention(self, directory: Path, rule: Dict) -> List[ComplianceIssue]:
        """Check for data retention policies."""
        issues = []
        
        # Look for retention policy documentation or code
        retention_files = list(directory.rglob('*retention*')) + list(directory.rglob('*policy*'))
        
        if not retention_files:
            issues.append(ComplianceIssue(
                standard=ComplianceStandard.GDPR,
                requirement=rule['name'],
                description='No data retention policy documentation found',
                severity=rule['severity'],
                file_path=str(directory),
                recommendation='Create and document data retention policies'
            ))
        
        return issues
    
    def _check_consent_mechanism(self, directory: Path, rule: Dict) -> List[ComplianceIssue]:
        """Check for consent tracking mechanism."""
        # This would check for consent management code
        return []  # Placeholder
    
    def _check_privacy_controls(self, directory: Path, rule: Dict) -> List[ComplianceIssue]:
        """Check for privacy by design implementation."""
        # This would check for privacy controls
        return []  # Placeholder
    
    def _check_deletion_capability(self, directory: Path, rule: Dict) -> List[ComplianceIssue]:
        """Check for data deletion capability."""
        # This would check for deletion APIs
        return []  # Placeholder
    
    def _check_data_export(self, directory: Path, rule: Dict) -> List[ComplianceIssue]:
        """Check for data export capability."""
        # This would check for export APIs
        return []  # Placeholder
    
    def _check_phi_encryption(self, directory: Path, rule: Dict) -> List[ComplianceIssue]:
        """Check for PHI encryption."""
        # This would check for health data encryption
        return []  # Placeholder
    
    def _check_access_logging(self, directory: Path, rule: Dict) -> List[ComplianceIssue]:
        """Check for access logging."""
        # This would check for comprehensive logging
        return []  # Placeholder
    
    def _check_access_controls(self, directory: Path, rule: Dict) -> List[ComplianceIssue]:
        """Check for access controls."""
        # This would check for authentication/authorization
        return []  # Placeholder
    
    def _check_monitoring(self, directory: Path, rule: Dict) -> List[ComplianceIssue]:
        """Check for system monitoring."""
        # This would check for monitoring implementation
        return []  # Placeholder


class SecurityReportGenerator:
    """
    Generates comprehensive security reports.
    """
    
    def __init__(self):
        pass
    
    def generate_report(self, security_issues: List[SecurityIssue], 
                       compliance_issues: List[ComplianceIssue],
                       output_format: str = 'json') -> str:
        """Generate comprehensive security report."""
        
        # Aggregate data
        report_data = {
            'timestamp': time.time(),
            'summary': self._generate_summary(security_issues, compliance_issues),
            'security_issues': [issue.to_dict() for issue in security_issues],
            'compliance_issues': [self._compliance_issue_to_dict(issue) for issue in compliance_issues],
            'recommendations': self._generate_recommendations(security_issues, compliance_issues),
            'risk_score': self._calculate_risk_score(security_issues, compliance_issues)
        }
        
        if output_format.lower() == 'json':
            return json.dumps(report_data, indent=2, default=str)
        elif output_format.lower() == 'html':
            return self._generate_html_report(report_data)
        else:
            return str(report_data)
    
    def _generate_summary(self, security_issues: List[SecurityIssue], 
                         compliance_issues: List[ComplianceIssue]) -> Dict[str, Any]:
        """Generate report summary."""
        security_severity_counts = {}
        for issue in security_issues:
            severity = issue.severity.value
            security_severity_counts[severity] = security_severity_counts.get(severity, 0) + 1
        
        compliance_severity_counts = {}
        for issue in compliance_issues:
            severity = issue.severity.value
            compliance_severity_counts[severity] = compliance_severity_counts.get(severity, 0) + 1
        
        return {
            'total_security_issues': len(security_issues),
            'total_compliance_issues': len(compliance_issues),
            'security_severity_breakdown': security_severity_counts,
            'compliance_severity_breakdown': compliance_severity_counts,
            'critical_security_issues': security_severity_counts.get('critical', 0),
            'high_security_issues': security_severity_counts.get('high', 0)
        }
    
    def _compliance_issue_to_dict(self, issue: ComplianceIssue) -> Dict[str, Any]:
        """Convert compliance issue to dictionary."""
        return {
            'standard': issue.standard.value,
            'requirement': issue.requirement,
            'description': issue.description,
            'severity': issue.severity.value,
            'file_path': issue.file_path,
            'recommendation': issue.recommendation
        }
    
    def _generate_recommendations(self, security_issues: List[SecurityIssue], 
                                compliance_issues: List[ComplianceIssue]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Security recommendations
        critical_security = [i for i in security_issues if i.severity == SecuritySeverity.CRITICAL]
        if critical_security:
            recommendations.append(f"Address {len(critical_security)} critical security issues immediately")
        
        high_security = [i for i in security_issues if i.severity == SecuritySeverity.HIGH]
        if high_security:
            recommendations.append(f"Review and fix {len(high_security)} high-severity security issues")
        
        # Compliance recommendations  
        critical_compliance = [i for i in compliance_issues if i.severity == SecuritySeverity.CRITICAL]
        if critical_compliance:
            recommendations.append(f"Address {len(critical_compliance)} critical compliance issues")
        
        return recommendations
    
    def _calculate_risk_score(self, security_issues: List[SecurityIssue], 
                            compliance_issues: List[ComplianceIssue]) -> float:
        """Calculate overall risk score (0-100)."""
        score = 0.0
        
        # Security issues scoring
        for issue in security_issues:
            if issue.severity == SecuritySeverity.CRITICAL:
                score += 20
            elif issue.severity == SecuritySeverity.HIGH:
                score += 10
            elif issue.severity == SecuritySeverity.MEDIUM:
                score += 5
            elif issue.severity == SecuritySeverity.LOW:
                score += 1
        
        # Compliance issues scoring
        for issue in compliance_issues:
            if issue.severity == SecuritySeverity.CRITICAL:
                score += 15
            elif issue.severity == SecuritySeverity.HIGH:
                score += 8
            elif issue.severity == SecuritySeverity.MEDIUM:
                score += 3
        
        return min(100.0, score)  # Cap at 100
    
    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html = f"""
        <html>
        <head><title>Security Scan Report</title></head>
        <body>
        <h1>Security Scan Report</h1>
        <h2>Summary</h2>
        <p>Risk Score: {data['risk_score']:.1f}/100</p>
        <p>Total Security Issues: {data['summary']['total_security_issues']}</p>
        <p>Total Compliance Issues: {data['summary']['total_compliance_issues']}</p>
        
        <h2>Recommendations</h2>
        <ul>
        {"".join(f"<li>{rec}</li>" for rec in data['recommendations'])}
        </ul>
        </body>
        </html>
        """
        return html


# Example usage and testing
if __name__ == "__main__":
    print("Testing Security Scanner...")
    
    # Test static analysis scanner
    print("\n1. Testing Static Analysis Scanner:")
    scanner = StaticAnalysisScanner()
    
    # Create a test file with security issues
    test_code = '''
import os
import subprocess

# Hardcoded credentials
password = "supersecret123"
api_key = "sk_test_abcdef1234567890"

def vulnerable_function(user_input):
    # Command injection vulnerability
    os.system("ls " + user_input)
    
    # SQL injection vulnerability  
    query = "SELECT * FROM users WHERE name = '%s'" % user_input
    
    # Dangerous function
    result = eval(user_input)
    
    return result

# Debug code left in production
DEBUG = True
if DEBUG:
    print("Password:", password)
'''
    
    # Write test file
    test_file = Path("test_security.py")
    test_file.write_text(test_code)
    
    try:
        # Scan the test file
        issues = scanner.scan_file(test_file)
        
        print(f"  ✓ Found {len(issues)} security issues")
        
        # Show issue types
        categories = set(issue.category for issue in issues)
        print(f"  ✓ Issue categories: {categories}")
        
        # Show summary
        summary = scanner.get_summary()
        print(f"  ✓ Summary: {summary['total_issues']} total, {summary.get('critical_issues', 0)} critical")
        
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()
    
    # Test compliance checker
    print("\n2. Testing Compliance Checker:")
    compliance_checker = ComplianceChecker()
    
    # Create test directory structure
    test_dir = Path("test_compliance")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Check compliance
        compliance_issues = compliance_checker.check_compliance(
            test_dir, 
            [ComplianceStandard.GDPR, ComplianceStandard.CCPA]
        )
        
        print(f"  ✓ Found {len(compliance_issues)} compliance issues")
        
    finally:
        # Clean up test directory
        if test_dir.exists() and test_dir.is_dir():
            test_dir.rmdir()
    
    # Test report generator
    print("\n3. Testing Report Generator:")
    report_generator = SecurityReportGenerator()
    
    # Generate report with mock data
    mock_security_issues = [
        SecurityIssue(
            issue_id="test123",
            severity=SecuritySeverity.HIGH,
            category="test_category",
            description="Test security issue",
            file_path="test.py",
            line_number=10,
            recommendation="Fix the issue"
        )
    ]
    
    mock_compliance_issues = [
        ComplianceIssue(
            standard=ComplianceStandard.GDPR,
            requirement="encryption",
            description="Missing encryption",
            severity=SecuritySeverity.MEDIUM,
            file_path="test.py",
            recommendation="Add encryption"
        )
    ]
    
    report = report_generator.generate_report(
        mock_security_issues,
        mock_compliance_issues,
        output_format='json'
    )
    
    # Parse report to verify
    report_data = json.loads(report)
    assert 'summary' in report_data
    assert 'risk_score' in report_data
    print(f"  ✓ Generated report with risk score: {report_data['risk_score']}")
    
    print("\n✓ Security scanner test completed!")