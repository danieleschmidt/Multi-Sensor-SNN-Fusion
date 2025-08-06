"""
Validation Tests for SNN-Fusion

This module provides validation tests that don't require external dependencies,
focusing on code structure, configuration validation, and basic functionality.
"""

import sys
import os
import importlib.util
from pathlib import Path
from typing import Dict, List, Any
import json
import inspect
import ast


class CodeValidator:
    """Validates code structure and basic functionality."""
    
    def __init__(self):
        self.src_path = Path(__file__).parent.parent
        self.results = []
    
    def validate_module_structure(self) -> Dict[str, Any]:
        """Validate the overall module structure."""
        results = {
            'test_name': 'Module Structure Validation',
            'status': 'passed',
            'details': []
        }
        
        expected_modules = [
            'utils/config.py',
            'utils/error_handling.py', 
            'utils/security_enhanced.py',
            'utils/backup_recovery.py',
            'utils/graceful_degradation.py',
            'monitoring/health.py',
            'optimization/performance.py',
            'scaling/load_balancer.py',
            'scaling/concurrent_processing.py',
            'testing/test_framework.py'
        ]
        
        missing_modules = []
        for module in expected_modules:
            module_path = self.src_path / module
            if not module_path.exists():
                missing_modules.append(module)
        
        if missing_modules:
            results['status'] = 'failed'
            results['details'].append(f"Missing modules: {missing_modules}")
        else:
            results['details'].append("All expected modules found")
        
        return results
    
    def validate_class_definitions(self) -> Dict[str, Any]:
        """Validate that key classes are properly defined."""
        results = {
            'test_name': 'Class Definition Validation',
            'status': 'passed',
            'details': []
        }
        
        expected_classes = {
            'utils/config.py': ['SNNFusionConfig', 'DatasetConfig', 'ModelConfig'],
            'utils/error_handling.py': ['ErrorHandler', 'SNNError'],
            'utils/security_enhanced.py': ['InputSanitizer', 'SecurityValidator'],
            'monitoring/health.py': ['HealthMonitor', 'SystemHealth'],
            'scaling/load_balancer.py': ['LoadBalancer', 'WorkerNode'],
            'scaling/concurrent_processing.py': ['ConcurrentProcessor', 'ProcessingTask']
        }
        
        for module_path, classes in expected_classes.items():
            full_path = self.src_path / module_path
            if not full_path.exists():
                continue
                
            try:
                # Parse file to find class definitions
                with open(full_path, 'r') as f:
                    tree = ast.parse(f.read())
                
                found_classes = [node.name for node in ast.walk(tree) 
                               if isinstance(node, ast.ClassDef)]
                
                missing_classes = [cls for cls in classes if cls not in found_classes]
                if missing_classes:
                    results['status'] = 'failed'
                    results['details'].append(
                        f"{module_path}: Missing classes {missing_classes}"
                    )
                else:
                    results['details'].append(f"{module_path}: All classes found")
                    
            except Exception as e:
                results['status'] = 'failed'
                results['details'].append(f"{module_path}: Parse error - {e}")
        
        return results
    
    def validate_function_signatures(self) -> Dict[str, Any]:
        """Validate that key functions have proper signatures."""
        results = {
            'test_name': 'Function Signature Validation',
            'status': 'passed', 
            'details': []
        }
        
        # Check specific function patterns in files
        function_patterns = {
            'utils/error_handling.py': ['handle_error', 'recover_from_error'],
            'utils/security_enhanced.py': ['sanitize_string', 'validate_input'],
            'monitoring/health.py': ['check_health', 'get_system_health'],
            'scaling/load_balancer.py': ['submit_task', 'get_cluster_status'],
            'scaling/concurrent_processing.py': ['submit_task', 'process_batch_async']
        }
        
        for module_path, functions in function_patterns.items():
            full_path = self.src_path / module_path
            if not full_path.exists():
                continue
                
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                
                found_functions = []
                for func in functions:
                    if f'def {func}(' in content:
                        found_functions.append(func)
                
                missing_functions = [f for f in functions if f not in found_functions]
                if missing_functions:
                    results['status'] = 'failed'
                    results['details'].append(
                        f"{module_path}: Missing functions {missing_functions}"
                    )
                else:
                    results['details'].append(f"{module_path}: All functions found")
                    
            except Exception as e:
                results['status'] = 'failed'
                results['details'].append(f"{module_path}: Read error - {e}")
        
        return results
    
    def validate_docstrings(self) -> Dict[str, Any]:
        """Validate that modules have proper documentation."""
        results = {
            'test_name': 'Documentation Validation',
            'status': 'passed',
            'details': []
        }
        
        module_files = [
            'utils/config.py',
            'utils/error_handling.py',
            'utils/security_enhanced.py',
            'monitoring/health.py',
            'optimization/performance.py',
            'scaling/load_balancer.py',
            'scaling/concurrent_processing.py'
        ]
        
        for module_path in module_files:
            full_path = self.src_path / module_path
            if not full_path.exists():
                continue
                
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Check for module docstring
                if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
                    results['status'] = 'failed'
                    results['details'].append(f"{module_path}: Missing module docstring")
                else:
                    results['details'].append(f"{module_path}: Has module docstring")
                    
            except Exception as e:
                results['status'] = 'failed'
                results['details'].append(f"{module_path}: Read error - {e}")
        
        return results
    
    def validate_imports(self) -> Dict[str, Any]:
        """Validate import statements are properly structured."""
        results = {
            'test_name': 'Import Structure Validation',
            'status': 'passed',
            'details': []
        }
        
        init_files = [
            'utils/__init__.py',
            'monitoring/__init__.py',
            'optimization/__init__.py', 
            'scaling/__init__.py',
            'testing/__init__.py'
        ]
        
        for init_path in init_files:
            full_path = self.src_path / init_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    # Check for __all__ definition
                    if '__all__' in content:
                        results['details'].append(f"{init_path}: Has __all__ definition")
                    else:
                        results['details'].append(f"{init_path}: Missing __all__ definition")
                        
                except Exception as e:
                    results['status'] = 'failed'
                    results['details'].append(f"{init_path}: Read error - {e}")
            else:
                results['details'].append(f"{init_path}: File not found")
        
        return results
    
    def validate_error_handling(self) -> Dict[str, Any]:
        """Validate error handling patterns."""
        results = {
            'test_name': 'Error Handling Validation',
            'status': 'passed',
            'details': []
        }
        
        # Check that modules use proper exception handling
        module_files = [
            'utils/error_handling.py',
            'utils/security_enhanced.py',
            'monitoring/health.py',
            'scaling/load_balancer.py'
        ]
        
        for module_path in module_files:
            full_path = self.src_path / module_path
            if not full_path.exists():
                continue
                
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Check for try-except blocks
                try_count = content.count('try:')
                except_count = content.count('except')
                
                if try_count > 0 and except_count > 0:
                    results['details'].append(
                        f"{module_path}: Has {try_count} try blocks, {except_count} except blocks"
                    )
                else:
                    results['details'].append(f"{module_path}: Limited error handling")
                    
            except Exception as e:
                results['status'] = 'failed'
                results['details'].append(f"{module_path}: Read error - {e}")
        
        return results
    
    def run_all_validations(self) -> List[Dict[str, Any]]:
        """Run all validation tests."""
        validations = [
            self.validate_module_structure(),
            self.validate_class_definitions(),
            self.validate_function_signatures(),
            self.validate_docstrings(),
            self.validate_imports(),
            self.validate_error_handling()
        ]
        
        return validations


class ConfigurationValidator:
    """Validates configuration structures and patterns."""
    
    def validate_config_structure(self) -> Dict[str, Any]:
        """Validate configuration class structures."""
        results = {
            'test_name': 'Configuration Structure Validation',
            'status': 'passed',
            'details': []
        }
        
        # Mock configuration validation
        expected_config_fields = {
            'SNNFusionConfig': ['dataset', 'model', 'training', 'stdp'],
            'DatasetConfig': ['name', 'path', 'batch_size'],
            'ModelConfig': ['architecture', 'n_neurons', 'connections']
        }
        
        config_file = Path(__file__).parent.parent / 'utils' / 'config.py'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    content = f.read()
                
                for config_class, fields in expected_config_fields.items():
                    if config_class in content:
                        found_fields = [field for field in fields if field in content]
                        if len(found_fields) >= len(fields) // 2:  # At least half the fields
                            results['details'].append(
                                f"{config_class}: Found {len(found_fields)}/{len(fields)} fields"
                            )
                        else:
                            results['details'].append(
                                f"{config_class}: Missing fields {set(fields) - set(found_fields)}"
                            )
                    else:
                        results['status'] = 'failed'
                        results['details'].append(f"{config_class}: Class not found")
                        
            except Exception as e:
                results['status'] = 'failed'
                results['details'].append(f"Config file read error: {e}")
        else:
            results['status'] = 'failed'
            results['details'].append("Config file not found")
        
        return results


def run_validation_suite():
    """Run the complete validation suite."""
    print("🧪 Running SNN-Fusion Validation Suite...")
    print("=" * 60)
    
    # Code validation
    code_validator = CodeValidator()
    code_results = code_validator.run_all_validations()
    
    # Configuration validation  
    config_validator = ConfigurationValidator()
    config_results = [config_validator.validate_config_structure()]
    
    # Combine results
    all_results = code_results + config_results
    
    # Print results
    passed = 0
    total = len(all_results)
    
    for result in all_results:
        status_icon = "✅" if result['status'] == 'passed' else "❌"
        print(f"\n{status_icon} {result['test_name']}")
        
        if result['status'] == 'passed':
            passed += 1
            
        for detail in result['details']:
            print(f"   {detail}")
    
    print("\n" + "=" * 60)
    print(f"📊 Validation Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All validations passed!")
        return True
    else:
        print("⚠️  Some validations failed. Review the details above.")
        return False


if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)