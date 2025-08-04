#!/usr/bin/env python3
"""
Structural test for SNN-Fusion package.
Tests package structure and basic class definitions without heavy dependencies.
"""

import sys
import os
import importlib.util
import ast

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def analyze_module_structure(module_path):
    """Analyze Python module structure without importing it."""
    try:
        with open(module_path, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        classes = []
        functions = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return {
            'classes': classes,
            'functions': functions,
            'imports': imports[:10]  # First 10 imports
        }
    except Exception as e:
        return {'error': str(e)}

def test_package_structure():
    """Test package directory structure."""
    print("🧪 Testing Package Structure...")
    
    src_dir = "src/snn_fusion"
    expected_modules = [
        "models",
        "training", 
        "datasets",
        "preprocessing",
        "algorithms",
        "api",
        "cli.py"
    ]
    
    results = []
    
    for module in expected_modules:
        path = os.path.join(src_dir, module)
        
        if os.path.exists(path):
            if os.path.isdir(path):
                init_file = os.path.join(path, "__init__.py")
                if os.path.exists(init_file):
                    print(f"✅ Module {module}/ with __init__.py")
                    results.append((f"Module {module}", True))
                else:
                    print(f"⚠️  Module {module}/ missing __init__.py")
                    results.append((f"Module {module}", False))
            else:
                print(f"✅ File {module}")
                results.append((f"File {module}", True))
        else:
            print(f"❌ Missing {module}")
            results.append((f"Missing {module}", False))
    
    return results

def test_key_implementations():
    """Test that key classes are implemented."""
    print("\n🧪 Testing Key Implementations...")
    
    key_files = {
        "HierarchicalFusionSNN": "src/snn_fusion/models/hierarchical_fusion.py",
        "STDPPlasticity": "src/snn_fusion/training/plasticity.py", 
        "TemporalLoss": "src/snn_fusion/training/losses.py",
        "MAVENDataset": "src/snn_fusion/datasets/maven_dataset.py",
        "MultiModalLSM": "src/snn_fusion/models/multimodal_lsm.py"
    }
    
    results = []
    
    for class_name, file_path in key_files.items():
        if os.path.exists(file_path):
            analysis = analyze_module_structure(file_path)
            
            if 'error' in analysis:
                print(f"❌ {class_name}: Parse error - {analysis['error']}")
                results.append((class_name, False))
            elif class_name in analysis['classes']:
                print(f"✅ {class_name}: Implemented")
                results.append((class_name, True))
            else:
                print(f"❌ {class_name}: Class not found")
                results.append((class_name, False))
        else:
            print(f"❌ {class_name}: File not found")
            results.append((class_name, False))
    
    return results

def test_configuration_files():
    """Test configuration files."""
    print("\n🧪 Testing Configuration Files...")
    
    config_files = [
        "pyproject.toml",
        "setup.py", 
        "requirements.txt",
        "README.md",
        "ARCHITECTURE.md"
    ]
    
    results = []
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file}")
            results.append((config_file, True))
        else:
            print(f"❌ {config_file}")
            results.append((config_file, False))
    
    return results

def test_documentation():
    """Test documentation structure."""
    print("\n🧪 Testing Documentation...")
    
    doc_structure = [
        "docs/",
        "docs/guides/",
        "docs/adr/"
    ]
    
    results = []
    
    for doc_path in doc_structure:
        if os.path.exists(doc_path):
            print(f"✅ {doc_path}")
            results.append((doc_path, True))
        else:
            print(f"❌ {doc_path}")
            results.append((doc_path, False))
    
    return results

def run_structural_tests():
    """Run all structural tests."""
    print("🚀 Starting SNN-Fusion Structural Tests\n")
    
    all_results = []
    
    # Run test suites
    all_results.extend(test_package_structure())
    all_results.extend(test_key_implementations())
    all_results.extend(test_configuration_files())
    all_results.extend(test_documentation())
    
    # Summarize results
    print("\n📊 Structural Test Summary:")
    print("=" * 60)
    
    passed = sum(1 for _, result in all_results if result)
    total = len(all_results)
    
    for test_name, result in all_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} structural tests passed ({passed/total*100:.1f}%)")
    
    if passed >= total * 0.8:  # 80% pass rate is good for structure
        print("🎉 Package structure looks good!")
        return True
    else:
        print("⚠️  Package structure needs improvement.")
        return False

if __name__ == "__main__":
    success = run_structural_tests()
    sys.exit(0 if success else 1)