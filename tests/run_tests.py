#!/usr/bin/env python3
"""
Test runner for AI Inference Platform core components.

This script runs all unit tests and generates coverage reports.
"""
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_tests():
    """Run all tests with coverage reporting."""
    print("🧪 Running AI Inference Platform Core Tests")
    print("=" * 50)
    
    # Test command with coverage
    cmd = [
        "python", "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=core",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--asyncio-mode=auto"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            print("📊 Coverage report generated in htmlcov/index.html")
        else:
            print(f"\n❌ Tests failed with exit code {result.returncode}")
            
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def run_specific_test_module(module_name):
    """Run tests for a specific module."""
    print(f"🧪 Running tests for {module_name}")
    print("=" * 50)
    
    cmd = [
        "python", "-m", "pytest",
        f"tests/unit/{module_name}",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test module
        module = sys.argv[1]
        exit_code = run_specific_test_module(module)
    else:
        # Run all tests
        exit_code = run_tests()
    
    sys.exit(exit_code)
