#!/usr/bin/env python3
"""
Comprehensive Test Runner for AI Inference Platform
Runs all tests in proper directory structure.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_test_suite(test_path, description):
    """Run a test suite and return results."""
    print(f"\n🧪 Running {description}...")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"✅ {description}: PASSED")
            return True, result.stdout.count("PASSED"), 0
        else:
            print(f"❌ {description}: FAILED")
            print(f"Error output: {result.stderr}")
            return False, result.stdout.count("PASSED"), result.stdout.count("FAILED")
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description}: TIMEOUT")
        return False, 0, 1
    except Exception as e:
        print(f"💥 {description}: ERROR - {e}")
        return False, 0, 1

def run_standalone_test(test_file, description):
    """Run a standalone test file."""
    print(f"\n🔧 Running {description}...")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"✅ {description}: PASSED")
            # Try to extract test count from output
            output = result.stdout
            if "Tests run:" in output:
                lines = output.split('\n')
                for line in lines:
                    if "Tests run:" in line and "Passed:" in line:
                        parts = line.split()
                        passed = int(parts[parts.index("Passed:") + 1])
                        failed = int(parts[parts.index("Failed:") + 1])
                        return True, passed, failed
            return True, 1, 0
        else:
            print(f"❌ {description}: FAILED")
            print(f"Error: {result.stderr}")
            return False, 0, 1
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description}: TIMEOUT")
        return False, 0, 1
    except Exception as e:
        print(f"💥 {description}: ERROR - {e}")
        return False, 0, 1

def main():
    """Run all tests in the project."""
    start_time = time.time()
    
    print("🚀 AI Inference Platform - Comprehensive Test Suite")
    print("=" * 80)
    
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"
    
    total_passed = 0
    total_failed = 0
    suite_results = []
    
    # Test suites to run
    test_suites = [
        (tests_dir / "executors", "Executor Tests"),
        (tests_dir / "flow_engine", "Flow Engine Tests"),
        (tests_dir / "integration", "Integration Tests"),
    ]
    
    # Run pytest-based test suites
    for test_path, description in test_suites:
        if test_path.exists():
            success, passed, failed = run_test_suite(test_path, description)
            suite_results.append((description, success, passed, failed))
            total_passed += passed
            total_failed += failed
        else:
            print(f"⚠️  {description}: Directory not found - {test_path}")
    
    # Run standalone tests
    standalone_tests = [
        (tests_dir / "executors" / "test_data_combiner_minimal.py", "Data Combiner Standalone Tests"),
        (tests_dir / "flow_engine" / "test_flow_engine_integration.py", "Flow Engine Integration Tests"),
    ]
    
    for test_file, description in standalone_tests:
        if test_file.exists():
            success, passed, failed = run_standalone_test(test_file, description)
            suite_results.append((description, success, passed, failed))
            total_passed += passed
            total_failed += failed
        else:
            print(f"⚠️  {description}: File not found - {test_file}")
    
    # Summary
    duration = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("📊 TEST EXECUTION SUMMARY")
    print("=" * 80)
    
    for description, success, passed, failed in suite_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} {description:30} ({passed:3d} passed, {failed:3d} failed)")
    
    print("-" * 80)
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success Rate: {(total_passed/(total_passed + total_failed)*100):.1f}%" if (total_passed + total_failed) > 0 else "N/A")
    print(f"Duration: {duration:.2f}s")
    
    # Overall status
    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED - PLATFORM READY FOR PRODUCTION!")
        return 0
    else:
        print(f"\n⚠️  {total_failed} TESTS FAILED - REVIEW REQUIRED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
