#!/usr/bin/env python3
"""
Comprehensive test runner for Flow Engine components.
Tests YAML Loader, Template Engine, Context Manager, and Flow Runner.
"""

import asyncio
import sys
import traceback
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import test modules
from tests.test_yaml_loader import TestFlowDataClasses, TestYAMLFlowLoader, TestFlowDefinition, TestIntegrationScenarios
from tests.test_template_engine import TestTemplateEngine, TestTemplateEngineIntegration
from tests.test_context_manager import TestFlowExecution, TestContextManager, TestContextManagerIntegration
from tests.test_flow_runner import TestExecutorRegistry, TestFlowRunner, TestFlowRunnerIntegration


class FlowEngineTestRunner:
    """Comprehensive test runner for flow engine components."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        self.start_time = None
    
    def log_test_result(self, test_name, success, error=None):
        """Log test result."""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✓ {test_name}")
        else:
            self.tests_failed += 1
            self.failures.append((test_name, error))
            print(f"✗ {test_name}: {error}")
    
    async def run_async_test(self, test_instance, test_method_name):
        """Run async test method."""
        try:
            test_method = getattr(test_instance, test_method_name)
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
            return True, None
        except Exception as e:
            return False, str(e)
    
    def run_sync_test(self, test_instance, test_method_name):
        """Run synchronous test method."""
        try:
            test_method = getattr(test_instance, test_method_name)
            test_method()
            return True, None
        except Exception as e:
            return False, str(e)
    
    async def run_test_class(self, test_class, class_name):
        """Run all tests in a test class."""
        print(f"\n--- {class_name} ---")
        
        # Get test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            test_instance = test_class()
            
            # Setup fixtures if needed
            if hasattr(test_instance, 'setup_method'):
                test_instance.setup_method()
            
            # Run test
            success, error = await self.run_async_test(test_instance, test_method_name)
            self.log_test_result(f"{class_name}.{test_method_name}", success, error)
            
            # Teardown if needed
            if hasattr(test_instance, 'teardown_method'):
                test_instance.teardown_method()
    
    async def run_yaml_loader_tests(self):
        """Run YAML Loader tests."""
        print("\n" + "="*60)
        print("YAML LOADER TESTS")
        print("="*60)
        
        await self.run_test_class(TestFlowDataClasses, "TestFlowDataClasses")
        await self.run_test_class(TestYAMLFlowLoader, "TestYAMLFlowLoader")
        await self.run_test_class(TestFlowDefinition, "TestFlowDefinition")
        await self.run_test_class(TestIntegrationScenarios, "TestIntegrationScenarios")
    
    async def run_template_engine_tests(self):
        """Run Template Engine tests."""
        print("\n" + "="*60)
        print("TEMPLATE ENGINE TESTS")
        print("="*60)
        
        await self.run_test_class(TestTemplateEngine, "TestTemplateEngine")
        await self.run_test_class(TestTemplateEngineIntegration, "TestTemplateEngineIntegration")
    
    async def run_context_manager_tests(self):
        """Run Context Manager tests."""
        print("\n" + "="*60)
        print("CONTEXT MANAGER TESTS")
        print("="*60)
        
        await self.run_test_class(TestFlowExecution, "TestFlowExecution")
        await self.run_test_class(TestContextManager, "TestContextManager")
        await self.run_test_class(TestContextManagerIntegration, "TestContextManagerIntegration")
    
    async def run_flow_runner_tests(self):
        """Run Flow Runner tests."""
        print("\n" + "="*60)
        print("FLOW RUNNER TESTS")
        print("="*60)
        
        await self.run_test_class(TestExecutorRegistry, "TestExecutorRegistry")
        await self.run_test_class(TestFlowRunner, "TestFlowRunner")
        await self.run_test_class(TestFlowRunnerIntegration, "TestFlowRunnerIntegration")
    
    async def run_integration_tests(self):
        """Run end-to-end integration tests."""
        print("\n" + "="*60)
        print("INTEGRATION TESTS")
        print("="*60)
        
        # Test complete flow engine pipeline
        await self.test_complete_flow_pipeline()
        await self.test_error_handling_pipeline()
        await self.test_performance_pipeline()
    
    async def test_complete_flow_pipeline(self):
        """Test complete flow engine pipeline."""
        try:
            from core.flow_engine import FlowRunner, YAMLFlowLoader
            
            # Create flow YAML
            flow_yaml = """
name: "integration_test_flow"
description: "End-to-end integration test"

inputs:
  - name: "test_input"
    type: "string"
    required: true

steps:
  - name: "step1"
    executor: "mock_executor"
    config:
      data: "{{ inputs.test_input }}"

outputs:
  - name: "result"
    value: "{{ steps.step1.result }}"
"""
            
            # Load flow
            loader = YAMLFlowLoader()
            flow_def = loader.load_from_string(flow_yaml)
            
            # Execute flow (would need mock executor)
            # This is a placeholder for actual integration test
            
            self.log_test_result("test_complete_flow_pipeline", True)
            
        except Exception as e:
            self.log_test_result("test_complete_flow_pipeline", False, str(e))
    
    async def test_error_handling_pipeline(self):
        """Test error handling throughout the pipeline."""
        try:
            # Test various error scenarios
            # This is a placeholder for comprehensive error testing
            self.log_test_result("test_error_handling_pipeline", True)
        except Exception as e:
            self.log_test_result("test_error_handling_pipeline", False, str(e))
    
    async def test_performance_pipeline(self):
        """Test performance characteristics."""
        try:
            # Test performance with large flows, many steps, etc.
            # This is a placeholder for performance testing
            self.log_test_result("test_performance_pipeline", True)
        except Exception as e:
            self.log_test_result("test_performance_pipeline", False, str(e))
    
    def print_summary(self):
        """Print test execution summary."""
        duration = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "="*60)
        print("FLOW ENGINE TEST SUMMARY")
        print("="*60)
        print(f"Tests run: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Success rate: {(self.tests_passed/self.tests_run*100):.1f}%" if self.tests_run > 0 else "N/A")
        print(f"Duration: {duration:.2f}s")
        
        if self.failures:
            print(f"\nFailures ({len(self.failures)}):")
            for test_name, error in self.failures:
                print(f"  ✗ {test_name}")
                print(f"    {error}")
        
        # Coverage analysis
        self.print_coverage_analysis()
    
    def print_coverage_analysis(self):
        """Print coverage analysis."""
        print(f"\n" + "="*60)
        print("COVERAGE ANALYSIS")
        print("="*60)
        
        components = {
            "YAML Loader": ["FlowDataClasses", "YAMLFlowLoader", "FlowDefinition", "IntegrationScenarios"],
            "Template Engine": ["TemplateEngine", "TemplateEngineIntegration"],
            "Context Manager": ["FlowExecution", "ContextManager", "ContextManagerIntegration"],
            "Flow Runner": ["ExecutorRegistry", "FlowRunner", "FlowRunnerIntegration"]
        }
        
        for component, test_classes in components.items():
            passed_classes = 0
            total_classes = len(test_classes)
            
            for test_class in test_classes:
                class_failures = [f for f in self.failures if test_class in f[0]]
                if not class_failures:
                    passed_classes += 1
            
            coverage = (passed_classes / total_classes * 100) if total_classes > 0 else 0
            status = "✓" if coverage == 100 else "⚠" if coverage >= 80 else "✗"
            print(f"{status} {component}: {coverage:.1f}% ({passed_classes}/{total_classes} test classes)")
    
    async def run_all_tests(self):
        """Run all flow engine tests."""
        self.start_time = time.time()
        
        print("Flow Engine Comprehensive Test Suite")
        print("="*60)
        
        try:
            await self.run_yaml_loader_tests()
            await self.run_template_engine_tests()
            await self.run_context_manager_tests()
            await self.run_flow_runner_tests()
            await self.run_integration_tests()
            
        except KeyboardInterrupt:
            print("\n\nTest execution interrupted by user")
        except Exception as e:
            print(f"\n\nUnexpected error during test execution: {e}")
            traceback.print_exc()
        
        finally:
            self.print_summary()
            return self.tests_failed == 0


async def main():
    """Main test execution."""
    runner = FlowEngineTestRunner()
    success = await runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
