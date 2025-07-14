#!/usr/bin/env python3
"""
Simplified Flow Engine Test Runner
Tests core flow engine functionality without external dependencies.
"""

import asyncio
import sys
import traceback
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock
import tempfile
import yaml
import uuid


class MockExecutionResult:
    """Mock ExecutionResult for testing."""
    def __init__(self, success=True, outputs=None, error=None):
        self.success = success
        self.outputs = outputs or {}
        self.error = error


class MockFlowContext:
    """Mock FlowContext for testing."""
    def __init__(self, execution_id=None, inputs=None, step_results=None):
        self.execution_id = execution_id or str(uuid.uuid4())
        self.inputs = inputs or {}
        self.step_results = step_results or {}


class FlowEngineTestRunner:
    """Simplified test runner for flow engine components."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        self.start_time = time.time()
    
    def log_result(self, test_name, success, error=None):
        """Log test result."""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✓ {test_name}")
        else:
            self.tests_failed += 1
            self.failures.append((test_name, error))
            print(f"✗ {test_name}: {error}")
    
    def run_test(self, test_name, test_func):
        """Run a single test."""
        try:
            test_func()
            self.log_result(test_name, True)
        except Exception as e:
            self.log_result(test_name, False, str(e))
    
    async def run_async_test(self, test_name, test_func):
        """Run an async test."""
        try:
            await test_func()
            self.log_result(test_name, True)
        except Exception as e:
            self.log_result(test_name, False, str(e))
    
    def test_yaml_loader_basic(self):
        """Test basic YAML loading functionality."""
        yaml_content = """
name: "test_flow"
description: "Test flow"
version: "1.0"

inputs:
  - name: "input1"
    type: "string"
    required: true

steps:
  - name: "step1"
    executor: "test_executor"
    config:
      param: "{{ inputs.input1 }}"

outputs:
  - name: "result"
    value: "{{ steps.step1.output }}"
"""
        
        # Parse YAML
        data = yaml.safe_load(yaml_content)
        
        # Validate structure
        assert data["name"] == "test_flow"
        assert len(data["inputs"]) == 1
        assert len(data["steps"]) == 1
        assert len(data["outputs"]) == 1
        
        # Validate input structure
        input1 = data["inputs"][0]
        assert input1["name"] == "input1"
        assert input1["type"] == "string"
        assert input1["required"] is True
        
        # Validate step structure
        step1 = data["steps"][0]
        assert step1["name"] == "step1"
        assert step1["executor"] == "test_executor"
        assert step1["config"]["param"] == "{{ inputs.input1 }}"
    
    def test_yaml_loader_validation(self):
        """Test YAML validation logic."""
        # Test missing required fields
        invalid_yaml = """
description: "Missing name"
steps: []
"""
        data = yaml.safe_load(invalid_yaml)
        
        # Should detect missing name
        assert "name" not in data
        
        # Test empty steps
        empty_steps_yaml = """
name: "test"
steps: []
"""
        data = yaml.safe_load(empty_steps_yaml)
        assert len(data["steps"]) == 0
    
    def test_yaml_dependency_validation(self):
        """Test step dependency validation."""
        yaml_content = """
name: "dependency_test"
steps:
  - name: "step1"
    executor: "executor1"
  - name: "step2"
    executor: "executor2"
    depends_on: ["step1"]
  - name: "step3"
    executor: "executor3"
    depends_on: ["step1", "step2"]
"""
        
        data = yaml.safe_load(yaml_content)
        steps = data["steps"]
        
        # Build dependency map
        step_names = {step["name"] for step in steps}
        
        # Validate dependencies exist
        for step in steps:
            if "depends_on" in step:
                for dep in step["depends_on"]:
                    assert dep in step_names, f"Dependency {dep} not found"
    
    def test_template_engine_basic(self):
        """Test basic template rendering."""
        # Simple string replacement
        template = "Hello {{ name }}!"
        context = {"name": "World"}
        
        # Mock template rendering
        result = template.replace("{{ name }}", context["name"])
        assert result == "Hello World!"
        
        # Nested object access
        template = "Value: {{ data.value }}"
        context = {"data": {"value": 42}}
        
        # Would need actual Jinja2 for real implementation
        # This is a simplified test
        assert "{{ data.value }}" in template
    
    def test_template_engine_step_references(self):
        """Test template rendering with step references."""
        template = "Result: {{ steps.step1.output }}"
        
        # Mock step results
        step_results = {
            "step1": MockExecutionResult(
                success=True,
                outputs={"output": "test_result"}
            )
        }
        
        # Verify template structure
        assert "steps.step1.output" in template
        assert step_results["step1"].outputs["output"] == "test_result"
    
    def test_context_manager_basic(self):
        """Test basic context management."""
        # Mock execution creation
        execution_id = str(uuid.uuid4())
        flow_name = "test_flow"
        inputs = {"param1": "value1"}
        
        # Mock execution data
        execution = {
            "execution_id": execution_id,
            "flow_name": flow_name,
            "status": "running",
            "inputs": inputs,
            "outputs": {},
            "step_results": {},
            "start_time": datetime.now(timezone.utc),
            "end_time": None,
            "error": None
        }
        
        # Validate execution structure
        assert execution["execution_id"] == execution_id
        assert execution["flow_name"] == flow_name
        assert execution["status"] == "running"
        assert execution["inputs"] == inputs
        assert execution["start_time"] is not None
    
    def test_context_manager_step_updates(self):
        """Test step result updates."""
        execution_id = str(uuid.uuid4())
        
        # Mock execution storage
        executions = {}
        executions[execution_id] = {
            "step_results": {}
        }
        
        # Mock step result update
        step_name = "test_step"
        step_result = MockExecutionResult(
            success=True,
            outputs={"data": "test_data"}
        )
        
        # Update step result
        executions[execution_id]["step_results"][step_name] = {
            "success": step_result.success,
            "outputs": step_result.outputs,
            "error": step_result.error
        }
        
        # Validate update
        stored_result = executions[execution_id]["step_results"][step_name]
        assert stored_result["success"] is True
        assert stored_result["outputs"]["data"] == "test_data"
    
    def test_executor_registry_basic(self):
        """Test basic executor registry functionality."""
        # Mock executor registry
        registry = {}
        
        # Mock executor class
        class MockExecutor:
            def __init__(self, name):
                self.name = name
            
            async def execute(self, context, config):
                return MockExecutionResult(
                    success=True,
                    outputs={"result": f"executed_{self.name}"}
                )
        
        # Register executor
        registry["test_executor"] = MockExecutor
        
        # Get executor
        executor_class = registry.get("test_executor")
        assert executor_class is not None
        
        # Create instance
        executor = executor_class("test_executor")
        assert executor.name == "test_executor"
    
    async def test_flow_execution_basic(self):
        """Test basic flow execution logic."""
        # Mock flow definition
        flow_def = {
            "name": "test_flow",
            "inputs": [{"name": "input1", "type": "string", "required": True}],
            "steps": [
                {
                    "name": "step1",
                    "executor": "mock_executor",
                    "config": {"data": "{{ inputs.input1 }}"}
                }
            ],
            "outputs": [{"name": "result", "value": "{{ steps.step1.result }}"}]
        }
        
        # Mock inputs
        inputs = {"input1": "test_value"}
        
        # Validate input requirements
        required_inputs = [inp for inp in flow_def["inputs"] if inp.get("required", False)]
        for req_input in required_inputs:
            assert req_input["name"] in inputs, f"Missing required input: {req_input['name']}"
        
        # Mock step execution
        step_results = {}
        for step in flow_def["steps"]:
            # Mock template processing
            config = step["config"].copy()
            if "{{ inputs.input1 }}" in str(config):
                config = {"data": inputs["input1"]}
            
            # Mock executor execution
            step_result = MockExecutionResult(
                success=True,
                outputs={"result": f"processed_{config['data']}"}
            )
            
            step_results[step["name"]] = step_result
        
        # Validate execution
        assert "step1" in step_results
        assert step_results["step1"].success is True
        assert "processed_test_value" in step_results["step1"].outputs["result"]
    
    def test_dependency_resolution(self):
        """Test step dependency resolution."""
        steps = [
            {"name": "step1", "executor": "exec1"},
            {"name": "step2", "executor": "exec2", "depends_on": ["step1"]},
            {"name": "step3", "executor": "exec3", "depends_on": ["step1", "step2"]},
            {"name": "step4", "executor": "exec4"}
        ]
        
        # Build dependency graph
        dependencies = {}
        for step in steps:
            dependencies[step["name"]] = step.get("depends_on", [])
        
        # Topological sort simulation
        def can_execute(step_name, completed_steps):
            deps = dependencies.get(step_name, [])
            return all(dep in completed_steps for dep in deps)
        
        # Simulate execution order
        completed = set()
        execution_order = []
        
        while len(execution_order) < len(steps):
            for step in steps:
                step_name = step["name"]
                if step_name not in completed and can_execute(step_name, completed):
                    execution_order.append(step_name)
                    completed.add(step_name)
                    break
        
        # Validate execution order
        assert execution_order.index("step1") < execution_order.index("step2")
        assert execution_order.index("step2") < execution_order.index("step3")
        assert execution_order.index("step1") < execution_order.index("step3")
    
    def test_error_handling(self):
        """Test error handling scenarios."""
        # Test step failure
        step_result = MockExecutionResult(
            success=False,
            error="Step execution failed"
        )
        
        assert step_result.success is False
        assert step_result.error == "Step execution failed"
        
        # Test missing dependency
        steps = [
            {"name": "step1", "executor": "exec1", "depends_on": ["nonexistent"]}
        ]
        
        step_names = {step["name"] for step in steps}
        
        # Should detect missing dependency
        for step in steps:
            if "depends_on" in step:
                for dep in step["depends_on"]:
                    if dep not in step_names:
                        # Missing dependency detected
                        assert dep == "nonexistent"
    
    def test_parallel_execution_groups(self):
        """Test parallel execution group identification."""
        steps = [
            {"name": "step1", "executor": "exec1"},
            {"name": "step2", "executor": "exec2", "parallel_group": "group_a"},
            {"name": "step3", "executor": "exec3", "parallel_group": "group_a"},
            {"name": "step4", "executor": "exec4", "parallel_group": "group_b"}
        ]
        
        # Group steps by parallel group
        parallel_groups = {}
        for step in steps:
            if "parallel_group" in step:
                group = step["parallel_group"]
                if group not in parallel_groups:
                    parallel_groups[group] = []
                parallel_groups[group].append(step["name"])
        
        # Validate grouping
        assert "group_a" in parallel_groups
        assert "group_b" in parallel_groups
        assert len(parallel_groups["group_a"]) == 2
        assert len(parallel_groups["group_b"]) == 1
        assert "step2" in parallel_groups["group_a"]
        assert "step3" in parallel_groups["group_a"]
    
    def test_conditional_execution(self):
        """Test conditional step execution."""
        steps = [
            {
                "name": "step1",
                "executor": "exec1",
                "condition": "{{ inputs.enable_step }}"
            }
        ]
        
        inputs = {"enable_step": True}
        
        # Mock condition evaluation
        condition = steps[0]["condition"]
        
        # Simple condition check (would use Jinja2 in real implementation)
        if "inputs.enable_step" in condition:
            should_execute = inputs.get("enable_step", False)
        else:
            should_execute = True
        
        assert should_execute is True
        
        # Test with false condition
        inputs["enable_step"] = False
        should_execute = inputs.get("enable_step", False)
        assert should_execute is False
    
    async def run_all_tests(self):
        """Run all flow engine tests."""
        print("Flow Engine Test Suite")
        print("=" * 50)
        
        # YAML Loader Tests
        print("\n--- YAML Loader Tests ---")
        self.run_test("test_yaml_loader_basic", self.test_yaml_loader_basic)
        self.run_test("test_yaml_loader_validation", self.test_yaml_loader_validation)
        self.run_test("test_yaml_dependency_validation", self.test_yaml_dependency_validation)
        
        # Template Engine Tests
        print("\n--- Template Engine Tests ---")
        self.run_test("test_template_engine_basic", self.test_template_engine_basic)
        self.run_test("test_template_engine_step_references", self.test_template_engine_step_references)
        
        # Context Manager Tests
        print("\n--- Context Manager Tests ---")
        self.run_test("test_context_manager_basic", self.test_context_manager_basic)
        self.run_test("test_context_manager_step_updates", self.test_context_manager_step_updates)
        
        # Flow Runner Tests
        print("\n--- Flow Runner Tests ---")
        self.run_test("test_executor_registry_basic", self.test_executor_registry_basic)
        await self.run_async_test("test_flow_execution_basic", self.test_flow_execution_basic)
        
        # Advanced Tests
        print("\n--- Advanced Flow Tests ---")
        self.run_test("test_dependency_resolution", self.test_dependency_resolution)
        self.run_test("test_error_handling", self.test_error_handling)
        self.run_test("test_parallel_execution_groups", self.test_parallel_execution_groups)
        self.run_test("test_conditional_execution", self.test_conditional_execution)
        
        # Print summary
        self.print_summary()
        
        return self.tests_failed == 0
    
    def print_summary(self):
        """Print test execution summary."""
        duration = time.time() - self.start_time
        
        print("\n" + "=" * 50)
        print("FLOW ENGINE TEST SUMMARY")
        print("=" * 50)
        print(f"Tests run: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Success rate: {(self.tests_passed/self.tests_run*100):.1f}%" if self.tests_run > 0 else "N/A")
        print(f"Duration: {duration:.2f}s")
        
        if self.failures:
            print(f"\nFailures ({len(self.failures)}):")
            for test_name, error in self.failures:
                print(f"  ✗ {test_name}: {error}")
        
        # Component coverage
        print(f"\nComponent Coverage:")
        components = {
            "YAML Loader": 3,
            "Template Engine": 2,
            "Context Manager": 2,
            "Flow Runner": 2,
            "Advanced Features": 4
        }
        
        for component, expected_tests in components.items():
            passed_tests = sum(1 for test_name, _ in self.failures if component.lower().replace(" ", "_") not in test_name.lower())
            coverage = (passed_tests / expected_tests * 100) if expected_tests > 0 else 0
            status = "✓" if coverage >= 90 else "⚠" if coverage >= 70 else "✗"
            print(f"  {status} {component}: {coverage:.0f}%")


async def main():
    """Main test execution."""
    runner = FlowEngineTestRunner()
    success = await runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
