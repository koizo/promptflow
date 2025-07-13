#!/usr/bin/env python3
"""
Test Phase 1: Core Framework Implementation

Tests the base executor framework, flow engine components,
and YAML flow loading functionality.
"""

import asyncio
import tempfile
from pathlib import Path
import yaml
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.executors.base_executor import BaseExecutor, ExecutionResult, FlowContext
from core.flow_engine.yaml_loader import YAMLFlowLoader
from core.flow_engine.template_engine import TemplateEngine
from core.flow_engine.context_manager import ContextManager
from core.flow_engine.flow_runner import FlowRunner, ExecutorRegistry


class TestExecutor(BaseExecutor):
    """Simple test executor for validation."""
    
    async def execute(self, context: FlowContext, config: dict) -> ExecutionResult:
        """Test execution that returns configured message."""
        message = config.get("message", "Hello from test executor!")
        
        return ExecutionResult(
            success=True,
            outputs={"message": message, "input_count": len(context.inputs)}
        )
    
    def get_required_config_keys(self):
        return []
    
    def get_optional_config_keys(self):
        return ["message"]


async def test_base_executor():
    """Test base executor functionality."""
    print("🧪 Testing Base Executor...")
    
    # Create test context
    context = FlowContext("test_flow", {"input1": "value1", "input2": "value2"})
    
    # Create test executor
    executor = TestExecutor("test_executor")
    
    # Test execution
    config = {"message": "Test message from config"}
    result = await executor._safe_execute(context, config)
    
    assert result.success, f"Execution failed: {result.error}"
    assert result.outputs["message"] == "Test message from config"
    assert result.outputs["input_count"] == 2
    assert result.execution_time is not None
    
    print("✅ Base Executor test passed")


def test_yaml_loader():
    """Test YAML flow loader."""
    print("🧪 Testing YAML Flow Loader...")
    
    # Create test flow YAML
    test_flow = {
        "name": "test_flow",
        "version": "1.0.0",
        "description": "Test flow for validation",
        "inputs": [
            {"name": "input1", "type": "string", "required": True},
            {"name": "input2", "type": "string", "default": "default_value"}
        ],
        "steps": [
            {
                "name": "step1",
                "executor": "test_executor",
                "config": {"message": "{{ inputs.input1 }}"}
            },
            {
                "name": "step2", 
                "executor": "test_executor",
                "depends_on": ["step1"],
                "config": {"message": "Step 2 depends on step 1"}
            }
        ],
        "outputs": [
            {"name": "result", "value": "{{ steps.step2.message }}"}
        ]
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_flow, f)
        temp_path = Path(f.name)
    
    try:
        # Load flow
        loader = YAMLFlowLoader()
        flow_def = loader.load_flow(temp_path)
        
        # Validate loaded flow
        assert flow_def.name == "test_flow"
        assert len(flow_def.inputs) == 2
        assert len(flow_def.steps) == 2
        assert len(flow_def.outputs) == 1
        
        # Test dependency validation
        errors = flow_def.validate_dependencies()
        assert len(errors) == 0, f"Dependency errors: {errors}"
        
        # Test execution order
        execution_order = flow_def.get_execution_order()
        assert len(execution_order) == 2  # Two levels
        assert execution_order[0] == ["step1"]  # First level
        assert execution_order[1] == ["step2"]  # Second level
        
        print("✅ YAML Flow Loader test passed")
        
    finally:
        # Clean up
        temp_path.unlink()


def test_template_engine():
    """Test template engine."""
    print("🧪 Testing Template Engine...")
    
    engine = TemplateEngine()
    
    # Test simple template
    template = "Hello {{ name }}!"
    context = {"name": "World"}
    result = engine.render_template(template, context)
    assert result == "Hello World!"
    
    # Test complex template with nested data
    template = "Input: {{ inputs.value }}, Step result: {{ steps.step1.output }}"
    context = {
        "inputs": {"value": "test_input"},
        "steps": {"step1": {"output": "step_output"}}
    }
    result = engine.render_template(template, context)
    assert result == "Input: test_input, Step result: step_output"
    
    # Test config rendering
    config = {
        "message": "Hello {{ inputs.name }}",
        "count": "{{ steps.counter.value }}",
        "nested": {
            "value": "{{ inputs.nested_value }}"
        }
    }
    context = {
        "inputs": {"name": "Test", "nested_value": "nested"},
        "steps": {"counter": {"value": 42}}
    }
    
    rendered = engine.render_config(config, context)
    assert rendered["message"] == "Hello Test"
    assert rendered["count"] == "42"
    assert rendered["nested"]["value"] == "nested"
    
    print("✅ Template Engine test passed")


async def test_context_manager():
    """Test context manager."""
    print("🧪 Testing Context Manager...")
    
    manager = ContextManager(cleanup_interval=1)  # Short interval for testing
    await manager.start()
    
    try:
        # Create execution
        execution = manager.create_context("test_flow", {"input": "value"})
        assert execution.flow_name == "test_flow"
        assert execution.status == "running"
        
        # Add step result
        result = ExecutionResult(success=True, outputs={"output": "test"})
        manager.add_step_result(execution.execution_id, "step1", result)
        
        # Check step was added
        assert execution.context.has_step_completed("step1")
        assert len(execution.context.completed_steps) == 1
        
        # Update status
        manager.update_execution(execution.execution_id, "completed")
        assert execution.status == "completed"
        
        # Get stats
        stats = manager.get_execution_stats()
        assert stats["total_executions"] == 1
        assert stats["completed"] == 1
        
        print("✅ Context Manager test passed")
        
    finally:
        await manager.stop()


async def test_executor_registry():
    """Test executor registry."""
    print("🧪 Testing Executor Registry...")
    
    registry = ExecutorRegistry()
    
    # Register test executor
    registry.register_executor("test", TestExecutor)
    
    # Check registration
    assert "test" in registry.list_executors()
    
    # Get executor instance
    executor = registry.get_executor("test")
    assert isinstance(executor, TestExecutor)
    
    # Get executor info
    info = registry.get_executor_info("test")
    assert info["name"] == "test"
    assert "TestExecutor" in info["class"]
    
    print("✅ Executor Registry test passed")


async def test_integration():
    """Test integration of all components."""
    print("🧪 Testing Integration...")
    
    # Create temporary flow directory
    with tempfile.TemporaryDirectory() as temp_dir:
        flows_dir = Path(temp_dir)
        flow_dir = flows_dir / "test_flow"
        flow_dir.mkdir()
        
        # Create test flow
        flow_yaml = {
            "name": "test_flow",
            "description": "Integration test flow",
            "inputs": [
                {"name": "message", "type": "string", "default": "Hello Integration!"}
            ],
            "steps": [
                {
                    "name": "process",
                    "executor": "test",
                    "config": {"message": "{{ inputs.message }}"}
                }
            ],
            "outputs": [
                {"name": "result", "value": "{{ steps.process.message }}"}
            ]
        }
        
        # Write flow file
        with open(flow_dir / "flow.yaml", 'w') as f:
            yaml.dump(flow_yaml, f)
        
        # Create flow runner
        runner = FlowRunner(flows_dir)
        
        # Register test executor
        runner.executor_registry.register_executor("test", TestExecutor)
        
        await runner.start()
        
        try:
            # Check flow was loaded
            assert "test_flow" in runner.list_flows()
            
            # Get flow info
            info = runner.get_flow_info("test_flow")
            assert info["name"] == "test_flow"
            assert len(info["steps"]) == 1
            
            # Run flow
            result = await runner.run_flow("test_flow", {"message": "Integration Success!"})
            
            # Validate result
            assert result["success"] == True
            assert result["flow"] == "test_flow"
            assert len(result["steps_completed"]) == 1
            assert result["result"] == "Integration Success!"
            
            print("✅ Integration test passed")
            
        finally:
            await runner.stop()


async def main():
    """Run all tests."""
    print("🚀 Starting Phase 1 Core Framework Tests\n")
    
    try:
        # Test individual components
        await test_base_executor()
        test_yaml_loader()
        test_template_engine()
        await test_context_manager()
        await test_executor_registry()
        
        # Test integration
        await test_integration()
        
        print("\n🎉 All Phase 1 tests passed!")
        print("✅ Core Framework implementation is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
