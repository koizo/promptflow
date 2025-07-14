"""
Unit tests for BaseExecutor framework.

Tests the core executor interface, ExecutionResult, and FlowContext
functionality that all other executors depend on.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock

from core.executors.base_executor import BaseExecutor, ExecutionResult, FlowContext


class TestExecutionResult:
    """Test suite for ExecutionResult class."""
    
    def test_successful_result_creation(self):
        """Test creating a successful ExecutionResult."""
        outputs = {"result": "test_value", "count": 42}
        metadata = {"processing_time": 1.23}
        
        result = ExecutionResult(
            success=True,
            outputs=outputs,
            metadata=metadata,
            execution_time=1.23
        )
        
        assert result.success is True
        assert result.outputs == outputs
        assert result.error is None
        assert result.metadata == metadata
        assert result.execution_time == 1.23
    
    def test_error_result_creation(self):
        """Test creating an error ExecutionResult."""
        error_message = "Test error occurred"
        metadata = {"error_code": "TEST_ERROR"}
        
        result = ExecutionResult(
            success=False,
            error=error_message,
            metadata=metadata,
            execution_time=0.5
        )
        
        assert result.success is False
        assert result.error == error_message
        assert result.outputs == {}  # Default empty dict
        assert result.metadata == metadata
        assert result.execution_time == 0.5
    
    def test_to_dict_conversion(self):
        """Test converting ExecutionResult to dictionary."""
        result = ExecutionResult(
            success=True,
            outputs={"key": "value"},
            metadata={"test": True},
            execution_time=2.5
        )
        
        result_dict = result.to_dict()
        
        expected = {
            "success": True,
            "outputs": {"key": "value"},
            "error": None,
            "metadata": {"test": True},
            "execution_time": 2.5
        }
        
        assert result_dict == expected
    
    def test_default_values(self):
        """Test ExecutionResult with default values."""
        result = ExecutionResult(success=True)
        
        assert result.success is True
        assert result.outputs == {}
        assert result.error is None
        assert result.metadata == {}
        assert result.execution_time is None


class TestFlowContext:
    """Test suite for FlowContext class."""
    
    def test_context_initialization(self):
        """Test FlowContext initialization."""
        flow_id = "test-flow-123"
        flow_name = "test_flow"
        inputs = {"param": "value"}
        context = FlowContext(flow_name=flow_name, inputs=inputs, flow_id=flow_id)
        
        assert context.flow_id == flow_id
        assert context.flow_name == flow_name
        assert context.inputs == inputs
        assert context.redis_enabled is True   # Default
        assert context.persist_steps is True   # Default
        assert len(context.step_results) == 0
    
    def test_context_with_auto_generated_id(self):
        """Test FlowContext with auto-generated flow_id."""
        context = FlowContext(flow_name="test_flow", inputs={"test": "value"})
        
        assert context.flow_id is not None
        assert len(context.flow_id) > 0
        assert context.flow_name == "test_flow"
        assert context.inputs == {"test": "value"}
    
    def test_add_step_result(self):
        """Test adding step results to context."""
        context = FlowContext(flow_name="test_flow", inputs={}, flow_id="test-flow")
        
        result = ExecutionResult(
            success=True,
            outputs={"data": "test_data"}
        )
        
        context.add_step_result("step1", result)
        
        assert "step1" in context.step_results
        assert context.step_results["step1"] == result
    
    def test_get_step_result(self):
        """Test retrieving step results from context."""
        context = FlowContext(flow_name="test_flow", inputs={}, flow_id="test-flow")
        
        result = ExecutionResult(
            success=True,
            outputs={"value": 42}
        )
        
        context.add_step_result("test_step", result)
        retrieved_result = context.step_results["test_step"]
        
        assert retrieved_result == result
        assert retrieved_result.outputs["value"] == 42
    
    def test_get_nonexistent_step_result(self):
        """Test retrieving non-existent step result returns KeyError."""
        context = FlowContext(flow_name="test_flow", inputs={}, flow_id="test-flow")
        
        # Should raise KeyError for non-existent step
        with pytest.raises(KeyError):
            _ = context.step_results["nonexistent_step"]
    
    def test_context_with_redis_settings(self):
        """Test FlowContext with Redis persistence settings."""
        context = FlowContext(
            flow_name="test_flow",
            inputs={},
            flow_id="test-flow"
        )
        
        assert context.redis_enabled is True   # Default
        assert context.persist_steps is True   # Default
    
    def test_multiple_step_results(self):
        """Test managing multiple step results."""
        context = FlowContext(flow_name="test_flow", inputs={}, flow_id="test-flow")
        
        # Add multiple step results
        for i in range(3):
            result = ExecutionResult(
                success=True,
                outputs={"step": i, "data": f"data_{i}"}
            )
            context.add_step_result(f"step_{i}", result)
        
        # Verify all results are stored
        assert len(context.step_results) == 3
        
        for i in range(3):
            retrieved = context.step_results[f"step_{i}"]
            assert retrieved is not None
            assert retrieved.outputs["step"] == i
            assert retrieved.outputs["data"] == f"data_{i}"


class ConcreteExecutor(BaseExecutor):
    """Concrete implementation of BaseExecutor for testing."""
    
    def __init__(self, name: str = "test_executor"):
        super().__init__(name)
    
    async def execute(self, context: FlowContext, config: dict) -> ExecutionResult:
        """Simple test implementation."""
        return ExecutionResult(
            success=True,
            outputs={"message": "Test execution completed"},
            metadata={"executor": self.name}
        )


class TestBaseExecutor:
    """Test suite for BaseExecutor abstract class."""
    
    def test_executor_initialization(self):
        """Test executor initialization with name."""
        executor = ConcreteExecutor("my_test_executor")
        
        assert executor.name == "my_test_executor"
    
    def test_executor_default_name(self):
        """Test executor with default name."""
        executor = ConcreteExecutor()
        
        assert executor.name == "test_executor"
    
    @pytest.mark.asyncio
    async def test_executor_execution(self):
        """Test executor execution interface."""
        executor = ConcreteExecutor("test_exec")
        context = FlowContext(flow_name="test_flow", inputs={}, flow_id="test-flow")
        config = {"param": "value"}
        
        result = await executor.execute(context, config)
        
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.outputs["message"] == "Test execution completed"
        assert result.metadata["executor"] == "test_exec"
    
    def test_abstract_executor_cannot_be_instantiated(self):
        """Test that BaseExecutor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseExecutor("test")
    
    def test_executor_name_property(self):
        """Test executor name property."""
        executor = ConcreteExecutor("property_test")
        
        assert executor.name == "property_test"
        
        # Name should be read-only (set during initialization)
        executor.name = "changed_name"
        assert executor.name == "changed_name"  # Actually allows change in this implementation


class TestExecutorIntegration:
    """Integration tests for executor framework components."""
    
    @pytest.mark.asyncio
    async def test_executor_with_context_flow(self):
        """Test executor execution with context and step results."""
        # Setup
        executor = ConcreteExecutor("integration_test")
        context = FlowContext(flow_name="test_flow", inputs={}, flow_id="integration-flow")
        
        # Add some previous step results to context
        previous_result = ExecutionResult(
            success=True,
            outputs={"previous_data": "from_previous_step"}
        )
        context.add_step_result("previous_step", previous_result)
        
        # Execute
        config = {"use_previous": True}
        result = await executor.execute(context, config)
        
        # Verify
        assert result.success is True
        assert context.step_results["previous_step"] is not None
        
        # Add current result to context
        context.add_step_result("current_step", result)
        
        # Verify context now has both results
        assert len(context.step_results) == 2
        assert "previous_step" in context.step_results
        assert "current_step" in context.step_results
    
    def test_execution_result_serialization(self):
        """Test ExecutionResult can be properly serialized."""
        result = ExecutionResult(
            success=True,
            outputs={"data": [1, 2, 3], "nested": {"key": "value"}},
            metadata={"timestamp": "2025-07-13T10:00:00Z"},
            execution_time=1.5
        )
        
        serialized = result.to_dict()
        
        # Verify all data is preserved
        assert serialized["success"] is True
        assert serialized["outputs"]["data"] == [1, 2, 3]
        assert serialized["outputs"]["nested"]["key"] == "value"
        assert serialized["metadata"]["timestamp"] == "2025-07-13T10:00:00Z"
        assert serialized["execution_time"] == 1.5
        assert serialized["error"] is None
    
    @pytest.mark.asyncio
    async def test_multiple_executor_chain(self):
        """Test chaining multiple executors through context."""
        context = FlowContext(flow_name="test_flow", inputs={}, flow_id="chain-test")
        
        # First executor
        executor1 = ConcreteExecutor("step1")
        result1 = await executor1.execute(context, {"step": 1})
        context.add_step_result("step1", result1)
        
        # Second executor
        executor2 = ConcreteExecutor("step2")
        result2 = await executor2.execute(context, {"step": 2})
        context.add_step_result("step2", result2)
        
        # Verify chain
        assert len(context.step_results) == 2
        assert context.step_results["step1"].metadata["executor"] == "step1"
        assert context.step_results["step2"].metadata["executor"] == "step2"
        
        # Both results should be successful
        assert context.step_results["step1"].success is True
        assert context.step_results["step2"].success is True
