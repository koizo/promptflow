"""
Unit tests for BaseExecutor and related components.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from core.executors.base_executor import (
    BaseExecutor, 
    ExecutionResult, 
    FlowContext
)


class MockTestExecutor(BaseExecutor):
    """Test implementation of BaseExecutor for testing."""
    
    async def execute(self, context: FlowContext, config: dict) -> ExecutionResult:
        """Simple test execution."""
        message = config.get("message", "default message")
        delay = config.get("delay", 0)
        
        if delay > 0:
            await asyncio.sleep(delay)
        
        if config.get("should_fail", False):
            return ExecutionResult(
                success=False,
                error="Intentional test failure"
            )
        
        return ExecutionResult(
            success=True,
            outputs={
                "message": message,
                "input_count": len(context.inputs),
                "config_keys": list(config.keys())
            },
            metadata={"test_executor": True}
        )
    
    def get_required_config_keys(self):
        return ["message"]
    
    def get_optional_config_keys(self):
        return ["delay", "should_fail"]


class TestExecutionResult:
    """Test ExecutionResult class."""
    
    def test_successful_result_creation(self):
        """Test creating a successful execution result."""
        result = ExecutionResult(
            success=True,
            outputs={"key": "value"},
            metadata={"time": 1.5}
        )
        
        assert result.success is True
        assert result.outputs == {"key": "value"}
        assert result.error is None
        assert result.metadata == {"time": 1.5}
        assert result.execution_time is None
    
    def test_failed_result_creation(self):
        """Test creating a failed execution result."""
        result = ExecutionResult(
            success=False,
            error="Something went wrong",
            metadata={"attempt": 1}
        )
        
        assert result.success is False
        assert result.outputs == {}
        assert result.error == "Something went wrong"
        assert result.metadata == {"attempt": 1}
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = ExecutionResult(
            success=True,
            outputs={"data": "test"},
            metadata={"source": "test"},
            execution_time=2.5
        )
        
        result_dict = result.to_dict()
        expected = {
            "success": True,
            "outputs": {"data": "test"},
            "error": None,
            "metadata": {"source": "test"},
            "execution_time": 2.5
        }
        
        assert result_dict == expected


class TestFlowContext:
    """Test FlowContext class."""
    
    def test_context_creation(self, sample_flow_context):
        """Test creating a flow context."""
        context = sample_flow_context
        
        assert context.flow_id == "test_flow_123"
        assert context.inputs == {"input_text": "Hello World", "param1": "value1"}
        assert context.step_results == {}
        assert context.completed_steps == []
        assert context.failed_steps == []
    
    def test_add_step_result(self, sample_flow_context):
        """Test adding step results to context."""
        context = sample_flow_context
        
        result = ExecutionResult(success=True, outputs={"data": "test"})
        context.add_step_result("step1", result)
        
        assert "step1" in context.step_results
        assert context.step_results["step1"] == result
    
    def test_get_step_output(self, sample_flow_context):
        """Test getting step output from context."""
        context = sample_flow_context
        
        result = ExecutionResult(success=True, outputs={"message": "hello", "count": 5})
        context.add_step_result("step1", result)
        
        # Test getting existing output
        assert context.get_step_output("step1", "message") == "hello"
        assert context.get_step_output("step1", "count") == 5
        
        # Test getting all outputs
        all_outputs = context.get_step_output("step1")
        assert all_outputs == {"message": "hello", "count": 5}
        
        # Test getting from non-existent step
        with pytest.raises(ValueError):
            context.get_step_output("missing_step", "key")
        
        # Test getting non-existent output key
        with pytest.raises(ValueError):
            context.get_step_output("step1", "missing_key")
    
    def test_context_serialization(self, sample_flow_context):
        """Test context methods."""
        context = sample_flow_context
        result = ExecutionResult(success=True, outputs={"data": "test"})
        context.add_step_result("step1", result)
        
        # Test execution summary
        summary = context.get_execution_summary()
        
        assert summary["flow_name"] == "test_flow_123"  # This is the flow_id from fixture
        assert "step1" in summary["completed_steps"]
        assert summary["total_steps"] == 1
        assert summary["success_rate"] == 1.0


class TestBaseExecutor:
    """Test BaseExecutor abstract class and implementations."""
    
    @pytest.fixture
    def test_executor(self):
        """Create a test executor instance."""
        return MockTestExecutor()
    
    def test_executor_creation(self, test_executor):
        """Test creating an executor instance."""
        assert isinstance(test_executor, BaseExecutor)
        assert test_executor.get_required_config_keys() == ["message"]
        assert "delay" in test_executor.get_optional_config_keys()
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, test_executor, sample_flow_context):
        """Test successful executor execution."""
        config = {"message": "test message", "delay": 0}
        
        result = await test_executor.execute(sample_flow_context, config)
        
        assert result.success is True
        assert result.outputs["message"] == "test message"
        assert result.outputs["input_count"] == 2  # Two inputs in sample context
        assert result.metadata["test_executor"] is True
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_failed_execution(self, test_executor, sample_flow_context):
        """Test failed executor execution."""
        config = {"message": "test", "should_fail": True}
        
        result = await test_executor.execute(sample_flow_context, config)
        
        assert result.success is False
        assert result.error == "Intentional test failure"
        assert result.outputs == {}
    
    @pytest.mark.asyncio
    async def test_execution_with_delay(self, test_executor, sample_flow_context):
        """Test executor execution with delay."""
        config = {"message": "delayed", "delay": 0.1}
        
        start_time = datetime.now()
        result = await test_executor.execute(sample_flow_context, config)
        end_time = datetime.now()
        
        assert result.success is True
        assert (end_time - start_time).total_seconds() >= 0.1
    
    def test_validate_config_success(self, test_executor):
        """Test successful config validation."""
        config = {"message": "required value", "delay": 1}
        
        # Should not raise exception
        test_executor.validate_config(config)
    
    def test_validate_config_missing_required(self, test_executor):
        """Test config validation - base validate_config doesn't check required keys."""
        config = {"delay": 1}  # Missing required "message"
        
        # The base validate_config method doesn't do anything by default
        # The actual validation happens in _safe_execute
        test_executor.validate_config(config)  # Should not raise
    
    @pytest.mark.asyncio
    async def test_safe_execute_missing_required(self, test_executor, sample_flow_context):
        """Test safe execution with missing required keys."""
        config = {"delay": 1}  # Missing required "message"
        
        result = await test_executor._safe_execute(sample_flow_context, config)
        
        assert result.success is False
        assert "Missing required configuration keys" in result.error
    
    def test_validate_config_unknown_keys(self, test_executor):
        """Test config validation with unknown keys - this is allowed by default."""
        config = {
            "message": "required",
            "unknown_key": "value",
            "another_unknown": "value"
        }
        
        # Should not raise exception - unknown keys are allowed by default
        test_executor.validate_config(config)
    
    @pytest.mark.asyncio
    async def test_safe_execute_with_validation(self, test_executor, sample_flow_context):
        """Test safe execute method with validation."""
        config = {"message": "validated execution"}
        
        result = await test_executor._safe_execute(sample_flow_context, config)
        
        assert result.success is True
        assert result.outputs["message"] == "validated execution"
        assert result.execution_time is not None
        assert result.metadata["executor"] == test_executor.name
    
    @pytest.mark.asyncio
    async def test_safe_execute_invalid_config(self, test_executor, sample_flow_context):
        """Test safe execute with invalid config."""
        config = {"invalid_key": "value"}  # Missing required "message"
        
        result = await test_executor._safe_execute(sample_flow_context, config)
        
        assert result.success is False
        assert "Missing required configuration keys" in result.error
    
    @pytest.mark.asyncio
    async def test_safe_execute_with_timing(self, test_executor, sample_flow_context):
        """Test execution timing measurement."""
        config = {"message": "timed execution", "delay": 0.1}
        
        result = await test_executor._safe_execute(sample_flow_context, config)
        
        assert result.success is True
        assert result.execution_time is not None
        assert result.execution_time >= 0.1
    
    def test_get_info(self, test_executor):
        """Test getting executor information."""
        info = test_executor.get_info()
        
        assert info["name"] == "MockTestExecutor"
        assert info["required_config"] == ["message"]
        assert "delay" in info["optional_config"]
        assert "description" in info
    
    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        
        class IncompleteExecutor(BaseExecutor):
            pass
        
        # Should not be able to instantiate incomplete executor
        with pytest.raises(TypeError):
            IncompleteExecutor()


@pytest.mark.asyncio
class TestExecutorIntegration:
    """Integration tests for executor components."""
    
    async def test_multiple_executors_in_sequence(self, sample_flow_context):
        """Test running multiple executors in sequence."""
        executor1 = MockTestExecutor()
        executor2 = MockTestExecutor()
        
        # First execution
        config1 = {"message": "first execution"}
        result1 = await executor1.execute(sample_flow_context, config1)
        sample_flow_context.add_step_result("step1", result1)
        
        # Second execution using first result
        config2 = {"message": f"second execution, first had {result1.outputs['input_count']} inputs"}
        result2 = await executor2.execute(sample_flow_context, config2)
        
        assert result1.success is True
        assert result2.success is True
        assert "first had 2 inputs" in result2.outputs["message"]
    
    async def test_executor_error_handling(self, sample_flow_context):
        """Test executor error handling in sequence."""
        executor1 = MockTestExecutor()
        executor2 = MockTestExecutor()
        
        # First execution fails
        config1 = {"message": "will fail", "should_fail": True}
        result1 = await executor1.execute(sample_flow_context, config1)
        
        # Second execution should still work
        config2 = {"message": "should succeed"}
        result2 = await executor2.execute(sample_flow_context, config2)
        
        assert result1.success is False
        assert result2.success is True
    
    async def test_concurrent_executor_execution(self, sample_flow_context):
        """Test concurrent execution of multiple executors."""
        executor1 = MockTestExecutor()
        executor2 = MockTestExecutor()
        
        config1 = {"message": "concurrent 1", "delay": 0.1}
        config2 = {"message": "concurrent 2", "delay": 0.1}
        
        # Run concurrently
        start_time = datetime.now()
        results = await asyncio.gather(
            executor1.execute(sample_flow_context, config1),
            executor2.execute(sample_flow_context, config2)
        )
        end_time = datetime.now()
        
        # Should complete in roughly 0.1 seconds (concurrent), not 0.2 (sequential)
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 0.15  # Allow some overhead
        
        assert all(result.success for result in results)
        assert results[0].outputs["message"] == "concurrent 1"
        assert results[1].outputs["message"] == "concurrent 2"
