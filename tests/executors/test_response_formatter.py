"""
Simplified unit tests for ResponseFormatter executor.

Tests the actual ResponseFormatter implementation behavior.
"""

import pytest
import uuid
from unittest.mock import Mock

from core.executors.response_formatter import ResponseFormatter
from core.executors.base_executor import FlowContext, ExecutionResult


class TestResponseFormatterSimple:
    """Simplified test suite for ResponseFormatter executor."""
    
    @pytest.fixture
    def response_formatter(self):
        """Create ResponseFormatter instance for testing."""
        return ResponseFormatter()
    
    @pytest.fixture
    def mock_context(self):
        """Create mock FlowContext for testing."""
        return FlowContext(
            flow_name="test_flow",
            inputs={"test_param": "test_value"},
            flow_id=f"test-flow-{uuid.uuid4().hex[:8]}"
        )
    
    @pytest.mark.asyncio
    async def test_standard_template_formatting(self, response_formatter, mock_context):
        """Test standard template formatting."""
        config = {"template": "standard"}
        result = await response_formatter.execute(mock_context, config)
        
        assert result.success is True
        assert "flow" in result.outputs
        assert "success" in result.outputs
        assert result.outputs["flow"] == "test_flow"
        assert result.outputs["success"] is True
    
    @pytest.mark.asyncio
    async def test_minimal_template_formatting(self, response_formatter, mock_context):
        """Test minimal template formatting."""
        config = {"template": "minimal"}
        result = await response_formatter.execute(mock_context, config)
        
        assert result.success is True
        assert "flow" in result.outputs
        assert "success" in result.outputs
        assert result.outputs["flow"] == "test_flow"
    
    @pytest.mark.asyncio
    async def test_detailed_template_formatting(self, response_formatter, mock_context):
        """Test detailed template formatting."""
        config = {"template": "detailed"}
        result = await response_formatter.execute(mock_context, config)
        
        assert result.success is True
        assert "flow" in result.outputs
        assert "step_details" in result.outputs
        assert result.outputs["flow"] == "test_flow"
    
    @pytest.mark.asyncio
    async def test_custom_fields_inclusion(self, response_formatter, mock_context):
        """Test custom fields inclusion."""
        config = {
            "template": "standard",
            "custom_fields": {
                "custom_field": "custom_value",
                "environment": "test"
            }
        }
        result = await response_formatter.execute(mock_context, config)
        
        assert result.success is True
        assert "custom_field" in result.outputs
        assert "environment" in result.outputs
        assert result.outputs["custom_field"] == "custom_value"
        assert result.outputs["environment"] == "test"
    
    @pytest.mark.asyncio
    async def test_metadata_inclusion(self, response_formatter, mock_context):
        """Test metadata inclusion."""
        config = {
            "template": "standard",
            "include_metadata": True
        }
        result = await response_formatter.execute(mock_context, config)
        
        assert result.success is True
        assert "metadata" in result.outputs
        assert isinstance(result.outputs["metadata"], dict)
    
    @pytest.mark.asyncio
    async def test_steps_inclusion(self, response_formatter, mock_context):
        """Test steps inclusion."""
        config = {
            "template": "standard",
            "include_steps": True
        }
        result = await response_formatter.execute(mock_context, config)
        
        assert result.success is True
        assert "steps" in result.outputs
        assert isinstance(result.outputs["steps"], dict)
    
    @pytest.mark.asyncio
    async def test_success_message(self, response_formatter, mock_context):
        """Test success message inclusion."""
        config = {
            "template": "standard",
            "success_message": "Flow completed successfully"
        }
        result = await response_formatter.execute(mock_context, config)
        
        assert result.success is True
        assert "message" in result.outputs
        assert result.outputs["message"] == "Flow completed successfully"
    
    @pytest.mark.asyncio
    async def test_default_configuration(self, response_formatter, mock_context):
        """Test response formatter with default configuration."""
        config = {}  # Empty config should use defaults
        result = await response_formatter.execute(mock_context, config)
        
        assert result.success is True
        assert "flow" in result.outputs
        assert "success" in result.outputs
    
    def test_response_formatter_name(self, response_formatter):
        """Test ResponseFormatter name property."""
        assert response_formatter.name == "ResponseFormatter"
    
    @pytest.mark.asyncio
    async def test_with_step_results(self, response_formatter):
        """Test ResponseFormatter with step results in context."""
        context = FlowContext(
            flow_name="test_flow_with_steps",
            inputs={"input": "test"},
            flow_id="test-flow-123"
        )
        
        # Add some step results
        step_result = ExecutionResult(
            success=True,
            outputs={"result": "step_output"}
        )
        context.add_step_result("test_step", step_result)
        
        config = {"template": "detailed"}
        result = await response_formatter.execute(context, config)
        
        assert result.success is True
        assert "step_details" in result.outputs
        assert "test_step" in result.outputs["step_details"]
    
    @pytest.mark.asyncio
    async def test_execution_metadata(self, response_formatter, mock_context):
        """Test that execution metadata is recorded."""
        config = {"template": "standard"}
        result = await response_formatter.execute(mock_context, config)
        
        assert result.success is True
        assert result.metadata is not None
        assert "formatter_template" in result.metadata
        assert result.metadata["formatter_template"] == "standard"
