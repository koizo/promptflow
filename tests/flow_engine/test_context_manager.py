"""
Comprehensive test suite for Context Manager.
Tests flow execution context, state management, and step tracking.
"""

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

from core.flow_engine.context_manager import ContextManager, FlowExecution
from core.executors.base_executor import ExecutionResult, FlowContext


class TestFlowExecution:
    """Test suite for FlowExecution data class."""
    
    def test_flow_execution_creation(self):
        """Test FlowExecution creation with various parameters."""
        execution_id = str(uuid.uuid4())
        flow_name = "test_flow"
        
        execution = FlowExecution(
            execution_id=execution_id,
            flow_name=flow_name,
            status="running",
            inputs={"param1": "value1"},
            outputs={},
            step_results={},
            error=None,
            start_time=datetime.now(timezone.utc),
            end_time=None
        )
        
        assert execution.execution_id == execution_id
        assert execution.flow_name == flow_name
        assert execution.status == "running"
        assert execution.inputs == {"param1": "value1"}
        assert execution.outputs == {}
        assert execution.step_results == {}
        assert execution.error is None
        assert execution.start_time is not None
        assert execution.end_time is None
    
    def test_flow_execution_completion(self):
        """Test FlowExecution completion tracking."""
        execution = FlowExecution(
            execution_id="test-id",
            flow_name="test_flow",
            status="completed",
            inputs={},
            outputs={"result": "success"},
            step_results={"step1": {"success": True}},
            error=None,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc)
        )
        
        assert execution.status == "completed"
        assert execution.outputs == {"result": "success"}
        assert execution.end_time is not None
        assert execution.error is None
    
    def test_flow_execution_failure(self):
        """Test FlowExecution failure tracking."""
        execution = FlowExecution(
            execution_id="test-id",
            flow_name="test_flow",
            status="failed",
            inputs={},
            outputs={},
            step_results={"step1": {"success": False}},
            error="Step execution failed",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc)
        )
        
        assert execution.status == "failed"
        assert execution.error == "Step execution failed"
        assert execution.end_time is not None


class TestContextManager:
    """Test suite for ContextManager class."""
    
    @pytest.fixture
    def context_manager(self):
        """Create ContextManager instance."""
        return ContextManager()
    
    @pytest.fixture
    def sample_inputs(self):
        """Sample input values for testing."""
        return {
            "text": "Hello World",
            "threshold": 0.8,
            "enable_advanced": True,
            "tags": ["test", "example"]
        }
    
    def test_create_execution_context(self, context_manager, sample_inputs):
        """Test creating new execution context."""
        flow_name = "test_flow"
        execution_id = context_manager.create_execution_context(flow_name, sample_inputs)
        
        assert execution_id is not None
        assert isinstance(execution_id, str)
        
        # Verify execution was stored
        execution = context_manager.get_execution(execution_id)
        assert execution is not None
        assert execution.flow_name == flow_name
        assert execution.inputs == sample_inputs
        assert execution.status == "running"
        assert execution.start_time is not None
        assert execution.end_time is None
    
    def test_get_flow_context(self, context_manager, sample_inputs):
        """Test getting FlowContext for execution."""
        flow_name = "test_flow"
        execution_id = context_manager.create_execution_context(flow_name, sample_inputs)
        
        flow_context = context_manager.get_flow_context(execution_id)
        
        assert isinstance(flow_context, FlowContext)
        assert flow_context.execution_id == execution_id
        assert flow_context.inputs == sample_inputs
        assert flow_context.step_results == {}
    
    def test_update_step_result(self, context_manager, sample_inputs):
        """Test updating step execution results."""
        execution_id = context_manager.create_execution_context("test_flow", sample_inputs)
        
        # Create mock execution result
        step_result = ExecutionResult(
            success=True,
            outputs={"text": "processed", "confidence": 0.95},
            error=None
        )
        
        # Update step result
        context_manager.update_step_result(execution_id, "step1", step_result)
        
        # Verify step result was stored
        flow_context = context_manager.get_flow_context(execution_id)
        assert "step1" in flow_context.step_results
        assert flow_context.step_results["step1"] == step_result
        
        # Verify execution was updated
        execution = context_manager.get_execution(execution_id)
        assert "step1" in execution.step_results
    
    def test_update_multiple_step_results(self, context_manager, sample_inputs):
        """Test updating multiple step results."""
        execution_id = context_manager.create_execution_context("test_flow", sample_inputs)
        
        # Add multiple step results
        for i in range(3):
            step_result = ExecutionResult(
                success=True,
                outputs={"result": f"step_{i}_output"},
                error=None
            )
            context_manager.update_step_result(execution_id, f"step_{i}", step_result)
        
        # Verify all steps were stored
        flow_context = context_manager.get_flow_context(execution_id)
        assert len(flow_context.step_results) == 3
        
        for i in range(3):
            assert f"step_{i}" in flow_context.step_results
            assert flow_context.step_results[f"step_{i}"].outputs["result"] == f"step_{i}_output"
    
    def test_complete_execution_success(self, context_manager, sample_inputs):
        """Test completing execution successfully."""
        execution_id = context_manager.create_execution_context("test_flow", sample_inputs)
        
        # Add some step results
        step_result = ExecutionResult(success=True, outputs={"data": "result"})
        context_manager.update_step_result(execution_id, "final_step", step_result)
        
        # Complete execution
        final_outputs = {"final_result": "success"}
        context_manager.complete_execution(execution_id, final_outputs)
        
        # Verify execution completion
        execution = context_manager.get_execution(execution_id)
        assert execution.status == "completed"
        assert execution.outputs == final_outputs
        assert execution.end_time is not None
        assert execution.error is None
    
    def test_fail_execution(self, context_manager, sample_inputs):
        """Test failing execution with error."""
        execution_id = context_manager.create_execution_context("test_flow", sample_inputs)
        
        # Fail execution
        error_message = "Step execution failed"
        context_manager.fail_execution(execution_id, error_message)
        
        # Verify execution failure
        execution = context_manager.get_execution(execution_id)
        assert execution.status == "failed"
        assert execution.error == error_message
        assert execution.end_time is not None
    
    def test_get_execution_nonexistent(self, context_manager):
        """Test getting non-existent execution."""
        result = context_manager.get_execution("nonexistent-id")
        assert result is None
    
    def test_get_flow_context_nonexistent(self, context_manager):
        """Test getting flow context for non-existent execution."""
        with pytest.raises(ValueError, match="Execution not found"):
            context_manager.get_flow_context("nonexistent-id")
    
    def test_update_step_result_nonexistent_execution(self, context_manager):
        """Test updating step result for non-existent execution."""
        step_result = ExecutionResult(success=True, outputs={})
        
        with pytest.raises(ValueError, match="Execution not found"):
            context_manager.update_step_result("nonexistent-id", "step1", step_result)
    
    def test_complete_execution_nonexistent(self, context_manager):
        """Test completing non-existent execution."""
        with pytest.raises(ValueError, match="Execution not found"):
            context_manager.complete_execution("nonexistent-id", {})
    
    def test_fail_execution_nonexistent(self, context_manager):
        """Test failing non-existent execution."""
        with pytest.raises(ValueError, match="Execution not found"):
            context_manager.fail_execution("nonexistent-id", "error")
    
    def test_list_executions_empty(self, context_manager):
        """Test listing executions when none exist."""
        executions = context_manager.list_executions()
        assert executions == []
    
    def test_list_executions_with_data(self, context_manager, sample_inputs):
        """Test listing executions with data."""
        # Create multiple executions
        execution_ids = []
        for i in range(3):
            execution_id = context_manager.create_execution_context(f"flow_{i}", sample_inputs)
            execution_ids.append(execution_id)
        
        # List executions
        executions = context_manager.list_executions()
        assert len(executions) == 3
        
        # Verify all executions are present
        returned_ids = [exec.execution_id for exec in executions]
        for execution_id in execution_ids:
            assert execution_id in returned_ids
    
    def test_list_executions_by_flow_name(self, context_manager, sample_inputs):
        """Test listing executions filtered by flow name."""
        # Create executions for different flows
        flow_a_id = context_manager.create_execution_context("flow_a", sample_inputs)
        flow_b_id = context_manager.create_execution_context("flow_b", sample_inputs)
        flow_a_id2 = context_manager.create_execution_context("flow_a", sample_inputs)
        
        # List executions for specific flow
        flow_a_executions = context_manager.list_executions(flow_name="flow_a")
        assert len(flow_a_executions) == 2
        
        returned_ids = [exec.execution_id for exec in flow_a_executions]
        assert flow_a_id in returned_ids
        assert flow_a_id2 in returned_ids
        assert flow_b_id not in returned_ids
    
    def test_list_executions_by_status(self, context_manager, sample_inputs):
        """Test listing executions filtered by status."""
        # Create executions with different statuses
        running_id = context_manager.create_execution_context("test_flow", sample_inputs)
        
        completed_id = context_manager.create_execution_context("test_flow", sample_inputs)
        context_manager.complete_execution(completed_id, {"result": "success"})
        
        failed_id = context_manager.create_execution_context("test_flow", sample_inputs)
        context_manager.fail_execution(failed_id, "error")
        
        # List by status
        running_executions = context_manager.list_executions(status="running")
        assert len(running_executions) == 1
        assert running_executions[0].execution_id == running_id
        
        completed_executions = context_manager.list_executions(status="completed")
        assert len(completed_executions) == 1
        assert completed_executions[0].execution_id == completed_id
        
        failed_executions = context_manager.list_executions(status="failed")
        assert len(failed_executions) == 1
        assert failed_executions[0].execution_id == failed_id
    
    def test_cleanup_old_executions(self, context_manager, sample_inputs):
        """Test cleanup of old executions."""
        # Create executions
        execution_ids = []
        for i in range(5):
            execution_id = context_manager.create_execution_context(f"flow_{i}", sample_inputs)
            execution_ids.append(execution_id)
        
        # Complete some executions
        context_manager.complete_execution(execution_ids[0], {"result": "success"})
        context_manager.complete_execution(execution_ids[1], {"result": "success"})
        context_manager.fail_execution(execution_ids[2], "error")
        
        # Cleanup completed and failed executions
        cleaned_count = context_manager.cleanup_executions(max_age_hours=0)
        
        # Should clean up completed and failed executions
        assert cleaned_count >= 3
        
        # Running executions should remain
        remaining_executions = context_manager.list_executions()
        running_count = len([e for e in remaining_executions if e.status == "running"])
        assert running_count == 2
    
    def test_get_execution_statistics(self, context_manager, sample_inputs):
        """Test getting execution statistics."""
        # Create executions with different statuses
        for i in range(2):
            execution_id = context_manager.create_execution_context("test_flow", sample_inputs)
            if i == 0:
                context_manager.complete_execution(execution_id, {"result": "success"})
            else:
                context_manager.fail_execution(execution_id, "error")
        
        # Create running execution
        context_manager.create_execution_context("test_flow", sample_inputs)
        
        stats = context_manager.get_execution_statistics()
        
        assert stats["total"] == 3
        assert stats["running"] == 1
        assert stats["completed"] == 1
        assert stats["failed"] == 1
    
    def test_get_execution_statistics_by_flow(self, context_manager, sample_inputs):
        """Test getting execution statistics by flow name."""
        # Create executions for different flows
        flow_a_id = context_manager.create_execution_context("flow_a", sample_inputs)
        context_manager.complete_execution(flow_a_id, {"result": "success"})
        
        flow_b_id = context_manager.create_execution_context("flow_b", sample_inputs)
        context_manager.fail_execution(flow_b_id, "error")
        
        context_manager.create_execution_context("flow_a", sample_inputs)  # Running
        
        # Get stats for specific flow
        flow_a_stats = context_manager.get_execution_statistics(flow_name="flow_a")
        assert flow_a_stats["total"] == 2
        assert flow_a_stats["running"] == 1
        assert flow_a_stats["completed"] == 1
        assert flow_a_stats["failed"] == 0
        
        flow_b_stats = context_manager.get_execution_statistics(flow_name="flow_b")
        assert flow_b_stats["total"] == 1
        assert flow_b_stats["failed"] == 1
    
    def test_concurrent_execution_contexts(self, context_manager, sample_inputs):
        """Test handling multiple concurrent execution contexts."""
        execution_ids = []
        
        # Create multiple concurrent executions
        for i in range(10):
            execution_id = context_manager.create_execution_context(f"flow_{i % 3}", sample_inputs)
            execution_ids.append(execution_id)
        
        # Update step results concurrently
        for i, execution_id in enumerate(execution_ids):
            step_result = ExecutionResult(
                success=True,
                outputs={"step_data": f"data_{i}"},
                error=None
            )
            context_manager.update_step_result(execution_id, f"step_{i}", step_result)
        
        # Verify all executions are properly maintained
        for i, execution_id in enumerate(execution_ids):
            flow_context = context_manager.get_flow_context(execution_id)
            assert f"step_{i}" in flow_context.step_results
            assert flow_context.step_results[f"step_{i}"].outputs["step_data"] == f"data_{i}"
    
    def test_step_result_overwrite(self, context_manager, sample_inputs):
        """Test overwriting step results."""
        execution_id = context_manager.create_execution_context("test_flow", sample_inputs)
        
        # Add initial step result
        initial_result = ExecutionResult(success=True, outputs={"data": "initial"})
        context_manager.update_step_result(execution_id, "step1", initial_result)
        
        # Overwrite with new result
        updated_result = ExecutionResult(success=True, outputs={"data": "updated"})
        context_manager.update_step_result(execution_id, "step1", updated_result)
        
        # Verify result was overwritten
        flow_context = context_manager.get_flow_context(execution_id)
        assert flow_context.step_results["step1"].outputs["data"] == "updated"
    
    def test_execution_timing(self, context_manager, sample_inputs):
        """Test execution timing tracking."""
        execution_id = context_manager.create_execution_context("test_flow", sample_inputs)
        
        # Get initial execution
        initial_execution = context_manager.get_execution(execution_id)
        start_time = initial_execution.start_time
        
        # Complete execution
        context_manager.complete_execution(execution_id, {"result": "success"})
        
        # Verify timing
        final_execution = context_manager.get_execution(execution_id)
        assert final_execution.start_time == start_time
        assert final_execution.end_time is not None
        assert final_execution.end_time >= start_time
    
    def test_large_step_results(self, context_manager, sample_inputs):
        """Test handling large step results."""
        execution_id = context_manager.create_execution_context("test_flow", sample_inputs)
        
        # Create large step result
        large_data = {"items": [f"item_{i}" for i in range(1000)]}
        large_result = ExecutionResult(
            success=True,
            outputs=large_data,
            error=None
        )
        
        # Store large result
        context_manager.update_step_result(execution_id, "large_step", large_result)
        
        # Verify large result was stored correctly
        flow_context = context_manager.get_flow_context(execution_id)
        assert "large_step" in flow_context.step_results
        assert len(flow_context.step_results["large_step"].outputs["items"]) == 1000
    
    def test_execution_context_isolation(self, context_manager, sample_inputs):
        """Test that execution contexts are properly isolated."""
        # Create two separate executions
        execution_id1 = context_manager.create_execution_context("flow_1", {"param": "value1"})
        execution_id2 = context_manager.create_execution_context("flow_2", {"param": "value2"})
        
        # Add different step results to each
        result1 = ExecutionResult(success=True, outputs={"data": "execution1"})
        result2 = ExecutionResult(success=True, outputs={"data": "execution2"})
        
        context_manager.update_step_result(execution_id1, "step1", result1)
        context_manager.update_step_result(execution_id2, "step1", result2)
        
        # Verify contexts are isolated
        context1 = context_manager.get_flow_context(execution_id1)
        context2 = context_manager.get_flow_context(execution_id2)
        
        assert context1.inputs["param"] == "value1"
        assert context2.inputs["param"] == "value2"
        assert context1.step_results["step1"].outputs["data"] == "execution1"
        assert context2.step_results["step1"].outputs["data"] == "execution2"


class TestContextManagerIntegration:
    """Integration tests for ContextManager with realistic scenarios."""
    
    @pytest.fixture
    def context_manager(self):
        return ContextManager()
    
    def test_document_processing_workflow_context(self, context_manager):
        """Test context management for document processing workflow."""
        inputs = {
            "document_file": "test_document.pdf",
            "analysis_level": "comprehensive",
            "languages": ["en", "es"]
        }
        
        execution_id = context_manager.create_execution_context("document_analysis", inputs)
        
        # Simulate OCR step
        ocr_result = ExecutionResult(
            success=True,
            outputs={
                "text": "Document content here...",
                "confidence": 0.95,
                "language": "en",
                "pages": 3
            }
        )
        context_manager.update_step_result(execution_id, "extract_text", ocr_result)
        
        # Simulate sentiment analysis step
        sentiment_result = ExecutionResult(
            success=True,
            outputs={
                "sentiment": "positive",
                "confidence": 0.87,
                "emotions": {"positive": 0.6, "neutral": 0.3, "negative": 0.1}
            }
        )
        context_manager.update_step_result(execution_id, "analyze_sentiment", sentiment_result)
        
        # Simulate LLM analysis step
        llm_result = ExecutionResult(
            success=True,
            outputs={
                "summary": "This document discusses...",
                "key_points": ["Point 1", "Point 2", "Point 3"],
                "recommendations": ["Recommendation 1", "Recommendation 2"]
            }
        )
        context_manager.update_step_result(execution_id, "llm_analysis", llm_result)
        
        # Simulate data combination step
        combined_result = ExecutionResult(
            success=True,
            outputs={
                "document_analysis": {
                    "text": "Document content here...",
                    "sentiment": "positive",
                    "summary": "This document discusses...",
                    "confidence_scores": {
                        "ocr": 0.95,
                        "sentiment": 0.87
                    }
                }
            }
        )
        context_manager.update_step_result(execution_id, "combine_results", combined_result)
        
        # Complete execution
        final_outputs = {
            "analysis_result": combined_result.outputs["document_analysis"],
            "processing_metadata": {
                "steps_completed": 4,
                "total_processing_time": "45.2s",
                "languages_detected": ["en"]
            }
        }
        context_manager.complete_execution(execution_id, final_outputs)
        
        # Verify complete workflow context
        execution = context_manager.get_execution(execution_id)
        assert execution.status == "completed"
        assert len(execution.step_results) == 4
        assert "extract_text" in execution.step_results
        assert "analyze_sentiment" in execution.step_results
        assert "llm_analysis" in execution.step_results
        assert "combine_results" in execution.step_results
        
        # Verify final outputs
        assert "analysis_result" in execution.outputs
        assert "processing_metadata" in execution.outputs
    
    def test_error_recovery_workflow_context(self, context_manager):
        """Test context management for error recovery workflow."""
        inputs = {"data": "test_input", "retry_count": 3}
        execution_id = context_manager.create_execution_context("resilient_flow", inputs)
        
        # Simulate primary processing failure
        primary_result = ExecutionResult(
            success=False,
            outputs={},
            error="Network timeout after 30 seconds"
        )
        context_manager.update_step_result(execution_id, "primary_processing", primary_result)
        
        # Simulate successful fallback processing
        fallback_result = ExecutionResult(
            success=True,
            outputs={
                "result": "fallback_data",
                "source": "cache",
                "confidence": 0.7
            }
        )
        context_manager.update_step_result(execution_id, "fallback_processing", fallback_result)
        
        # Simulate result combination with error handling
        recovery_result = ExecutionResult(
            success=True,
            outputs={
                "final_result": "fallback_data",
                "recovery_info": {
                    "primary_failed": True,
                    "fallback_used": True,
                    "confidence_reduced": True
                }
            }
        )
        context_manager.update_step_result(execution_id, "error_recovery", recovery_result)
        
        # Complete with partial success
        final_outputs = {
            "result": recovery_result.outputs["final_result"],
            "warnings": ["Primary processing failed, used fallback data"],
            "recovery_info": recovery_result.outputs["recovery_info"]
        }
        context_manager.complete_execution(execution_id, final_outputs)
        
        # Verify error recovery context
        execution = context_manager.get_execution(execution_id)
        assert execution.status == "completed"  # Completed despite partial failure
        
        # Verify step results include both failure and recovery
        flow_context = context_manager.get_flow_context(execution_id)
        assert not flow_context.step_results["primary_processing"].success
        assert flow_context.step_results["fallback_processing"].success
        assert flow_context.step_results["error_recovery"].success
    
    def test_parallel_processing_context(self, context_manager):
        """Test context management for parallel processing scenarios."""
        inputs = {"batch_data": ["item1", "item2", "item3"]}
        execution_id = context_manager.create_execution_context("parallel_flow", inputs)
        
        # Simulate parallel processing steps
        parallel_results = []
        for i in range(3):
            result = ExecutionResult(
                success=True,
                outputs={
                    "processed_item": f"processed_item_{i}",
                    "processing_time": f"{i + 1}.2s",
                    "confidence": 0.8 + (i * 0.05)
                }
            )
            context_manager.update_step_result(execution_id, f"parallel_step_{i}", result)
            parallel_results.append(result)
        
        # Simulate aggregation step
        aggregation_result = ExecutionResult(
            success=True,
            outputs={
                "aggregated_results": [r.outputs["processed_item"] for r in parallel_results],
                "total_items": 3,
                "average_confidence": 0.825,
                "total_processing_time": "6.6s"
            }
        )
        context_manager.update_step_result(execution_id, "aggregate_results", aggregation_result)
        
        # Complete execution
        context_manager.complete_execution(execution_id, aggregation_result.outputs)
        
        # Verify parallel processing context
        execution = context_manager.get_execution(execution_id)
        assert len(execution.step_results) == 4  # 3 parallel + 1 aggregation
        
        # Verify all parallel steps are present
        for i in range(3):
            assert f"parallel_step_{i}" in execution.step_results
            assert execution.step_results[f"parallel_step_{i}"]["success"] is True
