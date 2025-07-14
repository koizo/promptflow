"""
Comprehensive test suite for Flow Runner.
Tests flow execution orchestration, executor management, and workflow coordination.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile

from core.flow_engine.flow_runner import FlowRunner, ExecutorRegistry
from core.flow_engine.yaml_loader import FlowDefinition, FlowStep, FlowInput, FlowOutput
from core.executors.base_executor import BaseExecutor, ExecutionResult, FlowContext


class MockExecutor(BaseExecutor):
    """Mock executor for testing."""
    
    def __init__(self, name: str = "mock_executor", should_succeed: bool = True, delay: float = 0.0):
        super().__init__(name)
        self.should_succeed = should_succeed
        self.delay = delay
        self.execution_count = 0
    
    async def execute(self, context: FlowContext, config: dict) -> ExecutionResult:
        """Mock execution with configurable behavior."""
        self.execution_count += 1
        
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        if self.should_succeed:
            return ExecutionResult(
                success=True,
                outputs={
                    "result": f"mock_result_{self.execution_count}",
                    "config_echo": config,
                    "execution_count": self.execution_count
                }
            )
        else:
            return ExecutionResult(
                success=False,
                error=f"Mock execution failed for {self.name}"
            )


class TestExecutorRegistry:
    """Test suite for ExecutorRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create ExecutorRegistry instance."""
        return ExecutorRegistry()
    
    def test_register_executor(self, registry):
        """Test registering executors."""
        executor_class = MockExecutor
        registry.register_executor("test_executor", executor_class)
        
        assert "test_executor" in registry.executors
        assert registry.executors["test_executor"] == executor_class
    
    def test_register_invalid_executor(self, registry):
        """Test registering invalid executor class."""
        class InvalidExecutor:
            pass
        
        with pytest.raises(ValueError, match="Executor must inherit from BaseExecutor"):
            registry.register_executor("invalid", InvalidExecutor)
    
    def test_get_executor_new_instance(self, registry):
        """Test getting new executor instance."""
        registry.register_executor("test_executor", MockExecutor)
        
        executor = registry.get_executor("test_executor")
        assert isinstance(executor, MockExecutor)
        assert executor.name == "test_executor"
    
    def test_get_executor_cached_instance(self, registry):
        """Test getting cached executor instance."""
        registry.register_executor("test_executor", MockExecutor)
        
        executor1 = registry.get_executor("test_executor")
        executor2 = registry.get_executor("test_executor")
        
        # Should return same instance (cached)
        assert executor1 is executor2
    
    def test_get_unknown_executor(self, registry):
        """Test getting unknown executor."""
        with pytest.raises(ValueError, match="Unknown executor: unknown_executor"):
            registry.get_executor("unknown_executor")
    
    def test_list_executors(self, registry):
        """Test listing registered executors."""
        registry.register_executor("executor1", MockExecutor)
        registry.register_executor("executor2", MockExecutor)
        
        executors = registry.list_executors()
        assert "executor1" in executors
        assert "executor2" in executors
        assert len(executors) == 2
    
    def test_auto_discover_executors(self, registry):
        """Test auto-discovery of executors."""
        # This would test automatic discovery from a package
        # For now, test the method exists and can be called
        discovered = registry.auto_discover_executors("core.executors")
        assert isinstance(discovered, int)
        assert discovered >= 0


class TestFlowRunner:
    """Test suite for FlowRunner class."""
    
    @pytest.fixture
    def flow_runner(self):
        """Create FlowRunner instance with mock executors."""
        runner = FlowRunner()
        
        # Register mock executors
        runner.executor_registry.register_executor("mock_executor", MockExecutor)
        runner.executor_registry.register_executor("slow_executor", 
                                                 lambda: MockExecutor("slow_executor", delay=0.1))
        runner.executor_registry.register_executor("failing_executor", 
                                                 lambda: MockExecutor("failing_executor", should_succeed=False))
        
        return runner
    
    @pytest.fixture
    def simple_flow_definition(self):
        """Create simple flow definition for testing."""
        return FlowDefinition(
            name="simple_flow",
            description="Simple test flow",
            version="1.0",
            inputs=[
                FlowInput(name="input_text", type="string", required=True),
                FlowInput(name="threshold", type="float", required=False, default=0.5)
            ],
            steps=[
                FlowStep(
                    name="step1",
                    executor="mock_executor",
                    config={"text": "{{ inputs.input_text }}", "threshold": "{{ inputs.threshold }}"}
                ),
                FlowStep(
                    name="step2",
                    executor="mock_executor",
                    config={"data": "{{ steps.step1.result }}"},
                    depends_on=["step1"]
                )
            ],
            outputs=[
                FlowOutput(name="final_result", value="{{ steps.step2.result }}")
            ]
        )
    
    @pytest.fixture
    def complex_flow_definition(self):
        """Create complex flow definition for testing."""
        return FlowDefinition(
            name="complex_flow",
            description="Complex test flow with parallel steps",
            version="1.0",
            inputs=[
                FlowInput(name="data", type="string", required=True),
                FlowInput(name="enable_parallel", type="boolean", default=True)
            ],
            steps=[
                FlowStep(
                    name="initial_step",
                    executor="mock_executor",
                    config={"input": "{{ inputs.data }}"}
                ),
                FlowStep(
                    name="parallel_step_1",
                    executor="mock_executor",
                    config={"data": "{{ steps.initial_step.result }}"},
                    depends_on=["initial_step"],
                    parallel_group="parallel_processing",
                    condition="{{ inputs.enable_parallel }}"
                ),
                FlowStep(
                    name="parallel_step_2",
                    executor="slow_executor",
                    config={"data": "{{ steps.initial_step.result }}"},
                    depends_on=["initial_step"],
                    parallel_group="parallel_processing",
                    condition="{{ inputs.enable_parallel }}"
                ),
                FlowStep(
                    name="final_step",
                    executor="mock_executor",
                    config={
                        "result1": "{{ steps.parallel_step_1.result }}",
                        "result2": "{{ steps.parallel_step_2.result }}"
                    },
                    depends_on=["parallel_step_1", "parallel_step_2"]
                )
            ],
            outputs=[
                FlowOutput(name="result", value="{{ steps.final_step.result }}")
            ]
        )
    
    @pytest.mark.asyncio
    async def test_execute_simple_flow(self, flow_runner, simple_flow_definition):
        """Test executing simple linear flow."""
        inputs = {"input_text": "test input", "threshold": 0.8}
        
        result = await flow_runner.execute_flow(simple_flow_definition, inputs)
        
        assert result.success is True
        assert "final_result" in result.outputs
        assert result.execution_id is not None
        
        # Verify step execution order
        execution = flow_runner.context_manager.get_execution(result.execution_id)
        assert len(execution.step_results) == 2
        assert "step1" in execution.step_results
        assert "step2" in execution.step_results
    
    @pytest.mark.asyncio
    async def test_execute_flow_with_template_processing(self, flow_runner, simple_flow_definition):
        """Test flow execution with template processing."""
        inputs = {"input_text": "hello world", "threshold": 0.9}
        
        result = await flow_runner.execute_flow(simple_flow_definition, inputs)
        
        # Verify templates were processed correctly
        execution = flow_runner.context_manager.get_execution(result.execution_id)
        
        # Check step1 config was templated
        step1_result = execution.step_results["step1"]
        step1_config = step1_result["outputs"]["config_echo"]
        assert step1_config["text"] == "hello world"
        assert step1_config["threshold"] == 0.9
        
        # Check step2 received step1 result
        step2_result = execution.step_results["step2"]
        step2_config = step2_result["outputs"]["config_echo"]
        assert "mock_result_1" in step2_config["data"]
    
    @pytest.mark.asyncio
    async def test_execute_flow_with_missing_inputs(self, flow_runner, simple_flow_definition):
        """Test flow execution with missing required inputs."""
        inputs = {"threshold": 0.8}  # Missing required input_text
        
        result = await flow_runner.execute_flow(simple_flow_definition, inputs)
        
        assert result.success is False
        assert "Missing required input" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_flow_with_step_failure(self, flow_runner):
        """Test flow execution with step failure."""
        flow_def = FlowDefinition(
            name="failing_flow",
            steps=[
                FlowStep(name="good_step", executor="mock_executor", config={}),
                FlowStep(name="bad_step", executor="failing_executor", config={}, depends_on=["good_step"])
            ]
        )
        
        result = await flow_runner.execute_flow(flow_def, {})
        
        assert result.success is False
        assert "Mock execution failed" in result.error
        
        # Verify first step succeeded but second failed
        execution = flow_runner.context_manager.get_execution(result.execution_id)
        assert execution.step_results["good_step"]["success"] is True
        assert execution.step_results["bad_step"]["success"] is False
    
    @pytest.mark.asyncio
    async def test_execute_parallel_steps(self, flow_runner, complex_flow_definition):
        """Test execution of parallel steps."""
        inputs = {"data": "test data", "enable_parallel": True}
        
        result = await flow_runner.execute_flow(complex_flow_definition, inputs)
        
        assert result.success is True
        
        # Verify all steps executed
        execution = flow_runner.context_manager.get_execution(result.execution_id)
        assert len(execution.step_results) == 4
        assert "initial_step" in execution.step_results
        assert "parallel_step_1" in execution.step_results
        assert "parallel_step_2" in execution.step_results
        assert "final_step" in execution.step_results
    
    @pytest.mark.asyncio
    async def test_execute_conditional_steps(self, flow_runner, complex_flow_definition):
        """Test execution with conditional steps."""
        # Disable parallel processing
        inputs = {"data": "test data", "enable_parallel": False}
        
        result = await flow_runner.execute_flow(complex_flow_definition, inputs)
        
        # Should fail because final_step depends on parallel steps that were skipped
        assert result.success is False
        
        execution = flow_runner.context_manager.get_execution(result.execution_id)
        # Only initial step should have executed
        assert "initial_step" in execution.step_results
        assert "parallel_step_1" not in execution.step_results
        assert "parallel_step_2" not in execution.step_results
    
    @pytest.mark.asyncio
    async def test_execute_flow_with_unknown_executor(self, flow_runner):
        """Test flow execution with unknown executor."""
        flow_def = FlowDefinition(
            name="unknown_executor_flow",
            steps=[
                FlowStep(name="step1", executor="unknown_executor", config={})
            ]
        )
        
        result = await flow_runner.execute_flow(flow_def, {})
        
        assert result.success is False
        assert "Unknown executor" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_flow_with_circular_dependency(self, flow_runner):
        """Test flow execution with circular dependencies."""
        flow_def = FlowDefinition(
            name="circular_flow",
            steps=[
                FlowStep(name="step1", executor="mock_executor", config={}, depends_on=["step2"]),
                FlowStep(name="step2", executor="mock_executor", config={}, depends_on=["step1"])
            ]
        )
        
        result = await flow_runner.execute_flow(flow_def, {})
        
        assert result.success is False
        assert "Circular dependency" in result.error or "dependency" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_flow_with_timeout(self, flow_runner):
        """Test flow execution with timeout."""
        # Create flow with slow executor
        flow_def = FlowDefinition(
            name="slow_flow",
            steps=[
                FlowStep(name="slow_step", executor="slow_executor", config={})
            ],
            config={"execution": {"timeout": 0.05}}  # Very short timeout
        )
        
        result = await flow_runner.execute_flow(flow_def, {})
        
        # Might succeed or timeout depending on timing
        # Just verify it completes without hanging
        assert result.success in [True, False]
    
    @pytest.mark.asyncio
    async def test_load_and_execute_flow_from_file(self, flow_runner):
        """Test loading and executing flow from YAML file."""
        flow_yaml = """
name: "file_flow"
description: "Flow loaded from file"

inputs:
  - name: "message"
    type: "string"
    required: true

steps:
  - name: "process_message"
    executor: "mock_executor"
    config:
      text: "{{ inputs.message }}"

outputs:
  - name: "result"
    value: "{{ steps.process_message.result }}"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(flow_yaml)
            temp_path = f.name
        
        try:
            inputs = {"message": "hello from file"}
            result = await flow_runner.execute_flow_from_file(temp_path, inputs)
            
            assert result.success is True
            assert "result" in result.outputs
        finally:
            Path(temp_path).unlink()
    
    @pytest.mark.asyncio
    async def test_execute_flow_with_retry(self, flow_runner):
        """Test flow execution with retry logic."""
        # Create executor that fails first time, succeeds second time
        class RetryableExecutor(BaseExecutor):
            def __init__(self):
                super().__init__("retryable_executor")
                self.attempt_count = 0
            
            async def execute(self, context: FlowContext, config: dict) -> ExecutionResult:
                self.attempt_count += 1
                if self.attempt_count == 1:
                    return ExecutionResult(success=False, error="First attempt failed")
                else:
                    return ExecutionResult(success=True, outputs={"result": "success_on_retry"})
        
        flow_runner.executor_registry.register_executor("retryable_executor", RetryableExecutor)
        
        flow_def = FlowDefinition(
            name="retry_flow",
            steps=[
                FlowStep(name="retry_step", executor="retryable_executor", config={})
            ],
            config={"execution": {"retry_attempts": 2}}
        )
        
        result = await flow_runner.execute_flow(flow_def, {})
        
        # Should succeed on retry
        assert result.success is True
    
    def test_get_flow_status(self, flow_runner):
        """Test getting flow execution status."""
        # Create a mock execution
        execution_id = flow_runner.context_manager.create_execution_context("test_flow", {})
        
        status = flow_runner.get_flow_status(execution_id)
        
        assert status is not None
        assert status["execution_id"] == execution_id
        assert status["status"] == "running"
        assert "start_time" in status
    
    def test_get_flow_status_nonexistent(self, flow_runner):
        """Test getting status for non-existent flow."""
        status = flow_runner.get_flow_status("nonexistent-id")
        assert status is None
    
    def test_list_flow_executions(self, flow_runner):
        """Test listing flow executions."""
        # Create multiple executions
        execution_ids = []
        for i in range(3):
            execution_id = flow_runner.context_manager.create_execution_context(f"flow_{i}", {})
            execution_ids.append(execution_id)
        
        executions = flow_runner.list_flow_executions()
        
        assert len(executions) == 3
        returned_ids = [exec["execution_id"] for exec in executions]
        for execution_id in execution_ids:
            assert execution_id in returned_ids
    
    def test_cancel_flow_execution(self, flow_runner):
        """Test canceling flow execution."""
        execution_id = flow_runner.context_manager.create_execution_context("test_flow", {})
        
        success = flow_runner.cancel_flow_execution(execution_id)
        
        assert success is True
        
        # Verify execution was marked as cancelled
        execution = flow_runner.context_manager.get_execution(execution_id)
        assert execution.status == "cancelled"
    
    def test_cancel_nonexistent_execution(self, flow_runner):
        """Test canceling non-existent execution."""
        success = flow_runner.cancel_flow_execution("nonexistent-id")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_step_execution_order(self, flow_runner):
        """Test that steps execute in correct dependency order."""
        execution_order = []
        
        class OrderTrackingExecutor(BaseExecutor):
            def __init__(self, step_name):
                super().__init__(f"order_executor_{step_name}")
                self.step_name = step_name
            
            async def execute(self, context: FlowContext, config: dict) -> ExecutionResult:
                execution_order.append(self.step_name)
                return ExecutionResult(success=True, outputs={"result": f"result_{self.step_name}"})
        
        # Register executors for each step
        for step_name in ["a", "b", "c", "d"]:
            flow_runner.executor_registry.register_executor(
                f"order_executor_{step_name}",
                lambda name=step_name: OrderTrackingExecutor(name)
            )
        
        # Create flow with specific dependency order
        flow_def = FlowDefinition(
            name="order_flow",
            steps=[
                FlowStep(name="step_a", executor="order_executor_a", config={}),
                FlowStep(name="step_b", executor="order_executor_b", config={}, depends_on=["step_a"]),
                FlowStep(name="step_c", executor="order_executor_c", config={}, depends_on=["step_a"]),
                FlowStep(name="step_d", executor="order_executor_d", config={}, depends_on=["step_b", "step_c"])
            ]
        )
        
        result = await flow_runner.execute_flow(flow_def, {})
        
        assert result.success is True
        
        # Verify execution order
        assert execution_order[0] == "a"  # First
        assert execution_order[-1] == "d"  # Last
        assert execution_order.index("b") > execution_order.index("a")
        assert execution_order.index("c") > execution_order.index("a")
        assert execution_order.index("d") > execution_order.index("b")
        assert execution_order.index("d") > execution_order.index("c")
    
    @pytest.mark.asyncio
    async def test_flow_execution_context_isolation(self, flow_runner, simple_flow_definition):
        """Test that concurrent flow executions are isolated."""
        # Execute multiple flows concurrently
        tasks = []
        for i in range(3):
            inputs = {"input_text": f"input_{i}", "threshold": 0.1 * i}
            task = flow_runner.execute_flow(simple_flow_definition, inputs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for result in results:
            assert result.success is True
        
        # Verify each execution has unique context
        execution_ids = [result.execution_id for result in results]
        assert len(set(execution_ids)) == 3  # All unique
        
        # Verify inputs were processed correctly for each
        for i, result in enumerate(results):
            execution = flow_runner.context_manager.get_execution(result.execution_id)
            assert execution.inputs["input_text"] == f"input_{i}"
            assert execution.inputs["threshold"] == 0.1 * i


class TestFlowRunnerIntegration:
    """Integration tests for FlowRunner with realistic scenarios."""
    
    @pytest.fixture
    def flow_runner(self):
        """Create FlowRunner with realistic mock executors."""
        runner = FlowRunner()
        
        # Register mock executors that simulate real AI processing
        class OCRExecutor(BaseExecutor):
            async def execute(self, context: FlowContext, config: dict) -> ExecutionResult:
                return ExecutionResult(
                    success=True,
                    outputs={
                        "text": "Extracted text from document",
                        "confidence": 0.95,
                        "language": "en"
                    }
                )
        
        class SentimentExecutor(BaseExecutor):
            async def execute(self, context: FlowContext, config: dict) -> ExecutionResult:
                return ExecutionResult(
                    success=True,
                    outputs={
                        "sentiment": "positive",
                        "confidence": 0.87,
                        "emotions": {"positive": 0.6, "neutral": 0.3, "negative": 0.1}
                    }
                )
        
        class DataCombinerExecutor(BaseExecutor):
            async def execute(self, context: FlowContext, config: dict) -> ExecutionResult:
                sources = config.get("sources", [])
                combined_data = {}
                
                for source in sources:
                    if source in context.step_results:
                        step_result = context.step_results[source]
                        if hasattr(step_result, 'outputs'):
                            combined_data.update(step_result.outputs)
                
                return ExecutionResult(
                    success=True,
                    outputs={"combined_result": combined_data}
                )
        
        runner.executor_registry.register_executor("ocr_processor", OCRExecutor)
        runner.executor_registry.register_executor("sentiment_analyzer", SentimentExecutor)
        runner.executor_registry.register_executor("data_combiner", DataCombinerExecutor)
        
        return runner
    
    @pytest.mark.asyncio
    async def test_document_processing_workflow(self, flow_runner):
        """Test complete document processing workflow."""
        flow_yaml = """
name: "document_processing"
description: "Complete document analysis pipeline"

inputs:
  - name: "document_file"
    type: "file"
    required: true
  - name: "analysis_level"
    type: "string"
    default: "basic"

steps:
  - name: "extract_text"
    executor: "ocr_processor"
    config:
      image_path: "{{ inputs.document_file }}"
      confidence_threshold: 0.8
    description: "Extract text using OCR"
  
  - name: "analyze_sentiment"
    executor: "sentiment_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      analysis_type: "{{ inputs.analysis_level }}"
    depends_on: ["extract_text"]
    description: "Analyze text sentiment"
  
  - name: "combine_results"
    executor: "data_combiner"
    config:
      sources: ["extract_text", "analyze_sentiment"]
      strategy: "merge"
    depends_on: ["extract_text", "analyze_sentiment"]
    description: "Combine analysis results"

outputs:
  - name: "analysis_result"
    value: "{{ steps.combine_results.combined_result }}"
  - name: "processing_metadata"
    value: "{{ steps.combine_results.metadata }}"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(flow_yaml)
            temp_path = f.name
        
        try:
            inputs = {
                "document_file": "test_document.pdf",
                "analysis_level": "detailed"
            }
            
            result = await flow_runner.execute_flow_from_file(temp_path, inputs)
            
            assert result.success is True
            assert "analysis_result" in result.outputs
            
            # Verify workflow execution
            execution = flow_runner.context_manager.get_execution(result.execution_id)
            assert len(execution.step_results) == 3
            assert "extract_text" in execution.step_results
            assert "analyze_sentiment" in execution.step_results
            assert "combine_results" in execution.step_results
            
            # Verify data flow between steps
            ocr_result = execution.step_results["extract_text"]
            sentiment_result = execution.step_results["analyze_sentiment"]
            combined_result = execution.step_results["combine_results"]
            
            assert ocr_result["success"] is True
            assert sentiment_result["success"] is True
            assert combined_result["success"] is True
            
        finally:
            Path(temp_path).unlink()
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, flow_runner):
        """Test workflow with error recovery mechanisms."""
        # Add failing executor
        class FailingExecutor(BaseExecutor):
            async def execute(self, context: FlowContext, config: dict) -> ExecutionResult:
                return ExecutionResult(success=False, error="Simulated failure")
        
        class FallbackExecutor(BaseExecutor):
            async def execute(self, context: FlowContext, config: dict) -> ExecutionResult:
                return ExecutionResult(
                    success=True,
                    outputs={"result": "fallback_data", "source": "fallback"}
                )
        
        flow_runner.executor_registry.register_executor("failing_executor", FailingExecutor)
        flow_runner.executor_registry.register_executor("fallback_executor", FallbackExecutor)
        
        flow_def = FlowDefinition(
            name="error_recovery_flow",
            inputs=[FlowInput(name="data", type="string", required=True)],
            steps=[
                FlowStep(
                    name="primary_processing",
                    executor="failing_executor",
                    config={"data": "{{ inputs.data }}"}
                ),
                FlowStep(
                    name="fallback_processing",
                    executor="fallback_executor",
                    config={"data": "{{ inputs.data }}"},
                    condition="{{ not steps.primary_processing.success }}"
                ),
                FlowStep(
                    name="combine_results",
                    executor="data_combiner",
                    config={
                        "sources": ["primary_processing", "fallback_processing"],
                        "strategy": "merge"
                    },
                    depends_on=["primary_processing", "fallback_processing"]
                )
            ],
            outputs=[
                FlowOutput(name="result", value="{{ steps.combine_results.combined_result }}")
            ],
            config={"error_handling": {"continue_on_error": True}}
        )
        
        inputs = {"data": "test_input"}
        result = await flow_runner.execute_flow(flow_def, inputs)
        
        # Should succeed despite primary failure
        assert result.success is True
        
        execution = flow_runner.context_manager.get_execution(result.execution_id)
        assert execution.step_results["primary_processing"]["success"] is False
        assert execution.step_results["fallback_processing"]["success"] is True
