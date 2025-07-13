"""
Unit tests for Flow Runner and Executor Registry.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from core.flow_engine.flow_runner import (
    FlowRunner,
    ExecutorRegistry,
    FlowExecutionError
)
from core.flow_engine.yaml_loader import FlowDefinition, FlowStep, FlowInput, FlowOutput
from core.executors.base_executor import BaseExecutor, ExecutionResult, FlowContext
from core.state_store import StateStore


class MockExecutor(BaseExecutor):
    """Mock executor for testing."""
    
    def __init__(self, name="mock_executor", should_fail=False, delay=0):
        self.name = name
        self.should_fail = should_fail
        self.delay = delay
        self.execution_count = 0
    
    async def execute(self, context: FlowContext, config: dict) -> ExecutionResult:
        """Mock execution."""
        self.execution_count += 1
        
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        if self.should_fail:
            return ExecutionResult(
                success=False,
                error=f"Mock executor {self.name} failed intentionally"
            )
        
        message = config.get("message", f"Output from {self.name}")
        return ExecutionResult(
            success=True,
            outputs={
                "message": message,
                "execution_count": self.execution_count,
                "config_received": config
            },
            metadata={"executor_name": self.name}
        )
    
    def get_required_config_keys(self):
        return []
    
    def get_optional_config_keys(self):
        return ["message"]


class TestExecutorRegistry:
    """Test ExecutorRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create an executor registry."""
        return ExecutorRegistry()
    
    def test_registry_creation(self, registry):
        """Test creating an executor registry."""
        assert isinstance(registry, ExecutorRegistry)
        assert len(registry.executors) == 0
        assert len(registry.instances) == 0
    
    def test_register_executor(self, registry):
        """Test registering an executor."""
        registry.register_executor("test_executor", MockExecutor)
        
        assert "test_executor" in registry.executors
        assert registry.executors["test_executor"] == MockExecutor
    
    def test_register_invalid_executor(self, registry):
        """Test registering invalid executor class."""
        class NotAnExecutor:
            pass
        
        with pytest.raises(ValueError) as exc_info:
            registry.register_executor("invalid", NotAnExecutor)
        
        assert "must inherit from BaseExecutor" in str(exc_info.value)
    
    def test_get_executor(self, registry):
        """Test getting executor instance."""
        registry.register_executor("test_executor", MockExecutor)
        
        executor = registry.get_executor("test_executor")
        
        assert isinstance(executor, MockExecutor)
        assert executor.name == "mock_executor"
    
    def test_get_executor_caching(self, registry):
        """Test that executor instances are cached."""
        registry.register_executor("test_executor", MockExecutor)
        
        executor1 = registry.get_executor("test_executor")
        executor2 = registry.get_executor("test_executor")
        
        # Should return the same instance
        assert executor1 is executor2
    
    def test_get_unknown_executor(self, registry):
        """Test getting unknown executor."""
        with pytest.raises(ValueError) as exc_info:
            registry.get_executor("unknown_executor")
        
        assert "Unknown executor: unknown_executor" in str(exc_info.value)
    
    def test_list_executors(self, registry):
        """Test listing registered executors."""
        registry.register_executor("executor1", MockExecutor)
        registry.register_executor("executor2", MockExecutor)
        
        executors = registry.list_executors()
        
        assert "executor1" in executors
        assert "executor2" in executors
        assert len(executors) == 2
    
    def test_has_executor(self, registry):
        """Test checking if executor exists."""
        registry.register_executor("test_executor", MockExecutor)
        
        assert registry.has_executor("test_executor") is True
        assert registry.has_executor("unknown_executor") is False


class TestFlowRunner:
    """Test FlowRunner class."""
    
    @pytest.fixture
    def mock_state_store(self):
        """Create a mock state store."""
        store = Mock(spec=StateStore)
        store.save_flow_state = AsyncMock()
        store.get_flow_state = AsyncMock()
        store.save_step_result = AsyncMock()
        store.get_step_result = AsyncMock()
        return store
    
    @pytest.fixture
    def flow_runner(self, mock_state_store):
        """Create a flow runner with mocked dependencies."""
        runner = FlowRunner(state_store=mock_state_store)
        
        # Register mock executors
        runner.executor_registry.register_executor("mock_executor", MockExecutor)
        runner.executor_registry.register_executor("failing_executor", 
                                                 lambda: MockExecutor("failing", should_fail=True))
        runner.executor_registry.register_executor("slow_executor",
                                                 lambda: MockExecutor("slow", delay=0.1))
        
        return runner
    
    @pytest.fixture
    def simple_flow(self):
        """Create a simple flow definition."""
        return FlowDefinition(
            name="simple_flow",
            description="A simple test flow",
            inputs=[
                FlowInput(name="input_text", type="string", required=True)
            ],
            steps=[
                FlowStep(
                    name="step1",
                    executor="mock_executor",
                    config={"message": "{{ inputs.input_text }}"}
                )
            ],
            outputs=[
                FlowOutput(name="result", value="{{ steps.step1.message }}")
            ]
        )
    
    @pytest.fixture
    def complex_flow(self):
        """Create a complex flow with dependencies."""
        return FlowDefinition(
            name="complex_flow",
            description="A complex test flow with dependencies",
            inputs=[
                FlowInput(name="input1", type="string", required=True),
                FlowInput(name="input2", type="string", required=False, default="default")
            ],
            steps=[
                FlowStep(
                    name="step1",
                    executor="mock_executor",
                    config={"message": "Processing {{ inputs.input1 }}"}
                ),
                FlowStep(
                    name="step2",
                    executor="mock_executor",
                    config={"message": "Result from step1: {{ steps.step1.message }}"},
                    depends_on=["step1"]
                ),
                FlowStep(
                    name="step3",
                    executor="mock_executor",
                    config={"message": "Final: {{ steps.step2.message }}"},
                    depends_on=["step2"]
                )
            ],
            outputs=[
                FlowOutput(name="final_result", value="{{ steps.step3.message }}")
            ]
        )
    
    def test_flow_runner_creation(self, flow_runner):
        """Test creating a flow runner."""
        assert isinstance(flow_runner, FlowRunner)
        assert isinstance(flow_runner.executor_registry, ExecutorRegistry)
        assert flow_runner.template_engine is not None
        assert flow_runner.yaml_loader is not None
    
    @pytest.mark.asyncio
    async def test_execute_simple_flow(self, flow_runner, simple_flow):
        """Test executing a simple flow."""
        inputs = {"input_text": "Hello World"}
        
        result = await flow_runner.execute_flow(simple_flow, inputs)
        
        assert result.success is True
        assert result.outputs["result"] == "Hello World"
        assert "step1" in result.step_results
        assert result.step_results["step1"].success is True
    
    @pytest.mark.asyncio
    async def test_execute_complex_flow(self, flow_runner, complex_flow):
        """Test executing a complex flow with dependencies."""
        inputs = {"input1": "test data", "input2": "optional"}
        
        result = await flow_runner.execute_flow(complex_flow, inputs)
        
        assert result.success is True
        assert len(result.step_results) == 3
        
        # Check execution order and data flow
        step1_result = result.step_results["step1"]
        step2_result = result.step_results["step2"]
        step3_result = result.step_results["step3"]
        
        assert "Processing test data" in step1_result.outputs["message"]
        assert "Result from step1" in step2_result.outputs["message"]
        assert "Final:" in step3_result.outputs["message"]
    
    @pytest.mark.asyncio
    async def test_execute_flow_with_failure(self, flow_runner):
        """Test executing flow with failing step."""
        failing_flow = FlowDefinition(
            name="failing_flow",
            steps=[
                FlowStep(name="good_step", executor="mock_executor", config={}),
                FlowStep(name="bad_step", executor="failing_executor", config={})
            ]
        )
        
        result = await flow_runner.execute_flow(failing_flow, {})
        
        assert result.success is False
        assert "good_step" in result.step_results
        assert "bad_step" in result.step_results
        assert result.step_results["good_step"].success is True
        assert result.step_results["bad_step"].success is False
    
    @pytest.mark.asyncio
    async def test_execute_flow_with_missing_executor(self, flow_runner):
        """Test executing flow with unknown executor."""
        invalid_flow = FlowDefinition(
            name="invalid_flow",
            steps=[
                FlowStep(name="invalid_step", executor="unknown_executor", config={})
            ]
        )
        
        result = await flow_runner.execute_flow(invalid_flow, {})
        
        assert result.success is False
        assert "Unknown executor" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_flow_with_template_error(self, flow_runner):
        """Test executing flow with template rendering error."""
        template_error_flow = FlowDefinition(
            name="template_error_flow",
            steps=[
                FlowStep(
                    name="template_step",
                    executor="mock_executor",
                    config={"message": "{{ nonexistent.variable }}"}
                )
            ]
        )
        
        result = await flow_runner.execute_flow(template_error_flow, {})
        
        assert result.success is False
        assert "template" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_step(self, flow_runner, simple_flow):
        """Test executing a single step."""
        step = simple_flow.steps[0]
        context = FlowContext("test_flow", {"input_text": "test"})
        
        result = await flow_runner.execute_step(step, context)
        
        assert result.success is True
        assert result.outputs["message"] == "test"
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, flow_runner):
        """Test parallel execution of independent steps."""
        parallel_flow = FlowDefinition(
            name="parallel_flow",
            steps=[
                FlowStep(name="parallel1", executor="slow_executor", config={}, parallel_group="group1"),
                FlowStep(name="parallel2", executor="slow_executor", config={}, parallel_group="group1"),
                FlowStep(name="sequential", executor="mock_executor", config={}, depends_on=["parallel1", "parallel2"])
            ]
        )
        
        start_time = datetime.now()
        result = await flow_runner.execute_flow(parallel_flow, {})
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        
        assert result.success is True
        # Should complete in roughly 0.1 seconds (parallel) + overhead, not 0.2 (sequential)
        assert execution_time < 0.15
        assert len(result.step_results) == 3
    
    @pytest.mark.asyncio
    async def test_conditional_execution(self, flow_runner):
        """Test conditional step execution."""
        conditional_flow = FlowDefinition(
            name="conditional_flow",
            inputs=[FlowInput(name="enable_step", type="boolean", required=True)],
            steps=[
                FlowStep(name="always_run", executor="mock_executor", config={}),
                FlowStep(
                    name="conditional_step",
                    executor="mock_executor",
                    config={},
                    condition="{{ inputs.enable_step }}"
                )
            ]
        )
        
        # Test with condition true
        result_true = await flow_runner.execute_flow(conditional_flow, {"enable_step": True})
        assert result_true.success is True
        assert len(result_true.step_results) == 2
        
        # Test with condition false
        result_false = await flow_runner.execute_flow(conditional_flow, {"enable_step": False})
        assert result_false.success is True
        assert len(result_false.step_results) == 1  # Only always_run should execute
        assert "conditional_step" not in result_false.step_results
    
    @pytest.mark.asyncio
    async def test_flow_state_persistence(self, flow_runner, simple_flow, mock_state_store):
        """Test flow state persistence."""
        inputs = {"input_text": "test"}
        
        await flow_runner.execute_flow(simple_flow, inputs)
        
        # Verify state store was called
        mock_state_store.save_flow_state.assert_called()
        mock_state_store.save_step_result.assert_called()
    
    @pytest.mark.asyncio
    async def test_resume_flow_execution(self, flow_runner, complex_flow, mock_state_store):
        """Test resuming flow execution from saved state."""
        # Mock existing state
        mock_state_store.get_flow_state.return_value = {
            "flow_id": "test_flow",
            "status": "running",
            "completed_steps": ["step1"],
            "step_results": {
                "step1": {
                    "success": True,
                    "outputs": {"message": "Previous result"}
                }
            }
        }
        
        inputs = {"input1": "test"}
        
        # This would typically resume from step2
        result = await flow_runner.execute_flow(complex_flow, inputs, flow_id="test_flow")
        
        # Verify it attempted to get existing state
        mock_state_store.get_flow_state.assert_called_with("test_flow")
    
    def test_get_execution_order(self, flow_runner, complex_flow):
        """Test getting execution order for flow steps."""
        execution_order = flow_runner.get_execution_order(complex_flow)
        
        step1_index = execution_order.index("step1")
        step2_index = execution_order.index("step2")
        step3_index = execution_order.index("step3")
        
        assert step1_index < step2_index < step3_index
    
    def test_validate_flow_inputs(self, flow_runner, simple_flow):
        """Test flow input validation."""
        # Valid inputs
        valid_inputs = {"input_text": "test"}
        flow_runner.validate_flow_inputs(simple_flow, valid_inputs)  # Should not raise
        
        # Missing required input
        invalid_inputs = {}
        with pytest.raises(FlowExecutionError) as exc_info:
            flow_runner.validate_flow_inputs(simple_flow, invalid_inputs)
        
        assert "Required input" in str(exc_info.value)
    
    def test_apply_input_defaults(self, flow_runner):
        """Test applying default values to inputs."""
        flow_with_defaults = FlowDefinition(
            name="defaults_flow",
            inputs=[
                FlowInput(name="required", type="string", required=True),
                FlowInput(name="optional", type="string", required=False, default="default_value"),
                FlowInput(name="optional_no_default", type="string", required=False)
            ],
            steps=[]
        )
        
        inputs = {"required": "test"}
        result_inputs = flow_runner.apply_input_defaults(flow_with_defaults, inputs)
        
        assert result_inputs["required"] == "test"
        assert result_inputs["optional"] == "default_value"
        assert "optional_no_default" not in result_inputs
    
    @pytest.mark.asyncio
    async def test_flow_timeout(self, flow_runner):
        """Test flow execution timeout."""
        # Create a flow with very slow executor
        slow_flow = FlowDefinition(
            name="slow_flow",
            steps=[
                FlowStep(name="slow_step", executor="slow_executor", config={})
            ]
        )
        
        # Set a very short timeout
        with patch.object(flow_runner, 'execution_timeout', 0.05):
            result = await flow_runner.execute_flow(slow_flow, {})
            
            # Should timeout and fail
            assert result.success is False
            assert "timeout" in result.error.lower()
    
    def test_get_flow_info(self, flow_runner, simple_flow):
        """Test getting flow information."""
        info = flow_runner.get_flow_info(simple_flow)
        
        assert info["name"] == "simple_flow"
        assert info["description"] == "A simple test flow"
        assert len(info["inputs"]) == 1
        assert len(info["steps"]) == 1
        assert len(info["outputs"]) == 1
        assert "executors_required" in info


class TestFlowExecutionError:
    """Test FlowExecutionError exception."""
    
    def test_flow_execution_error_creation(self):
        """Test creating flow execution error."""
        error = FlowExecutionError("Flow execution failed")
        
        assert str(error) == "Flow execution failed"
        assert isinstance(error, Exception)
    
    def test_flow_execution_error_with_context(self):
        """Test flow execution error with context."""
        error = FlowExecutionError("Step failed", step_name="step1", flow_name="test_flow")
        
        error_msg = str(error)
        assert "Step failed" in error_msg


@pytest.mark.integration
class TestFlowRunnerIntegration:
    """Integration tests for flow runner."""
    
    @pytest.mark.asyncio
    async def test_realistic_document_processing_flow(self, mock_state_store):
        """Test a realistic document processing flow."""
        runner = FlowRunner(state_store=mock_state_store)
        
        # Register mock executors that simulate real behavior
        class MockFileHandler(MockExecutor):
            async def execute(self, context, config):
                return ExecutionResult(
                    success=True,
                    outputs={"temp_path": "/tmp/document.pdf", "file_type": "pdf"}
                )
        
        class MockDocumentExtractor(MockExecutor):
            async def execute(self, context, config):
                return ExecutionResult(
                    success=True,
                    outputs={"text": "Extracted document text", "page_count": 3}
                )
        
        class MockLLMAnalyzer(MockExecutor):
            async def execute(self, context, config):
                text = config.get("text", "")
                return ExecutionResult(
                    success=True,
                    outputs={
                        "analysis": f"Analysis of: {text[:50]}...",
                        "summary": "Document summary",
                        "key_points": ["Point 1", "Point 2"]
                    }
                )
        
        runner.executor_registry.register_executor("file_handler", MockFileHandler)
        runner.executor_registry.register_executor("document_extractor", MockDocumentExtractor)
        runner.executor_registry.register_executor("llm_analyzer", MockLLMAnalyzer)
        
        # Define realistic flow
        flow = FlowDefinition(
            name="document_analysis",
            inputs=[
                FlowInput(name="file_path", type="string", required=True),
                FlowInput(name="analysis_type", type="string", required=False, default="summary")
            ],
            steps=[
                FlowStep(
                    name="handle_file",
                    executor="file_handler",
                    config={"file_path": "{{ inputs.file_path }}"}
                ),
                FlowStep(
                    name="extract_text",
                    executor="document_extractor",
                    config={"file_path": "{{ steps.handle_file.temp_path }}"},
                    depends_on=["handle_file"]
                ),
                FlowStep(
                    name="analyze_content",
                    executor="llm_analyzer",
                    config={
                        "text": "{{ steps.extract_text.text }}",
                        "analysis_type": "{{ inputs.analysis_type }}"
                    },
                    depends_on=["extract_text"]
                )
            ],
            outputs=[
                FlowOutput(name="analysis_result", value="{{ steps.analyze_content.analysis }}"),
                FlowOutput(name="summary", value="{{ steps.analyze_content.summary }}")
            ]
        )
        
        # Execute the flow
        inputs = {"file_path": "/path/to/document.pdf", "analysis_type": "detailed"}
        result = await runner.execute_flow(flow, inputs)
        
        # Verify results
        assert result.success is True
        assert len(result.step_results) == 3
        assert "Analysis of: Extracted document text" in result.outputs["analysis_result"]
        assert result.outputs["summary"] == "Document summary"
        
        # Verify execution order
        assert all(step in result.step_results for step in ["handle_file", "extract_text", "analyze_content"])
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_partial_execution(self, mock_state_store):
        """Test error recovery and partial execution scenarios."""
        runner = FlowRunner(state_store=mock_state_store)
        
        # Register executors with different behaviors
        runner.executor_registry.register_executor("success_executor", MockExecutor)
        runner.executor_registry.register_executor("failure_executor", 
                                                 lambda: MockExecutor("failing", should_fail=True))
        
        # Flow with mixed success/failure
        flow = FlowDefinition(
            name="mixed_flow",
            steps=[
                FlowStep(name="step1", executor="success_executor", config={}),
                FlowStep(name="step2", executor="failure_executor", config={}, depends_on=["step1"]),
                FlowStep(name="step3", executor="success_executor", config={})  # Independent step
            ]
        )
        
        result = await runner.execute_flow(flow, {})
        
        # Should have partial success
        assert result.success is False  # Overall failure due to step2
        assert result.step_results["step1"].success is True
        assert result.step_results["step2"].success is False
        assert result.step_results["step3"].success is True  # Independent step still runs
        
        # Verify error information is preserved
        assert "failed intentionally" in result.step_results["step2"].error
