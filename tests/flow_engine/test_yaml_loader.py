"""
Comprehensive test suite for YAML Flow Loader.
Tests flow definition loading, parsing, validation, and error handling.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from core.flow_engine.yaml_loader import (
    YAMLFlowLoader, FlowDefinition, FlowStep, FlowInput, FlowOutput
)


class TestFlowDataClasses:
    """Test flow data classes and their validation."""
    
    def test_flow_input_creation(self):
        """Test FlowInput creation with various parameters."""
        # Basic required input
        input1 = FlowInput(name="test_input", type="string")
        assert input1.name == "test_input"
        assert input1.type == "string"
        assert input1.required is True
        assert input1.default is None
        assert input1.description == ""
        
        # Optional input with defaults
        input2 = FlowInput(
            name="optional_input",
            type="integer",
            required=False,
            default=42,
            description="Test input",
            enum=[1, 2, 3, 42]
        )
        assert input2.required is False
        assert input2.default == 42
        assert input2.description == "Test input"
        assert input2.enum == [1, 2, 3, 42]
    
    def test_flow_output_creation(self):
        """Test FlowOutput creation."""
        output = FlowOutput(
            name="result",
            value="{{ steps.final.output }}",
            description="Final result"
        )
        assert output.name == "result"
        assert output.value == "{{ steps.final.output }}"
        assert output.description == "Final result"
    
    def test_flow_step_creation(self):
        """Test FlowStep creation with various configurations."""
        # Basic step
        step1 = FlowStep(name="basic_step", executor="test_executor")
        assert step1.name == "basic_step"
        assert step1.executor == "test_executor"
        assert step1.config == {}
        assert step1.depends_on == []
        assert step1.condition is None
        assert step1.parallel_group is None
        
        # Complex step
        step2 = FlowStep(
            name="complex_step",
            executor="advanced_executor",
            config={"param1": "value1", "param2": 42},
            depends_on=["step1", "step2"],
            condition="{{ inputs.enable_step }}",
            parallel_group="group_a",
            outputs=["result1", "result2"],
            description="Complex processing step"
        )
        assert step2.config == {"param1": "value1", "param2": 42}
        assert step2.depends_on == ["step1", "step2"]
        assert step2.condition == "{{ inputs.enable_step }}"
        assert step2.parallel_group == "group_a"
        assert step2.outputs == ["result1", "result2"]
        assert step2.description == "Complex processing step"


class TestYAMLFlowLoader:
    """Test suite for YAMLFlowLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create YAMLFlowLoader instance."""
        return YAMLFlowLoader()
    
    @pytest.fixture
    def sample_flow_yaml(self):
        """Sample valid flow YAML content."""
        return """
name: "test_flow"
description: "Test flow for validation"
version: "1.0"

inputs:
  - name: "input_text"
    type: "string"
    required: true
    description: "Input text to process"
  - name: "threshold"
    type: "float"
    required: false
    default: 0.5
    description: "Processing threshold"

steps:
  - name: "process_text"
    executor: "text_processor"
    config:
      text: "{{ inputs.input_text }}"
      threshold: "{{ inputs.threshold }}"
    description: "Process the input text"
  
  - name: "analyze_result"
    executor: "result_analyzer"
    config:
      data: "{{ steps.process_text.result }}"
    depends_on: ["process_text"]
    description: "Analyze processing results"

outputs:
  - name: "final_result"
    value: "{{ steps.analyze_result.analysis }}"
    description: "Final analysis result"

config:
  execution:
    mode: "async"
    timeout: 300
  
  error_handling:
    continue_on_error: false

tags:
  - "text-processing"
  - "analysis"
"""
    
    @pytest.fixture
    def invalid_flow_yaml(self):
        """Invalid flow YAML for error testing."""
        return """
name: "invalid_flow"
# Missing required fields
steps:
  - name: "step1"
    # Missing executor
    config: {}
"""
    
    def test_load_from_string_valid(self, loader, sample_flow_yaml):
        """Test loading valid YAML from string."""
        flow_def = loader.load_from_string(sample_flow_yaml)
        
        assert isinstance(flow_def, FlowDefinition)
        assert flow_def.name == "test_flow"
        assert flow_def.description == "Test flow for validation"
        assert flow_def.version == "1.0"
        
        # Check inputs
        assert len(flow_def.inputs) == 2
        input1 = flow_def.inputs[0]
        assert input1.name == "input_text"
        assert input1.type == "string"
        assert input1.required is True
        
        input2 = flow_def.inputs[1]
        assert input2.name == "threshold"
        assert input2.type == "float"
        assert input2.required is False
        assert input2.default == 0.5
        
        # Check steps
        assert len(flow_def.steps) == 2
        step1 = flow_def.steps[0]
        assert step1.name == "process_text"
        assert step1.executor == "text_processor"
        assert step1.config["text"] == "{{ inputs.input_text }}"
        
        step2 = flow_def.steps[1]
        assert step2.name == "analyze_result"
        assert step2.depends_on == ["process_text"]
        
        # Check outputs
        assert len(flow_def.outputs) == 1
        output = flow_def.outputs[0]
        assert output.name == "final_result"
        assert output.value == "{{ steps.analyze_result.analysis }}"
        
        # Check config
        assert flow_def.config["execution"]["mode"] == "async"
        assert flow_def.config["execution"]["timeout"] == 300
        
        # Check tags
        assert "text-processing" in flow_def.tags
        assert "analysis" in flow_def.tags
    
    def test_load_from_file_valid(self, loader, sample_flow_yaml):
        """Test loading valid YAML from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(sample_flow_yaml)
            temp_path = f.name
        
        try:
            flow_def = loader.load_from_file(temp_path)
            assert flow_def.name == "test_flow"
            assert len(flow_def.steps) == 2
        finally:
            Path(temp_path).unlink()
    
    def test_load_from_file_not_found(self, loader):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            loader.load_from_file("non_existent_file.yaml")
    
    def test_load_invalid_yaml_syntax(self, loader):
        """Test loading YAML with syntax errors."""
        invalid_yaml = """
name: "test"
steps:
  - name: "step1"
    config:
      invalid: [unclosed list
"""
        with pytest.raises(yaml.YAMLError):
            loader.load_from_string(invalid_yaml)
    
    def test_load_missing_required_fields(self, loader):
        """Test loading YAML missing required fields."""
        # Missing name
        yaml_no_name = """
description: "Test flow"
steps: []
"""
        with pytest.raises(ValueError, match="Missing required field: name"):
            loader.load_from_string(yaml_no_name)
        
        # Missing steps
        yaml_no_steps = """
name: "test_flow"
description: "Test flow"
"""
        with pytest.raises(ValueError, match="Missing required field: steps"):
            loader.load_from_string(yaml_no_steps)
    
    def test_load_empty_steps(self, loader):
        """Test loading YAML with empty steps list."""
        yaml_empty_steps = """
name: "test_flow"
description: "Test flow"
steps: []
"""
        with pytest.raises(ValueError, match="Flow must have at least one step"):
            loader.load_from_string(yaml_empty_steps)
    
    def test_load_invalid_step_format(self, loader):
        """Test loading YAML with invalid step format."""
        yaml_invalid_step = """
name: "test_flow"
steps:
  - name: "step1"
    # Missing executor
    config: {}
"""
        with pytest.raises(ValueError, match="Step 'step1' missing required field: executor"):
            loader.load_from_string(yaml_invalid_step)
    
    def test_validate_step_dependencies_valid(self, loader, sample_flow_yaml):
        """Test validation of valid step dependencies."""
        flow_def = loader.load_from_string(sample_flow_yaml)
        # Should not raise any exception
        loader.validate_dependencies(flow_def)
    
    def test_validate_step_dependencies_invalid(self, loader):
        """Test validation of invalid step dependencies."""
        yaml_invalid_deps = """
name: "test_flow"
steps:
  - name: "step1"
    executor: "executor1"
    depends_on: ["non_existent_step"]
  - name: "step2"
    executor: "executor2"
"""
        flow_def = loader.load_from_string(yaml_invalid_deps)
        with pytest.raises(ValueError, match="Step 'step1' depends on non-existent step: non_existent_step"):
            loader.validate_dependencies(flow_def)
    
    def test_validate_circular_dependencies(self, loader):
        """Test detection of circular dependencies."""
        yaml_circular = """
name: "test_flow"
steps:
  - name: "step1"
    executor: "executor1"
    depends_on: ["step2"]
  - name: "step2"
    executor: "executor2"
    depends_on: ["step1"]
"""
        flow_def = loader.load_from_string(yaml_circular)
        with pytest.raises(ValueError, match="Circular dependency detected"):
            loader.validate_dependencies(flow_def)
    
    def test_validate_self_dependency(self, loader):
        """Test detection of self-dependencies."""
        yaml_self_dep = """
name: "test_flow"
steps:
  - name: "step1"
    executor: "executor1"
    depends_on: ["step1"]
"""
        flow_def = loader.load_from_string(yaml_self_dep)
        with pytest.raises(ValueError, match="Step 'step1' cannot depend on itself"):
            loader.validate_dependencies(flow_def)
    
    def test_get_execution_order_simple(self, loader):
        """Test execution order calculation for simple dependencies."""
        yaml_simple = """
name: "test_flow"
steps:
  - name: "step1"
    executor: "executor1"
  - name: "step2"
    executor: "executor2"
    depends_on: ["step1"]
  - name: "step3"
    executor: "executor3"
    depends_on: ["step2"]
"""
        flow_def = loader.load_from_string(yaml_simple)
        execution_order = loader.get_execution_order(flow_def)
        
        # Should be in dependency order
        step_names = [step.name for step in execution_order]
        assert step_names == ["step1", "step2", "step3"]
    
    def test_get_execution_order_complex(self, loader):
        """Test execution order calculation for complex dependencies."""
        yaml_complex = """
name: "test_flow"
steps:
  - name: "step1"
    executor: "executor1"
  - name: "step2"
    executor: "executor2"
  - name: "step3"
    executor: "executor3"
    depends_on: ["step1", "step2"]
  - name: "step4"
    executor: "executor4"
    depends_on: ["step1"]
  - name: "step5"
    executor: "executor5"
    depends_on: ["step3", "step4"]
"""
        flow_def = loader.load_from_string(yaml_complex)
        execution_order = loader.get_execution_order(flow_def)
        
        step_names = [step.name for step in execution_order]
        
        # step1 and step2 should come first (no dependencies)
        assert step_names.index("step1") < step_names.index("step3")
        assert step_names.index("step2") < step_names.index("step3")
        assert step_names.index("step1") < step_names.index("step4")
        
        # step3 and step4 should come before step5
        assert step_names.index("step3") < step_names.index("step5")
        assert step_names.index("step4") < step_names.index("step5")
    
    def test_get_parallel_groups(self, loader):
        """Test identification of parallel execution groups."""
        yaml_parallel = """
name: "test_flow"
steps:
  - name: "step1"
    executor: "executor1"
    parallel_group: "group_a"
  - name: "step2"
    executor: "executor2"
    parallel_group: "group_a"
  - name: "step3"
    executor: "executor3"
    parallel_group: "group_b"
  - name: "step4"
    executor: "executor4"
    # No parallel group
"""
        flow_def = loader.load_from_string(yaml_parallel)
        parallel_groups = loader.get_parallel_groups(flow_def)
        
        assert "group_a" in parallel_groups
        assert "group_b" in parallel_groups
        assert len(parallel_groups["group_a"]) == 2
        assert len(parallel_groups["group_b"]) == 1
        
        group_a_names = [step.name for step in parallel_groups["group_a"]]
        assert "step1" in group_a_names
        assert "step2" in group_a_names
    
    def test_load_with_conditions(self, loader):
        """Test loading steps with conditional execution."""
        yaml_conditions = """
name: "test_flow"
inputs:
  - name: "enable_step"
    type: "boolean"
    default: true

steps:
  - name: "step1"
    executor: "executor1"
    condition: "{{ inputs.enable_step }}"
  - name: "step2"
    executor: "executor2"
    condition: "{{ steps.step1.success }}"
    depends_on: ["step1"]
"""
        flow_def = loader.load_from_string(yaml_conditions)
        
        assert flow_def.steps[0].condition == "{{ inputs.enable_step }}"
        assert flow_def.steps[1].condition == "{{ steps.step1.success }}"
    
    def test_load_with_outputs_list(self, loader):
        """Test loading steps with output specifications."""
        yaml_outputs = """
name: "test_flow"
steps:
  - name: "step1"
    executor: "executor1"
    outputs: ["result1", "result2", "metadata"]
"""
        flow_def = loader.load_from_string(yaml_outputs)
        
        assert flow_def.steps[0].outputs == ["result1", "result2", "metadata"]
    
    def test_load_minimal_flow(self, loader):
        """Test loading minimal valid flow."""
        minimal_yaml = """
name: "minimal_flow"
steps:
  - name: "only_step"
    executor: "simple_executor"
"""
        flow_def = loader.load_from_string(minimal_yaml)
        
        assert flow_def.name == "minimal_flow"
        assert flow_def.description == ""
        assert flow_def.version == "1.0"  # Default version
        assert len(flow_def.steps) == 1
        assert len(flow_def.inputs) == 0
        assert len(flow_def.outputs) == 0
        assert flow_def.config == {}
        assert flow_def.tags == []
    
    def test_load_with_enum_validation(self, loader):
        """Test loading inputs with enum validation."""
        yaml_enum = """
name: "enum_flow"
inputs:
  - name: "mode"
    type: "string"
    enum: ["fast", "accurate", "balanced"]
    default: "balanced"

steps:
  - name: "process"
    executor: "processor"
    config:
      mode: "{{ inputs.mode }}"
"""
        flow_def = loader.load_from_string(yaml_enum)
        
        mode_input = flow_def.inputs[0]
        assert mode_input.enum == ["fast", "accurate", "balanced"]
        assert mode_input.default == "balanced"
    
    def test_error_handling_invalid_yaml_structure(self, loader):
        """Test error handling for invalid YAML structure."""
        # Test with non-dict root
        with pytest.raises(ValueError, match="Flow definition must be a dictionary"):
            loader.load_from_string("- invalid_root_list")
        
        # Test with invalid steps structure
        yaml_invalid_steps = """
name: "test"
steps: "not_a_list"
"""
        with pytest.raises(ValueError, match="Steps must be a list"):
            loader.load_from_string(yaml_invalid_steps)
    
    def test_duplicate_step_names(self, loader):
        """Test detection of duplicate step names."""
        yaml_duplicate = """
name: "test_flow"
steps:
  - name: "duplicate_step"
    executor: "executor1"
  - name: "duplicate_step"
    executor: "executor2"
"""
        with pytest.raises(ValueError, match="Duplicate step name: duplicate_step"):
            loader.load_from_string(yaml_duplicate)
    
    def test_load_from_path_object(self, loader, sample_flow_yaml):
        """Test loading from Path object."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(sample_flow_yaml)
            temp_path = Path(f.name)
        
        try:
            flow_def = loader.load_from_file(temp_path)
            assert flow_def.name == "test_flow"
        finally:
            temp_path.unlink()
    
    @patch("builtins.open", mock_open(read_data="invalid: yaml: content:"))
    def test_file_read_error_handling(self, loader):
        """Test error handling when file reading fails."""
        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(yaml.YAMLError):
                loader.load_from_file("test.yaml")


class TestFlowDefinition:
    """Test FlowDefinition class methods."""
    
    @pytest.fixture
    def sample_flow_def(self):
        """Create sample FlowDefinition for testing."""
        return FlowDefinition(
            name="test_flow",
            description="Test flow",
            version="1.0",
            inputs=[
                FlowInput(name="input1", type="string"),
                FlowInput(name="input2", type="integer", required=False, default=10)
            ],
            steps=[
                FlowStep(name="step1", executor="executor1"),
                FlowStep(name="step2", executor="executor2", depends_on=["step1"])
            ],
            outputs=[
                FlowOutput(name="result", value="{{ steps.step2.output }}")
            ],
            config={"timeout": 300},
            tags=["test", "example"]
        )
    
    def test_get_step_by_name(self, sample_flow_def):
        """Test getting step by name."""
        step = sample_flow_def.get_step_by_name("step1")
        assert step is not None
        assert step.name == "step1"
        assert step.executor == "executor1"
        
        # Test non-existent step
        assert sample_flow_def.get_step_by_name("non_existent") is None
    
    def test_get_input_by_name(self, sample_flow_def):
        """Test getting input by name."""
        input1 = sample_flow_def.get_input_by_name("input1")
        assert input1 is not None
        assert input1.name == "input1"
        assert input1.type == "string"
        
        # Test non-existent input
        assert sample_flow_def.get_input_by_name("non_existent") is None
    
    def test_get_required_inputs(self, sample_flow_def):
        """Test getting required inputs."""
        required = sample_flow_def.get_required_inputs()
        assert len(required) == 1
        assert required[0].name == "input1"
    
    def test_get_optional_inputs(self, sample_flow_def):
        """Test getting optional inputs."""
        optional = sample_flow_def.get_optional_inputs()
        assert len(optional) == 1
        assert optional[0].name == "input2"
        assert optional[0].default == 10
    
    def test_validate_input_values_valid(self, sample_flow_def):
        """Test validation of valid input values."""
        inputs = {"input1": "test_value", "input2": 20}
        # Should not raise exception
        sample_flow_def.validate_input_values(inputs)
    
    def test_validate_input_values_missing_required(self, sample_flow_def):
        """Test validation with missing required inputs."""
        inputs = {"input2": 20}  # Missing required input1
        with pytest.raises(ValueError, match="Missing required input: input1"):
            sample_flow_def.validate_input_values(inputs)
    
    def test_validate_input_values_with_defaults(self, sample_flow_def):
        """Test validation uses default values for optional inputs."""
        inputs = {"input1": "test_value"}  # input2 should use default
        validated = sample_flow_def.validate_input_values(inputs)
        assert validated["input1"] == "test_value"
        assert validated["input2"] == 10  # Default value
    
    def test_to_dict(self, sample_flow_def):
        """Test conversion to dictionary."""
        flow_dict = sample_flow_def.to_dict()
        
        assert flow_dict["name"] == "test_flow"
        assert flow_dict["description"] == "Test flow"
        assert flow_dict["version"] == "1.0"
        assert len(flow_dict["inputs"]) == 2
        assert len(flow_dict["steps"]) == 2
        assert len(flow_dict["outputs"]) == 1
        assert flow_dict["config"]["timeout"] == 300
        assert "test" in flow_dict["tags"]


class TestIntegrationScenarios:
    """Integration tests for complex flow scenarios."""
    
    @pytest.fixture
    def loader(self):
        return YAMLFlowLoader()
    
    def test_complex_workflow_loading(self, loader):
        """Test loading a complex real-world workflow."""
        complex_yaml = """
name: "document_processing_pipeline"
description: "Complete document analysis with OCR, sentiment, and LLM processing"
version: "2.1"

inputs:
  - name: "document_file"
    type: "file"
    required: true
    description: "Document to process"
  - name: "analysis_depth"
    type: "string"
    enum: ["basic", "detailed", "comprehensive"]
    default: "detailed"
  - name: "languages"
    type: "array"
    default: ["en"]
    description: "Languages for OCR processing"

steps:
  - name: "extract_text"
    executor: "ocr_processor"
    config:
      image_path: "{{ inputs.document_file }}"
      languages: "{{ inputs.languages }}"
      confidence_threshold: 0.8
    outputs: ["text", "confidence", "language"]
    description: "Extract text using OCR"
  
  - name: "analyze_sentiment"
    executor: "sentiment_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      analysis_type: "{{ inputs.analysis_depth }}"
    depends_on: ["extract_text"]
    condition: "{{ steps.extract_text.confidence > 0.7 }}"
    parallel_group: "analysis"
    description: "Analyze text sentiment"
  
  - name: "classify_document"
    executor: "document_classifier"
    config:
      text: "{{ steps.extract_text.text }}"
      confidence_threshold: 0.6
    depends_on: ["extract_text"]
    parallel_group: "analysis"
    description: "Classify document type"
  
  - name: "llm_analysis"
    executor: "llm_analyzer"
    config:
      text: "{{ steps.extract_text.text }}"
      prompt: "Analyze this document and provide insights"
      max_tokens: 500
    depends_on: ["extract_text"]
    condition: "{{ inputs.analysis_depth == 'comprehensive' }}"
    description: "Advanced LLM analysis"
  
  - name: "combine_results"
    executor: "data_combiner"
    config:
      sources: ["extract_text", "analyze_sentiment", "classify_document", "llm_analysis"]
      strategy: "structured"
      structure_template:
        extracted_text: "0.text"
        confidence: "0.confidence"
        sentiment: "1.sentiment"
        document_type: "2.classification"
        insights: "3.analysis"
    depends_on: ["analyze_sentiment", "classify_document"]
    description: "Combine all analysis results"

outputs:
  - name: "analysis_result"
    value: "{{ steps.combine_results.structured_data }}"
    description: "Complete document analysis"
  - name: "processing_metadata"
    value: "{{ steps.combine_results.combination_metadata }}"
    description: "Processing statistics"

config:
  execution:
    mode: "async"
    timeout: 600
    retry_attempts: 2
  
  error_handling:
    continue_on_error: false
    fallback_strategy: "partial_results"
  
  caching:
    enabled: true
    ttl: 3600

tags:
  - "document-processing"
  - "multi-modal"
  - "enterprise"
  - "production"
"""
        
        flow_def = loader.load_from_string(complex_yaml)
        
        # Validate basic structure
        assert flow_def.name == "document_processing_pipeline"
        assert flow_def.version == "2.1"
        assert len(flow_def.inputs) == 3
        assert len(flow_def.steps) == 5
        assert len(flow_def.outputs) == 2
        
        # Validate complex input with enum
        analysis_input = flow_def.get_input_by_name("analysis_depth")
        assert analysis_input.enum == ["basic", "detailed", "comprehensive"]
        assert analysis_input.default == "detailed"
        
        # Validate parallel groups
        parallel_groups = loader.get_parallel_groups(flow_def)
        assert "analysis" in parallel_groups
        assert len(parallel_groups["analysis"]) == 2
        
        # Validate execution order
        execution_order = loader.get_execution_order(flow_def)
        step_names = [step.name for step in execution_order]
        
        # extract_text should be first
        assert step_names[0] == "extract_text"
        
        # combine_results should be last
        assert step_names[-1] == "combine_results"
        
        # Validate dependencies
        loader.validate_dependencies(flow_def)
        
        # Validate conditional steps
        llm_step = flow_def.get_step_by_name("llm_analysis")
        assert llm_step.condition == "{{ inputs.analysis_depth == 'comprehensive' }}"
        
        sentiment_step = flow_def.get_step_by_name("analyze_sentiment")
        assert sentiment_step.condition == "{{ steps.extract_text.confidence > 0.7 }}"
    
    def test_error_recovery_workflow(self, loader):
        """Test workflow designed for error recovery scenarios."""
        error_recovery_yaml = """
name: "resilient_processing"
description: "Workflow with multiple fallback strategies"

inputs:
  - name: "input_data"
    type: "string"
    required: true

steps:
  - name: "primary_processing"
    executor: "primary_processor"
    config:
      data: "{{ inputs.input_data }}"
      strict_mode: true
  
  - name: "fallback_processing"
    executor: "fallback_processor"
    config:
      data: "{{ inputs.input_data }}"
      lenient_mode: true
    condition: "{{ not steps.primary_processing.success }}"
  
  - name: "combine_results"
    executor: "data_combiner"
    config:
      sources: ["primary_processing", "fallback_processing"]
      strategy: "merge"
      merge_strategy: "keep_first"
    depends_on: ["primary_processing", "fallback_processing"]

outputs:
  - name: "result"
    value: "{{ steps.combine_results.combined_result }}"
"""
        
        flow_def = loader.load_from_string(error_recovery_yaml)
        
        # Validate conditional fallback
        fallback_step = flow_def.get_step_by_name("fallback_processing")
        assert fallback_step.condition == "{{ not steps.primary_processing.success }}"
        
        # Validate execution order handles conditional dependencies
        execution_order = loader.get_execution_order(flow_def)
        step_names = [step.name for step in execution_order]
        
        # Should handle conditional dependencies properly
        assert "primary_processing" in step_names
        assert "fallback_processing" in step_names
        assert "combine_results" in step_names
