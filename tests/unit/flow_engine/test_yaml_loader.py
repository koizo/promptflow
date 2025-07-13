"""
Unit tests for YAML Flow Loader.
"""
import pytest
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from core.flow_engine.yaml_loader import (
    YAMLFlowLoader,
    FlowDefinition,
    FlowStep,
    FlowInput,
    FlowOutput
)


class TestFlowInput:
    """Test FlowInput dataclass."""
    
    def test_flow_input_creation(self):
        """Test creating a flow input."""
        input_def = FlowInput(
            name="test_input",
            type="string",
            required=True,
            description="Test input parameter"
        )
        
        assert input_def.name == "test_input"
        assert input_def.type == "string"
        assert input_def.required is True
        assert input_def.default is None
        assert input_def.description == "Test input parameter"
    
    def test_flow_input_with_default(self):
        """Test creating a flow input with default value."""
        input_def = FlowInput(
            name="optional_input",
            type="integer",
            required=False,
            default=42
        )
        
        assert input_def.name == "optional_input"
        assert input_def.required is False
        assert input_def.default == 42
    
    def test_flow_input_with_enum(self):
        """Test creating a flow input with enum values."""
        input_def = FlowInput(
            name="choice_input",
            type="string",
            enum=["option1", "option2", "option3"]
        )
        
        assert input_def.enum == ["option1", "option2", "option3"]


class TestFlowOutput:
    """Test FlowOutput dataclass."""
    
    def test_flow_output_creation(self):
        """Test creating a flow output."""
        output_def = FlowOutput(
            name="result",
            value="{{ steps.final_step.output }}",
            description="Final result"
        )
        
        assert output_def.name == "result"
        assert output_def.value == "{{ steps.final_step.output }}"
        assert output_def.description == "Final result"


class TestFlowStep:
    """Test FlowStep dataclass."""
    
    def test_flow_step_creation(self):
        """Test creating a flow step."""
        step = FlowStep(
            name="process_data",
            executor="llm_analyzer",
            config={"prompt": "Analyze this data"},
            description="Data processing step"
        )
        
        assert step.name == "process_data"
        assert step.executor == "llm_analyzer"
        assert step.config == {"prompt": "Analyze this data"}
        assert step.depends_on == []
        assert step.condition is None
        assert step.parallel_group is None
    
    def test_flow_step_with_dependencies(self):
        """Test creating a flow step with dependencies."""
        step = FlowStep(
            name="combine_results",
            executor="data_combiner",
            config={"sources": ["step1", "step2"]},
            depends_on=["step1", "step2"]
        )
        
        assert step.depends_on == ["step1", "step2"]
    
    def test_flow_step_with_condition(self):
        """Test creating a flow step with condition."""
        step = FlowStep(
            name="conditional_step",
            executor="test_executor",
            config={},
            condition="{{ inputs.enable_feature == true }}"
        )
        
        assert step.condition == "{{ inputs.enable_feature == true }}"
    
    def test_flow_step_with_parallel_group(self):
        """Test creating a flow step with parallel group."""
        step = FlowStep(
            name="parallel_step",
            executor="test_executor",
            config={},
            parallel_group="group_a"
        )
        
        assert step.parallel_group == "group_a"


class TestFlowDefinition:
    """Test FlowDefinition dataclass."""
    
    def test_flow_definition_creation(self, sample_flow_definition):
        """Test creating a flow definition."""
        flow_def = sample_flow_definition
        
        assert flow_def.name == "test_flow"
        assert flow_def.description == "A test flow for unit testing"
        assert len(flow_def.inputs) == 2
        assert len(flow_def.steps) == 2
        assert len(flow_def.outputs) == 1
    
    def test_flow_definition_minimal(self):
        """Test creating a minimal flow definition."""
        flow_def = FlowDefinition(
            name="minimal_flow",
            steps=[
                FlowStep(name="single_step", executor="test_executor", config={})
            ]
        )
        
        assert flow_def.name == "minimal_flow"
        assert flow_def.description == ""
        assert flow_def.inputs == []
        assert len(flow_def.steps) == 1
        assert flow_def.outputs == []


class TestYAMLFlowLoader:
    """Test YAMLFlowLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create a YAML flow loader instance."""
        return YAMLFlowLoader()
    
    @pytest.fixture
    def sample_yaml_content(self, sample_yaml_flow):
        """Get sample YAML content as string."""
        return yaml.dump(sample_yaml_flow)
    
    def test_loader_creation(self, loader):
        """Test creating a YAML flow loader."""
        assert isinstance(loader, YAMLFlowLoader)
    
    def test_load_from_string(self, loader, sample_yaml_content):
        """Test loading flow from YAML string."""
        flow_def = loader.load_from_string(sample_yaml_content)
        
        assert isinstance(flow_def, FlowDefinition)
        assert flow_def.name == "test_flow"
        assert len(flow_def.inputs) == 1
        assert len(flow_def.steps) == 1
        assert len(flow_def.outputs) == 1
    
    def test_load_from_file(self, loader, temp_dir, sample_yaml_content):
        """Test loading flow from YAML file."""
        yaml_file = temp_dir / "test_flow.yaml"
        yaml_file.write_text(sample_yaml_content)
        
        flow_def = loader.load_from_file(yaml_file)
        
        assert isinstance(flow_def, FlowDefinition)
        assert flow_def.name == "test_flow"
    
    def test_load_from_directory(self, loader, temp_dir, sample_yaml_content):
        """Test loading flow from directory with flow.yaml."""
        flow_dir = temp_dir / "test_flow"
        flow_dir.mkdir()
        
        flow_file = flow_dir / "flow.yaml"
        flow_file.write_text(sample_yaml_content)
        
        flow_def = loader.load_from_directory(flow_dir)
        
        assert isinstance(flow_def, FlowDefinition)
        assert flow_def.name == "test_flow"
    
    def test_load_invalid_yaml(self, loader):
        """Test loading invalid YAML content."""
        invalid_yaml = "invalid: yaml: content: ["
        
        with pytest.raises(FlowValidationError) as exc_info:
            loader.load_from_string(invalid_yaml)
        
        assert "Failed to parse YAML" in str(exc_info.value)
    
    def test_load_missing_required_fields(self, loader):
        """Test loading YAML with missing required fields."""
        incomplete_yaml = yaml.dump({
            "description": "Missing name and steps"
        })
        
        with pytest.raises(FlowValidationError) as exc_info:
            loader.load_from_string(incomplete_yaml)
        
        assert "Missing required field" in str(exc_info.value)
    
    def test_load_invalid_step_structure(self, loader):
        """Test loading YAML with invalid step structure."""
        invalid_step_yaml = yaml.dump({
            "name": "test_flow",
            "steps": [
                {
                    "name": "step1",
                    # Missing required "executor" field
                    "config": {}
                }
            ]
        })
        
        with pytest.raises(FlowValidationError) as exc_info:
            loader.load_from_string(invalid_step_yaml)
        
        assert "executor" in str(exc_info.value).lower()
    
    def test_validate_flow_definition(self, loader, sample_flow_definition):
        """Test flow definition validation."""
        # Should not raise exception for valid flow
        loader.validate_flow_definition(sample_flow_definition)
    
    def test_validate_flow_definition_duplicate_step_names(self, loader):
        """Test validation with duplicate step names."""
        flow_def = FlowDefinition(
            name="test_flow",
            steps=[
                FlowStep(name="duplicate", executor="executor1", config={}),
                FlowStep(name="duplicate", executor="executor2", config={})
            ]
        )
        
        with pytest.raises(FlowValidationError) as exc_info:
            loader.validate_flow_definition(flow_def)
        
        assert "Duplicate step name" in str(exc_info.value)
    
    def test_validate_flow_definition_invalid_dependencies(self, loader):
        """Test validation with invalid step dependencies."""
        flow_def = FlowDefinition(
            name="test_flow",
            steps=[
                FlowStep(
                    name="step1", 
                    executor="executor1", 
                    config={},
                    depends_on=["nonexistent_step"]
                )
            ]
        )
        
        with pytest.raises(FlowValidationError) as exc_info:
            loader.validate_flow_definition(flow_def)
        
        assert "Unknown dependency" in str(exc_info.value)
    
    def test_validate_flow_definition_circular_dependencies(self, loader):
        """Test validation with circular dependencies."""
        flow_def = FlowDefinition(
            name="test_flow",
            steps=[
                FlowStep(name="step1", executor="executor1", config={}, depends_on=["step2"]),
                FlowStep(name="step2", executor="executor2", config={}, depends_on=["step1"])
            ]
        )
        
        with pytest.raises(FlowValidationError) as exc_info:
            loader.validate_flow_definition(flow_def)
        
        assert "Circular dependency" in str(exc_info.value)
    
    def test_validate_flow_definition_duplicate_input_names(self, loader):
        """Test validation with duplicate input names."""
        flow_def = FlowDefinition(
            name="test_flow",
            inputs=[
                FlowInput(name="duplicate", type="string"),
                FlowInput(name="duplicate", type="integer")
            ],
            steps=[FlowStep(name="step1", executor="executor1", config={})]
        )
        
        with pytest.raises(FlowValidationError) as exc_info:
            loader.validate_flow_definition(flow_def)
        
        assert "Duplicate input name" in str(exc_info.value)
    
    def test_validate_flow_definition_duplicate_output_names(self, loader):
        """Test validation with duplicate output names."""
        flow_def = FlowDefinition(
            name="test_flow",
            steps=[FlowStep(name="step1", executor="executor1", config={})],
            outputs=[
                FlowOutput(name="duplicate", value="value1"),
                FlowOutput(name="duplicate", value="value2")
            ]
        )
        
        with pytest.raises(FlowValidationError) as exc_info:
            loader.validate_flow_definition(flow_def)
        
        assert "Duplicate output name" in str(exc_info.value)
    
    def test_get_execution_order(self, loader, sample_flow_definition):
        """Test getting execution order for steps."""
        execution_order = loader.get_execution_order(sample_flow_definition)
        
        # step1 should come before step2 (step2 depends on step1)
        step1_index = execution_order.index("step1")
        step2_index = execution_order.index("step2")
        
        assert step1_index < step2_index
    
    def test_get_execution_order_complex_dependencies(self, loader):
        """Test execution order with complex dependencies."""
        flow_def = FlowDefinition(
            name="complex_flow",
            steps=[
                FlowStep(name="step_a", executor="executor", config={}),
                FlowStep(name="step_b", executor="executor", config={}, depends_on=["step_a"]),
                FlowStep(name="step_c", executor="executor", config={}, depends_on=["step_a"]),
                FlowStep(name="step_d", executor="executor", config={}, depends_on=["step_b", "step_c"])
            ]
        )
        
        execution_order = loader.get_execution_order(flow_def)
        
        # step_a should be first
        assert execution_order[0] == "step_a"
        
        # step_b and step_c should come after step_a
        step_a_index = execution_order.index("step_a")
        step_b_index = execution_order.index("step_b")
        step_c_index = execution_order.index("step_c")
        step_d_index = execution_order.index("step_d")
        
        assert step_a_index < step_b_index
        assert step_a_index < step_c_index
        assert step_b_index < step_d_index
        assert step_c_index < step_d_index
    
    def test_get_parallel_groups(self, loader):
        """Test getting parallel execution groups."""
        flow_def = FlowDefinition(
            name="parallel_flow",
            steps=[
                FlowStep(name="step1", executor="executor", config={}, parallel_group="group_a"),
                FlowStep(name="step2", executor="executor", config={}, parallel_group="group_a"),
                FlowStep(name="step3", executor="executor", config={}, parallel_group="group_b"),
                FlowStep(name="step4", executor="executor", config={})
            ]
        )
        
        parallel_groups = loader.get_parallel_groups(flow_def)
        
        assert "group_a" in parallel_groups
        assert "group_b" in parallel_groups
        assert len(parallel_groups["group_a"]) == 2
        assert len(parallel_groups["group_b"]) == 1
        assert "step1" in parallel_groups["group_a"]
        assert "step2" in parallel_groups["group_a"]
        assert "step3" in parallel_groups["group_b"]
    
    def test_load_from_nonexistent_file(self, loader, temp_dir):
        """Test loading from non-existent file."""
        nonexistent_file = temp_dir / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError):
            loader.load_from_file(nonexistent_file)
    
    def test_load_from_directory_missing_flow_yaml(self, loader, temp_dir):
        """Test loading from directory without flow.yaml."""
        empty_dir = temp_dir / "empty_flow"
        empty_dir.mkdir()
        
        with pytest.raises(FileNotFoundError):
            loader.load_from_directory(empty_dir)


class TestFlowValidationError:
    """Test FlowValidationError exception."""
    
    def test_validation_error_creation(self):
        """Test creating flow validation error."""
        error = FlowValidationError("Test validation error")
        
        assert str(error) == "Test validation error"
        assert isinstance(error, Exception)
    
    def test_validation_error_with_context(self):
        """Test validation error with context information."""
        error = FlowValidationError("Invalid step configuration", step_name="step1")
        
        # The error should contain the context information
        error_msg = str(error)
        assert "Invalid step configuration" in error_msg


@pytest.mark.integration
class TestYAMLLoaderIntegration:
    """Integration tests for YAML loader with real files."""
    
    def test_load_complete_flow_example(self, loader, temp_dir):
        """Test loading a complete, realistic flow example."""
        complete_flow = {
            "name": "document_analysis_flow",
            "description": "Complete document analysis with OCR and LLM",
            "inputs": [
                {
                    "name": "file_path",
                    "type": "string",
                    "required": True,
                    "description": "Path to document file"
                },
                {
                    "name": "analysis_type",
                    "type": "string",
                    "required": False,
                    "default": "summary",
                    "enum": ["summary", "detailed", "extraction"],
                    "description": "Type of analysis to perform"
                }
            ],
            "steps": [
                {
                    "name": "handle_file",
                    "executor": "file_handler",
                    "config": {
                        "file_path": "{{ inputs.file_path }}",
                        "validate_format": True
                    }
                },
                {
                    "name": "extract_text",
                    "executor": "document_extractor",
                    "config": {
                        "file_path": "{{ steps.handle_file.temp_path }}",
                        "extract_images": True
                    },
                    "depends_on": ["handle_file"]
                },
                {
                    "name": "ocr_images",
                    "executor": "ocr_processor",
                    "config": {
                        "images": "{{ steps.extract_text.images }}",
                        "provider": "tesseract"
                    },
                    "depends_on": ["extract_text"],
                    "condition": "{{ steps.extract_text.has_images }}"
                },
                {
                    "name": "combine_text",
                    "executor": "data_combiner",
                    "config": {
                        "text_sources": [
                            "{{ steps.extract_text.text }}",
                            "{{ steps.ocr_images.text }}"
                        ]
                    },
                    "depends_on": ["extract_text", "ocr_images"]
                },
                {
                    "name": "analyze_content",
                    "executor": "llm_analyzer",
                    "config": {
                        "text": "{{ steps.combine_text.combined_text }}",
                        "analysis_type": "{{ inputs.analysis_type }}",
                        "model": "mistral"
                    },
                    "depends_on": ["combine_text"]
                }
            ],
            "outputs": [
                {
                    "name": "analysis_result",
                    "value": "{{ steps.analyze_content.analysis }}",
                    "description": "Final analysis result"
                },
                {
                    "name": "extracted_text",
                    "value": "{{ steps.combine_text.combined_text }}",
                    "description": "All extracted text"
                }
            ]
        }
        
        # Save to file and load
        yaml_file = temp_dir / "complete_flow.yaml"
        yaml_file.write_text(yaml.dump(complete_flow))
        
        flow_def = loader.load_from_file(yaml_file)
        
        # Validate the loaded flow
        assert flow_def.name == "document_analysis_flow"
        assert len(flow_def.inputs) == 2
        assert len(flow_def.steps) == 5
        assert len(flow_def.outputs) == 2
        
        # Check execution order
        execution_order = loader.get_execution_order(flow_def)
        assert execution_order.index("handle_file") < execution_order.index("extract_text")
        assert execution_order.index("extract_text") < execution_order.index("combine_text")
        
        # Validate the flow
        loader.validate_flow_definition(flow_def)
