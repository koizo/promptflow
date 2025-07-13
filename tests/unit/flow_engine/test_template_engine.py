"""
Unit tests for Template Engine.
"""
import pytest
from unittest.mock import Mock

from core.flow_engine.template_engine import (
    TemplateEngine,
    TemplateError,
    TemplateContext
)
from core.executors.base_executor import ExecutionResult, FlowContext


class TestTemplateContext:
    """Test TemplateContext class."""
    
    def test_template_context_creation(self):
        """Test creating a template context."""
        inputs = {"name": "John", "age": 30}
        step_results = {
            "step1": ExecutionResult(success=True, outputs={"result": "processed"})
        }
        
        context = TemplateContext(inputs=inputs, step_results=step_results)
        
        assert context.inputs == inputs
        assert context.step_results == step_results
        assert context.config == {}
        assert context.metadata == {}
    
    def test_template_context_with_config(self):
        """Test creating template context with config."""
        context = TemplateContext(
            inputs={"test": "value"},
            config={"timeout": 30, "model": "gpt-4"}
        )
        
        assert context.config["timeout"] == 30
        assert context.config["model"] == "gpt-4"
    
    def test_template_context_to_dict(self):
        """Test converting template context to dictionary."""
        inputs = {"input1": "value1"}
        step_results = {
            "step1": ExecutionResult(success=True, outputs={"output1": "result1"})
        }
        config = {"setting": "value"}
        metadata = {"flow_id": "test_123"}
        
        context = TemplateContext(
            inputs=inputs,
            step_results=step_results,
            config=config,
            metadata=metadata
        )
        
        context_dict = context.to_dict()
        
        assert context_dict["inputs"] == inputs
        assert context_dict["config"] == config
        assert context_dict["metadata"] == metadata
        assert "step1" in context_dict["steps"]
        assert context_dict["steps"]["step1"]["output1"] == "result1"


class TestTemplateEngine:
    """Test TemplateEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create a template engine instance."""
        return TemplateEngine()
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample template context."""
        return TemplateContext(
            inputs={"name": "Alice", "count": 5},
            step_results={
                "step1": ExecutionResult(
                    success=True,
                    outputs={"message": "Hello", "number": 42}
                ),
                "step2": ExecutionResult(
                    success=True,
                    outputs={"processed": True, "items": ["a", "b", "c"]}
                )
            },
            config={"model": "mistral", "temperature": 0.7},
            metadata={"flow_id": "test_flow", "user_id": "user123"}
        )
    
    def test_engine_creation(self, engine):
        """Test creating a template engine."""
        assert isinstance(engine, TemplateEngine)
        assert engine.jinja_env is not None
    
    def test_render_simple_template(self, engine, sample_context):
        """Test rendering a simple template."""
        template = "Hello {{ inputs.name }}!"
        result = engine.render(template, sample_context)
        
        assert result == "Hello Alice!"
    
    def test_render_step_output_template(self, engine, sample_context):
        """Test rendering template with step outputs."""
        template = "Step1 said: {{ steps.step1.message }}"
        result = engine.render(template, sample_context)
        
        assert result == "Step1 said: Hello"
    
    def test_render_config_template(self, engine, sample_context):
        """Test rendering template with config values."""
        template = "Using model {{ config.model }} with temp {{ config.temperature }}"
        result = engine.render(template, sample_context)
        
        assert result == "Using model mistral with temp 0.7"
    
    def test_render_metadata_template(self, engine, sample_context):
        """Test rendering template with metadata."""
        template = "Flow {{ metadata.flow_id }} for user {{ metadata.user_id }}"
        result = engine.render(template, sample_context)
        
        assert result == "Flow test_flow for user user123"
    
    def test_render_complex_template(self, engine, sample_context):
        """Test rendering complex template with multiple references."""
        template = """
        User: {{ inputs.name }}
        Count: {{ inputs.count }}
        Step1 Number: {{ steps.step1.number }}
        Step2 Items: {{ steps.step2.items | join(', ') }}
        Model: {{ config.model }}
        """.strip()
        
        result = engine.render(template, sample_context)
        
        assert "User: Alice" in result
        assert "Count: 5" in result
        assert "Step1 Number: 42" in result
        assert "Step2 Items: a, b, c" in result
        assert "Model: mistral" in result
    
    def test_render_with_filters(self, engine, sample_context):
        """Test rendering template with Jinja filters."""
        template = "Name: {{ inputs.name | upper }}, Items: {{ steps.step2.items | length }}"
        result = engine.render(template, sample_context)
        
        assert result == "Name: ALICE, Items: 3"
    
    def test_render_with_conditionals(self, engine, sample_context):
        """Test rendering template with conditional logic."""
        template = """
        {% if steps.step2.processed %}
        Processing completed successfully!
        {% else %}
        Processing failed.
        {% endif %}
        """.strip()
        
        result = engine.render(template, sample_context)
        
        assert "Processing completed successfully!" in result
        assert "Processing failed." not in result
    
    def test_render_with_loops(self, engine, sample_context):
        """Test rendering template with loops."""
        template = """
        Items:
        {% for item in steps.step2.items %}
        - {{ item }}
        {% endfor %}
        """.strip()
        
        result = engine.render(template, sample_context)
        
        assert "- a" in result
        assert "- b" in result
        assert "- c" in result
    
    def test_render_invalid_template_syntax(self, engine, sample_context):
        """Test rendering template with invalid syntax."""
        template = "Hello {{ inputs.name"  # Missing closing brace
        
        with pytest.raises(TemplateError) as exc_info:
            engine.render(template, sample_context)
        
        assert "Template syntax error" in str(exc_info.value)
    
    def test_render_undefined_variable(self, engine, sample_context):
        """Test rendering template with undefined variable."""
        template = "Hello {{ inputs.nonexistent }}!"
        
        with pytest.raises(TemplateError) as exc_info:
            engine.render(template, sample_context)
        
        assert "Undefined variable" in str(exc_info.value)
    
    def test_render_undefined_step(self, engine, sample_context):
        """Test rendering template with undefined step."""
        template = "Result: {{ steps.nonexistent_step.output }}"
        
        with pytest.raises(TemplateError) as exc_info:
            engine.render(template, sample_context)
        
        assert "Undefined variable" in str(exc_info.value)
    
    def test_render_config_dict(self, engine, sample_context):
        """Test rendering a configuration dictionary."""
        config_template = {
            "message": "Hello {{ inputs.name }}!",
            "count": "{{ inputs.count }}",
            "result": "{{ steps.step1.message }} - {{ steps.step1.number }}",
            "nested": {
                "model": "{{ config.model }}",
                "processed": "{{ steps.step2.processed }}"
            }
        }
        
        result = engine.render_config(config_template, sample_context)
        
        assert result["message"] == "Hello Alice!"
        assert result["count"] == "5"
        assert result["result"] == "Hello - 42"
        assert result["nested"]["model"] == "mistral"
        assert result["nested"]["processed"] == "True"
    
    def test_render_config_list(self, engine, sample_context):
        """Test rendering a configuration list."""
        config_template = [
            "{{ inputs.name }}",
            "{{ steps.step1.number }}",
            {
                "item": "{{ steps.step2.items[0] }}",
                "count": "{{ inputs.count }}"
            }
        ]
        
        result = engine.render_config(config_template, sample_context)
        
        assert result[0] == "Alice"
        assert result[1] == "42"
        assert result[2]["item"] == "a"
        assert result[2]["count"] == "5"
    
    def test_render_config_mixed_types(self, engine, sample_context):
        """Test rendering config with mixed template and non-template values."""
        config_template = {
            "templated": "Hello {{ inputs.name }}!",
            "static": "This is static",
            "number": 123,
            "boolean": True,
            "null_value": None,
            "mixed_list": [
                "{{ inputs.name }}",
                "static string",
                456
            ]
        }
        
        result = engine.render_config(config_template, sample_context)
        
        assert result["templated"] == "Hello Alice!"
        assert result["static"] == "This is static"
        assert result["number"] == 123
        assert result["boolean"] is True
        assert result["null_value"] is None
        assert result["mixed_list"][0] == "Alice"
        assert result["mixed_list"][1] == "static string"
        assert result["mixed_list"][2] == 456
    
    def test_render_empty_template(self, engine, sample_context):
        """Test rendering empty template."""
        result = engine.render("", sample_context)
        assert result == ""
    
    def test_render_whitespace_template(self, engine, sample_context):
        """Test rendering template with only whitespace."""
        result = engine.render("   \n\t  ", sample_context)
        assert result == "   \n\t  "
    
    def test_custom_filters(self, engine, sample_context):
        """Test custom Jinja filters if any are added."""
        # Test that standard filters work
        template = "{{ inputs.name | length }}"
        result = engine.render(template, sample_context)
        assert result == "5"  # Length of "Alice"
    
    def test_render_with_missing_step_output(self, engine):
        """Test rendering when step exists but output key doesn't."""
        context = TemplateContext(
            inputs={"name": "Bob"},
            step_results={
                "step1": ExecutionResult(success=True, outputs={"existing": "value"})
            }
        )
        
        template = "{{ steps.step1.nonexistent }}"
        
        with pytest.raises(TemplateError):
            engine.render(template, context)
    
    def test_render_failed_step_result(self, engine):
        """Test rendering template with failed step result."""
        context = TemplateContext(
            inputs={"name": "Charlie"},
            step_results={
                "failed_step": ExecutionResult(
                    success=False,
                    error="Step failed",
                    outputs={}
                )
            }
        )
        
        # Should still be able to access the step, even if it failed
        template = "Step success: {{ steps.failed_step.success }}"
        result = engine.render(template, context)
        
        assert "Step success: False" in result
    
    def test_render_step_metadata(self, engine):
        """Test rendering step metadata."""
        context = TemplateContext(
            inputs={"test": "value"},
            step_results={
                "step1": ExecutionResult(
                    success=True,
                    outputs={"result": "data"},
                    metadata={"execution_time": 1.5, "model": "test_model"}
                )
            }
        )
        
        template = "Execution time: {{ steps.step1.execution_time }}, Model: {{ steps.step1.model }}"
        result = engine.render(template, context)
        
        assert "Execution time: 1.5" in result
        assert "Model: test_model" in result


class TestTemplateError:
    """Test TemplateError exception."""
    
    def test_template_error_creation(self):
        """Test creating template error."""
        error = TemplateError("Template rendering failed")
        
        assert str(error) == "Template rendering failed"
        assert isinstance(error, Exception)
    
    def test_template_error_with_template(self):
        """Test template error with template information."""
        template = "{{ invalid.template }}"
        error = TemplateError(f"Error in template: {template}")
        
        assert template in str(error)


@pytest.mark.integration
class TestTemplateEngineIntegration:
    """Integration tests for template engine."""
    
    def test_realistic_flow_templating(self):
        """Test templating in a realistic flow scenario."""
        engine = TemplateEngine()
        
        # Simulate a document analysis flow
        context = TemplateContext(
            inputs={
                "document_path": "/path/to/document.pdf",
                "analysis_type": "summary",
                "user_id": "user123"
            },
            step_results={
                "extract_text": ExecutionResult(
                    success=True,
                    outputs={
                        "text": "This is the extracted document text...",
                        "page_count": 5,
                        "word_count": 1250
                    },
                    metadata={"extraction_time": 2.3}
                ),
                "analyze_content": ExecutionResult(
                    success=True,
                    outputs={
                        "summary": "Document summary here...",
                        "key_points": ["Point 1", "Point 2", "Point 3"],
                        "sentiment": "neutral"
                    },
                    metadata={"model_used": "mistral", "analysis_time": 4.7}
                )
            },
            config={
                "max_summary_length": 500,
                "include_metadata": True
            },
            metadata={
                "flow_id": "doc_analysis_001",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        )
        
        # Test complex response template
        response_template = {
            "user_id": "{{ inputs.user_id }}",
            "document": "{{ inputs.document_path }}",
            "analysis_type": "{{ inputs.analysis_type }}",
            "results": {
                "summary": "{{ steps.analyze_content.summary }}",
                "key_points": "{{ steps.analyze_content.key_points | join('; ') }}",
                "document_stats": {
                    "pages": "{{ steps.extract_text.page_count }}",
                    "words": "{{ steps.extract_text.word_count }}",
                    "sentiment": "{{ steps.analyze_content.sentiment }}"
                }
            },
            "processing_info": {
                "extraction_time": "{{ steps.extract_text.extraction_time }}s",
                "analysis_time": "{{ steps.analyze_content.analysis_time }}s",
                "model_used": "{{ steps.analyze_content.model_used }}",
                "flow_id": "{{ metadata.flow_id }}"
            }
        }
        
        result = engine.render_config(response_template, context)
        
        # Verify the rendered result
        assert result["user_id"] == "user123"
        assert result["document"] == "/path/to/document.pdf"
        assert result["analysis_type"] == "summary"
        assert result["results"]["summary"] == "Document summary here..."
        assert result["results"]["key_points"] == "Point 1; Point 2; Point 3"
        assert result["results"]["document_stats"]["pages"] == "5"
        assert result["results"]["document_stats"]["words"] == "1250"
        assert result["results"]["document_stats"]["sentiment"] == "neutral"
        assert result["processing_info"]["extraction_time"] == "2.3s"
        assert result["processing_info"]["analysis_time"] == "4.7s"
        assert result["processing_info"]["model_used"] == "mistral"
        assert result["processing_info"]["flow_id"] == "doc_analysis_001"
    
    def test_conditional_step_execution_template(self):
        """Test template for conditional step execution."""
        engine = TemplateEngine()
        
        context = TemplateContext(
            inputs={"enable_ocr": True, "file_type": "pdf"},
            step_results={
                "file_analysis": ExecutionResult(
                    success=True,
                    outputs={"has_images": True, "text_extractable": False}
                )
            }
        )
        
        # Template for conditional execution
        condition_template = "{{ inputs.enable_ocr and steps.file_analysis.has_images and not steps.file_analysis.text_extractable }}"
        
        result = engine.render(condition_template, context)
        
        # Should evaluate to "True" since all conditions are met
        assert result == "True"
    
    def test_error_handling_in_complex_template(self):
        """Test error handling in complex template scenarios."""
        engine = TemplateEngine()
        
        context = TemplateContext(
            inputs={"name": "test"},
            step_results={}
        )
        
        # Template that references non-existent step
        complex_template = {
            "valid": "{{ inputs.name }}",
            "invalid": "{{ steps.nonexistent.output }}",
            "nested": {
                "also_invalid": "{{ steps.missing.data }}"
            }
        }
        
        with pytest.raises(TemplateError):
            engine.render_config(complex_template, context)
