"""
Comprehensive test suite for Template Engine.
Tests Jinja2 template processing, context resolution, and error handling.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from core.flow_engine.template_engine import TemplateEngine
from core.executors.base_executor import ExecutionResult, FlowContext


class TestTemplateEngine:
    """Test suite for TemplateEngine class."""
    
    @pytest.fixture
    def template_engine(self):
        """Create TemplateEngine instance."""
        return TemplateEngine()
    
    @pytest.fixture
    def sample_context(self):
        """Create sample flow context for template testing."""
        context = Mock(spec=FlowContext)
        
        # Mock step results
        step1_result = Mock()
        step1_result.success = True
        step1_result.outputs = {
            "text": "Hello World",
            "confidence": 0.95,
            "metadata": {"language": "en", "words": 2}
        }
        
        step2_result = Mock()
        step2_result.success = True
        step2_result.outputs = {
            "sentiment": "positive",
            "score": 0.8,
            "emotions": {"joy": 0.7, "neutral": 0.3}
        }
        
        failed_result = Mock()
        failed_result.success = False
        failed_result.error = "Processing failed"
        failed_result.outputs = {}
        
        context.step_results = {
            "extract_text": step1_result,
            "analyze_sentiment": step2_result,
            "failed_step": failed_result
        }
        
        return context
    
    @pytest.fixture
    def sample_inputs(self):
        """Sample input values for template testing."""
        return {
            "user_name": "Alice",
            "threshold": 0.7,
            "enable_advanced": True,
            "tags": ["important", "urgent"],
            "config": {
                "mode": "production",
                "debug": False
            }
        }
    
    def test_render_simple_template(self, template_engine, sample_inputs):
        """Test rendering simple templates with input values."""
        # String template
        template = "Hello {{ inputs.user_name }}!"
        result = template_engine.render(template, inputs=sample_inputs)
        assert result == "Hello Alice!"
        
        # Numeric template
        template = "Threshold: {{ inputs.threshold }}"
        result = template_engine.render(template, inputs=sample_inputs)
        assert result == "Threshold: 0.7"
        
        # Boolean template
        template = "Advanced: {{ inputs.enable_advanced }}"
        result = template_engine.render(template, inputs=sample_inputs)
        assert result == "Advanced: True"
    
    def test_render_step_references(self, template_engine, sample_context):
        """Test rendering templates with step result references."""
        # Simple step output reference
        template = "Extracted: {{ steps.extract_text.text }}"
        result = template_engine.render(template, context=sample_context)
        assert result == "Extracted: Hello World"
        
        # Nested step output reference
        template = "Language: {{ steps.extract_text.metadata.language }}"
        result = template_engine.render(template, context=sample_context)
        assert result == "Language: en"
        
        # Multiple step references
        template = "Text: {{ steps.extract_text.text }}, Sentiment: {{ steps.analyze_sentiment.sentiment }}"
        result = template_engine.render(template, context=sample_context)
        assert result == "Text: Hello World, Sentiment: positive"
    
    def test_render_complex_expressions(self, template_engine, sample_inputs, sample_context):
        """Test rendering complex Jinja2 expressions."""
        # Conditional expression
        template = "{% if inputs.enable_advanced %}Advanced Mode{% else %}Basic Mode{% endif %}"
        result = template_engine.render(template, inputs=sample_inputs, context=sample_context)
        assert result == "Advanced Mode"
        
        # Loop expression
        template = "Tags: {% for tag in inputs.tags %}{{ tag }}{% if not loop.last %}, {% endif %}{% endfor %}"
        result = template_engine.render(template, inputs=sample_inputs)
        assert result == "Tags: important, urgent"
        
        # Mathematical expression
        template = "Score: {{ (steps.analyze_sentiment.score * 100) | round(1) }}%"
        result = template_engine.render(template, context=sample_context)
        assert result == "Score: 80.0%"
    
    def test_render_with_filters(self, template_engine, sample_context):
        """Test rendering with Jinja2 filters."""
        # String filters
        template = "{{ steps.extract_text.text | upper }}"
        result = template_engine.render(template, context=sample_context)
        assert result == "HELLO WORLD"
        
        template = "{{ steps.extract_text.text | lower }}"
        result = template_engine.render(template, context=sample_context)
        assert result == "hello world"
        
        # Numeric filters
        template = "{{ steps.analyze_sentiment.score | round(2) }}"
        result = template_engine.render(template, context=sample_context)
        assert result == "0.8"
        
        # List filters
        template = "{{ inputs.tags | join(', ') }}"
        result = template_engine.render(template, inputs={"tags": ["a", "b", "c"]})
        assert result == "a, b, c"
    
    def test_render_config_object(self, template_engine, sample_inputs):
        """Test rendering entire configuration objects."""
        config_template = {
            "user": "{{ inputs.user_name }}",
            "threshold": "{{ inputs.threshold }}",
            "advanced": "{{ inputs.enable_advanced }}",
            "nested": {
                "mode": "{{ inputs.config.mode }}",
                "debug": "{{ inputs.config.debug }}"
            }
        }
        
        result = template_engine.render_config(config_template, inputs=sample_inputs)
        
        assert result["user"] == "Alice"
        assert result["threshold"] == 0.7
        assert result["advanced"] is True
        assert result["nested"]["mode"] == "production"
        assert result["nested"]["debug"] is False
    
    def test_render_config_with_lists(self, template_engine, sample_inputs):
        """Test rendering configuration with lists and complex structures."""
        config_template = {
            "tags": "{{ inputs.tags }}",
            "processed_tags": [
                "{{ inputs.tags[0] | upper }}",
                "{{ inputs.tags[1] | upper }}"
            ],
            "conditional_list": [
                "{% if inputs.enable_advanced %}advanced{% endif %}",
                "basic"
            ]
        }
        
        result = template_engine.render_config(config_template, inputs=sample_inputs)
        
        assert result["tags"] == ["important", "urgent"]
        assert result["processed_tags"] == ["IMPORTANT", "URGENT"]
        assert result["conditional_list"] == ["advanced", "basic"]
    
    def test_render_step_success_checks(self, template_engine, sample_context):
        """Test rendering templates that check step success status."""
        # Success check
        template = "{% if steps.extract_text.success %}Text extracted successfully{% endif %}"
        result = template_engine.render(template, context=sample_context)
        assert result == "Text extracted successfully"
        
        # Failure check
        template = "{% if not steps.failed_step.success %}Step failed: {{ steps.failed_step.error }}{% endif %}"
        result = template_engine.render(template, context=sample_context)
        assert result == "Step failed: Processing failed"
        
        # Conditional processing based on success
        template = "{{ steps.extract_text.text if steps.extract_text.success else 'No text available' }}"
        result = template_engine.render(template, context=sample_context)
        assert result == "Hello World"
    
    def test_render_with_custom_functions(self, template_engine):
        """Test rendering with custom template functions."""
        # Test built-in functions
        template = "Current time: {{ now() }}"
        result = template_engine.render(template)
        assert "Current time:" in result
        
        # Test custom functions if any are added
        template = "{{ 'hello' | title }}"
        result = template_engine.render(template)
        assert result == "Hello"
    
    def test_error_handling_undefined_variable(self, template_engine):
        """Test error handling for undefined variables."""
        template = "{{ inputs.nonexistent_variable }}"
        
        with pytest.raises(Exception):  # Should raise TemplateError or similar
            template_engine.render(template, inputs={})
    
    def test_error_handling_invalid_syntax(self, template_engine):
        """Test error handling for invalid template syntax."""
        # Unclosed tag
        template = "{{ inputs.user_name"
        
        with pytest.raises(Exception):  # Should raise TemplateSyntaxError
            template_engine.render(template, inputs={"user_name": "Alice"})
        
        # Invalid expression
        template = "{{ inputs.user_name | invalid_filter }}"
        
        with pytest.raises(Exception):  # Should raise TemplateError
            template_engine.render(template, inputs={"user_name": "Alice"})
    
    def test_render_with_none_values(self, template_engine):
        """Test rendering with None values."""
        inputs = {"value": None, "text": "hello"}
        
        # None value handling
        template = "Value: {{ inputs.value or 'default' }}"
        result = template_engine.render(template, inputs=inputs)
        assert result == "Value: default"
        
        # None in conditional
        template = "{% if inputs.value %}Has value{% else %}No value{% endif %}"
        result = template_engine.render(template, inputs=inputs)
        assert result == "No value"
    
    def test_render_with_empty_context(self, template_engine):
        """Test rendering with empty or minimal context."""
        # No context provided
        template = "Static text"
        result = template_engine.render(template)
        assert result == "Static text"
        
        # Empty inputs
        template = "Default: {{ inputs.value | default('fallback') }}"
        result = template_engine.render(template, inputs={})
        assert result == "Default: fallback"
    
    def test_render_nested_step_access(self, template_engine, sample_context):
        """Test accessing deeply nested step result data."""
        template = "Joy level: {{ steps.analyze_sentiment.emotions.joy }}"
        result = template_engine.render(template, context=sample_context)
        assert result == "Joy level: 0.7"
        
        # Multiple levels of nesting
        template = "Word count: {{ steps.extract_text.metadata.words }}"
        result = template_engine.render(template, context=sample_context)
        assert result == "Word count: 2"
    
    def test_render_array_access(self, template_engine):
        """Test accessing array elements in templates."""
        inputs = {
            "items": ["first", "second", "third"],
            "data": [{"name": "A", "value": 1}, {"name": "B", "value": 2}]
        }
        
        # Array index access
        template = "First item: {{ inputs.items[0] }}"
        result = template_engine.render(template, inputs=inputs)
        assert result == "First item: first"
        
        # Array of objects access
        template = "First name: {{ inputs.data[0].name }}"
        result = template_engine.render(template, inputs=inputs)
        assert result == "First name: A"
        
        # Array length
        template = "Count: {{ inputs.items | length }}"
        result = template_engine.render(template, inputs=inputs)
        assert result == "Count: 3"
    
    def test_render_config_preserves_types(self, template_engine):
        """Test that render_config preserves data types correctly."""
        config_template = {
            "string_value": "{{ inputs.text }}",
            "number_value": "{{ inputs.number }}",
            "boolean_value": "{{ inputs.flag }}",
            "list_value": "{{ inputs.items }}",
            "dict_value": "{{ inputs.config }}"
        }
        
        inputs = {
            "text": "hello",
            "number": 42,
            "flag": True,
            "items": [1, 2, 3],
            "config": {"key": "value"}
        }
        
        result = template_engine.render_config(config_template, inputs=inputs)
        
        assert isinstance(result["string_value"], str)
        assert isinstance(result["number_value"], int)
        assert isinstance(result["boolean_value"], bool)
        assert isinstance(result["list_value"], list)
        assert isinstance(result["dict_value"], dict)
        
        assert result["string_value"] == "hello"
        assert result["number_value"] == 42
        assert result["boolean_value"] is True
        assert result["list_value"] == [1, 2, 3]
        assert result["dict_value"] == {"key": "value"}
    
    def test_render_with_special_characters(self, template_engine):
        """Test rendering templates with special characters."""
        inputs = {
            "text": "Hello, 世界! 🌍",
            "path": "/path/with spaces/file.txt",
            "json": '{"key": "value with quotes"}'
        }
        
        # Unicode characters
        template = "Message: {{ inputs.text }}"
        result = template_engine.render(template, inputs=inputs)
        assert result == "Message: Hello, 世界! 🌍"
        
        # Paths with spaces
        template = "Path: {{ inputs.path }}"
        result = template_engine.render(template, inputs=inputs)
        assert result == "Path: /path/with spaces/file.txt"
        
        # JSON strings
        template = "JSON: {{ inputs.json }}"
        result = template_engine.render(template, inputs=inputs)
        assert result == 'JSON: {"key": "value with quotes"}'
    
    def test_performance_with_large_context(self, template_engine):
        """Test template rendering performance with large context."""
        # Create large context
        large_context = Mock(spec=FlowContext)
        large_context.step_results = {}
        
        # Add many step results
        for i in range(100):
            step_result = Mock()
            step_result.success = True
            step_result.outputs = {
                "data": f"result_{i}",
                "score": i * 0.01,
                "metadata": {"index": i, "processed": True}
            }
            large_context.step_results[f"step_{i}"] = step_result
        
        # Simple template that doesn't access all data
        template = "First result: {{ steps.step_0.data }}"
        result = template_engine.render(template, context=large_context)
        assert result == "First result: result_0"
        
        # Template that accesses multiple steps
        template = "Results: {{ steps.step_0.data }}, {{ steps.step_50.data }}, {{ steps.step_99.data }}"
        result = template_engine.render(template, context=large_context)
        assert result == "Results: result_0, result_50, result_99"
    
    def test_template_caching(self, template_engine):
        """Test template compilation caching."""
        template = "Hello {{ inputs.name }}!"
        
        # First render
        result1 = template_engine.render(template, inputs={"name": "Alice"})
        assert result1 == "Hello Alice!"
        
        # Second render with same template (should use cache)
        result2 = template_engine.render(template, inputs={"name": "Bob"})
        assert result2 == "Hello Bob!"
        
        # Verify template was cached (implementation detail)
        assert len(template_engine._template_cache) > 0
    
    def test_render_config_recursive(self, template_engine):
        """Test recursive rendering of nested configuration structures."""
        config_template = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "{{ inputs.deep_value }}",
                        "computed": "{{ inputs.base * 2 }}"
                    },
                    "array": [
                        "{{ inputs.item1 }}",
                        "{{ inputs.item2 }}"
                    ]
                },
                "simple": "{{ inputs.simple }}"
            }
        }
        
        inputs = {
            "deep_value": "nested",
            "base": 10,
            "item1": "first",
            "item2": "second",
            "simple": "value"
        }
        
        result = template_engine.render_config(config_template, inputs=inputs)
        
        assert result["level1"]["level2"]["level3"]["value"] == "nested"
        assert result["level1"]["level2"]["level3"]["computed"] == 20
        assert result["level1"]["level2"]["array"] == ["first", "second"]
        assert result["level1"]["simple"] == "value"


class TestTemplateEngineIntegration:
    """Integration tests for TemplateEngine with realistic scenarios."""
    
    @pytest.fixture
    def template_engine(self):
        return TemplateEngine()
    
    def test_document_processing_templates(self, template_engine):
        """Test templates for document processing workflow."""
        # Mock realistic document processing context
        context = Mock(spec=FlowContext)
        
        ocr_result = Mock()
        ocr_result.success = True
        ocr_result.outputs = {
            "text": "Invoice #12345\nDate: 2024-01-15\nAmount: $150.00",
            "confidence": 0.95,
            "language": "en",
            "pages": 1
        }
        
        sentiment_result = Mock()
        sentiment_result.success = True
        sentiment_result.outputs = {
            "sentiment": "neutral",
            "confidence": 0.82,
            "emotions": {"neutral": 0.82, "positive": 0.12, "negative": 0.06}
        }
        
        context.step_results = {
            "extract_text": ocr_result,
            "analyze_sentiment": sentiment_result
        }
        
        inputs = {
            "document_type": "invoice",
            "analysis_level": "detailed",
            "output_format": "json"
        }
        
        # Template for LLM analysis prompt
        prompt_template = """
Analyze this {{ inputs.document_type }} document:

Text: {{ steps.extract_text.text }}
Confidence: {{ steps.extract_text.confidence }}
Sentiment: {{ steps.analyze_sentiment.sentiment }}

Provide {{ inputs.analysis_level }} analysis in {{ inputs.output_format }} format.
"""
        
        result = template_engine.render(prompt_template, inputs=inputs, context=context)
        
        assert "Invoice #12345" in result
        assert "Confidence: 0.95" in result
        assert "Sentiment: neutral" in result
        assert "detailed analysis" in result
        assert "json format" in result
    
    def test_conditional_workflow_templates(self, template_engine):
        """Test templates for conditional workflow execution."""
        context = Mock(spec=FlowContext)
        
        # Mock step that might fail
        risky_step = Mock()
        risky_step.success = False
        risky_step.error = "Network timeout"
        risky_step.outputs = {}
        
        # Mock fallback step
        fallback_step = Mock()
        fallback_step.success = True
        fallback_step.outputs = {"result": "fallback_data", "source": "cache"}
        
        context.step_results = {
            "primary_processing": risky_step,
            "fallback_processing": fallback_step
        }
        
        # Conditional template for result selection
        result_template = """
{%- if steps.primary_processing.success -%}
{{ steps.primary_processing.result }}
{%- elif steps.fallback_processing.success -%}
{{ steps.fallback_processing.result }} (from {{ steps.fallback_processing.source }})
{%- else -%}
No data available
{%- endif -%}
"""
        
        result = template_engine.render(result_template, context=context)
        assert result.strip() == "fallback_data (from cache)"
    
    def test_multi_step_aggregation_templates(self, template_engine):
        """Test templates for aggregating multiple step results."""
        context = Mock(spec=FlowContext)
        
        # Mock multiple analysis steps
        for i in range(3):
            step_result = Mock()
            step_result.success = True
            step_result.outputs = {
                "score": 0.7 + (i * 0.1),
                "category": f"category_{i}",
                "confidence": 0.8 + (i * 0.05)
            }
            context.step_results[f"analysis_{i}"] = step_result
        
        # Template for aggregating scores
        aggregation_template = """
{%- set scores = [] -%}
{%- for step_name, step_result in steps.items() -%}
  {%- if step_name.startswith('analysis_') and step_result.success -%}
    {%- set _ = scores.append(step_result.score) -%}
  {%- endif -%}
{%- endfor -%}
Average Score: {{ (scores | sum / scores | length) | round(2) }}
Total Analyses: {{ scores | length }}
Scores: {{ scores | join(', ') }}
"""
        
        result = template_engine.render(aggregation_template, context=context)
        
        assert "Average Score: 0.8" in result
        assert "Total Analyses: 3" in result
        assert "0.7, 0.8, 0.9" in result
    
    def test_error_recovery_templates(self, template_engine):
        """Test templates for error recovery scenarios."""
        context = Mock(spec=FlowContext)
        
        # Mix of successful and failed steps
        success_step = Mock()
        success_step.success = True
        success_step.outputs = {"data": "success_data"}
        
        failed_step = Mock()
        failed_step.success = False
        failed_step.error = "Processing failed"
        failed_step.outputs = {}
        
        context.step_results = {
            "step_1": success_step,
            "step_2": failed_step,
            "step_3": success_step
        }
        
        # Template for error reporting
        error_report_template = """
{%- set failed_steps = [] -%}
{%- set successful_steps = [] -%}
{%- for step_name, step_result in steps.items() -%}
  {%- if step_result.success -%}
    {%- set _ = successful_steps.append(step_name) -%}
  {%- else -%}
    {%- set _ = failed_steps.append({'name': step_name, 'error': step_result.error}) -%}
  {%- endif -%}
{%- endfor -%}
Execution Summary:
- Successful: {{ successful_steps | length }} steps
- Failed: {{ failed_steps | length }} steps
{%- if failed_steps %}

Failed Steps:
{%- for failed in failed_steps %}
- {{ failed.name }}: {{ failed.error }}
{%- endfor %}
{%- endif %}
"""
        
        result = template_engine.render(error_report_template, context=context)
        
        assert "Successful: 2 steps" in result
        assert "Failed: 1 steps" in result
        assert "step_2: Processing failed" in result
