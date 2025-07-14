"""
Comprehensive test suite for DataCombiner executor.
Tests all combination strategies, error scenarios, and edge cases.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch

from core.executors.data_combiner import DataCombiner
from core.executors.base_executor import ExecutionResult, FlowContext


class TestDataCombiner:
    """Test suite for DataCombiner executor."""
    
    @pytest.fixture
    def combiner(self):
        """Create DataCombiner instance."""
        return DataCombiner()
    
    @pytest.fixture
    def mock_context(self):
        """Create mock flow context with sample step results."""
        context = Mock(spec=FlowContext)
        
        # Mock step results
        step1_result = Mock()
        step1_result.success = True
        step1_result.outputs = {"text": "Hello", "score": 0.8}
        step1_result.timestamp = "2024-01-01T00:00:00"
        
        step2_result = Mock()
        step2_result.success = True
        step2_result.outputs = {"text": "World", "score": 0.9}
        step2_result.timestamp = "2024-01-01T00:01:00"
        
        step3_result = Mock()
        step3_result.success = True
        step3_result.outputs = {"items": ["a", "b", "c"], "count": 3}
        
        context.step_results = {
            "step1": step1_result,
            "step2": step2_result,
            "step3": step3_result
        }
        
        return context
    
    @pytest.mark.asyncio
    async def test_merge_strategy_basic(self, combiner, mock_context):
        """Test basic merge strategy with dictionary combination."""
        config = {
            "sources": ["step1", "step2"],
            "strategy": "merge",
            "output_key": "merged_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        assert "merged_data" in result.outputs
        merged = result.outputs["merged_data"]
        assert merged["text"] == "World"  # Last value wins (overwrite)
        assert merged["score"] == 0.9
    
    @pytest.mark.asyncio
    async def test_merge_strategy_keep_first(self, combiner, mock_context):
        """Test merge strategy with keep_first conflict resolution."""
        config = {
            "sources": ["step1", "step2"],
            "strategy": "merge",
            "merge_strategy": "keep_first",
            "output_key": "merged_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        merged = result.outputs["merged_data"]
        assert merged["text"] == "Hello"  # First value kept
        assert merged["score"] == 0.8
    
    @pytest.mark.asyncio
    async def test_merge_strategy_combine(self, combiner, mock_context):
        """Test merge strategy with combine conflict resolution."""
        config = {
            "sources": ["step1", "step2"],
            "strategy": "merge",
            "merge_strategy": "combine",
            "output_key": "merged_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        merged = result.outputs["merged_data"]
        assert merged["text"] == "Hello World"  # Strings combined
        assert merged["score"] == [0.8, 0.9]  # Values combined in list
    
    @pytest.mark.asyncio
    async def test_concat_strategy(self, combiner, mock_context):
        """Test concatenation strategy with list combination."""
        # Add list data to context
        step4_result = Mock()
        step4_result.outputs = ["x", "y", "z"]
        mock_context.step_results["step4"] = step4_result
        
        config = {
            "sources": ["step3", "step4"],
            "strategy": "concat",
            "output_key": "concatenated_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        concat_data = result.outputs["concatenated_data"]
        # step3.outputs is a dict, so it gets appended as single item
        # step4.outputs is a list, so it gets extended
        assert len(concat_data) == 4  # 1 dict + 3 list items
        assert concat_data[0] == {"items": ["a", "b", "c"], "count": 3}
        assert concat_data[1:] == ["x", "y", "z"]
    
    @pytest.mark.asyncio
    async def test_join_strategy_basic(self, combiner, mock_context):
        """Test join strategy with text combination."""
        config = {
            "sources": ["step1", "step2"],
            "strategy": "join",
            "output_key": "joined_text"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        joined = result.outputs["joined_text"]
        # Should extract 'text' field from dictionaries
        assert "Hello" in joined
        assert "World" in joined
    
    @pytest.mark.asyncio
    async def test_join_strategy_custom_separator(self, combiner, mock_context):
        """Test join strategy with custom separator."""
        config = {
            "sources": ["step1", "step2"],
            "strategy": "join",
            "join_separator": " | ",
            "output_key": "joined_text"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        joined = result.outputs["joined_text"]
        assert " | " in joined
    
    @pytest.mark.asyncio
    async def test_structured_strategy_default(self, combiner, mock_context):
        """Test structured strategy with default structure."""
        config = {
            "sources": ["step1", "step2"],
            "strategy": "structured",
            "output_key": "structured_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        structured = result.outputs["structured_data"]
        assert "source_0" in structured
        assert "source_1" in structured
        assert structured["source_0"] == {"text": "Hello", "score": 0.8}
        assert structured["source_1"] == {"text": "World", "score": 0.9}
    
    @pytest.mark.asyncio
    async def test_structured_strategy_template(self, combiner, mock_context):
        """Test structured strategy with custom template."""
        config = {
            "sources": ["step1", "step2"],
            "strategy": "structured",
            "structure_template": {
                "first_text": "0.text",
                "second_score": "1.score",
                "full_first": 0
            },
            "output_key": "structured_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        structured = result.outputs["structured_data"]
        assert structured["first_text"] == "Hello"
        assert structured["second_score"] == 0.9
        assert structured["full_first"] == {"text": "Hello", "score": 0.8}
    
    @pytest.mark.asyncio
    async def test_aggregate_strategy_basic(self, combiner, mock_context):
        """Test aggregate strategy with numeric data."""
        config = {
            "sources": ["step1", "step2"],
            "strategy": "aggregate",
            "aggregations": ["count", "sum", "avg", "min", "max"],
            "output_key": "aggregated_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        agg = result.outputs["aggregated_data"]
        assert agg["count"] == 2  # Two numeric values (0.8, 0.9)
        assert agg["sum"] == 1.7
        assert agg["average"] == 0.85
        assert agg["minimum"] == 0.8
        assert agg["maximum"] == 0.9
        assert agg["total_sources"] == 2
    
    @pytest.mark.asyncio
    async def test_aggregate_strategy_with_text(self, combiner, mock_context):
        """Test aggregate strategy including text aggregation."""
        config = {
            "sources": ["step1", "step2"],
            "strategy": "aggregate",
            "aggregations": ["count", "sum", "concat_text"],
            "output_key": "aggregated_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        agg = result.outputs["aggregated_data"]
        assert agg["text_count"] == 2
        assert "Hello" in agg["combined_text"]
        assert "World" in agg["combined_text"]
    
    @pytest.mark.asyncio
    async def test_direct_values_as_sources(self, combiner, mock_context):
        """Test using direct values instead of step references."""
        config = {
            "sources": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
                "Direct string value"
            ],
            "strategy": "merge",
            "output_key": "combined_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        combined = result.outputs["combined_data"]
        assert combined["name"] == "Bob"  # Last value wins
        assert combined["age"] == 25
        assert "source_0" in combined  # String value added with index
    
    @pytest.mark.asyncio
    async def test_include_metadata(self, combiner, mock_context):
        """Test including combination metadata in output."""
        config = {
            "sources": ["step1", "step2"],
            "strategy": "merge",
            "include_metadata": True,
            "output_key": "combined_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        assert "combination_metadata" in result.outputs
        metadata = result.outputs["combination_metadata"]
        assert metadata["sources_count"] == 2
        assert metadata["strategy"] == "merge"
        assert "timestamp" in metadata
        assert "source_info" in metadata
        assert "step1" in metadata["source_info"]
        assert "step2" in metadata["source_info"]
    
    @pytest.mark.asyncio
    async def test_transformations_rename(self, combiner, mock_context):
        """Test output transformations - renaming keys."""
        config = {
            "sources": ["step1"],
            "strategy": "merge",
            "output_key": "data",
            "transform": {
                "rename": {"data": "renamed_data"}
            }
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        assert "data" not in result.outputs
        assert "renamed_data" in result.outputs
    
    @pytest.mark.asyncio
    async def test_transformations_include_only(self, combiner, mock_context):
        """Test output transformations - include only specific keys."""
        config = {
            "sources": ["step1"],
            "strategy": "merge",
            "output_key": "data",
            "include_metadata": True,
            "transform": {
                "include_only": ["data"]
            }
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        assert "data" in result.outputs
        assert "combination_metadata" not in result.outputs
    
    @pytest.mark.asyncio
    async def test_transformations_exclude(self, combiner, mock_context):
        """Test output transformations - exclude specific keys."""
        config = {
            "sources": ["step1"],
            "strategy": "merge",
            "output_key": "data",
            "include_metadata": True,
            "transform": {
                "exclude": ["combination_metadata"]
            }
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        assert "data" in result.outputs
        assert "combination_metadata" not in result.outputs
    
    @pytest.mark.asyncio
    async def test_transformations_format(self, combiner, mock_context):
        """Test output transformations - formatting values."""
        config = {
            "sources": [{"message": "hello world"}],
            "strategy": "merge",
            "output_key": "data",
            "transform": {
                "format": {
                    "data": "json"
                }
            }
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        # Data should be JSON string
        assert isinstance(result.outputs["data"], str)
        parsed = json.loads(result.outputs["data"])
        assert parsed["message"] == "hello world"
    
    @pytest.mark.asyncio
    async def test_error_no_sources(self, combiner, mock_context):
        """Test error handling when no sources specified."""
        config = {
            "strategy": "merge"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is False
        assert "No sources specified" in result.error
    
    @pytest.mark.asyncio
    async def test_error_empty_sources(self, combiner, mock_context):
        """Test error handling when sources list is empty."""
        config = {
            "sources": [],
            "strategy": "merge"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is False
        assert "No sources specified" in result.error
    
    @pytest.mark.asyncio
    async def test_error_invalid_step_reference(self, combiner, mock_context):
        """Test handling of invalid step references."""
        config = {
            "sources": ["nonexistent_step"],
            "strategy": "merge"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is False
        assert "No valid source data found" in result.error
    
    @pytest.mark.asyncio
    async def test_error_unknown_strategy(self, combiner, mock_context):
        """Test error handling for unknown combination strategy."""
        config = {
            "sources": ["step1"],
            "strategy": "unknown_strategy"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is False
        assert "Unknown combination strategy" in result.error
    
    @pytest.mark.asyncio
    async def test_mixed_source_types(self, combiner, mock_context):
        """Test combining step references with direct values."""
        config = {
            "sources": [
                "step1",  # Step reference
                {"direct": "value"},  # Direct dict
                "direct string",  # Direct string
                42  # Direct number
            ],
            "strategy": "concat",
            "output_key": "mixed_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        mixed = result.outputs["mixed_data"]
        assert len(mixed) == 4
        assert mixed[0] == {"text": "Hello", "score": 0.8}  # From step1
        assert mixed[1] == {"direct": "value"}
        assert mixed[2] == "direct string"
        assert mixed[3] == 42
    
    @pytest.mark.asyncio
    async def test_step_result_without_outputs(self, combiner, mock_context):
        """Test handling step results that don't have outputs attribute."""
        # Add a step result without outputs
        raw_result = {"raw": "data"}
        mock_context.step_results["raw_step"] = raw_result
        
        config = {
            "sources": ["raw_step"],
            "strategy": "merge",
            "output_key": "combined_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        combined = result.outputs["combined_data"]
        assert "source_0" in combined
        assert combined["source_0"] == {"raw": "data"}
    
    @pytest.mark.asyncio
    async def test_join_strategy_with_content_field(self, combiner, mock_context):
        """Test join strategy extracting 'content' field."""
        # Add step with content field
        step_content = Mock()
        step_content.outputs = {"content": "Content text", "other": "data"}
        mock_context.step_results["content_step"] = step_content
        
        config = {
            "sources": ["content_step"],
            "strategy": "join",
            "output_key": "joined_text"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        assert result.outputs["joined_text"] == "Content text"
    
    @pytest.mark.asyncio
    async def test_join_strategy_fallback_to_json(self, combiner, mock_context):
        """Test join strategy falling back to JSON for complex objects."""
        # Add step without text or content fields
        step_complex = Mock()
        step_complex.outputs = {"complex": {"nested": "data"}, "number": 123}
        mock_context.step_results["complex_step"] = step_complex
        
        config = {
            "sources": ["complex_step"],
            "strategy": "join",
            "output_key": "joined_text"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        joined = result.outputs["joined_text"]
        # Should be JSON representation
        assert "complex" in joined
        assert "nested" in joined
    
    @pytest.mark.asyncio
    async def test_structured_template_invalid_reference(self, combiner, mock_context):
        """Test structured strategy with invalid template references."""
        config = {
            "sources": ["step1"],
            "strategy": "structured",
            "structure_template": {
                "invalid_index": "5.text",  # Index out of range
                "invalid_path": "0.nonexistent.field",  # Invalid path
                "valid": "0.text"
            },
            "output_key": "structured_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        structured = result.outputs["structured_data"]
        assert structured["invalid_index"] is None
        assert structured["invalid_path"] is None
        assert structured["valid"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_aggregate_with_no_numeric_data(self, combiner, mock_context):
        """Test aggregate strategy when no numeric data is available."""
        # Create step with only text data
        text_step = Mock()
        text_step.outputs = {"message": "Only text here"}
        mock_context.step_results["text_only"] = text_step
        
        config = {
            "sources": ["text_only"],
            "strategy": "aggregate",
            "output_key": "aggregated_data"
        }
        
        result = await combiner.execute(mock_context, config)
        
        assert result.success is True
        agg = result.outputs["aggregated_data"]
        assert agg["text_count"] == 1
        assert agg["total_sources"] == 1
        # Numeric aggregations should not be present
        assert "count" not in agg
        assert "sum" not in agg
    
    @pytest.mark.asyncio
    async def test_transformation_format_string_operations(self, combiner, mock_context):
        """Test string formatting transformations."""
        config = {
            "sources": [{"text": "Hello World"}],
            "strategy": "merge",
            "output_key": "data",
            "transform": {
                "format": {
                    "data": "upper"
                }
            }
        }
        
        # Since data is a dict, string formatting won't apply
        result = await combiner.execute(mock_context, config)
        assert result.success is True
        
        # Test with string output
        config["sources"] = ["Hello World"]
        config["transform"]["format"]["data"] = "upper"
        
        result = await combiner.execute(mock_context, config)
        assert result.success is True
        # String should be in the merged result under source_0
    
    def test_get_required_config(self, combiner):
        """Test required configuration keys."""
        required = combiner.get_required_config()
        assert required == ['sources']
    
    def test_get_optional_config(self, combiner):
        """Test optional configuration keys and defaults."""
        optional = combiner.get_optional_config()
        assert optional['strategy'] == 'merge'
        assert optional['output_key'] == 'combined_result'
        assert optional['join_separator'] == ' '
        assert optional['merge_strategy'] == 'overwrite'
        assert optional['include_metadata'] is False
        assert optional['transform'] == {}
        assert optional['structure_template'] == {}
        assert optional['aggregations'] == ['count', 'sum']
    
    def test_executor_name_default(self):
        """Test default executor name."""
        combiner = DataCombiner()
        assert combiner.name == "data_combiner"
    
    def test_executor_name_custom(self):
        """Test custom executor name."""
        combiner = DataCombiner("custom_combiner")
        assert combiner.name == "custom_combiner"
    
    @pytest.mark.asyncio
    async def test_exception_handling(self, combiner, mock_context):
        """Test general exception handling."""
        # Mock an exception in strategy application
        with patch.object(combiner, '_apply_strategy', side_effect=Exception("Test error")):
            config = {
                "sources": ["step1"],
                "strategy": "merge"
            }
            
            result = await combiner.execute(mock_context, config)
            
            assert result.success is False
            assert "Data combination failed" in result.error
            assert "Test error" in result.error


class TestDataCombinerIntegration:
    """Integration tests for DataCombiner with realistic scenarios."""
    
    @pytest.fixture
    def realistic_context(self):
        """Create realistic flow context with AI processing results."""
        context = Mock(spec=FlowContext)
        
        # OCR result
        ocr_result = Mock()
        ocr_result.success = True
        ocr_result.outputs = {
            "text": "Invoice #12345\nDate: 2024-01-15\nAmount: $150.00",
            "confidence": 0.95,
            "language": "en"
        }
        
        # Sentiment analysis result
        sentiment_result = Mock()
        sentiment_result.success = True
        sentiment_result.outputs = {
            "sentiment": "neutral",
            "confidence": 0.82,
            "emotions": {"neutral": 0.82, "positive": 0.12, "negative": 0.06}
        }
        
        # LLM analysis result
        llm_result = Mock()
        llm_result.success = True
        llm_result.outputs = {
            "summary": "This is an invoice document with payment details",
            "key_entities": ["Invoice", "Date", "Amount"],
            "document_type": "invoice"
        }
        
        context.step_results = {
            "ocr_extraction": ocr_result,
            "sentiment_analysis": sentiment_result,
            "llm_analysis": llm_result
        }
        
        return context
    
    @pytest.mark.asyncio
    async def test_document_processing_pipeline_combination(self, realistic_context):
        """Test combining results from a complete document processing pipeline."""
        combiner = DataCombiner("document_combiner")
        
        config = {
            "sources": ["ocr_extraction", "sentiment_analysis", "llm_analysis"],
            "strategy": "structured",
            "structure_template": {
                "extracted_text": "0.text",
                "text_confidence": "0.confidence",
                "sentiment_score": "1.sentiment",
                "sentiment_confidence": "1.confidence",
                "document_summary": "2.summary",
                "document_type": "2.document_type",
                "key_entities": "2.key_entities"
            },
            "output_key": "document_analysis_result",
            "include_metadata": True
        }
        
        result = await combiner.execute(realistic_context, config)
        
        assert result.success is True
        
        # Verify structured output
        doc_result = result.outputs["document_analysis_result"]
        assert doc_result["extracted_text"] == "Invoice #12345\nDate: 2024-01-15\nAmount: $150.00"
        assert doc_result["text_confidence"] == 0.95
        assert doc_result["sentiment_score"] == "neutral"
        assert doc_result["sentiment_confidence"] == 0.82
        assert doc_result["document_summary"] == "This is an invoice document with payment details"
        assert doc_result["document_type"] == "invoice"
        assert doc_result["key_entities"] == ["Invoice", "Date", "Amount"]
        
        # Verify metadata
        assert "combination_metadata" in result.outputs
        metadata = result.outputs["combination_metadata"]
        assert metadata["sources_count"] == 3
        assert metadata["strategy"] == "structured"
    
    @pytest.mark.asyncio
    async def test_multi_document_aggregation(self, realistic_context):
        """Test aggregating results from multiple similar documents."""
        # Add more document results
        doc2_ocr = Mock()
        doc2_ocr.outputs = {"text": "Receipt #67890", "confidence": 0.88}
        doc3_ocr = Mock()
        doc3_ocr.outputs = {"text": "Contract #11111", "confidence": 0.92}
        
        realistic_context.step_results.update({
            "doc2_ocr": doc2_ocr,
            "doc3_ocr": doc3_ocr
        })
        
        combiner = DataCombiner("multi_doc_combiner")
        
        config = {
            "sources": ["ocr_extraction", "doc2_ocr", "doc3_ocr"],
            "strategy": "aggregate",
            "aggregations": ["count", "avg", "min", "max", "concat_text"],
            "output_key": "multi_document_summary",
            "transform": {
                "rename": {"multi_document_summary": "aggregated_documents"}
            }
        }
        
        result = await combiner.execute(realistic_context, config)
        
        assert result.success is True
        
        # Verify aggregation
        agg_docs = result.outputs["aggregated_documents"]
        assert agg_docs["count"] == 3  # Three confidence scores
        assert agg_docs["average"] == (0.95 + 0.88 + 0.92) / 3
        assert agg_docs["minimum"] == 0.88
        assert agg_docs["maximum"] == 0.95
        assert agg_docs["text_count"] == 3
        assert "Invoice" in agg_docs["combined_text"]
        assert "Receipt" in agg_docs["combined_text"]
        assert "Contract" in agg_docs["combined_text"]
    
    @pytest.mark.asyncio
    async def test_error_recovery_combination(self, realistic_context):
        """Test combining results when some steps failed."""
        # Add a failed step result
        failed_step = Mock()
        failed_step.success = False
        failed_step.outputs = {}
        realistic_context.step_results["failed_processing"] = failed_step
        
        combiner = DataCombiner("error_recovery_combiner")
        
        config = {
            "sources": ["ocr_extraction", "failed_processing", "sentiment_analysis"],
            "strategy": "merge",
            "output_key": "recovered_results",
            "include_metadata": True
        }
        
        result = await combiner.execute(realistic_context, config)
        
        assert result.success is True
        
        # Should successfully combine available results
        recovered = result.outputs["recovered_results"]
        assert "text" in recovered  # From OCR
        assert "sentiment" in recovered  # From sentiment analysis
        
        # Metadata should show the failed step
        metadata = result.outputs["combination_metadata"]
        assert "failed_processing" in metadata["source_info"]
        assert metadata["source_info"]["failed_processing"]["success"] is False
