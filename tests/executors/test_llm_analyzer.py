"""
Unit tests for LLMAnalyzer executor.

Tests LLM-based text analysis, custom prompts, multiple models,
chunked processing, and provider integration.
"""

import pytest
import uuid
from unittest.mock import Mock, patch, AsyncMock

from core.executors.llm_analyzer import LLMAnalyzer
from core.executors.base_executor import FlowContext, ExecutionResult


class TestLLMAnalyzer:
    """Test suite for LLMAnalyzer executor."""
    
    @pytest.fixture
    def llm_analyzer(self):
        """Create LLMAnalyzer instance for testing."""
        return LLMAnalyzer()
    
    @pytest.fixture
    def mock_context(self):
        """Create mock FlowContext for testing."""
        return FlowContext(
            flow_name="test_flow",
            inputs={"test_param": "test_value"},
            flow_id=f"test-flow-{uuid.uuid4().hex[:8]}"
        )
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for testing."""
        mock_response = Mock()
        mock_response.content = "This is a comprehensive analysis of the provided text. The content appears to be positive and informative."
        return mock_response
    
    @pytest.fixture
    def sample_text_config(self):
        """Sample configuration for single text analysis."""
        return {
            "text": "This is a sample text for analysis. It contains multiple sentences and should provide good material for LLM analysis.",
            "prompt": "Analyze the sentiment and key themes in this text",
            "model": "mistral",
            "provider": "ollama"
        }
    
    @pytest.fixture
    def sample_chunks_config(self):
        """Sample configuration for chunked text analysis."""
        return {
            "chunks": [
                {"text": "First chunk of text for analysis."},
                {"text": "Second chunk with different content."},
                {"text": "Third chunk completing the analysis."}
            ],
            "prompt": "Analyze the main themes in this text chunk",
            "model": "mistral",
            "provider": "ollama",
            "combine_chunks": True
        }
    
    @pytest.mark.asyncio
    async def test_successful_single_text_analysis(self, llm_analyzer, mock_context, sample_text_config, mock_llm_response):
        """Test successful single text analysis."""
        with patch.object(llm_analyzer, '_get_llm_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_manager.return_value = mock_manager
            
            result = await llm_analyzer.execute(mock_context, sample_text_config)
            
            assert result.success is True
            assert "analysis" in result.outputs
            assert "prompt" in result.outputs
            assert "model" in result.outputs
            assert "provider" in result.outputs
            assert "text_length" in result.outputs
            assert "analysis_type" in result.outputs
            
            assert result.outputs["analysis"] == mock_llm_response.content
            assert result.outputs["model"] == "mistral"
            assert result.outputs["provider"] == "ollama"
            assert result.outputs["analysis_type"] == "single_text"
            
            # Verify metadata
            assert "input_characters" in result.metadata
            assert "output_characters" in result.metadata
            assert "model_used" in result.metadata
            assert "provider_used" in result.metadata
    
    @pytest.mark.asyncio
    async def test_successful_chunked_analysis(self, llm_analyzer, mock_context, sample_chunks_config, mock_llm_response):
        """Test successful chunked text analysis."""
        with patch.object(llm_analyzer, '_get_llm_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_manager.return_value = mock_manager
            
            result = await llm_analyzer.execute(mock_context, sample_chunks_config)
            
            assert result.success is True
            assert "chunk_analyses" in result.outputs
            assert "combined_analysis" in result.outputs
            assert "total_chunks" in result.outputs
            assert "successful_analyses" in result.outputs
            assert "analysis_type" in result.outputs
            
            assert result.outputs["analysis_type"] == "chunked_text"
            assert result.outputs["total_chunks"] == 3
            assert result.outputs["successful_analyses"] == 3
            assert len(result.outputs["chunk_analyses"]) == 3
            
            # Verify chunk analysis structure
            for i, chunk_analysis in enumerate(result.outputs["chunk_analyses"]):
                assert "chunk_index" in chunk_analysis
                assert "analysis" in chunk_analysis
                assert "chunk_text_length" in chunk_analysis
                assert chunk_analysis["chunk_index"] == i
    
    @pytest.mark.asyncio
    async def test_missing_text_and_chunks(self, llm_analyzer, mock_context):
        """Test error when both text and chunks are missing."""
        config = {
            "prompt": "Analyze this text"
        }
        
        result = await llm_analyzer.execute(mock_context, config)
        
        assert result.success is False
        assert "Either 'text' or 'chunks' must be provided" in result.error
    
    @pytest.mark.asyncio
    async def test_missing_prompt(self, llm_analyzer, mock_context):
        """Test error when prompt is missing."""
        config = {
            "text": "Sample text for analysis"
        }
        
        result = await llm_analyzer.execute(mock_context, config)
        
        assert result.success is False
        assert "Analysis prompt is required" in result.error
    
    @pytest.mark.asyncio
    async def test_default_model_and_provider(self, llm_analyzer, mock_context, mock_llm_response):
        """Test default model and provider selection."""
        config = {
            "text": "Sample text",
            "prompt": "Analyze this text"
        }
        
        with patch.object(llm_analyzer, '_get_llm_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_manager.return_value = mock_manager
            
            result = await llm_analyzer.execute(mock_context, config)
            
            assert result.success is True
            assert result.outputs["model"] == "mistral"  # Default model
            assert result.outputs["provider"] == "ollama"  # Default provider
    
    @pytest.mark.asyncio
    async def test_custom_model_and_provider(self, llm_analyzer, mock_context, mock_llm_response):
        """Test custom model and provider configuration."""
        config = {
            "text": "Sample text",
            "prompt": "Analyze this text",
            "model": "gpt-4",
            "provider": "openai"
        }
        
        with patch.object(llm_analyzer, '_get_llm_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_manager.return_value = mock_manager
            
            result = await llm_analyzer.execute(mock_context, config)
            
            assert result.success is True
            assert result.outputs["model"] == "gpt-4"
            assert result.outputs["provider"] == "openai"
    
    @pytest.mark.asyncio
    async def test_chunked_analysis_without_combination(self, llm_analyzer, mock_context, mock_llm_response):
        """Test chunked analysis without combining results."""
        config = {
            "chunks": [
                {"text": "First chunk"},
                {"text": "Second chunk"}
            ],
            "prompt": "Analyze this chunk",
            "combine_chunks": False
        }
        
        with patch.object(llm_analyzer, '_get_llm_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_manager.return_value = mock_manager
            
            result = await llm_analyzer.execute(mock_context, config)
            
            assert result.success is True
            assert result.outputs["combined_analysis"] is None
            assert len(result.outputs["chunk_analyses"]) == 2
    
    @pytest.mark.asyncio
    async def test_empty_chunks_handling(self, llm_analyzer, mock_context, mock_llm_response):
        """Test handling of empty chunks."""
        config = {
            "chunks": [
                {"text": "Valid chunk"},
                {"text": ""},  # Empty chunk
                {"text": "   "},  # Whitespace only
                {"text": "Another valid chunk"}
            ],
            "prompt": "Analyze this chunk"
        }
        
        with patch.object(llm_analyzer, '_get_llm_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_manager.return_value = mock_manager
            
            result = await llm_analyzer.execute(mock_context, config)
            
            assert result.success is True
            # Should only process non-empty chunks
            assert len(result.outputs["chunk_analyses"]) == 2
            assert result.outputs["successful_analyses"] == 2
    
    @pytest.mark.asyncio
    async def test_llm_generation_error(self, llm_analyzer, mock_context):
        """Test handling of LLM generation errors."""
        config = {
            "text": "Sample text",
            "prompt": "Analyze this text"
        }
        
        with patch.object(llm_analyzer, '_get_llm_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.generate = AsyncMock(side_effect=Exception("LLM service unavailable"))
            mock_get_manager.return_value = mock_manager
            
            result = await llm_analyzer.execute(mock_context, config)
            
            assert result.success is False
            assert "LLM analysis failed" in result.error
            assert "LLM service unavailable" in result.error
    
    @pytest.mark.asyncio
    async def test_chunk_analysis_partial_failure(self, llm_analyzer, mock_context, mock_llm_response):
        """Test handling when some chunks fail to analyze."""
        config = {
            "chunks": [
                {"text": "First chunk"},
                {"text": "Second chunk"},
                {"text": "Third chunk"}
            ],
            "prompt": "Analyze this chunk"
        }
        
        with patch.object(llm_analyzer, '_get_llm_manager') as mock_get_manager:
            mock_manager = Mock()
            # First call succeeds, second fails, third succeeds
            mock_manager.generate = AsyncMock(side_effect=[
                mock_llm_response,
                Exception("Temporary failure"),
                mock_llm_response
            ])
            mock_get_manager.return_value = mock_manager
            
            result = await llm_analyzer.execute(mock_context, config)
            
            assert result.success is True
            assert len(result.outputs["chunk_analyses"]) == 3
            assert result.outputs["successful_analyses"] == 2
            
            # Check that failed chunk is marked with error
            failed_chunk = result.outputs["chunk_analyses"][1]
            assert failed_chunk.get("error") is True
            assert "Analysis failed" in failed_chunk["analysis"]
    
    @pytest.mark.asyncio
    async def test_chunk_metadata_preservation(self, llm_analyzer, mock_context, mock_llm_response):
        """Test that chunk metadata is preserved in results."""
        config = {
            "chunks": [
                {
                    "text": "First chunk",
                    "metadata": {"source": "document1.pdf", "page": 1}
                },
                {
                    "text": "Second chunk",
                    "metadata": {"source": "document2.pdf", "page": 2}
                }
            ],
            "prompt": "Analyze this chunk"
        }
        
        with patch.object(llm_analyzer, '_get_llm_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.generate = AsyncMock(return_value=mock_llm_response)
            mock_get_manager.return_value = mock_manager
            
            result = await llm_analyzer.execute(mock_context, config)
            
            assert result.success is True
            
            # Check metadata preservation
            chunk1 = result.outputs["chunk_analyses"][0]
            chunk2 = result.outputs["chunk_analyses"][1]
            
            assert "chunk_metadata" in chunk1
            assert "chunk_metadata" in chunk2
            assert chunk1["chunk_metadata"]["source"] == "document1.pdf"
            assert chunk2["chunk_metadata"]["page"] == 2
    
    def test_llm_analyzer_name(self, llm_analyzer):
        """Test LLMAnalyzer name property."""
        assert llm_analyzer.name == "LLMAnalyzer"
    
    def test_get_llm_manager_singleton(self, llm_analyzer):
        """Test that LLM manager is created as singleton."""
        with patch('core.executors.llm_analyzer.LLMManager') as mock_manager_class:
            mock_instance = Mock()
            mock_manager_class.return_value = mock_instance
            
            # First call should create manager
            manager1 = llm_analyzer._get_llm_manager()
            assert manager1 == mock_instance
            
            # Second call should return same instance
            manager2 = llm_analyzer._get_llm_manager()
            assert manager2 == mock_instance
            assert manager1 is manager2
            
            # Manager should only be created once
            mock_manager_class.assert_called_once()
    
    def test_required_config_keys(self, llm_analyzer):
        """Test required configuration keys."""
        required_keys = llm_analyzer.get_required_config_keys()
        assert "prompt" in required_keys
    
    def test_optional_config_keys(self, llm_analyzer):
        """Test optional configuration keys."""
        optional_keys = llm_analyzer.get_optional_config_keys()
        expected_keys = ["text", "chunks", "model", "provider", "combine_chunks"]
        
        for key in expected_keys:
            assert key in optional_keys
    
    def test_config_validation_success(self, llm_analyzer):
        """Test successful configuration validation."""
        valid_configs = [
            {"prompt": "Analyze", "text": "Sample text"},
            {"prompt": "Analyze", "chunks": [{"text": "chunk1"}]},
            {"prompt": "Analyze", "text": "Sample", "combine_chunks": True}
        ]
        
        for config in valid_configs:
            # Should not raise exception
            llm_analyzer.validate_config(config)
    
    def test_config_validation_failures(self, llm_analyzer):
        """Test configuration validation failures."""
        invalid_configs = [
            {"prompt": "Analyze"},  # Missing text and chunks
            {"prompt": "Analyze", "chunks": "not_a_list"},  # Invalid chunks format
            {"prompt": "Analyze", "text": "Sample", "combine_chunks": "not_boolean"}  # Invalid boolean
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                llm_analyzer.validate_config(config)
    
    def test_get_info(self, llm_analyzer):
        """Test executor information."""
        info = llm_analyzer.get_info()
        
        assert "capabilities" in info
        assert "supported_providers" in info
        assert "supported_models" in info
        
        # Check capabilities
        capabilities = info["capabilities"]
        assert "Text analysis using Large Language Models" in capabilities
        assert "Custom prompt support" in capabilities
        assert "Chunked text processing" in capabilities
        
        # Check supported providers
        assert "ollama" in info["supported_providers"]
        assert "openai" in info["supported_providers"]
        
        # Check supported models
        assert "mistral" in info["supported_models"]["ollama"]
        assert "gpt-4" in info["supported_models"]["openai"]


class TestLLMAnalyzerIntegration:
    """Integration tests for LLMAnalyzer with flow context."""
    
    @pytest.mark.asyncio
    async def test_llm_analyzer_in_flow_context(self):
        """Test LLMAnalyzer as part of a flow context."""
        llm_analyzer = LLMAnalyzer()
        context = FlowContext(
            flow_name="text_analysis_flow",
            inputs={"document": "Sample document content"},
            flow_id="integration-test"
        )
        
        # Add previous step result (e.g., from document extraction)
        previous_result = ExecutionResult(
            success=True,
            outputs={"extracted_text": "This is extracted text from a document. It contains important information."}
        )
        context.add_step_result("document_extraction", previous_result)
        
        config = {
            "text": context.step_results["document_extraction"].outputs["extracted_text"],
            "prompt": "Summarize the key points and identify the main themes in this text"
        }
        
        with patch.object(llm_analyzer, '_get_llm_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_response = Mock()
            mock_response.content = "The text discusses important information with key themes including documentation and content analysis."
            mock_manager.generate = AsyncMock(return_value=mock_response)
            mock_get_manager.return_value = mock_manager
            
            result = await llm_analyzer.execute(context, config)
            
            # Add result to context
            context.add_step_result("llm_analysis", result)
            
            # Verify integration
            assert result.success is True
            assert context.step_results["llm_analysis"] is not None
            
            # Verify the result can be used by subsequent steps
            analysis_result = context.step_results["llm_analysis"]
            assert "analysis" in analysis_result.outputs
            assert analysis_result.outputs["analysis"] == mock_response.content
