"""
Unit tests for LLM Analyzer Executor.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch

from core.executors.llm_analyzer import LLMAnalyzer
from core.executors.base_executor import FlowContext, ExecutionResult
from core.llm.llm_manager import LLMManager
from core.llm.base_provider import LLMResponse


class TestLLMAnalyzer:
    """Test LLMAnalyzer executor."""
    
    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager."""
        manager = Mock(spec=LLMManager)
        manager.generate = AsyncMock()
        return manager
    
    @pytest.fixture
    def llm_analyzer(self, mock_llm_manager):
        """Create LLM analyzer with mocked dependencies."""
        analyzer = LLMAnalyzer()
        analyzer.llm_manager = mock_llm_manager
        return analyzer
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample flow context."""
        return FlowContext(
            flow_id="test_flow",
            inputs={"text": "Sample text to analyze", "user_prompt": "Analyze this text"},
            config={"model": "mistral", "temperature": 0.7}
        )
    
    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        return LLMResponse(
            content="This is the analysis result from the LLM",
            model="mistral",
            usage={"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
            finish_reason="stop",
            metadata={"response_time": 2.5}
        )
    
    def test_llm_analyzer_creation(self, llm_analyzer):
        """Test creating LLM analyzer."""
        assert isinstance(llm_analyzer, LLMAnalyzer)
        assert llm_analyzer.llm_manager is not None
    
    def test_get_required_config_keys(self, llm_analyzer):
        """Test getting required configuration keys."""
        required_keys = llm_analyzer.get_required_config_keys()
        
        assert "text" in required_keys
        assert "prompt" in required_keys
    
    def test_get_optional_config_keys(self, llm_analyzer):
        """Test getting optional configuration keys."""
        optional_keys = llm_analyzer.get_optional_config_keys()
        
        expected_optional = [
            "model", "provider", "temperature", "max_tokens", 
            "system_prompt", "analysis_type", "output_format"
        ]
        
        for key in expected_optional:
            assert key in optional_keys
    
    @pytest.mark.asyncio
    async def test_execute_basic_analysis(self, llm_analyzer, sample_context, mock_llm_response):
        """Test basic text analysis execution."""
        config = {
            "text": "Sample text to analyze",
            "prompt": "Analyze this text and provide insights"
        }
        
        llm_analyzer.llm_manager.generate.return_value = mock_llm_response
        
        result = await llm_analyzer.execute(sample_context, config)
        
        assert result.success is True
        assert result.outputs["analysis"] == mock_llm_response.content
        assert result.outputs["model_used"] == mock_llm_response.model
        assert result.outputs["token_usage"] == mock_llm_response.usage
        assert result.metadata["finish_reason"] == mock_llm_response.finish_reason
        
        # Verify LLM manager was called correctly
        llm_analyzer.llm_manager.generate.assert_called_once()
        call_args = llm_analyzer.llm_manager.generate.call_args
        assert call_args.kwargs["prompt"] == config["prompt"]
    
    @pytest.mark.asyncio
    async def test_execute_with_system_prompt(self, llm_analyzer, sample_context, mock_llm_response):
        """Test execution with system prompt."""
        config = {
            "text": "Sample text",
            "prompt": "Analyze this text",
            "system_prompt": "You are an expert text analyst"
        }
        
        llm_analyzer.llm_manager.generate.return_value = mock_llm_response
        
        result = await llm_analyzer.execute(sample_context, config)
        
        assert result.success is True
        
        # Verify system prompt was passed
        call_args = llm_analyzer.llm_manager.generate.call_args
        assert call_args.kwargs["system_prompt"] == config["system_prompt"]
    
    @pytest.mark.asyncio
    async def test_execute_with_custom_model(self, llm_analyzer, sample_context, mock_llm_response):
        """Test execution with custom model and provider."""
        config = {
            "text": "Sample text",
            "prompt": "Analyze this",
            "model": "llama3.2",
            "provider": "ollama",
            "temperature": 0.5,
            "max_tokens": 1000
        }
        
        llm_analyzer.llm_manager.generate.return_value = mock_llm_response
        
        result = await llm_analyzer.execute(sample_context, config)
        
        assert result.success is True
        
        # Verify model parameters were passed
        call_args = llm_analyzer.llm_manager.generate.call_args
        assert call_args.kwargs["model"] == "llama3.2"
        assert call_args.kwargs["provider"] == "ollama"
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["max_tokens"] == 1000
    
    @pytest.mark.asyncio
    async def test_execute_with_analysis_type(self, llm_analyzer, sample_context, mock_llm_response):
        """Test execution with predefined analysis types."""
        config = {
            "text": "This is a great product! I love it.",
            "analysis_type": "sentiment"
        }
        
        llm_analyzer.llm_manager.generate.return_value = mock_llm_response
        
        result = await llm_analyzer.execute(sample_context, config)
        
        assert result.success is True
        
        # Verify that a sentiment analysis prompt was generated
        call_args = llm_analyzer.llm_manager.generate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "sentiment" in prompt.lower()
        assert config["text"] in prompt
    
    @pytest.mark.asyncio
    async def test_execute_with_output_format(self, llm_analyzer, sample_context, mock_llm_response):
        """Test execution with specific output format."""
        config = {
            "text": "Sample text for analysis",
            "prompt": "Analyze this text",
            "output_format": "json"
        }
        
        llm_analyzer.llm_manager.generate.return_value = mock_llm_response
        
        result = await llm_analyzer.execute(sample_context, config)
        
        assert result.success is True
        
        # Verify output format instruction was added to prompt
        call_args = llm_analyzer.llm_manager.generate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "json" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_execute_llm_failure(self, llm_analyzer, sample_context):
        """Test execution when LLM generation fails."""
        config = {
            "text": "Sample text",
            "prompt": "Analyze this"
        }
        
        # Mock LLM manager to raise exception
        llm_analyzer.llm_manager.generate.side_effect = Exception("LLM service unavailable")
        
        result = await llm_analyzer.execute(sample_context, config)
        
        assert result.success is False
        assert "LLM service unavailable" in result.error
        assert result.outputs == {}
    
    @pytest.mark.asyncio
    async def test_execute_empty_text(self, llm_analyzer, sample_context, mock_llm_response):
        """Test execution with empty text."""
        config = {
            "text": "",
            "prompt": "Analyze this text"
        }
        
        llm_analyzer.llm_manager.generate.return_value = mock_llm_response
        
        result = await llm_analyzer.execute(sample_context, config)
        
        # Should still succeed but with warning
        assert result.success is True
        assert "warning" in result.metadata
        assert "empty" in result.metadata["warning"].lower()
    
    @pytest.mark.asyncio
    async def test_execute_very_long_text(self, llm_analyzer, sample_context, mock_llm_response):
        """Test execution with very long text."""
        long_text = "This is a test. " * 10000  # Very long text
        config = {
            "text": long_text,
            "prompt": "Summarize this text",
            "max_tokens": 500
        }
        
        llm_analyzer.llm_manager.generate.return_value = mock_llm_response
        
        result = await llm_analyzer.execute(sample_context, config)
        
        assert result.success is True
        
        # Should handle long text appropriately
        call_args = llm_analyzer.llm_manager.generate.call_args
        assert call_args.kwargs["max_tokens"] == 500
    
    @pytest.mark.asyncio
    async def test_execute_with_context_from_previous_steps(self, llm_analyzer, mock_llm_response):
        """Test execution using context from previous steps."""
        # Create context with previous step results
        context = FlowContext(
            flow_id="test_flow",
            inputs={"user_query": "What is the sentiment?"},
            config={}
        )
        
        # Add previous step result
        previous_result = ExecutionResult(
            success=True,
            outputs={"extracted_text": "I love this product! It's amazing."}
        )
        context.add_step_result("extract_text", previous_result)
        
        config = {
            "text": "{{ steps.extract_text.extracted_text }}",  # Template reference
            "prompt": "Analyze the sentiment of: {{ steps.extract_text.extracted_text }}"
        }
        
        # Mock template rendering (this would normally be done by flow runner)
        rendered_config = {
            "text": "I love this product! It's amazing.",
            "prompt": "Analyze the sentiment of: I love this product! It's amazing."
        }
        
        llm_analyzer.llm_manager.generate.return_value = mock_llm_response
        
        result = await llm_analyzer.execute(context, rendered_config)
        
        assert result.success is True
        assert result.outputs["analysis"] == mock_llm_response.content
    
    def test_build_analysis_prompt_sentiment(self, llm_analyzer):
        """Test building sentiment analysis prompt."""
        text = "I love this product!"
        prompt = llm_analyzer._build_analysis_prompt("sentiment", text)
        
        assert "sentiment" in prompt.lower()
        assert text in prompt
        assert "positive" in prompt.lower() or "negative" in prompt.lower()
    
    def test_build_analysis_prompt_summary(self, llm_analyzer):
        """Test building summary analysis prompt."""
        text = "This is a long document with many details..."
        prompt = llm_analyzer._build_analysis_prompt("summary", text)
        
        assert "summary" in prompt.lower() or "summarize" in prompt.lower()
        assert text in prompt
    
    def test_build_analysis_prompt_extraction(self, llm_analyzer):
        """Test building entity extraction prompt."""
        text = "John Smith works at Acme Corp in New York."
        prompt = llm_analyzer._build_analysis_prompt("extraction", text)
        
        assert "extract" in prompt.lower() or "entities" in prompt.lower()
        assert text in prompt
    
    def test_build_analysis_prompt_custom(self, llm_analyzer):
        """Test building custom analysis prompt."""
        text = "Sample text"
        prompt = llm_analyzer._build_analysis_prompt("custom_analysis", text)
        
        # Should fall back to generic analysis
        assert "analyze" in prompt.lower()
        assert text in prompt
    
    def test_format_output_json(self, llm_analyzer):
        """Test JSON output formatting."""
        raw_output = "This is positive sentiment"
        formatted = llm_analyzer._format_output(raw_output, "json")
        
        assert "json" in formatted.lower()
    
    def test_format_output_structured(self, llm_analyzer):
        """Test structured output formatting."""
        raw_output = "Key points: 1. Point one 2. Point two"
        formatted = llm_analyzer._format_output(raw_output, "structured")
        
        assert "structured" in formatted.lower() or "format" in formatted.lower()
    
    def test_format_output_default(self, llm_analyzer):
        """Test default output formatting."""
        raw_output = "This is the analysis result"
        formatted = llm_analyzer._format_output(raw_output, "text")
        
        assert formatted == raw_output  # Should return unchanged
    
    @pytest.mark.asyncio
    async def test_execute_with_multiple_analysis_types(self, llm_analyzer, sample_context, mock_llm_response):
        """Test execution with multiple analysis types."""
        config = {
            "text": "I love this product! It's the best thing ever.",
            "analysis_type": "sentiment,summary,extraction"
        }
        
        llm_analyzer.llm_manager.generate.return_value = mock_llm_response
        
        result = await llm_analyzer.execute(sample_context, config)
        
        assert result.success is True
        
        # Should handle multiple analysis types
        call_args = llm_analyzer.llm_manager.generate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "sentiment" in prompt.lower()
        assert "summary" in prompt.lower() or "summarize" in prompt.lower()
        assert "extract" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_execute_with_validation_error(self, llm_analyzer, sample_context):
        """Test execution with configuration validation error."""
        # Missing required 'text' field
        config = {
            "prompt": "Analyze this"
            # Missing "text" field
        }
        
        result = await llm_analyzer.execute_with_validation(sample_context, config)
        
        assert result.success is False
        assert "validation" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_performance_tracking(self, llm_analyzer, sample_context, mock_llm_response):
        """Test that execution time is tracked."""
        config = {
            "text": "Sample text",
            "prompt": "Analyze this"
        }
        
        # Add delay to mock response
        async def delayed_generate(*args, **kwargs):
            import asyncio
            await asyncio.sleep(0.1)
            return mock_llm_response
        
        llm_analyzer.llm_manager.generate = delayed_generate
        
        result = await llm_analyzer.execute_with_validation(sample_context, config)
        
        assert result.success is True
        assert result.execution_time is not None
        assert result.execution_time >= 0.1


@pytest.mark.integration
class TestLLMAnalyzerIntegration:
    """Integration tests for LLM Analyzer."""
    
    @pytest.mark.asyncio
    async def test_realistic_document_analysis_workflow(self):
        """Test realistic document analysis workflow."""
        # This would typically use a real LLM manager in integration tests
        analyzer = LLMAnalyzer()
        
        # Mock the LLM manager for this test
        mock_manager = Mock(spec=LLMManager)
        mock_response = LLMResponse(
            content="""
            Analysis Results:
            - Sentiment: Positive
            - Key Topics: Product quality, customer satisfaction
            - Summary: Customer expresses high satisfaction with product quality
            - Entities: Product (mentioned 3 times), Customer (mentioned 2 times)
            """,
            model="mistral",
            usage={"prompt_tokens": 50, "completion_tokens": 80, "total_tokens": 130},
            finish_reason="stop"
        )
        mock_manager.generate = AsyncMock(return_value=mock_response)
        analyzer.llm_manager = mock_manager
        
        # Simulate document analysis context
        context = FlowContext(
            flow_id="doc_analysis_001",
            inputs={
                "document_type": "customer_feedback",
                "analysis_requirements": "sentiment, topics, summary"
            },
            config={"model": "mistral", "temperature": 0.3}
        )
        
        # Add extracted text from previous step
        extraction_result = ExecutionResult(
            success=True,
            outputs={
                "text": "I absolutely love this product! The quality is outstanding and it exceeded all my expectations. The customer service was also fantastic. I would definitely recommend this to anyone looking for a high-quality solution.",
                "word_count": 32,
                "language": "en"
            }
        )
        context.add_step_result("extract_text", extraction_result)
        
        # Configuration for comprehensive analysis
        config = {
            "text": extraction_result.outputs["text"],
            "analysis_type": "sentiment,summary,extraction",
            "output_format": "structured",
            "model": "mistral",
            "temperature": 0.3,
            "system_prompt": "You are an expert document analyst specializing in customer feedback analysis."
        }
        
        # Execute analysis
        result = await analyzer.execute(context, config)
        
        # Verify results
        assert result.success is True
        assert result.outputs["analysis"] is not None
        assert result.outputs["model_used"] == "mistral"
        assert result.outputs["token_usage"]["total_tokens"] == 130
        assert result.metadata["finish_reason"] == "stop"
        
        # Verify the LLM was called with correct parameters
        mock_manager.generate.assert_called_once()
        call_kwargs = mock_manager.generate.call_args.kwargs
        assert call_kwargs["model"] == "mistral"
        assert call_kwargs["temperature"] == 0.3
        assert "sentiment" in call_kwargs["prompt"].lower()
        assert "summary" in call_kwargs["prompt"].lower()
        assert "extract" in call_kwargs["prompt"].lower()
    
    @pytest.mark.asyncio
    async def test_multi_step_analysis_chain(self):
        """Test chaining multiple LLM analysis steps."""
        analyzer = LLMAnalyzer()
        
        # Mock different responses for different analysis steps
        mock_manager = Mock(spec=LLMManager)
        
        responses = [
            LLMResponse(content="Sentiment: Positive (0.85 confidence)", model="mistral", usage={}, finish_reason="stop"),
            LLMResponse(content="Key topics: product quality, customer service, recommendations", model="mistral", usage={}, finish_reason="stop"),
            LLMResponse(content="Summary: Highly satisfied customer praising product quality and service", model="mistral", usage={}, finish_reason="stop")
        ]
        
        mock_manager.generate = AsyncMock(side_effect=responses)
        analyzer.llm_manager = mock_manager
        
        context = FlowContext(flow_id="chain_analysis", inputs={}, config={})
        
        text = "I absolutely love this product! The quality is outstanding and the customer service was fantastic."
        
        # Step 1: Sentiment Analysis
        result1 = await analyzer.execute(context, {
            "text": text,
            "analysis_type": "sentiment"
        })
        context.add_step_result("sentiment_analysis", result1)
        
        # Step 2: Topic Extraction
        result2 = await analyzer.execute(context, {
            "text": text,
            "analysis_type": "extraction"
        })
        context.add_step_result("topic_extraction", result2)
        
        # Step 3: Summary Generation
        result3 = await analyzer.execute(context, {
            "text": text,
            "analysis_type": "summary"
        })
        context.add_step_result("summary_generation", result3)
        
        # Verify all steps succeeded
        assert result1.success is True
        assert result2.success is True
        assert result3.success is True
        
        # Verify different analyses were performed
        assert "sentiment" in result1.outputs["analysis"].lower()
        assert "topics" in result2.outputs["analysis"].lower()
        assert "summary" in result3.outputs["analysis"].lower()
        
        # Verify all LLM calls were made
        assert mock_manager.generate.call_count == 3
