"""
Unit tests for SentimentAnalyzer executor.

Tests sentiment analysis with HuggingFace and LLM providers,
multiple analysis types, emotion detection, and result standardization.
"""

import pytest
import uuid
import json
from unittest.mock import Mock, patch, AsyncMock

from core.executors.sentiment_analyzer import SentimentAnalyzer
from core.executors.base_executor import FlowContext, ExecutionResult


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer executor."""
    
    @pytest.fixture
    def sentiment_analyzer(self):
        """Create SentimentAnalyzer instance for testing."""
        return SentimentAnalyzer()
    
    @pytest.fixture
    def mock_context(self):
        """Create mock FlowContext for testing."""
        return FlowContext(
            flow_name="test_flow",
            inputs={"test_param": "test_value"},
            flow_id=f"test-flow-{uuid.uuid4().hex[:8]}"
        )
    
    @pytest.fixture
    def sample_hf_config(self):
        """Sample configuration for HuggingFace provider."""
        return {
            "text": "I love this product! It's amazing and works perfectly.",
            "provider": "huggingface",
            "analysis_type": "basic",
            "device": "cpu"
        }
    
    @pytest.fixture
    def sample_llm_config(self):
        """Sample configuration for LLM provider."""
        return {
            "text": "I'm really disappointed with this service. The quality is terrible.",
            "provider": "llm",
            "analysis_type": "detailed",
            "llm_model": "mistral"
        }
    
    @pytest.fixture
    def mock_hf_basic_results(self):
        """Mock HuggingFace basic sentiment results."""
        return [[
            {"label": "POSITIVE", "score": 0.9234},
            {"label": "NEGATIVE", "score": 0.0456},
            {"label": "NEUTRAL", "score": 0.0310}
        ]]
    
    @pytest.fixture
    def mock_hf_emotion_results(self):
        """Mock HuggingFace emotion analysis results."""
        return [[
            {"label": "joy", "score": 0.8234},
            {"label": "anger", "score": 0.0456},
            {"label": "sadness", "score": 0.0310},
            {"label": "fear", "score": 0.0200},
            {"label": "surprise", "score": 0.0500},
            {"label": "disgust", "score": 0.0300}
        ]]
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for sentiment analysis."""
        return {
            "sentiment": "negative",
            "confidence": 0.85,
            "emotions": ["anger", "disappointment"],
            "key_phrases": [
                {"phrase": "really disappointed", "sentiment": "negative"},
                {"phrase": "quality is terrible", "sentiment": "negative"}
            ],
            "reasoning": "The text expresses strong dissatisfaction with service quality"
        }
    
    @pytest.mark.asyncio
    async def test_missing_text(self, sentiment_analyzer, mock_context):
        """Test error when text is missing."""
        config = {"provider": "huggingface"}
        
        result = await sentiment_analyzer.execute(mock_context, config)
        
        assert result.success is False
        assert "text is required" in result.error
    
    @pytest.mark.asyncio
    async def test_empty_text(self, sentiment_analyzer, mock_context):
        """Test error when text is empty."""
        config = {
            "text": "",
            "provider": "huggingface"
        }
        
        result = await sentiment_analyzer.execute(mock_context, config)
        
        assert result.success is False
        assert "text cannot be empty" in result.error
    
    @pytest.mark.asyncio
    async def test_text_too_long(self, sentiment_analyzer, mock_context):
        """Test error when text is too long."""
        config = {
            "text": "x" * 60000,  # 60K characters
            "provider": "huggingface"
        }
        
        result = await sentiment_analyzer.execute(mock_context, config)
        
        assert result.success is False
        assert "Text too long" in result.error
    
    @pytest.mark.asyncio
    async def test_unsupported_provider(self, sentiment_analyzer, mock_context):
        """Test error with unsupported provider."""
        config = {
            "text": "Sample text",
            "provider": "unsupported_provider"
        }
        
        result = await sentiment_analyzer.execute(mock_context, config)
        
        assert result.success is False
        assert "Provider must be" in result.error
    
    @pytest.mark.asyncio
    async def test_unsupported_analysis_type(self, sentiment_analyzer, mock_context):
        """Test error with unsupported analysis type."""
        config = {
            "text": "Sample text",
            "provider": "huggingface",
            "analysis_type": "unsupported_type"
        }
        
        result = await sentiment_analyzer.execute(mock_context, config)
        
        assert result.success is False
        assert "analysis_type must be" in result.error
    
    @pytest.mark.asyncio
    async def test_successful_hf_basic_analysis(self, sentiment_analyzer, mock_context, 
                                              sample_hf_config, mock_hf_basic_results):
        """Test successful HuggingFace basic sentiment analysis."""
        with patch.object(sentiment_analyzer, '_analyze_huggingface', 
                         return_value=ExecutionResult(
                             success=True,
                             outputs={
                                 'sentiment': 'positive',
                                 'confidence': 0.923,
                                 'all_scores': {
                                     'POSITIVE': 0.923,
                                     'NEGATIVE': 0.046,
                                     'NEUTRAL': 0.031
                                 }
                             }
                         )) as mock_analyze:
            
            result = await sentiment_analyzer.execute(mock_context, sample_hf_config)
            
            assert result.success is True
            assert "sentiment" in result.outputs
            assert "confidence" in result.outputs
            assert "processing_time_seconds" in result.outputs
            assert "provider" in result.outputs
            assert "analysis_type" in result.outputs
            assert "text_length" in result.outputs
            assert "model_used" in result.outputs
            
            assert result.outputs["sentiment"] == "positive"
            assert result.outputs["confidence"] == 0.923
            assert result.outputs["provider"] == "huggingface"
            assert result.outputs["analysis_type"] == "basic"
    
    @pytest.mark.asyncio
    async def test_successful_llm_analysis(self, sentiment_analyzer, mock_context, sample_llm_config):
        """Test successful LLM sentiment analysis."""
        with patch.object(sentiment_analyzer, '_analyze_llm', 
                         return_value=ExecutionResult(
                             success=True,
                             outputs={
                                 'sentiment': 'negative',
                                 'confidence': 0.85,
                                 'emotions': ['anger', 'disappointment'],
                                 'reasoning': 'Strong dissatisfaction expressed'
                             }
                         )) as mock_analyze:
            
            result = await sentiment_analyzer.execute(mock_context, sample_llm_config)
            
            assert result.success is True
            assert result.outputs["sentiment"] == "negative"
            assert result.outputs["provider"] == "llm"
            assert result.outputs["analysis_type"] == "detailed"
    
    @pytest.mark.asyncio
    async def test_default_configuration(self, sentiment_analyzer, mock_context):
        """Test sentiment analysis with default configuration."""
        config = {"text": "This is a neutral statement."}
        
        with patch.object(sentiment_analyzer, '_analyze_huggingface', 
                         return_value=ExecutionResult(
                             success=True,
                             outputs={
                                 'sentiment': 'neutral',
                                 'confidence': 0.75
                             }
                         )) as mock_analyze:
            
            result = await sentiment_analyzer.execute(mock_context, config)
            
            assert result.success is True
            assert result.outputs["provider"] == "huggingface"  # Default provider
            assert result.outputs["analysis_type"] == "basic"  # Default analysis type
    
    def test_sentiment_analyzer_name(self, sentiment_analyzer):
        """Test SentimentAnalyzer name property."""
        assert sentiment_analyzer.name == "sentiment_analyzer"
    
    def test_sentiment_prompts(self, sentiment_analyzer):
        """Test sentiment prompt templates."""
        prompts = sentiment_analyzer.SENTIMENT_PROMPTS
        
        assert "basic" in prompts
        assert "detailed" in prompts
        assert "comprehensive" in prompts
        assert "emotions" in prompts
        
        # Check that prompts contain expected placeholders
        for prompt_type, prompt_text in prompts.items():
            assert "{text}" in prompt_text
            assert "JSON" in prompt_text or "json" in prompt_text
    
    def test_hf_models_mapping(self, sentiment_analyzer):
        """Test HuggingFace models mapping."""
        models = sentiment_analyzer.HF_MODELS
        
        assert "basic" in models
        assert "emotions" in models
        assert "comprehensive" in models
        assert "detailed" in models
        
        # Check that all models are valid HuggingFace model names
        for model_type, model_name in models.items():
            assert "/" in model_name  # HF models have format "org/model"
    
    def test_get_model_name_hf(self, sentiment_analyzer):
        """Test model name extraction for HuggingFace provider."""
        config = {
            "provider": "huggingface",
            "analysis_type": "basic"
        }
        
        model_name = sentiment_analyzer._get_model_name("huggingface", config)
        assert model_name == sentiment_analyzer.HF_MODELS["basic"]
        
        # Test with custom model name
        config["hf_model_name"] = "custom/model"
        model_name = sentiment_analyzer._get_model_name("huggingface", config)
        assert model_name == "custom/model"
    
    def test_get_model_name_llm(self, sentiment_analyzer):
        """Test model name extraction for LLM provider."""
        config = {"llm_model": "gpt-4"}
        
        model_name = sentiment_analyzer._get_model_name("llm", config)
        assert model_name == "gpt-4"
        
        # Test default
        model_name = sentiment_analyzer._get_model_name("llm", {})
        assert model_name == "mistral"
    
    def test_get_executor_info(self, sentiment_analyzer):
        """Test executor information."""
        info = sentiment_analyzer.get_executor_info()
        
        assert "name" in info
        assert "description" in info
        assert "providers" in info
        assert "analysis_types" in info
        assert "capabilities" in info
        assert "supported_models" in info
        
        # Check providers
        assert "huggingface" in info["providers"]
        assert "llm" in info["providers"]
        
        # Check analysis types
        expected_types = ["basic", "detailed", "comprehensive", "emotions"]
        for analysis_type in expected_types:
            assert analysis_type in info["analysis_types"]
        
        # Check capabilities
        capabilities = info["capabilities"]
        assert "Sentiment analysis" in str(capabilities)
        assert "Emotion detection" in str(capabilities)


class TestSentimentAnalyzerHuggingFace:
    """Test suite for HuggingFace-specific functionality."""
    
    @pytest.fixture
    def sentiment_analyzer(self):
        return SentimentAnalyzer()
    
    def test_hf_import_error(self, sentiment_analyzer):
        """Test handling of missing HuggingFace libraries."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'transformers'")):
            # Should raise exception which gets caught by caller
            with pytest.raises(Exception) as exc_info:
                sentiment_analyzer._analyze_hf_sync(
                    "test text", "cardiffnlp/twitter-roberta-base-sentiment-latest", -1, "basic"
                )
            assert "HuggingFace processing failed" in str(exc_info.value)
    
    def test_process_basic_sentiment(self, sentiment_analyzer):
        """Test basic sentiment processing."""
        mock_results = [[
            {"label": "POSITIVE", "score": 0.9234},
            {"label": "NEGATIVE", "score": 0.0456},
            {"label": "NEUTRAL", "score": 0.0310}
        ]]
        
        result = sentiment_analyzer._process_basic_sentiment(mock_results)
        
        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.923
        assert "all_scores" in result
        assert result["all_scores"]["POSITIVE"] == 0.923
    
    def test_process_basic_sentiment_label_mapping(self, sentiment_analyzer):
        """Test sentiment label mapping for different model formats."""
        # Test LABEL_X format
        mock_results = [[
            {"label": "LABEL_0", "score": 0.8},  # negative
            {"label": "LABEL_1", "score": 0.1},  # neutral
            {"label": "LABEL_2", "score": 0.1}   # positive
        ]]
        
        result = sentiment_analyzer._process_basic_sentiment(mock_results)
        assert result["sentiment"] == "negative"
        assert result["confidence"] == 0.8
    
    def test_process_emotions(self, sentiment_analyzer):
        """Test emotion processing."""
        mock_results = [[
            {"label": "joy", "score": 0.8234},
            {"label": "anger", "score": 0.0456},
            {"label": "sadness", "score": 0.0310},
            {"label": "fear", "score": 0.0200}
        ]]
        
        result = sentiment_analyzer._process_emotions(mock_results)
        
        assert result["primary_emotion"] == "joy"
        assert result["confidence"] == 0.823
        assert result["emotional_intensity"] == "high"
        assert "emotion_scores" in result
        assert result["emotion_scores"]["joy"] == 0.823
    
    def test_process_emotions_intensity_levels(self, sentiment_analyzer):
        """Test emotion intensity level calculation."""
        # High intensity
        mock_results = [[{"label": "joy", "score": 0.8}]]
        result = sentiment_analyzer._process_emotions(mock_results)
        assert result["emotional_intensity"] == "high"
        
        # Medium intensity
        mock_results = [[{"label": "joy", "score": 0.6}]]
        result = sentiment_analyzer._process_emotions(mock_results)
        assert result["emotional_intensity"] == "medium"
        
        # Low intensity
        mock_results = [[{"label": "joy", "score": 0.3}]]
        result = sentiment_analyzer._process_emotions(mock_results)
        assert result["emotional_intensity"] == "low"
    
    def test_process_comprehensive(self, sentiment_analyzer):
        """Test comprehensive analysis processing."""
        mock_results = [[
            {"label": "POSITIVE", "score": 0.9},
            {"label": "NEGATIVE", "score": 0.1}
        ]]
        
        result = sentiment_analyzer._process_comprehensive(mock_results, "comprehensive")
        
        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.9
        assert "emotion_scores" in result
        assert "analysis_type" in result
        assert result["analysis_type"] == "comprehensive"


class TestSentimentAnalyzerLLM:
    """Test suite for LLM-specific functionality."""
    
    @pytest.fixture
    def sentiment_analyzer(self):
        return SentimentAnalyzer()
    
    @pytest.fixture
    def mock_context(self):
        return FlowContext(
            flow_name="test_flow",
            inputs={},
            flow_id="test-llm"
        )
    
    @pytest.mark.asyncio
    async def test_llm_analysis_success(self, sentiment_analyzer, mock_context):
        """Test successful LLM sentiment analysis."""
        mock_llm_analyzer = Mock()
        mock_llm_analyzer.execute = AsyncMock(return_value=ExecutionResult(
            success=True,
            outputs={
                "analysis": '{"sentiment": "positive", "confidence": 0.9, "reasoning": "Very positive language"}'
            }
        ))
        
        sentiment_analyzer._llm_analyzer = mock_llm_analyzer
        
        result = await sentiment_analyzer._analyze_llm(
            "I love this product!", 
            {"analysis_type": "basic", "llm_model": "mistral"}, 
            mock_context
        )
        
        assert result.success is True
        assert result.outputs["sentiment"] == "positive"
        assert result.outputs["confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_llm_json_parsing_with_markdown(self, sentiment_analyzer, mock_context):
        """Test LLM JSON parsing with markdown formatting."""
        mock_llm_analyzer = Mock()
        mock_llm_analyzer.execute = AsyncMock(return_value=ExecutionResult(
            success=True,
            outputs={
                "analysis": '```json\n{"sentiment": "negative", "confidence": 0.8}\n```'
            }
        ))
        
        sentiment_analyzer._llm_analyzer = mock_llm_analyzer
        
        result = await sentiment_analyzer._analyze_llm(
            "This is terrible", 
            {"analysis_type": "basic"}, 
            mock_context
        )
        
        assert result.success is True
        assert result.outputs["sentiment"] == "negative"
        assert result.outputs["confidence"] == 0.8
    
    @pytest.mark.asyncio
    async def test_llm_json_parsing_failure(self, sentiment_analyzer, mock_context):
        """Test LLM JSON parsing failure fallback."""
        mock_llm_analyzer = Mock()
        mock_llm_analyzer.execute = AsyncMock(return_value=ExecutionResult(
            success=True,
            outputs={
                "analysis": "This is not valid JSON response"
            }
        ))
        
        sentiment_analyzer._llm_analyzer = mock_llm_analyzer
        
        result = await sentiment_analyzer._analyze_llm(
            "Sample text", 
            {"analysis_type": "basic"}, 
            mock_context
        )
        
        assert result.success is True
        assert result.outputs["sentiment"] == "neutral"  # Fallback
        assert result.outputs["confidence"] == 0.5
        assert "note" in result.outputs
        assert "could not be parsed" in result.outputs["note"]
    
    def test_standardize_llm_response_basic(self, sentiment_analyzer):
        """Test LLM response standardization for basic analysis."""
        parsed_result = {
            "sentiment": "positive",
            "confidence": 0.85,
            "reasoning": "Positive language detected"
        }
        
        result = sentiment_analyzer._standardize_llm_response(parsed_result, "basic")
        
        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.85
        assert result["reasoning"] == "Positive language detected"
    
    def test_standardize_llm_response_emotions(self, sentiment_analyzer):
        """Test LLM response standardization for emotion analysis."""
        parsed_result = {
            "primary_emotion": "joy",
            "emotion_scores": {"joy": 0.8, "anger": 0.1},
            "confidence": 0.9,
            "emotional_intensity": "high"
        }
        
        result = sentiment_analyzer._standardize_llm_response(parsed_result, "emotions")
        
        assert result["primary_emotion"] == "joy"
        assert result["emotion_scores"]["joy"] == 0.8
        assert result["emotional_intensity"] == "high"
    
    def test_standardize_llm_response_comprehensive(self, sentiment_analyzer):
        """Test LLM response standardization for comprehensive analysis."""
        parsed_result = {
            "overall_sentiment": "mixed",  # Should map to 'sentiment'
            "confidence": 0.75,
            "emotion_scores": {"joy": 0.4, "anger": 0.3},
            "aspects": [{"aspect": "quality", "sentiment": "positive"}],
            "insights": "Mixed feelings about the product"
        }
        
        result = sentiment_analyzer._standardize_llm_response(parsed_result, "comprehensive")
        
        assert result["sentiment"] == "mixed"  # Mapped from overall_sentiment
        assert result["confidence"] == 0.75
        assert result["emotion_scores"]["joy"] == 0.4
        assert result["aspects"][0]["aspect"] == "quality"
        assert result["insights"] == "Mixed feelings about the product"


class TestSentimentAnalyzerIntegration:
    """Integration tests for SentimentAnalyzer with flow context."""
    
    @pytest.mark.asyncio
    async def test_sentiment_analyzer_in_flow_context(self):
        """Test SentimentAnalyzer as part of a flow context."""
        sentiment_analyzer = SentimentAnalyzer()
        context = FlowContext(
            flow_name="text_analysis_flow",
            inputs={"user_review": "Great product!"},
            flow_id="integration-test"
        )
        
        # Add previous step result (e.g., from text extraction)
        text_result = ExecutionResult(
            success=True,
            outputs={
                "extracted_text": "I absolutely love this product! The quality is outstanding and delivery was fast.",
                "text_length": 85
            }
        )
        context.add_step_result("text_extraction", text_result)
        
        config = {
            "text": context.step_results["text_extraction"].outputs["extracted_text"],
            "provider": "huggingface",
            "analysis_type": "detailed"
        }
        
        with patch.object(sentiment_analyzer, '_analyze_huggingface', 
                         return_value=ExecutionResult(
                             success=True,
                             outputs={
                                 'sentiment': 'positive',
                                 'confidence': 0.95,
                                 'all_scores': {
                                     'POSITIVE': 0.95,
                                     'NEGATIVE': 0.03,
                                     'NEUTRAL': 0.02
                                 }
                             }
                         )):
            
            result = await sentiment_analyzer.execute(context, config)
            
            # Add result to context
            context.add_step_result("sentiment_analysis", result)
            
            # Verify integration
            assert result.success is True
            assert context.step_results["sentiment_analysis"] is not None
            
            # Verify the result can be used by subsequent steps
            sentiment_result = context.step_results["sentiment_analysis"]
            assert sentiment_result.outputs["sentiment"] == "positive"
            assert sentiment_result.outputs["confidence"] == 0.95
            assert sentiment_result.outputs["provider"] == "huggingface"
            assert sentiment_result.outputs["analysis_type"] == "detailed"
