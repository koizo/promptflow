"""
Unit tests for VisionClassifier executor.

Tests image classification with HuggingFace and OpenAI providers,
model loading, image processing, and result standardization.
"""

import pytest
import uuid
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, mock_open

from core.executors.vision_classifier import VisionClassifier
from core.executors.base_executor import FlowContext, ExecutionResult


class TestVisionClassifier:
    """Test suite for VisionClassifier executor."""
    
    @pytest.fixture
    def vision_classifier(self):
        """Create VisionClassifier instance for testing."""
        return VisionClassifier()
    
    @pytest.fixture
    def mock_context(self):
        """Create mock FlowContext for testing."""
        return FlowContext(
            flow_name="test_flow",
            inputs={"test_param": "test_value"},
            flow_id=f"test-flow-{uuid.uuid4().hex[:8]}"
        )
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Sample image bytes for testing (minimal PNG)."""
        # Minimal 1x1 PNG image bytes
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
    
    @pytest.fixture
    def sample_hf_config(self):
        """Sample HuggingFace configuration."""
        return {
            "image": {"content": b"fake_image_data", "filename": "test.jpg"},
            "provider": "huggingface",
            "hf_model_name": "google/vit-base-patch16-224",
            "top_k": 5,
            "confidence_threshold": 0.1
        }
    
    @pytest.fixture
    def sample_openai_config(self):
        """Sample OpenAI configuration."""
        return {
            "image": {"content": b"fake_image_data", "filename": "test.jpg"},
            "provider": "openai",
            "openai_model": "gpt-4-vision-preview",
            "top_k": 3
        }
    
    @pytest.fixture
    def mock_hf_predictions(self):
        """Mock HuggingFace predictions."""
        return [
            {"label": "Egyptian cat", "score": 0.9234},
            {"label": "tabby, tabby cat", "score": 0.0456},
            {"label": "tiger cat", "score": 0.0234},
            {"label": "lynx, catamount", "score": 0.0076}
        ]
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = json.dumps({
            "predictions": [
                {"label": "cat", "confidence": 0.95, "description": "Domestic cat sitting"},
                {"label": "pet", "confidence": 0.87, "description": "Household pet"},
                {"label": "animal", "confidence": 0.82, "description": "Mammalian animal"}
            ],
            "analysis": "The image shows a domestic cat in a typical resting position."
        })
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response
    
    @pytest.mark.asyncio
    async def test_missing_image_data(self, vision_classifier, mock_context):
        """Test error when image data is missing."""
        config = {"provider": "huggingface"}
        
        result = await vision_classifier.execute(mock_context, config)
        
        assert result.success is False
        assert "Image data is required" in result.error
    
    @pytest.mark.asyncio
    async def test_invalid_image_dict_format(self, vision_classifier, mock_context):
        """Test error when image dictionary lacks content."""
        config = {
            "image": {"filename": "test.jpg"},  # Missing content
            "provider": "huggingface"
        }
        
        result = await vision_classifier.execute(mock_context, config)
        
        assert result.success is False
        assert "must contain 'content' field" in result.error
    
    @pytest.mark.asyncio
    async def test_unsupported_provider(self, vision_classifier, mock_context):
        """Test error with unsupported provider."""
        config = {
            "image": {"content": b"fake_data", "filename": "test.jpg"},
            "provider": "unsupported_provider"
        }
        
        with patch('tempfile.mkstemp', return_value=(1, '/tmp/test.jpg')), \
             patch('os.fdopen'), \
             patch('pathlib.Path.unlink'):
            
            result = await vision_classifier.execute(mock_context, config)
            
            assert result.success is False
            assert "Unsupported provider" in result.error
    
    @pytest.mark.asyncio
    async def test_successful_huggingface_classification(self, vision_classifier, mock_context, 
                                                       sample_hf_config, mock_hf_predictions):
        """Test successful HuggingFace image classification."""
        with patch('tempfile.mkstemp', return_value=(1, '/tmp/test.jpg')), \
             patch('os.fdopen'), \
             patch('pathlib.Path.unlink'), \
             patch.object(vision_classifier, '_classify_with_huggingface', 
                         return_value={
                             'predictions': [
                                 {'label': 'Egyptian cat', 'confidence': 0.9234, 'rank': 1},
                                 {'label': 'tabby cat', 'confidence': 0.0456, 'rank': 2}
                             ],
                             'top_prediction': {'label': 'Egyptian cat', 'confidence': 0.9234, 'rank': 1},
                             'confidence': 0.9234,
                             'provider': 'huggingface',
                             'model_used': 'google/vit-base-patch16-224'
                         }) as mock_classify:
            
            result = await vision_classifier.execute(mock_context, sample_hf_config)
            
            assert result.success is True
            assert "predictions" in result.outputs
            assert "top_prediction" in result.outputs
            assert "provider" in result.outputs
            assert "processing_time_seconds" in result.outputs
            
            assert result.outputs["provider"] == "huggingface"
            assert result.outputs["top_prediction"]["label"] == "Egyptian cat"
            assert result.outputs["confidence"] == 0.9234
    
    @pytest.mark.asyncio
    async def test_successful_openai_classification(self, vision_classifier, mock_context, 
                                                  sample_openai_config):
        """Test successful OpenAI image classification."""
        with patch('tempfile.mkstemp', return_value=(1, '/tmp/test.jpg')), \
             patch('os.fdopen'), \
             patch('pathlib.Path.unlink'), \
             patch.object(vision_classifier, '_classify_with_openai', 
                         return_value={
                             'predictions': [
                                 {'label': 'cat', 'confidence': 0.95, 'description': 'Domestic cat', 'rank': 1},
                                 {'label': 'pet', 'confidence': 0.87, 'description': 'Household pet', 'rank': 2}
                             ],
                             'top_prediction': {'label': 'cat', 'confidence': 0.95, 'description': 'Domestic cat', 'rank': 1},
                             'confidence': 0.95,
                             'provider': 'openai',
                             'model_used': 'gpt-4-vision-preview',
                             'analysis': 'The image shows a domestic cat.'
                         }) as mock_classify:
            
            result = await vision_classifier.execute(mock_context, sample_openai_config)
            
            assert result.success is True
            assert "predictions" in result.outputs
            assert "analysis" in result.outputs
            assert result.outputs["provider"] == "openai"
            assert result.outputs["top_prediction"]["label"] == "cat"
    
    @pytest.mark.asyncio
    async def test_file_path_image_input(self, vision_classifier, mock_context):
        """Test image input as file path."""
        config = {
            "image": "/path/to/test_image.jpg",
            "provider": "huggingface"
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch.object(vision_classifier, '_classify_with_huggingface', 
                         return_value={'predictions': [], 'provider': 'huggingface'}) as mock_classify:
            
            result = await vision_classifier.execute(mock_context, config)
            
            assert result.success is True
            mock_classify.assert_called_once_with("/path/to/test_image.jpg", config)
    
    @pytest.mark.asyncio
    async def test_nonexistent_file_path(self, vision_classifier, mock_context):
        """Test error with non-existent file path."""
        config = {
            "image": "/path/to/nonexistent.jpg",
            "provider": "huggingface"
        }
        
        with patch('pathlib.Path.exists', return_value=False):
            result = await vision_classifier.execute(mock_context, config)
            
            assert result.success is False
            assert "Image file not found" in result.error
    
    @pytest.mark.asyncio
    async def test_temp_file_cleanup(self, vision_classifier, mock_context, sample_hf_config):
        """Test temporary file cleanup after processing."""
        temp_path = '/tmp/test_cleanup.jpg'
        
        with patch('tempfile.mkstemp', return_value=(1, temp_path)), \
             patch('os.fdopen'), \
             patch('pathlib.Path.exists', return_value=True) as mock_exists, \
             patch('pathlib.Path.unlink') as mock_unlink, \
             patch.object(vision_classifier, '_classify_with_huggingface', 
                         return_value={'predictions': [], 'provider': 'huggingface'}):
            
            result = await vision_classifier.execute(mock_context, sample_hf_config)
            
            assert result.success is True
            mock_unlink.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_huggingface_model_validation(self, vision_classifier, mock_context):
        """Test HuggingFace model validation."""
        config = {
            "image": {"content": b"fake_data", "filename": "test.jpg"},
            "provider": "huggingface",
            "hf_model_name": "invalid/model-name"
        }
        
        with patch('tempfile.mkstemp', return_value=(1, '/tmp/test.jpg')), \
             patch('os.fdopen'), \
             patch('pathlib.Path.unlink'):
            
            result = await vision_classifier.execute(mock_context, config)
            
            assert result.success is False
            assert "Unsupported HuggingFace model" in result.error
    
    @pytest.mark.asyncio
    async def test_openai_model_validation(self, vision_classifier, mock_context):
        """Test OpenAI model validation."""
        config = {
            "image": {"content": b"fake_data", "filename": "test.jpg"},
            "provider": "openai",
            "openai_model": "invalid-model"
        }
        
        with patch('tempfile.mkstemp', return_value=(1, '/tmp/test.jpg')), \
             patch('os.fdopen'), \
             patch('pathlib.Path.unlink'):
            
            result = await vision_classifier.execute(mock_context, config)
            
            assert result.success is False
            assert "Unsupported OpenAI model" in result.error
    
    def test_vision_classifier_name(self, vision_classifier):
        """Test VisionClassifier name property."""
        assert vision_classifier.name == "vision_classifier"
    
    def test_supported_hf_models(self, vision_classifier):
        """Test supported HuggingFace models configuration."""
        models = vision_classifier.SUPPORTED_HF_MODELS
        
        assert "google/vit-base-patch16-224" in models
        assert "microsoft/resnet-50" in models
        assert "google/efficientnet-b0" in models
        assert "facebook/convnext-tiny-224" in models
        
        # Check model metadata structure
        vit_model = models["google/vit-base-patch16-224"]
        assert "name" in vit_model
        assert "type" in vit_model
        assert "size" in vit_model
        assert "speed" in vit_model
    
    def test_supported_openai_models(self, vision_classifier):
        """Test supported OpenAI models configuration."""
        models = vision_classifier.OPENAI_MODELS
        
        assert "gpt-4-vision-preview" in models
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
    
    def test_standardize_hf_results_empty(self, vision_classifier):
        """Test standardizing empty HuggingFace results."""
        result = vision_classifier._standardize_hf_results([], "test-model", "cpu")
        
        assert result["predictions"] == []
        assert result["top_prediction"] is None
        assert result["confidence"] == 0.0
        assert result["provider"] == "huggingface"
        assert result["model_used"] == "test-model"
        assert result["device"] == "cpu"
    
    def test_standardize_hf_results_with_predictions(self, vision_classifier, mock_hf_predictions):
        """Test standardizing HuggingFace results with predictions."""
        result = vision_classifier._standardize_hf_results(mock_hf_predictions, "test-model", "cuda")
        
        assert len(result["predictions"]) == 4
        assert result["top_prediction"]["label"] == "Egyptian cat"
        assert result["top_prediction"]["confidence"] == 0.9234
        assert result["confidence"] == 0.9234
        assert result["provider"] == "huggingface"
        assert result["device"] == "cuda"
        
        # Check prediction structure
        pred = result["predictions"][0]
        assert "label" in pred
        assert "confidence" in pred
        assert "rank" in pred
        assert pred["rank"] == 1
    
    def test_standardize_openai_results(self, vision_classifier):
        """Test standardizing OpenAI results."""
        parsed_response = {
            "predictions": [
                {"label": "cat", "confidence": 0.95, "description": "Domestic cat"},
                {"label": "pet", "confidence": 0.87, "description": "Household pet"}
            ],
            "analysis": "The image shows a cat."
        }
        
        result = vision_classifier._standardize_openai_results(
            parsed_response, "gpt-4-vision-preview", "raw response text"
        )
        
        assert len(result["predictions"]) == 2
        assert result["top_prediction"]["label"] == "cat"
        assert result["confidence"] == 0.95
        assert result["provider"] == "openai"
        assert result["model_used"] == "gpt-4-vision-preview"
        assert result["analysis"] == "The image shows a cat."
        
        # Check prediction structure
        pred = result["predictions"][0]
        assert "description" in pred
        assert pred["description"] == "Domestic cat"
    
    def test_parse_openai_text_response(self, vision_classifier):
        """Test parsing OpenAI text response fallback."""
        response_text = """
        This image shows a category of domestic animal.
        The main subject appears to be a cat type creature.
        The object in the image is clearly a feline.
        """
        
        result = vision_classifier._parse_openai_text_response(response_text, "gpt-4o")
        
        assert len(result["predictions"]) > 0
        assert result["provider"] == "openai"
        assert result["model_used"] == "gpt-4o"
        assert "parsing_method" in result["metadata"]
        assert result["metadata"]["parsing_method"] == "text_fallback"
    
    def test_parse_openai_text_response_no_categories(self, vision_classifier):
        """Test parsing OpenAI text response with no clear categories."""
        response_text = "This is a nice image with good lighting."
        
        result = vision_classifier._parse_openai_text_response(response_text, "gpt-4o")
        
        assert len(result["predictions"]) == 1
        assert result["predictions"][0]["label"] == "general_image"
        assert result["predictions"][0]["confidence"] == 0.7


class TestVisionClassifierHuggingFace:
    """Test suite for HuggingFace-specific functionality."""
    
    @pytest.fixture
    def vision_classifier(self):
        return VisionClassifier()
    
    def test_hf_supported_models_structure(self, vision_classifier):
        """Test HuggingFace supported models structure."""
        models = vision_classifier.SUPPORTED_HF_MODELS
        
        # Test that all models have required metadata
        for model_name, model_info in models.items():
            assert "name" in model_info
            assert "type" in model_info
            assert "size" in model_info
            assert "speed" in model_info
            
            # Test model types are valid
            assert model_info["type"] in ["transformer", "cnn"]
            
            # Test speed categories are valid
            assert model_info["speed"] in ["very_fast", "fast", "medium", "slow"]
    
    def test_hf_model_validation_logic(self, vision_classifier):
        """Test HuggingFace model validation logic."""
        # Valid model should not raise error
        valid_models = list(vision_classifier.SUPPORTED_HF_MODELS.keys())
        assert len(valid_models) > 0
        
        # Test that we have expected models
        expected_models = [
            "google/vit-base-patch16-224",
            "microsoft/resnet-50", 
            "google/efficientnet-b0",
            "facebook/convnext-tiny-224"
        ]
        
        for model in expected_models:
            assert model in vision_classifier.SUPPORTED_HF_MODELS


class TestVisionClassifierOpenAI:
    """Test suite for OpenAI-specific functionality."""
    
    @pytest.fixture
    def vision_classifier(self):
        return VisionClassifier()
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API response."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = json.dumps({
            "predictions": [
                {"label": "cat", "confidence": 0.95, "description": "Domestic cat sitting"},
                {"label": "pet", "confidence": 0.87, "description": "Household pet"},
                {"label": "animal", "confidence": 0.82, "description": "Mammalian animal"}
            ],
            "analysis": "The image shows a domestic cat in a typical resting position."
        })
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response
    
    @pytest.mark.asyncio
    async def test_openai_json_response_parsing(self, vision_classifier, mock_openai_response):
        """Test parsing structured JSON response from OpenAI."""
        config = {
            "openai_model": "gpt-4-vision-preview",
            "top_k": 3
        }
        
        with patch('builtins.open', mock_open(read_data=b"fake_image_data")), \
             patch('base64.b64encode', return_value=b"fake_base64"), \
             patch('pathlib.Path.suffix', return_value='.jpg'):
            
            vision_classifier.openai_client = Mock()
            vision_classifier.openai_client.chat.completions.create.return_value = mock_openai_response
            
            result = await vision_classifier._classify_with_openai("/fake/path.jpg", config)
            
            assert result["provider"] == "openai"
            assert len(result["predictions"]) == 3
            assert result["top_prediction"]["label"] == "cat"
            assert "analysis" in result
    
    @pytest.mark.asyncio
    async def test_openai_custom_prompt(self, vision_classifier):
        """Test OpenAI with custom classification prompt."""
        config = {
            "openai_model": "gpt-4o",
            "classification_prompt": "What type of animal is this?",
            "top_k": 2
        }
        
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = '{"predictions": [{"label": "dog", "confidence": 0.9}], "analysis": "It\'s a dog"}'
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        with patch('builtins.open', mock_open(read_data=b"fake_image_data")), \
             patch('base64.b64encode', return_value=b"fake_base64"), \
             patch('pathlib.Path.suffix', return_value='.jpg'):
            
            vision_classifier.openai_client = Mock()
            vision_classifier.openai_client.chat.completions.create.return_value = mock_response
            
            result = await vision_classifier._classify_with_openai("/fake/path.jpg", config)
            
            # Verify custom prompt was used
            call_args = vision_classifier.openai_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            prompt_text = messages[0]["content"][0]["text"]
            assert "What type of animal is this?" in prompt_text


class TestVisionClassifierIntegration:
    """Integration tests for VisionClassifier with flow context."""
    
    @pytest.mark.asyncio
    async def test_vision_classifier_in_flow_context(self):
        """Test VisionClassifier as part of a flow context."""
        vision_classifier = VisionClassifier()
        context = FlowContext(
            flow_name="image_analysis_flow",
            inputs={"image_file": "test_image.jpg"},
            flow_id="integration-test"
        )
        
        # Add previous step result (e.g., from file handler)
        file_result = ExecutionResult(
            success=True,
            outputs={
                "temp_path": "/tmp/uploaded_image.jpg",
                "filename": "test_image.jpg",
                "file_extension": ".jpg"
            }
        )
        context.add_step_result("file_upload", file_result)
        
        config = {
            "image": context.step_results["file_upload"].outputs["temp_path"],
            "provider": "huggingface",
            "hf_model_name": "google/vit-base-patch16-224",
            "top_k": 3
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch.object(vision_classifier, '_classify_with_huggingface', 
                         return_value={
                             'predictions': [
                                 {'label': 'golden retriever', 'confidence': 0.89, 'rank': 1},
                                 {'label': 'dog', 'confidence': 0.76, 'rank': 2}
                             ],
                             'top_prediction': {'label': 'golden retriever', 'confidence': 0.89, 'rank': 1},
                             'confidence': 0.89,
                             'provider': 'huggingface'
                         }):
            
            result = await vision_classifier.execute(context, config)
            
            # Add result to context
            context.add_step_result("image_classification", result)
            
            # Verify integration
            assert result.success is True
            assert context.step_results["image_classification"] is not None
            
            # Verify the result can be used by subsequent steps
            classification_result = context.step_results["image_classification"]
            assert classification_result.outputs["top_prediction"]["label"] == "golden retriever"
            assert classification_result.outputs["confidence"] == 0.89
