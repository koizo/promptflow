"""
Unit tests for WhisperProcessor executor.

Tests audio transcription with local Whisper, OpenAI API, and HuggingFace providers,
model loading, language detection, and timestamp generation.
"""

import pytest
import uuid
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, mock_open

from core.executors.whisper_processor import WhisperProcessor
from core.executors.base_executor import FlowContext, ExecutionResult


class TestWhisperProcessor:
    """Test suite for WhisperProcessor executor."""
    
    @pytest.fixture
    def whisper_processor(self):
        """Create WhisperProcessor instance for testing."""
        return WhisperProcessor()
    
    @pytest.fixture
    def mock_context(self):
        """Create mock FlowContext for testing."""
        return FlowContext(
            flow_name="test_flow",
            inputs={"test_param": "test_value"},
            flow_id=f"test-flow-{uuid.uuid4().hex[:8]}"
        )
    
    @pytest.fixture
    def sample_audio_file(self):
        """Create a temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Write minimal WAV header (44 bytes) + some dummy data
            wav_header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
            temp_file.write(wav_header)
            temp_file.flush()
            yield temp_file.name
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except:
            pass
    
    @pytest.fixture
    def sample_local_config(self, sample_audio_file):
        """Sample configuration for local Whisper."""
        return {
            "audio_path": sample_audio_file,
            "provider": "local",
            "model_size": "base",
            "language": "en",
            "include_timestamps": True,
            "temperature": 0.0
        }
    
    @pytest.fixture
    def sample_openai_config(self, sample_audio_file):
        """Sample configuration for OpenAI Whisper."""
        return {
            "audio_path": sample_audio_file,
            "provider": "openai",
            "language": "auto",
            "temperature": 0.2,
            "api_key": "test-api-key"
        }
    
    @pytest.fixture
    def sample_hf_config(self, sample_audio_file):
        """Sample configuration for HuggingFace Whisper."""
        return {
            "audio_path": sample_audio_file,
            "provider": "huggingface",
            "model_name": "openai/whisper-base",
            "language": "en",
            "device": "cpu",
            "include_timestamps": True
        }
    
    @pytest.fixture
    def mock_whisper_result(self):
        """Mock Whisper transcription result."""
        return {
            'text': 'This is a test transcription of the audio file.',
            'language': 'en',
            'segments': [
                {'start': 0.0, 'end': 2.5, 'text': 'This is a test'},
                {'start': 2.5, 'end': 5.0, 'text': 'transcription of the audio file.'}
            ],
            'duration': 5.0
        }
    
    @pytest.mark.asyncio
    async def test_missing_audio_path(self, whisper_processor, mock_context):
        """Test error when audio_path is missing."""
        config = {"provider": "local"}
        
        result = await whisper_processor.execute(mock_context, config)
        
        assert result.success is False
        assert "audio_path is required" in result.error
    
    @pytest.mark.asyncio
    async def test_nonexistent_audio_file(self, whisper_processor, mock_context):
        """Test error when audio file doesn't exist."""
        config = {
            "audio_path": "/nonexistent/audio.wav",
            "provider": "local"
        }
        
        result = await whisper_processor.execute(mock_context, config)
        
        assert result.success is False
        assert "Audio file not found" in result.error
    
    @pytest.mark.asyncio
    async def test_unsupported_language(self, whisper_processor, mock_context, sample_audio_file):
        """Test error with unsupported language."""
        config = {
            "audio_path": sample_audio_file,
            "language": "unsupported_lang",
            "provider": "local"
        }
        
        result = await whisper_processor.execute(mock_context, config)
        
        assert result.success is False
        assert "Unsupported language" in result.error
    
    @pytest.mark.asyncio
    async def test_unsupported_model_size(self, whisper_processor, mock_context, sample_audio_file):
        """Test error with unsupported model size."""
        config = {
            "audio_path": sample_audio_file,
            "model_size": "unsupported_size",
            "provider": "local"
        }
        
        result = await whisper_processor.execute(mock_context, config)
        
        assert result.success is False
        assert "Unsupported model size" in result.error
    
    @pytest.mark.asyncio
    async def test_unsupported_provider(self, whisper_processor, mock_context, sample_audio_file):
        """Test error with unsupported provider."""
        config = {
            "audio_path": sample_audio_file,
            "provider": "unsupported_provider"
        }
        
        result = await whisper_processor.execute(mock_context, config)
        
        assert result.success is False
        assert "Provider must be" in result.error
    
    @pytest.mark.asyncio
    async def test_openai_missing_api_key(self, whisper_processor, mock_context, sample_audio_file):
        """Test error when OpenAI API key is missing."""
        config = {
            "audio_path": sample_audio_file,
            "provider": "openai"
        }
        
        with patch.dict(os.environ, {}, clear=True):
            result = await whisper_processor.execute(mock_context, config)
            
            assert result.success is False
            assert "OpenAI API key required" in result.error
    
    @pytest.mark.asyncio
    async def test_successful_local_whisper_transcription(self, whisper_processor, mock_context, 
                                                        sample_local_config, mock_whisper_result):
        """Test successful local Whisper transcription."""
        with patch.object(whisper_processor, '_transcribe_audio_sync', 
                         return_value={
                             'success': True,
                             'transcript': 'This is a test transcription.',
                             'language': 'en',
                             'segments': [
                                 {'start': 0.0, 'end': 5.0, 'text': 'This is a test transcription.'}
                             ],
                             'duration': 5.0,
                             'confidence': 0.95
                         }) as mock_transcribe:
            
            result = await whisper_processor.execute(mock_context, sample_local_config)
            
            assert result.success is True
            assert "transcript" in result.outputs
            assert "language_detected" in result.outputs
            assert "confidence_avg" in result.outputs
            assert "word_count" in result.outputs
            assert "processing_time_seconds" in result.outputs
            assert "model_used" in result.outputs
            assert "audio_duration" in result.outputs
            assert "segments" in result.outputs
            assert "has_timestamps" in result.outputs
            
            assert result.outputs["transcript"] == "This is a test transcription."
            assert result.outputs["language_detected"] == "en"
            assert result.outputs["confidence_avg"] == 0.95
            assert result.outputs["model_used"] == "local/base"
            assert result.outputs["has_timestamps"] is True
            assert len(result.outputs["segments"]) == 1
    
    @pytest.mark.asyncio
    async def test_successful_openai_transcription(self, whisper_processor, mock_context, 
                                                 sample_openai_config):
        """Test successful OpenAI Whisper transcription."""
        with patch.object(whisper_processor, '_transcribe_audio_sync', 
                         return_value={
                             'success': True,
                             'transcript': 'OpenAI transcription result.',
                             'language': 'en',
                             'duration': 3.5,
                             'confidence': 0.98
                         }) as mock_transcribe:
            
            result = await whisper_processor.execute(mock_context, sample_openai_config)
            
            assert result.success is True
            assert result.outputs["transcript"] == "OpenAI transcription result."
            assert result.outputs["model_used"] == "openai/base"
            assert result.outputs["confidence_avg"] == 0.98
    
    @pytest.mark.asyncio
    async def test_successful_huggingface_transcription(self, whisper_processor, mock_context, 
                                                      sample_hf_config):
        """Test successful HuggingFace Whisper transcription."""
        with patch.object(whisper_processor, '_transcribe_audio_sync', 
                         return_value={
                             'success': True,
                             'transcript': 'HuggingFace transcription result.',
                             'language': 'en',
                             'segments': [],
                             'confidence': 0.95
                         }) as mock_transcribe:
            
            result = await whisper_processor.execute(mock_context, sample_hf_config)
            
            assert result.success is True
            assert result.outputs["transcript"] == "HuggingFace transcription result."
            assert result.outputs["model_used"] == "huggingface/base"
    
    @pytest.mark.asyncio
    async def test_transcription_without_timestamps(self, whisper_processor, mock_context, sample_audio_file):
        """Test transcription without timestamps."""
        config = {
            "audio_path": sample_audio_file,
            "provider": "local",
            "include_timestamps": False
        }
        
        with patch.object(whisper_processor, '_transcribe_audio_sync', 
                         return_value={
                             'success': True,
                             'transcript': 'Test transcription.',
                             'language': 'en',
                             'confidence': 0.95
                         }) as mock_transcribe:
            
            result = await whisper_processor.execute(mock_context, config)
            
            assert result.success is True
            assert result.outputs["has_timestamps"] is False
            assert "segments" not in result.outputs or not result.outputs["segments"]
    
    @pytest.mark.asyncio
    async def test_transcription_failure(self, whisper_processor, mock_context, sample_local_config):
        """Test handling of transcription failure."""
        with patch.object(whisper_processor, '_transcribe_audio_sync', 
                         return_value={
                             'success': False,
                             'error': 'Transcription failed due to corrupted audio'
                         }) as mock_transcribe:
            
            result = await whisper_processor.execute(mock_context, sample_local_config)
            
            assert result.success is False
            assert "Transcription failed due to corrupted audio" in result.error
    
    @pytest.mark.asyncio
    async def test_default_configuration(self, whisper_processor, mock_context, sample_audio_file):
        """Test transcription with default configuration."""
        config = {"audio_path": sample_audio_file}
        
        with patch.object(whisper_processor, '_transcribe_audio_sync', 
                         return_value={
                             'success': True,
                             'transcript': 'Default config transcription.',
                             'language': 'auto',
                             'confidence': 0.95
                         }) as mock_transcribe:
            
            result = await whisper_processor.execute(mock_context, config)
            
            assert result.success is True
            assert result.outputs["model_used"] == "local/base"  # Default provider and model
    
    def test_whisper_processor_name(self, whisper_processor):
        """Test WhisperProcessor name property."""
        assert whisper_processor.name == "whisper_processor"
    
    def test_supported_languages(self, whisper_processor):
        """Test supported languages configuration."""
        languages = whisper_processor.SUPPORTED_LANGUAGES
        
        assert "auto" in languages
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages
        assert "de" in languages
        assert len(languages) >= 10  # Should have multiple languages
    
    def test_supported_models(self, whisper_processor):
        """Test supported Whisper models."""
        models = whisper_processor.WHISPER_MODELS
        
        assert "tiny" in models
        assert "base" in models
        assert "small" in models
        assert "medium" in models
        assert "large" in models
    
    def test_get_info(self, whisper_processor):
        """Test executor information."""
        info = whisper_processor.get_info()
        
        assert "name" in info
        assert "required_config" in info
        assert "optional_config" in info
        assert "supported_languages" in info
        assert "supported_models" in info
        assert "providers" in info
        assert "capabilities" in info
        
        # Check required config
        assert "audio_path" in info["required_config"]
        
        # Check optional config
        optional_keys = ["language", "model_size", "provider", "include_timestamps", "temperature", "api_key"]
        for key in optional_keys:
            assert key in info["optional_config"]
        
        # Check providers
        assert "local" in info["providers"]
        assert "openai" in info["providers"]
        assert "huggingface" in info["providers"]
        
        # Check capabilities
        capabilities = info["capabilities"]
        assert "Speech-to-text transcription" in capabilities
        assert "Multi-language support" in capabilities
        assert "Timestamp generation" in capabilities


class TestWhisperProcessorLocalProvider:
    """Test suite for local Whisper provider functionality."""
    
    @pytest.fixture
    def whisper_processor(self):
        return WhisperProcessor()
    
    def test_local_whisper_import_error(self, whisper_processor):
        """Test handling of missing Whisper library."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'whisper'")):
            result = whisper_processor._transcribe_local_whisper(
                "/fake/audio.wav", "en", "base", True, 0.0
            )
            
            assert result["success"] is False
            assert "Whisper not installed" in result["error"]
    
    def test_local_whisper_configuration_validation(self, whisper_processor):
        """Test local Whisper configuration validation."""
        # Test that the method exists and has proper signature
        import inspect
        sig = inspect.signature(whisper_processor._transcribe_local_whisper)
        expected_params = ['audio_path', 'language', 'model_size', 'include_timestamps', 'temperature']
        
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params


class TestWhisperProcessorOpenAIProvider:
    """Test suite for OpenAI Whisper provider functionality."""
    
    @pytest.fixture
    def whisper_processor(self):
        return WhisperProcessor()
    
    def test_openai_import_error(self, whisper_processor):
        """Test handling of missing OpenAI library."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'openai'")):
            result = whisper_processor._transcribe_openai_whisper(
                "/fake/audio.wav", "en", 0.0, {"api_key": "test-key"}
            )
            
            assert result["success"] is False
            assert "OpenAI library not installed" in result["error"]
    
    def test_openai_configuration_validation(self, whisper_processor):
        """Test OpenAI configuration validation."""
        # Test that the method exists and has proper signature
        import inspect
        sig = inspect.signature(whisper_processor._transcribe_openai_whisper)
        expected_params = ['audio_path', 'language', 'temperature', 'config']
        
        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params


class TestWhisperProcessorHuggingFaceProvider:
    """Test suite for HuggingFace Whisper provider functionality."""
    
    @pytest.fixture
    def whisper_processor(self):
        return WhisperProcessor()
    
    def test_huggingface_import_error(self, whisper_processor):
        """Test handling of missing HuggingFace libraries."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'transformers'")):
            result = whisper_processor._transcribe_huggingface_whisper(
                "/fake/audio.wav", "en", {"model_name": "openai/whisper-base"}
            )
            
            assert result["success"] is False
            assert "HuggingFace transformers not installed" in result["error"]
    
    def test_huggingface_device_selection(self, whisper_processor):
        """Test HuggingFace device selection logic."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = {'text': 'HF transcription'}
        
        with patch('transformers.pipeline', return_value=mock_pipeline) as mock_pipe_create, \
             patch('torch.cuda.is_available', return_value=True):
            
            # Test auto device selection with CUDA available
            result = whisper_processor._transcribe_huggingface_whisper(
                "/fake/audio.wav", "en", {"device": "auto", "model_name": "openai/whisper-base"}
            )
            
            # Should use CUDA (device=0)
            mock_pipe_create.assert_called_once()
            call_kwargs = mock_pipe_create.call_args[1]
            assert call_kwargs["device"] == 0
    
    def test_huggingface_model_caching(self, whisper_processor):
        """Test HuggingFace model caching."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = {'text': 'Cached model test'}
        
        with patch('transformers.pipeline', return_value=mock_pipeline) as mock_pipe_create, \
             patch('torch.cuda.is_available', return_value=False):
            
            config = {"device": "cpu", "model_name": "openai/whisper-base"}
            
            # First call should create pipeline
            result1 = whisper_processor._transcribe_huggingface_whisper("/fake/audio1.wav", "en", config)
            assert mock_pipe_create.call_count == 1
            
            # Second call should use cached pipeline
            result2 = whisper_processor._transcribe_huggingface_whisper("/fake/audio2.wav", "en", config)
            assert mock_pipe_create.call_count == 1  # Still 1, pipeline was cached
    
    def test_huggingface_with_timestamps(self, whisper_processor):
        """Test HuggingFace transcription with timestamps."""
        mock_pipeline = Mock()
        mock_pipeline.return_value = {
            'text': 'HF transcription with timestamps',
            'chunks': [
                {'timestamp': [0.0, 2.0], 'text': 'HF transcription'},
                {'timestamp': [2.0, 4.0], 'text': 'with timestamps'}
            ]
        }
        
        with patch('transformers.pipeline', return_value=mock_pipeline), \
             patch('torch.cuda.is_available', return_value=False):
            
            result = whisper_processor._transcribe_huggingface_whisper(
                "/fake/audio.wav", "en", 
                {"model_name": "openai/whisper-base", "include_timestamps": True}
            )
            
            assert result["success"] is True
            assert len(result["segments"]) == 2
            assert result["segments"][0]["start"] == 0.0
            assert result["segments"][0]["end"] == 2.0


class TestWhisperProcessorIntegration:
    """Integration tests for WhisperProcessor with flow context."""
    
    @pytest.mark.asyncio
    async def test_whisper_processor_in_flow_context(self):
        """Test WhisperProcessor as part of a flow context."""
        whisper_processor = WhisperProcessor()
        context = FlowContext(
            flow_name="audio_transcription_flow",
            inputs={"audio_file": "speech.wav"},
            flow_id="integration-test"
        )
        
        # Add previous step result (e.g., from file handler)
        file_result = ExecutionResult(
            success=True,
            outputs={
                "temp_path": "/tmp/uploaded_audio.wav",
                "filename": "speech.wav",
                "file_extension": ".wav"
            }
        )
        context.add_step_result("file_upload", file_result)
        
        config = {
            "audio_path": context.step_results["file_upload"].outputs["temp_path"],
            "provider": "local",
            "language": "en",
            "include_timestamps": True
        }
        
        # Mock file existence check and transcription
        with patch('os.path.exists', return_value=True), \
             patch.object(whisper_processor, '_transcribe_audio_sync', 
                         return_value={
                             'success': True,
                             'transcript': 'This is the transcribed speech from the audio file.',
                             'language': 'en',
                             'segments': [
                                 {'start': 0.0, 'end': 3.0, 'text': 'This is the transcribed speech'},
                                 {'start': 3.0, 'end': 6.0, 'text': 'from the audio file.'}
                             ],
                             'duration': 6.0,
                             'confidence': 0.95
                         }):
            
            result = await whisper_processor.execute(context, config)
            
            # Add result to context
            context.add_step_result("speech_transcription", result)
            
            # Verify integration
            assert result.success is True
            assert context.step_results["speech_transcription"] is not None
            
            # Verify the result can be used by subsequent steps
            transcription_result = context.step_results["speech_transcription"]
            assert transcription_result.outputs["transcript"] == "This is the transcribed speech from the audio file."
            assert transcription_result.outputs["word_count"] == 9  # Correct word count
            assert transcription_result.outputs["has_timestamps"] is True
            assert len(transcription_result.outputs["segments"]) == 2
