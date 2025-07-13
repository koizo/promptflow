"""
Whisper Processor Executor

Speech-to-text processing using OpenAI Whisper models.
Supports both local Whisper and OpenAI API integration.
"""

import os
import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .base_executor import BaseExecutor, ExecutionResult, FlowContext

logger = logging.getLogger(__name__)


class WhisperProcessor(BaseExecutor):
    """
    Process audio files using Whisper speech-to-text models.
    
    Supports local Whisper models and OpenAI Whisper API.
    Provides transcription with timestamps, confidence scores, and language detection.
    """
    
    # Supported languages (subset of Whisper's 99 languages)
    SUPPORTED_LANGUAGES = [
        'auto', 'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar', 'hi'
    ]
    
    # Whisper model sizes
    WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large']
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.name = name or "whisper_processor"
        
        # Thread pool for CPU-intensive transcription
        self.whisper_thread_pool = ThreadPoolExecutor(
            max_workers=2,  # Whisper is memory-intensive
            thread_name_prefix='Whisper-Worker'
        )
        
        # Cache for loaded models
        self._loaded_models = {}
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Transcribe audio file using Whisper.
        
        Config parameters:
        - audio_path (required): Path to audio file
        - language (optional): Language code or 'auto' for detection (default: 'auto')
        - model_size (optional): Whisper model size (default: 'base')
        - provider (optional): 'local', 'openai', or 'huggingface' (default: 'local')
        - model_name (optional): HuggingFace model name (e.g., 'openai/whisper-large-v3')
        - include_timestamps (optional): Include word-level timestamps (default: True)
        - temperature (optional): Sampling temperature (default: 0.0)
        - api_key (optional): OpenAI API key (required for 'openai' provider)
        - device (optional): Device for HuggingFace models ('cpu', 'cuda', 'auto') (default: 'auto')
        """
        start_time = time.time()
        
        try:
            # Validate configuration
            validation_result = self._validate_config(config)
            if not validation_result.success:
                return validation_result
            
            audio_path = config.get('audio_path')
            language = config.get('language', 'auto')
            model_size = config.get('model_size', 'base')
            provider = config.get('provider', 'local')
            include_timestamps = config.get('include_timestamps', True)
            temperature = config.get('temperature', 0.0)
            
            logger.info(f"Starting Whisper transcription: {Path(audio_path).name} ({provider}/{model_size})")
            
            # Run transcription in background thread
            loop = asyncio.get_event_loop()
            transcription_result = await loop.run_in_executor(
                self.whisper_thread_pool,
                self._transcribe_audio_sync,
                audio_path, language, model_size, provider, include_timestamps, temperature, config
            )
            
            if not transcription_result['success']:
                return ExecutionResult(
                    success=False,
                    error=transcription_result['error']
                )
            
            processing_time = time.time() - start_time
            
            # Prepare outputs
            outputs = {
                "transcript": transcription_result['transcript'],
                "language_detected": transcription_result.get('language', language),
                "confidence_avg": transcription_result.get('confidence', 0.0),
                "word_count": len(transcription_result['transcript'].split()),
                "processing_time_seconds": processing_time,
                "model_used": f"{provider}/{model_size}",
                "audio_duration": transcription_result.get('duration', 0.0)
            }
            
            # Add timestamps if available
            if include_timestamps and 'segments' in transcription_result:
                outputs["segments"] = transcription_result['segments']
                outputs["has_timestamps"] = True
            else:
                outputs["has_timestamps"] = False
            
            logger.info(f"Whisper transcription completed in {processing_time:.2f}s: {len(outputs['transcript'])} characters")
            
            return ExecutionResult(
                success=True,
                outputs=outputs,
                metadata={
                    "executor": self.name,
                    "provider": provider,
                    "model": model_size,
                    "language": language,
                    "processing_time": processing_time,
                    "audio_file": Path(audio_path).name
                }
            )
            
        except Exception as e:
            logger.error(f"Whisper processing failed: {str(e)}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=f"Whisper processing failed: {str(e)}"
            )
    
    def _validate_config(self, config: Dict[str, Any]) -> ExecutionResult:
        """Validate Whisper configuration."""
        try:
            # Check required parameters
            if 'audio_path' not in config:
                return ExecutionResult(
                    success=False,
                    error="audio_path is required"
                )
            
            audio_path = config['audio_path']
            if not os.path.exists(audio_path):
                return ExecutionResult(
                    success=False,
                    error=f"Audio file not found: {audio_path}"
                )
            
            # Validate language
            language = config.get('language', 'auto')
            if language not in self.SUPPORTED_LANGUAGES:
                return ExecutionResult(
                    success=False,
                    error=f"Unsupported language: {language}. Supported: {self.SUPPORTED_LANGUAGES}"
                )
            
            # Validate model size
            model_size = config.get('model_size', 'base')
            if model_size not in self.WHISPER_MODELS:
                return ExecutionResult(
                    success=False,
                    error=f"Unsupported model size: {model_size}. Supported: {self.WHISPER_MODELS}"
                )
            
            # Validate provider
            provider = config.get('provider', 'local')
            if provider not in ['local', 'openai']:
                return ExecutionResult(
                    success=False,
                    error="Provider must be 'local' or 'openai'"
                )
            
            # Check OpenAI API key if using OpenAI provider
            if provider == 'openai' and not config.get('api_key') and not os.getenv('OPENAI_API_KEY'):
                return ExecutionResult(
                    success=False,
                    error="OpenAI API key required for 'openai' provider"
                )
            
            return ExecutionResult(success=True)
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Configuration validation failed: {str(e)}"
            )
    
    def _transcribe_audio_sync(self, audio_path: str, language: str, model_size: str, 
                              provider: str, include_timestamps: bool, temperature: float,
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous audio transcription (runs in thread pool)."""
        try:
            if provider == 'local':
                return self._transcribe_local_whisper(
                    audio_path, language, model_size, include_timestamps, temperature
                )
            elif provider == 'openai':
                return self._transcribe_openai_whisper(
                    audio_path, language, temperature, config
                )
            elif provider == 'huggingface':
                return self._transcribe_huggingface_whisper(
                    audio_path, language, config
                )
            else:
                return {
                    'success': False,
                    'error': f"Unknown provider: {provider}. Supported: local, openai, huggingface"
                }
                
        except Exception as e:
            logger.error(f"Synchronous transcription failed: {str(e)}")
            return {
                'success': False,
                'error': f"Transcription failed: {str(e)}"
            }
    
    def _transcribe_local_whisper(self, audio_path: str, language: str, model_size: str,
                                 include_timestamps: bool, temperature: float) -> Dict[str, Any]:
        """Transcribe using local Whisper model."""
        try:
            import whisper
            
            # Load model (with caching)
            model_key = model_size
            if model_key not in self._loaded_models:
                logger.info(f"Loading Whisper model: {model_size}")
                self._loaded_models[model_key] = whisper.load_model(model_size)
            
            model = self._loaded_models[model_key]
            
            # Transcribe audio
            transcribe_options = {
                'temperature': temperature,
                'word_timestamps': include_timestamps
            }
            
            if language != 'auto':
                transcribe_options['language'] = language
            
            result = model.transcribe(audio_path, **transcribe_options)
            
            # Extract segments with timestamps if requested
            segments = []
            if include_timestamps and 'segments' in result:
                for segment in result['segments']:
                    segments.append({
                        'start': segment.get('start', 0.0),
                        'end': segment.get('end', 0.0),
                        'text': segment.get('text', '').strip()
                    })
            
            return {
                'success': True,
                'transcript': result['text'].strip(),
                'language': result.get('language', language),
                'segments': segments,
                'duration': result.get('duration', 0.0),
                'confidence': 0.95  # Whisper doesn't provide confidence, use estimate
            }
            
        except ImportError:
            return {
                'success': False,
                'error': "Whisper not installed. Install with: pip install openai-whisper"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Local Whisper transcription failed: {str(e)}"
            }
    
    def _transcribe_openai_whisper(self, audio_path: str, language: str, 
                                  temperature: float, config: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe using OpenAI Whisper API."""
        try:
            import openai
            
            # Set API key
            api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
            openai.api_key = api_key
            
            # Open audio file
            with open(audio_path, 'rb') as audio_file:
                # Prepare API parameters
                api_params = {
                    'model': 'whisper-1',
                    'file': audio_file,
                    'temperature': temperature,
                    'response_format': 'verbose_json'
                }
                
                if language != 'auto':
                    api_params['language'] = language
                
                # Call OpenAI API
                response = openai.Audio.transcribe(**api_params)
            
            # Extract segments if available
            segments = []
            if 'segments' in response:
                for segment in response['segments']:
                    segments.append({
                        'start': segment.get('start', 0.0),
                        'end': segment.get('end', 0.0),
                        'text': segment.get('text', '').strip()
                    })
            
            return {
                'success': True,
                'transcript': response['text'].strip(),
                'language': response.get('language', language),
                'segments': segments,
                'duration': response.get('duration', 0.0),
                'confidence': 0.98  # OpenAI API typically high quality
            }
            
        except ImportError:
            return {
                'success': False,
                'error': "OpenAI library not installed. Install with: pip install openai"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"OpenAI Whisper API failed: {str(e)}"
            }
    
    def get_info(self) -> Dict[str, Any]:
        """Get executor information."""
        return {
            "name": "whisper_processor",
            "class": "WhisperProcessor",
            "module": "core.executors.whisper_processor", 
            "required_config": ["audio_path"],
            "optional_config": [
                "language",
                "model_size", 
                "provider",
                "include_timestamps",
                "temperature",
                "api_key"
            ],
            "description": """
    Process audio files using Whisper speech-to-text models.
    
    Supports both local Whisper models and OpenAI Whisper API.
    Provides transcription with timestamps, confidence scores, and language detection.
    Handles multiple audio formats and languages.
    """,
            "supported_languages": self.SUPPORTED_LANGUAGES,
            "supported_models": self.WHISPER_MODELS,
            "providers": ["local", "openai", "huggingface"],
            "capabilities": [
                "Speech-to-text transcription",
                "Multi-language support",
                "Timestamp generation",
                "Language detection", 
                "Multiple model sizes",
                "Local and cloud processing",
                "HuggingFace Transformers models"
            ]
        }
    
    def _transcribe_huggingface_whisper(self, audio_path: str, language: str, 
                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe using HuggingFace Whisper models."""
        try:
            # Import HuggingFace dependencies
            try:
                from transformers import AutomaticSpeechRecognitionPipeline, pipeline
                import torch
            except ImportError:
                return {
                    'success': False,
                    'error': "HuggingFace transformers not installed. Run: pip install transformers torch"
                }
            
            # Get model configuration
            model_name = config.get('model_name', 'openai/whisper-base')
            device = config.get('device', 'auto')
            include_timestamps = config.get('include_timestamps', True)
            
            # Determine device
            if device == 'auto':
                device = 0 if torch.cuda.is_available() else -1
            elif device == 'cuda':
                device = 0 if torch.cuda.is_available() else -1
            elif device == 'cpu':
                device = -1
            
            logger.info(f"Loading HuggingFace Whisper model: {model_name} on device: {device}")
            
            # Create or get cached pipeline
            cache_key = f"{model_name}_{device}"
            if cache_key not in self._loaded_models:
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=device,
                    return_timestamps=include_timestamps
                )
                self._loaded_models[cache_key] = pipe
            else:
                pipe = self._loaded_models[cache_key]
            
            # Prepare transcription parameters
            generate_kwargs = {}
            if language != 'auto':
                # Map language codes to HuggingFace format
                lang_map = {
                    'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german',
                    'it': 'italian', 'pt': 'portuguese', 'ru': 'russian', 'ja': 'japanese',
                    'ko': 'korean', 'zh': 'chinese', 'ar': 'arabic', 'hi': 'hindi'
                }
                if language in lang_map:
                    generate_kwargs['language'] = lang_map[language]
            
            # Transcribe audio
            logger.info(f"Transcribing audio with HuggingFace model: {model_name}")
            result = pipe(audio_path, generate_kwargs=generate_kwargs)
            
            # Process results
            if isinstance(result, dict):
                transcript = result.get('text', '')
                chunks = result.get('chunks', [])
            else:
                transcript = str(result)
                chunks = []
            
            # Build response
            response = {
                'success': True,
                'transcript': transcript.strip(),
                'language': language if language != 'auto' else 'detected',
                'confidence': 0.95,  # HuggingFace doesn't provide confidence scores
                'duration': 0.0  # Would need librosa to get actual duration
            }
            
            # Add timestamps if available
            if include_timestamps and chunks:
                segments = []
                for chunk in chunks:
                    if 'timestamp' in chunk:
                        segments.append({
                            'start': chunk['timestamp'][0] if chunk['timestamp'][0] else 0.0,
                            'end': chunk['timestamp'][1] if chunk['timestamp'][1] else 0.0,
                            'text': chunk.get('text', '').strip()
                        })
                response['segments'] = segments
            
            logger.info(f"HuggingFace transcription completed: {len(transcript)} characters")
            return response
            
        except Exception as e:
            logger.error(f"HuggingFace Whisper transcription failed: {str(e)}")
            return {
                'success': False,
                'error': f"HuggingFace transcription failed: {str(e)}"
            }
