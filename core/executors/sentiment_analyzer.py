"""
Sentiment Analysis Executor

Analyzes sentiment and emotions in text using multiple providers:
- HuggingFace: Specialized sentiment/emotion models
- LLM: Existing LLM infrastructure with sentiment prompts

Author: AI Inference Platform
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from .base_executor import BaseExecutor, ExecutionResult, FlowContext
from .llm_analyzer import LLMAnalyzer

logger = logging.getLogger(__name__)


class SentimentAnalyzer(BaseExecutor):
    """
    Analyze sentiment and emotions in text using HuggingFace models or LLM providers.
    
    Supports two providers:
    - huggingface: Fast, specialized sentiment/emotion models
    - llm: Reuse existing LLM infrastructure with sentiment prompts
    """
    
    # Sentiment prompt templates for LLM provider
    SENTIMENT_PROMPTS = {
        'basic': """
Analyze the sentiment of this text: "{text}"

Respond with JSON only:
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.85,
    "reasoning": "brief explanation"
}}
""",
        
        'detailed': """
Perform detailed sentiment analysis on: "{text}"

Provide JSON response only:
{{
    "sentiment": "positive|negative|neutral|mixed",
    "confidence": 0.85,
    "emotions": ["joy", "anger", "sadness", "fear", "surprise"],
    "key_phrases": [
        {{"phrase": "example phrase", "sentiment": "positive"}}
    ],
    "reasoning": "detailed explanation",
    "context": "any important context or nuances"
}}
""",
        
        'comprehensive': """
Comprehensive sentiment and emotion analysis for: "{text}"

JSON response format only:
{{
    "overall_sentiment": "positive|negative|neutral|mixed",
    "confidence": 0.85,
    "emotion_scores": {{
        "joy": 0.7,
        "anger": 0.1,
        "sadness": 0.05,
        "fear": 0.02,
        "surprise": 0.1,
        "disgust": 0.03
    }},
    "aspects": [
        {{"aspect": "product quality", "sentiment": "positive", "confidence": 0.9}},
        {{"aspect": "shipping", "sentiment": "negative", "confidence": 0.8}}
    ],
    "key_phrases": [
        {{"phrase": "excellent quality", "sentiment": "positive"}},
        {{"phrase": "slow delivery", "sentiment": "negative"}}
    ],
    "insights": "detailed analysis and recommendations"
}}
""",
        
        'emotions': """
Analyze the emotions in this text: "{text}"

Provide JSON response only:
{{
    "primary_emotion": "joy|anger|sadness|fear|surprise|disgust",
    "emotion_scores": {{
        "joy": 0.7,
        "anger": 0.1,
        "sadness": 0.05,
        "fear": 0.02,
        "surprise": 0.1,
        "disgust": 0.03
    }},
    "confidence": 0.85,
    "emotional_intensity": "low|medium|high",
    "reasoning": "explanation of detected emotions"
}}
"""
    }
    
    # HuggingFace model mappings
    HF_MODELS = {
        'basic': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'emotions': 'j-hartmann/emotion-english-distilroberta-base',
        'comprehensive': 'SamLowe/roberta-base-go_emotions',
        'detailed': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    }
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.name = name or "sentiment_analyzer"
        
        # Thread pool for CPU-intensive HuggingFace processing
        self.hf_thread_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix='SentimentHF-Worker'
        )
        
        # Cache for loaded HuggingFace models
        self._loaded_models = {}
        
        # LLM analyzer instance for LLM provider
        self._llm_analyzer = None
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Analyze sentiment using specified provider.
        
        Config parameters:
        - text (required): Text to analyze
        - provider (optional): 'huggingface' or 'llm' (default: 'huggingface')
        - analysis_type (optional): 'basic', 'detailed', 'comprehensive', 'emotions' (default: 'basic')
        - hf_model_name (optional): HuggingFace model name (default: auto-selected)
        - llm_model (optional): LLM model name (default: 'mistral')
        - device (optional): Device for HF models - 'auto', 'cpu', 'cuda' (default: 'auto')
        """
        start_time = time.time()
        
        try:
            # Validate configuration
            validation_result = self._validate_config(config)
            if not validation_result.success:
                return validation_result
            
            text = config.get('text')
            provider = config.get('provider', 'huggingface')
            analysis_type = config.get('analysis_type', 'basic')
            
            logger.info(f"Starting sentiment analysis: {len(text)} chars using {provider}/{analysis_type}")
            
            # Route to appropriate provider
            if provider == 'huggingface':
                result = await self._analyze_huggingface(text, config)
            elif provider == 'llm':
                result = await self._analyze_llm(text, config, context)
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unknown provider: {provider}. Supported: huggingface, llm"
                )
            
            processing_time = time.time() - start_time
            
            if not result.success:
                return result
            
            # Add metadata
            outputs = result.outputs
            outputs.update({
                "processing_time_seconds": processing_time,
                "provider": provider,
                "analysis_type": analysis_type,
                "text_length": len(text),
                "model_used": self._get_model_name(provider, config)
            })
            
            logger.info(f"Sentiment analysis completed in {processing_time:.2f}s: {outputs.get('sentiment', 'N/A')}")
            
            return ExecutionResult(
                success=True,
                outputs=outputs,
                metadata={
                    "executor": self.name,
                    "provider": provider,
                    "analysis_type": analysis_type,
                    "processing_time": processing_time,
                    "text_length": len(text)
                }
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=f"Sentiment analysis failed: {str(e)}"
            )
    
    def _validate_config(self, config: Dict[str, Any]) -> ExecutionResult:
        """Validate sentiment analysis configuration."""
        try:
            # Check required parameters
            if 'text' not in config:
                return ExecutionResult(
                    success=False,
                    error="text is required"
                )
            
            text = config['text']
            if not text or not text.strip():
                return ExecutionResult(
                    success=False,
                    error="text cannot be empty"
                )
            
            # Validate text length (reasonable limits)
            if len(text) > 50000:  # 50K characters max
                return ExecutionResult(
                    success=False,
                    error=f"Text too long: {len(text)} characters (max: 50,000)"
                )
            
            # Validate provider
            provider = config.get('provider', 'huggingface')
            if provider not in ['huggingface', 'llm']:
                return ExecutionResult(
                    success=False,
                    error="Provider must be 'huggingface' or 'llm'"
                )
            
            # Validate analysis type
            analysis_type = config.get('analysis_type', 'basic')
            if analysis_type not in ['basic', 'detailed', 'comprehensive', 'emotions']:
                return ExecutionResult(
                    success=False,
                    error="analysis_type must be 'basic', 'detailed', 'comprehensive', or 'emotions'"
                )
            
            return ExecutionResult(success=True)
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Configuration validation failed: {str(e)}"
            )
    
    async def _analyze_huggingface(self, text: str, config: Dict[str, Any]) -> ExecutionResult:
        """Analyze sentiment using HuggingFace models."""
        try:
            # Import HuggingFace dependencies
            try:
                from transformers import pipeline
                import torch
            except ImportError:
                return ExecutionResult(
                    success=False,
                    error="HuggingFace transformers not installed. Run: pip install transformers torch"
                )
            
            analysis_type = config.get('analysis_type', 'basic')
            device = config.get('device', 'auto')
            model_name = config.get('hf_model_name') or self.HF_MODELS.get(analysis_type)
            
            # Determine device
            if device == 'auto':
                device = 0 if torch.cuda.is_available() else -1
            elif device == 'cuda':
                device = 0 if torch.cuda.is_available() else -1
            elif device == 'cpu':
                device = -1
            
            logger.info(f"Using HuggingFace model: {model_name} on device: {device}")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.hf_thread_pool,
                self._analyze_hf_sync,
                text, model_name, device, analysis_type
            )
            
            return ExecutionResult(success=True, outputs=result)
            
        except Exception as e:
            logger.error(f"HuggingFace sentiment analysis failed: {str(e)}")
            return ExecutionResult(
                success=False,
                error=f"HuggingFace analysis failed: {str(e)}"
            )
    
    def _analyze_hf_sync(self, text: str, model_name: str, device: int, analysis_type: str) -> Dict[str, Any]:
        """Synchronous HuggingFace analysis (runs in thread pool)."""
        try:
            from transformers import pipeline
            
            # Create or get cached pipeline
            cache_key = f"{model_name}_{device}"
            if cache_key not in self._loaded_models:
                if 'emotion' in model_name.lower() or analysis_type in ['emotions', 'comprehensive']:
                    task = "text-classification"
                else:
                    task = "sentiment-analysis"
                
                pipe = pipeline(
                    task,
                    model=model_name,
                    device=device,
                    return_all_scores=True
                )
                self._loaded_models[cache_key] = pipe
            else:
                pipe = self._loaded_models[cache_key]
            
            # Analyze sentiment
            results = pipe(text)
            
            # Process results based on analysis type
            if analysis_type == 'basic':
                return self._process_basic_sentiment(results)
            elif analysis_type == 'emotions':
                return self._process_emotions(results)
            elif analysis_type in ['detailed', 'comprehensive']:
                return self._process_comprehensive(results, analysis_type)
            
        except Exception as e:
            raise Exception(f"HuggingFace processing failed: {str(e)}")
    
    def _process_basic_sentiment(self, results) -> Dict[str, Any]:
        """Process basic sentiment results."""
        if isinstance(results[0], list):
            results = results[0]
        
        # Find highest confidence result
        best_result = max(results, key=lambda x: x['score'])
        
        # Map labels to standard sentiment
        label_map = {
            'POSITIVE': 'positive',
            'NEGATIVE': 'negative', 
            'NEUTRAL': 'neutral',
            'LABEL_0': 'negative',  # Some models use LABEL_X
            'LABEL_1': 'neutral',
            'LABEL_2': 'positive'
        }
        
        sentiment = label_map.get(best_result['label'].upper(), best_result['label'].lower())
        
        return {
            'sentiment': sentiment,
            'confidence': round(best_result['score'], 3),
            'all_scores': {r['label']: round(r['score'], 3) for r in results}
        }
    
    def _process_emotions(self, results) -> Dict[str, Any]:
        """Process emotion analysis results."""
        if isinstance(results[0], list):
            results = results[0]
        
        # Convert to emotion scores
        emotion_scores = {}
        primary_emotion = None
        max_score = 0
        
        for result in results:
            emotion = result['label'].lower()
            score = round(result['score'], 3)
            emotion_scores[emotion] = score
            
            if score > max_score:
                max_score = score
                primary_emotion = emotion
        
        return {
            'primary_emotion': primary_emotion,
            'emotion_scores': emotion_scores,
            'confidence': max_score,
            'emotional_intensity': 'high' if max_score > 0.7 else 'medium' if max_score > 0.4 else 'low'
        }
    
    def _process_comprehensive(self, results, analysis_type: str) -> Dict[str, Any]:
        """Process comprehensive analysis results."""
        basic_result = self._process_basic_sentiment(results)
        
        if analysis_type == 'comprehensive':
            # Try to extract emotions if available
            emotions = {}
            if isinstance(results[0], list):
                for result in results[0]:
                    label = result['label'].lower()
                    if label in ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust']:
                        emotions[label] = round(result['score'], 3)
            
            return {
                'sentiment': basic_result['sentiment'],  # Use 'sentiment' instead of 'overall_sentiment'
                'confidence': basic_result['confidence'],
                'emotion_scores': emotions if emotions else {'joy': 0.0, 'anger': 0.0, 'sadness': 0.0},
                'all_scores': basic_result['all_scores'],
                'analysis_type': 'comprehensive'
            }
        
        return basic_result
    
    async def _analyze_llm(self, text: str, config: Dict[str, Any], context: FlowContext) -> ExecutionResult:
        """Analyze sentiment using LLM provider (reuse existing LLMAnalyzer)."""
        try:
            # Create LLMAnalyzer instance if not exists
            if not self._llm_analyzer:
                self._llm_analyzer = LLMAnalyzer()
            
            analysis_type = config.get('analysis_type', 'basic')
            llm_model = config.get('llm_model', 'mistral')
            
            # Build sentiment-specific prompt
            prompt_template = self.SENTIMENT_PROMPTS.get(analysis_type, self.SENTIMENT_PROMPTS['basic'])
            prompt = prompt_template.format(text=text)
            
            logger.info(f"Using LLM for sentiment analysis: {llm_model}/{analysis_type}")
            
            # Use existing LLM infrastructure
            llm_config = {
                'prompt': prompt,  # LLMAnalyzer expects 'prompt', not 'text'
                'model': llm_model,
                'temperature': 0.1,  # Low temperature for consistent analysis
                'max_tokens': 1000
            }
            
            result = await self._llm_analyzer.execute(context, llm_config)
            
            if not result.success:
                return result
            
            # Try to parse JSON response
            try:
                import json
                response_text = result.outputs.get('analysis', '').strip()
                
                # Extract JSON from response if wrapped in text
                if '```json' in response_text:
                    start = response_text.find('{')
                    end = response_text.rfind('}') + 1
                    response_text = response_text[start:end]
                elif response_text.startswith('```'):
                    lines = response_text.split('\n')
                    response_text = '\n'.join(lines[1:-1])
                
                parsed_result = json.loads(response_text)
                
                # Standardize the response format
                standardized = self._standardize_llm_response(parsed_result, analysis_type)
                
                return ExecutionResult(success=True, outputs=standardized)
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse LLM JSON response: {e}")
                # Fallback to text analysis
                return ExecutionResult(
                    success=True,
                    outputs={
                        'sentiment': 'neutral',
                        'confidence': 0.5,
                        'analysis': result.outputs.get('analysis', ''),
                        'raw_response': response_text,
                        'note': 'LLM response could not be parsed as JSON'
                    }
                )
                
        except Exception as e:
            logger.error(f"LLM sentiment analysis failed: {str(e)}")
            return ExecutionResult(
                success=False,
                error=f"LLM analysis failed: {str(e)}"
            )
    
    def _standardize_llm_response(self, parsed_result: Dict, analysis_type: str) -> Dict[str, Any]:
        """Standardize LLM response format."""
        standardized = {}
        
        # Basic sentiment mapping
        if 'sentiment' in parsed_result:
            standardized['sentiment'] = parsed_result['sentiment']
        elif 'overall_sentiment' in parsed_result:
            standardized['sentiment'] = parsed_result['overall_sentiment']  # Map to standard field
        else:
            standardized['sentiment'] = 'neutral'
        
        # Confidence
        standardized['confidence'] = parsed_result.get('confidence', 0.8)
        
        # Analysis type specific fields
        if analysis_type == 'emotions':
            standardized['primary_emotion'] = parsed_result.get('primary_emotion')
            standardized['emotion_scores'] = parsed_result.get('emotion_scores', {})
            standardized['emotional_intensity'] = parsed_result.get('emotional_intensity')
        
        elif analysis_type in ['detailed', 'comprehensive']:
            standardized['emotions'] = parsed_result.get('emotions', [])
            standardized['key_phrases'] = parsed_result.get('key_phrases', [])
            standardized['reasoning'] = parsed_result.get('reasoning', '')
            
            if analysis_type == 'comprehensive':
                standardized['emotion_scores'] = parsed_result.get('emotion_scores', {})
                standardized['aspects'] = parsed_result.get('aspects', [])
                standardized['insights'] = parsed_result.get('insights', '')
        
        # Always include reasoning if available
        if 'reasoning' in parsed_result:
            standardized['reasoning'] = parsed_result['reasoning']
        
        return standardized
    
    def _get_model_name(self, provider: str, config: Dict[str, Any]) -> str:
        """Get the model name used for analysis."""
        if provider == 'huggingface':
            analysis_type = config.get('analysis_type', 'basic')
            return config.get('hf_model_name') or self.HF_MODELS.get(analysis_type, 'unknown')
        elif provider == 'llm':
            return config.get('llm_model', 'mistral')
        return 'unknown'
    
    def get_executor_info(self) -> Dict[str, Any]:
        """Return executor information."""
        return {
            "name": self.name,
            "description": "Sentiment and emotion analysis using HuggingFace or LLM providers",
            "version": "1.0.0",
            "providers": ["huggingface", "llm"],
            "analysis_types": ["basic", "detailed", "comprehensive", "emotions"],
            "capabilities": [
                "Sentiment analysis (positive/negative/neutral)",
                "Emotion detection and scoring",
                "Multi-provider support",
                "Comprehensive text analysis",
                "Aspect-based sentiment analysis",
                "Key phrase extraction"
            ],
            "supported_models": {
                "huggingface": list(self.HF_MODELS.values()),
                "llm": ["mistral", "llama2", "openai", "custom"]
            }
        }
