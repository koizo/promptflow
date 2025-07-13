"""
Vision Classification Executor

Provides image classification capabilities using multiple providers:
- HuggingFace: Fast, local processing with various pre-trained models
- OpenAI: High accuracy vision analysis with natural language descriptions

Follows the same dual-provider pattern as SentimentAnalyzer for consistency.
"""

import asyncio
import base64
import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from PIL import Image
import torch
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification

from .base_executor import BaseExecutor, ExecutionResult, FlowContext

logger = logging.getLogger(__name__)


class VisionClassifier(BaseExecutor):
    """
    Image classification executor with dual provider support.
    
    Providers:
    - huggingface: Fast local processing with multiple model options
    - openai: High accuracy vision analysis with GPT-4V
    
    Supported HuggingFace Models:
    - google/vit-base-patch16-224 (Vision Transformer)
    - microsoft/resnet-50 (ResNet-50)
    - google/efficientnet-b0 (EfficientNet)
    - facebook/convnext-tiny-224 (ConvNeXT)
    """
    
    # Supported HuggingFace models with metadata
    SUPPORTED_HF_MODELS = {
        'google/vit-base-patch16-224': {
            'name': 'Vision Transformer (Base)',
            'type': 'transformer',
            'size': '86M parameters',
            'speed': 'fast'
        },
        'google/vit-large-patch16-224': {
            'name': 'Vision Transformer (Large)', 
            'type': 'transformer',
            'size': '304M parameters',
            'speed': 'medium'
        },
        'microsoft/resnet-50': {
            'name': 'ResNet-50',
            'type': 'cnn',
            'size': '25M parameters', 
            'speed': 'very_fast'
        },
        'microsoft/resnet-101': {
            'name': 'ResNet-101',
            'type': 'cnn',
            'size': '45M parameters',
            'speed': 'fast'
        },
        'google/efficientnet-b0': {
            'name': 'EfficientNet-B0',
            'type': 'cnn',
            'size': '5M parameters',
            'speed': 'very_fast'
        },
        'google/efficientnet-b4': {
            'name': 'EfficientNet-B4',
            'type': 'cnn', 
            'size': '19M parameters',
            'speed': 'medium'
        },
        'facebook/convnext-tiny-224': {
            'name': 'ConvNeXT Tiny',
            'type': 'transformer',
            'size': '28M parameters',
            'speed': 'fast'
        },
        'facebook/convnext-base-224': {
            'name': 'ConvNeXT Base',
            'type': 'transformer',
            'size': '89M parameters', 
            'speed': 'medium'
        }
    }
    
    # OpenAI Vision models
    OPENAI_MODELS = {
        'gpt-4-vision-preview': 'GPT-4 Vision (Preview)',
        'gpt-4o': 'GPT-4 Omni (Latest)',
        'gpt-4o-mini': 'GPT-4 Omni Mini (Fast)'
    }
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.name = name or "vision_classifier"
        self.hf_models = {}  # Model cache for HuggingFace
        self.openai_client = None
        
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Execute image classification with specified provider.
        
        Args:
            context: Flow context containing configuration and image data
            
        Returns:
            ExecutionResult with classification predictions
        """
        try:
            # Extract configuration
            image_data = config.get('image')
            provider = config.get('provider', 'huggingface').lower()
            
            if not image_data:
                return ExecutionResult(
                    success=False,
                    error="Image data is required for classification"
                )
            
            # Handle different image input formats
            image_path = None
            temp_file = None
            
            if isinstance(image_data, dict):
                # Image data comes as dictionary with content, filename, etc.
                if 'content' not in image_data:
                    return ExecutionResult(
                        success=False,
                        error="Image data dictionary must contain 'content' field"
                    )
                
                # Create temporary file from image content
                import tempfile
                import os
                
                # Get file extension from filename or content_type
                filename = image_data.get('filename', 'image.jpg')
                file_ext = Path(filename).suffix or '.jpg'
                
                # Create temporary file
                temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext)
                try:
                    with os.fdopen(temp_fd, 'wb') as tmp_file:
                        tmp_file.write(image_data['content'])
                    image_path = temp_path
                    temp_file = temp_path
                    logger.info(f"Created temporary file for image: {filename}")
                except Exception as e:
                    os.close(temp_fd)
                    return ExecutionResult(
                        success=False,
                        error=f"Failed to create temporary image file: {str(e)}"
                    )
            
            elif isinstance(image_data, (str, Path)):
                # Image data is a file path
                image_path = str(image_data)
                if not Path(image_path).exists():
                    return ExecutionResult(
                        success=False,
                        error=f"Image file not found: {image_path}"
                    )
            
            else:
                return ExecutionResult(
                    success=False,
                    error="Image must be either a file path or dictionary with content"
                )
            
            logger.info(f"Starting image classification using {provider}")
            start_time = time.time()
            
            try:
                # Route to appropriate provider
                if provider == 'huggingface':
                    result = await self._classify_with_huggingface(image_path, config)
                elif provider == 'openai':
                    result = await self._classify_with_openai(image_path, config)
                else:
                    return ExecutionResult(
                        success=False,
                        error=f"Unsupported provider: {provider}. Use 'huggingface' or 'openai'"
                    )
                
                processing_time = time.time() - start_time
                result['processing_time_seconds'] = round(processing_time, 3)
                
                logger.info(f"Image classification completed in {processing_time:.3f}s: {result.get('top_prediction', {}).get('label', 'unknown')}")
                
                return ExecutionResult(success=True, outputs=result)
                
            finally:
                # Clean up temporary file if created
                if temp_file and Path(temp_file).exists():
                    try:
                        Path(temp_file).unlink()
                        logger.debug(f"Cleaned up temporary file: {temp_file}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temporary file {temp_file}: {cleanup_error}")
            
        except Exception as e:
            logger.error(f"Image classification failed: {str(e)}")
            return ExecutionResult(
                success=False,
                error=f"Classification failed: {str(e)}"
            )
    
    async def _classify_with_huggingface(self, image_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify image using HuggingFace transformers models.
        
        Args:
            image_path: Path to image file
            config: Configuration dictionary
            
        Returns:
            Standardized classification results
        """
        model_name = config.get('hf_model_name', 'google/vit-base-patch16-224')
        device = config.get('device', 'auto')
        top_k = config.get('top_k', 5)
        confidence_threshold = config.get('confidence_threshold', 0.1)
        
        # Validate model
        if model_name not in self.SUPPORTED_HF_MODELS:
            raise ValueError(f"Unsupported HuggingFace model: {model_name}")
        
        # Determine device
        if device == 'auto':
            device_id = 0 if torch.cuda.is_available() else -1
            device_name = 'cuda' if device_id >= 0 else 'cpu'
        elif device == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device_id = -1
                device_name = 'cpu'
            else:
                device_id = 0
                device_name = 'cuda'
        else:  # cpu
            device_id = -1
            device_name = 'cpu'
        
        logger.info(f"Using HuggingFace model: {model_name} on device: {device_name}")
        
        # Load or get cached model
        model_key = f"{model_name}_{device_name}"
        if model_key not in self.hf_models:
            logger.info(f"Loading HuggingFace model: {model_name}")
            try:
                classifier = pipeline(
                    "image-classification",
                    model=model_name,
                    device=device_id
                )
                self.hf_models[model_key] = classifier
                logger.info(f"Model loaded successfully: {model_name}")
            except Exception as e:
                raise RuntimeError(f"Failed to load HuggingFace model {model_name}: {str(e)}")
        else:
            classifier = self.hf_models[model_key]
        
        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")
        
        # Perform classification
        try:
            # Get all predictions by setting top_k to None (returns all classes)
            predictions = classifier(image, top_k=None)
            
            # Filter by confidence threshold and limit to top_k
            filtered_predictions = [
                pred for pred in predictions 
                if pred['score'] >= confidence_threshold
            ]
            
            # Sort by score and take top_k
            top_predictions = sorted(filtered_predictions, key=lambda x: x['score'], reverse=True)[:top_k]
            
            if not top_predictions:
                logger.warning(f"No predictions above confidence threshold {confidence_threshold}")
                top_predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)[:1]
            
        except Exception as e:
            raise RuntimeError(f"Classification failed with model {model_name}: {str(e)}")
        
        # Standardize results
        return self._standardize_hf_results(top_predictions, model_name, device_name)
    
    async def _classify_with_openai(self, image_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify image using OpenAI Vision API.
        
        Args:
            image_path: Path to image file
            config: Configuration dictionary
            
        Returns:
            Standardized classification results
        """
        model = config.get('openai_model', 'gpt-4-vision-preview')
        custom_prompt = config.get('classification_prompt')
        top_k = config.get('top_k', 5)
        
        # Validate model
        if model not in self.OPENAI_MODELS:
            raise ValueError(f"Unsupported OpenAI model: {model}")
        
        # Initialize OpenAI client if needed
        if not self.openai_client:
            try:
                import openai
                self.openai_client = openai.OpenAI()
                logger.info("OpenAI client initialized")
            except ImportError:
                raise RuntimeError("OpenAI package not installed. Install with: pip install openai")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
        
        # Encode image to base64
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                
            # Determine image format
            image_format = Path(image_path).suffix.lower().lstrip('.')
            if image_format == 'jpg':
                image_format = 'jpeg'
                
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {str(e)}")
        
        # Create classification prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = f"""Analyze this image and provide a classification with the top {top_k} most likely categories.

Return your response in this exact JSON format:
{{
    "predictions": [
        {{"label": "category_name", "confidence": 0.95, "description": "brief description"}},
        {{"label": "category_name", "confidence": 0.87, "description": "brief description"}}
    ],
    "analysis": "Brief analysis of what you see in the image"
}}

Focus on the main subject/content of the image and provide confidence scores between 0 and 1."""
        
        logger.info(f"Using OpenAI model: {model}")
        
        # Make API call
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1  # Low temperature for consistent classification
            )
            
            response_text = response.choices[0].message.content
            logger.info("OpenAI Vision API call successful")
            
        except Exception as e:
            raise RuntimeError(f"OpenAI Vision API call failed: {str(e)}")
        
        # Parse response
        try:
            import json
            # Try to extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                # Fallback: create structured response from text
                return self._parse_openai_text_response(response_text, model)
            
            parsed_response = json.loads(json_text)
            
        except Exception as e:
            logger.warning(f"Failed to parse OpenAI JSON response: {str(e)}")
            return self._parse_openai_text_response(response_text, model)
        
        # Standardize results
        return self._standardize_openai_results(parsed_response, model, response_text)
    
    def _standardize_hf_results(self, predictions: List[Dict], model_name: str, device: str) -> Dict[str, Any]:
        """Standardize HuggingFace classification results."""
        if not predictions:
            return {
                'predictions': [],
                'top_prediction': None,
                'confidence': 0.0,
                'provider': 'huggingface',
                'model_used': model_name,
                'device': device,
                'metadata': {
                    'model_info': self.SUPPORTED_HF_MODELS.get(model_name, {}),
                    'total_predictions': 0
                }
            }
        
        # Format predictions
        formatted_predictions = [
            {
                'label': pred['label'],
                'confidence': round(pred['score'], 4),
                'rank': idx + 1
            }
            for idx, pred in enumerate(predictions)
        ]
        
        top_prediction = formatted_predictions[0] if formatted_predictions else None
        
        return {
            'predictions': formatted_predictions,
            'top_prediction': top_prediction,
            'confidence': top_prediction['confidence'] if top_prediction else 0.0,
            'provider': 'huggingface',
            'model_used': model_name,
            'device': device,
            'metadata': {
                'model_info': self.SUPPORTED_HF_MODELS.get(model_name, {}),
                'total_predictions': len(formatted_predictions),
                'all_scores': {pred['label']: pred['confidence'] for pred in formatted_predictions}
            }
        }
    
    def _standardize_openai_results(self, parsed_response: Dict, model: str, raw_response: str) -> Dict[str, Any]:
        """Standardize OpenAI Vision API results."""
        predictions = parsed_response.get('predictions', [])
        analysis = parsed_response.get('analysis', '')
        
        # Format predictions
        formatted_predictions = []
        for idx, pred in enumerate(predictions):
            formatted_predictions.append({
                'label': pred.get('label', f'category_{idx+1}'),
                'confidence': float(pred.get('confidence', 0.0)),
                'description': pred.get('description', ''),
                'rank': idx + 1
            })
        
        # Sort by confidence
        formatted_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        top_prediction = formatted_predictions[0] if formatted_predictions else None
        
        return {
            'predictions': formatted_predictions,
            'top_prediction': top_prediction,
            'confidence': top_prediction['confidence'] if top_prediction else 0.0,
            'provider': 'openai',
            'model_used': model,
            'analysis': analysis,
            'metadata': {
                'model_info': {'name': self.OPENAI_MODELS.get(model, model)},
                'total_predictions': len(formatted_predictions),
                'raw_response': raw_response[:500] + '...' if len(raw_response) > 500 else raw_response
            }
        }
    
    def _parse_openai_text_response(self, response_text: str, model: str) -> Dict[str, Any]:
        """Fallback parser for OpenAI text responses that aren't JSON."""
        logger.info("Parsing OpenAI text response as fallback")
        
        # Simple text parsing - look for category mentions
        lines = response_text.split('\n')
        predictions = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['category', 'class', 'type', 'object', 'subject']):
                # Extract potential category name
                words = line.split()
                if len(words) > 1:
                    predictions.append({
                        'label': ' '.join(words[-2:]).strip('.,!?'),
                        'confidence': 0.8,  # Default confidence
                        'description': line,
                        'rank': len(predictions) + 1
                    })
        
        # If no structured predictions found, create a general one
        if not predictions:
            predictions = [{
                'label': 'general_image',
                'confidence': 0.7,
                'description': 'Image classification completed',
                'rank': 1
            }]
        
        top_prediction = predictions[0] if predictions else None
        
        return {
            'predictions': predictions[:5],  # Limit to top 5
            'top_prediction': top_prediction,
            'confidence': top_prediction['confidence'] if top_prediction else 0.0,
            'provider': 'openai',
            'model_used': model,
            'analysis': response_text[:200] + '...' if len(response_text) > 200 else response_text,
            'metadata': {
                'model_info': {'name': self.OPENAI_MODELS.get(model, model)},
                'total_predictions': len(predictions),
                'parsing_method': 'text_fallback',
                'raw_response': response_text[:300] + '...' if len(response_text) > 300 else response_text
            }
        }
