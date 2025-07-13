"""
OCR Processor Executor

Reusable executor for extracting text from images using OCR.
Uses existing OCR providers (Tesseract, TrOCR) and supports
multiple languages and image formats.
"""

from typing import Dict, Any, List
from pathlib import Path
import logging

from .base_executor import BaseExecutor, ExecutionResult, FlowContext
from ..ocr.ocr_manager import OCRManager

logger = logging.getLogger(__name__)


class OCRProcessor(BaseExecutor):
    """
    Extract text from images using Optical Character Recognition.
    
    Supports multiple OCR providers (Tesseract, TrOCR) and languages.
    Can process various image formats and provide confidence scores.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.ocr_manager = None
    
    def _get_ocr_manager(self) -> OCRManager:
        """Get or create OCR manager."""
        if self.ocr_manager is None:
            self.ocr_manager = OCRManager()
        return self.ocr_manager
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Extract text from image using OCR.
        
        Config parameters:
        - image_path (required): Path to image file
        - provider (optional): OCR provider ('tesseract', 'huggingface')
        - languages (optional): List of languages for OCR (default: ['en'])
        - confidence_threshold (optional): Minimum confidence for text extraction
        - preprocess (optional): Whether to preprocess image (default: True)
        """
        try:
            image_path = config.get('image_path')
            if not image_path:
                return ExecutionResult(
                    success=False,
                    error="image_path is required for OCR processing"
                )
            
            image_path = Path(image_path)
            if not image_path.exists():
                return ExecutionResult(
                    success=False,
                    error=f"Image file not found: {image_path}"
                )
            
            # Get configuration
            provider = config.get('provider', 'tesseract')
            languages = config.get('languages', ['en'])
            confidence_threshold = config.get('confidence_threshold', 0.0)
            preprocess = config.get('preprocess', True)
            
            ocr_manager = self._get_ocr_manager()
            
            # Check if provider is available
            if not ocr_manager.is_provider_available(provider):
                available_providers = ocr_manager.get_available_providers()
                return ExecutionResult(
                    success=False,
                    error=f"OCR provider '{provider}' not available. Available: {available_providers}"
                )
            
            # Check supported formats
            supported_formats = ocr_manager.get_supported_formats(provider)
            file_extension = image_path.suffix.lower()
            if file_extension not in supported_formats:
                return ExecutionResult(
                    success=False,
                    error=f"Image format {file_extension} not supported by {provider}. Supported: {supported_formats}"
                )
            
            self.logger.info(f"Processing image {image_path} with {provider} OCR")
            
            # Perform OCR
            ocr_result = await ocr_manager.extract_text(
                image_path=str(image_path),
                provider=provider,
                languages=languages,
                preprocess=preprocess
            )
            
            # Filter by confidence if threshold is set
            filtered_text = ocr_result.text
            filtered_words = []
            
            if confidence_threshold > 0.0 and hasattr(ocr_result, 'word_confidences'):
                # Filter words by confidence
                words = ocr_result.text.split()
                confidences = ocr_result.word_confidences or []
                
                filtered_words = [
                    word for word, conf in zip(words, confidences)
                    if conf >= confidence_threshold
                ]
                filtered_text = ' '.join(filtered_words)
            
            # Calculate statistics
            total_words = len(ocr_result.text.split()) if ocr_result.text else 0
            filtered_word_count = len(filtered_words) if filtered_words else total_words
            
            return ExecutionResult(
                success=True,
                outputs={
                    "text": filtered_text,
                    "original_text": ocr_result.text,
                    "confidence": ocr_result.confidence,
                    "word_count": filtered_word_count,
                    "total_words": total_words,
                    "provider": provider,
                    "languages": languages,
                    "image_file": image_path.name,
                    "confidence_threshold": confidence_threshold,
                    "preprocessing_applied": preprocess
                },
                metadata={
                    "ocr_provider": provider,
                    "languages_used": languages,
                    "image_processed": str(image_path),
                    "extraction_confidence": ocr_result.confidence,
                    "character_count": len(filtered_text),
                    "words_extracted": filtered_word_count,
                    "confidence_filtering": confidence_threshold > 0.0
                }
            )
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {str(e)}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=f"OCR processing failed: {str(e)}"
            )
    
    def get_required_config_keys(self) -> List[str]:
        """Required configuration keys."""
        return ["image_path"]
    
    def get_optional_config_keys(self) -> List[str]:
        """Optional configuration keys."""
        return [
            "provider",
            "languages",
            "confidence_threshold",
            "preprocess"
        ]
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate executor configuration."""
        super().validate_config(config)
        
        # Validate languages is list
        if 'languages' in config:
            languages = config['languages']
            if not isinstance(languages, list):
                raise ValueError("languages must be a list")
            
            for lang in languages:
                if not isinstance(lang, str):
                    raise ValueError("All language codes must be strings")
        
        # Validate confidence_threshold is float between 0 and 1
        if 'confidence_threshold' in config:
            threshold = config['confidence_threshold']
            if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
                raise ValueError("confidence_threshold must be a number between 0.0 and 1.0")
        
        # Validate preprocess is boolean
        if 'preprocess' in config and not isinstance(config['preprocess'], bool):
            raise ValueError("preprocess must be a boolean")
        
        # Validate provider
        if 'provider' in config:
            provider = config['provider']
            valid_providers = ['tesseract', 'huggingface']
            if provider not in valid_providers:
                raise ValueError(f"provider must be one of: {valid_providers}")
    
    def get_supported_providers(self) -> List[str]:
        """Get list of supported OCR providers."""
        ocr_manager = self._get_ocr_manager()
        return ocr_manager.get_available_providers()
    
    def get_supported_languages(self, provider: str = 'tesseract') -> List[str]:
        """Get list of supported languages for a provider."""
        ocr_manager = self._get_ocr_manager()
        return ocr_manager.get_supported_languages(provider)
    
    def get_info(self) -> Dict[str, Any]:
        """Get executor information."""
        info = super().get_info()
        info.update({
            "capabilities": [
                "Text extraction from images",
                "Multiple OCR provider support",
                "Multi-language text recognition",
                "Confidence-based filtering",
                "Image preprocessing",
                "Format validation"
            ],
            "supported_providers": self.get_supported_providers(),
            "supported_formats": [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"],
            "supported_languages": {
                "tesseract": ["en", "es", "fr", "de", "pt", "it", "ru", "zh", "ja", "ko"],
                "huggingface": ["en", "multilingual"]
            }
        })
        return info
