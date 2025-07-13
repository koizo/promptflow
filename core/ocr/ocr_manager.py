"""
OCR Manager for handling different OCR providers
"""
import logging
from typing import Dict, Any, Optional, List
from .base_provider import BaseOCRProvider, OCRRequest, OCRResponse
from .huggingface_provider import HuggingFaceOCRProvider
from .tesseract_provider import TesseractProvider

logger = logging.getLogger(__name__)


class OCRManager:
    """Manager for OCR providers"""
    
    def __init__(self):
        self.providers: Dict[str, BaseOCRProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize OCR providers based on configuration"""
        
        # Initialize Hugging Face OCR provider (TrOCR)
        huggingface_config = {
            "model_name": "microsoft/trocr-base-printed",
            "device": "cpu",
            "use_gpu": False,
            "supported_formats": [".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".bmp"],
            "max_image_size": "10MB"
        }
        self.providers["huggingface"] = HuggingFaceOCRProvider(huggingface_config)
        
        # Initialize Tesseract provider as fallback
        tesseract_config = {
            "tesseract_cmd": None,  # Use system default
            "tesseract_config": "--oem 3 --psm 6",
            "supported_formats": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
            "max_image_size": "10MB"
        }
        self.providers["tesseract"] = TesseractProvider(tesseract_config)
        
        logger.info(f"Initialized {len(self.providers)} OCR providers")
    
    def get_provider(self, provider_name: str = "huggingface") -> BaseOCRProvider:
        """Get OCR provider by name"""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown OCR provider: {provider_name}")
        return self.providers[provider_name]
    
    async def extract_text(
        self,
        image_path: Optional[str] = None,
        image_data: Optional[bytes] = None,
        image_base64: Optional[str] = None,
        provider: str = "huggingface",
        languages: List[str] = ["en"],
        return_bboxes: bool = True,
        return_confidence: bool = True
    ) -> OCRResponse:
        """Extract text using specified provider"""
        try:
            ocr_provider = self.get_provider(provider)
            
            # Create request
            request = OCRRequest(
                image_path=image_path,
                image_data=image_data,
                image_base64=image_base64,
                languages=languages,
                return_bboxes=return_bboxes,
                return_confidence=return_confidence
            )
            
            # Extract text
            response = await ocr_provider.extract_text(request)
            logger.info(f"OCR completed using {provider}: {len(response.text_blocks)} text blocks found")
            
            return response
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise
    
    async def health_check(self, provider: str = "huggingface") -> bool:
        """Check health of specified provider"""
        try:
            ocr_provider = self.get_provider(provider)
            return await ocr_provider.health_check()
        except Exception as e:
            logger.error(f"Health check failed for {provider}: {e}")
            return False
    
    def get_supported_languages(self, provider: str = "huggingface") -> List[str]:
        """Get supported languages for specified provider"""
        try:
            ocr_provider = self.get_provider(provider)
            return ocr_provider.get_supported_languages()
        except Exception as e:
            logger.error(f"Failed to get languages for {provider}: {e}")
            return ["en"]
    
    def get_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    def get_supported_formats(self, provider_name: str = "tesseract") -> List[str]:
        """Get supported image formats for a provider"""
        try:
            provider = self.get_provider(provider_name)
            # Most OCR providers support these common image formats
            return [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif"]
        except Exception as e:
            logger.error(f"Failed to get supported formats: {e}")
            return [".jpg", ".jpeg", ".png"]
    
    def supports_format(self, file_extension: str, provider_name: str = "tesseract") -> bool:
        """Check if a file format is supported by the provider"""
        try:
            supported_formats = self.get_supported_formats(provider_name)
            return file_extension.lower() in [fmt.lower() for fmt in supported_formats]
        except Exception as e:
            logger.error(f"Format support check failed: {e}")
            return file_extension.lower() in [".jpg", ".jpeg", ".png"]
    
    def validate_image_format(self, file_path: str, provider: str = "huggingface") -> bool:
        """Validate image format for specified provider"""
        try:
            ocr_provider = self.get_provider(provider)
            return ocr_provider.validate_image_format(file_path)
        except Exception as e:
            logger.error(f"Format validation failed: {e}")
            return False


# Global OCR manager instance
ocr_manager = OCRManager()
