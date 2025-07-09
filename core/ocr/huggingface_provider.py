"""
Hugging Face OCR provider implementation using TrOCR
"""
import logging
import time
import numpy as np
from typing import Dict, Any, List
from PIL import Image
import io

from .base_provider import BaseOCRProvider, OCRRequest, OCRResponse, OCRTextBlock, OCRBoundingBox

logger = logging.getLogger(__name__)


class HuggingFaceOCRProvider(BaseOCRProvider):
    """Hugging Face OCR provider using TrOCR and other vision-to-text models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "microsoft/trocr-base-printed")
        self.device = config.get("device", "cpu")
        self.use_gpu = config.get("use_gpu", False)
        
        # Lazy loading of models
        self._processor = None
        self._model = None
    
    def _get_model_and_processor(self):
        """Lazy load Hugging Face TrOCR model and processor"""
        if self._processor is None or self._model is None:
            try:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                
                logger.info(f"Loading HF OCR model: {self.model_name}")
                
                # Load processor and model
                self._processor = TrOCRProcessor.from_pretrained(self.model_name)
                self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                
                # Move to specified device
                if self.use_gpu and self.device != "cpu":
                    import torch
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self._model = self._model.to(device)
                    logger.info(f"Model moved to {device}")
                
                logger.info(f"HF OCR model loaded successfully")
                
            except ImportError:
                raise RuntimeError("Transformers not installed. Install with: pip install transformers")
            except Exception as e:
                logger.error(f"Failed to load HF OCR model: {e}")
                raise RuntimeError(f"HF OCR model loading failed: {e}")
        
        return self._processor, self._model
    
    async def extract_text(self, request: OCRRequest) -> OCRResponse:
        """Extract text using Hugging Face TrOCR"""
        start_time = time.time()
        
        try:
            # Get model and processor
            processor, model = self._get_model_and_processor()
            
            # Prepare image data
            image_data = self.prepare_image_data(
                request.image_path,
                request.image_data,
                request.image_base64
            )
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image with TrOCR
            pixel_values = processor(image, return_tensors="pt").pixel_values
            
            # Move to device if using GPU
            if self.use_gpu and self.device != "cpu":
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                pixel_values = pixel_values.to(device)
            
            # Generate text
            generated_ids = model.generate(pixel_values, max_new_tokens=256)
            
            # Decode generated text
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Create text block
            text_block = OCRTextBlock(
                text=generated_text,
                confidence=1.0,  # TrOCR doesn't provide confidence scores
                bbox=None,  # TrOCR doesn't provide bounding boxes by default
                language="auto"
            )
            
            text_blocks = [text_block] if generated_text.strip() else []
            
            processing_time = time.time() - start_time
            
            return OCRResponse(
                text_blocks=text_blocks,
                full_text=generated_text,
                confidence_avg=1.0,
                processing_time=processing_time,
                image_info={
                    "width": image.width,
                    "height": image.height,
                    "format": image.format,
                    "mode": image.mode,
                    "model_used": self.model_name
                }
            )
            
        except Exception as e:
            logger.error(f"HF OCR extraction failed: {e}")
            raise RuntimeError(f"OCR extraction failed: {e}")
    
    async def health_check(self) -> bool:
        """Check Hugging Face OCR health"""
        try:
            # Try to load the model and processor
            self._get_model_and_processor()
            return True
        except Exception as e:
            logger.error(f"HF OCR health check failed: {e}")
            return False
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages for TrOCR models"""
        # TrOCR supports multiple languages depending on the model
        return [
            "en", "fr", "de", "es", "it", "pt", "nl", "ru", "ja", "ko", "zh", "ar",
            "hi", "th", "vi", "id", "ms", "tl", "sw", "tr", "pl", "cs", "sk", "hu",
            "ro", "bg", "hr", "sl", "et", "lv", "lt", "mt", "ga", "cy", "eu", "ca"
        ]
    
    def get_available_models(self) -> List[str]:
        """Get list of available HF OCR models"""
        return [
            "microsoft/trocr-base-printed",
            "microsoft/trocr-large-printed", 
            "microsoft/trocr-base-handwritten",
            "microsoft/trocr-large-handwritten",
            "microsoft/trocr-small-printed"
        ]
