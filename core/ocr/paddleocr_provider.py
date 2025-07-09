"""
PaddleOCR provider implementation
"""
import logging
import time
import numpy as np
from typing import Dict, Any, List
from PIL import Image
import io

from .base_provider import BaseOCRProvider, OCRRequest, OCRResponse, OCRTextBlock, OCRBoundingBox

logger = logging.getLogger(__name__)


class PaddleOCRProvider(BaseOCRProvider):
    """PaddleOCR provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.use_angle_cls = config.get("use_angle_cls", True)
        self.use_gpu = config.get("use_gpu", False)
        self.lang = config.get("default_language", "en")
        self.det_model_dir = config.get("det_model_dir")
        self.rec_model_dir = config.get("rec_model_dir")
        self.cls_model_dir = config.get("cls_model_dir")
        
        # Lazy loading of PaddleOCR
        self._ocr_engine = None
    
    def _get_ocr_engine(self):
        """Lazy load PaddleOCR engine"""
        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR
                
                # Initialize PaddleOCR
                init_kwargs = {
                    "use_angle_cls": self.use_angle_cls,
                    "lang": self.lang,
                    "use_gpu": self.use_gpu,
                    "show_log": False
                }
                
                # Add custom model paths if provided
                if self.det_model_dir:
                    init_kwargs["det_model_dir"] = self.det_model_dir
                if self.rec_model_dir:
                    init_kwargs["rec_model_dir"] = self.rec_model_dir
                if self.cls_model_dir:
                    init_kwargs["cls_model_dir"] = self.cls_model_dir
                
                self._ocr_engine = PaddleOCR(**init_kwargs)
                logger.info("PaddleOCR engine initialized successfully")
                
            except ImportError:
                raise RuntimeError("PaddleOCR not installed. Install with: pip install paddleocr")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                raise RuntimeError(f"PaddleOCR initialization failed: {e}")
        
        return self._ocr_engine
    
    async def extract_text(self, request: OCRRequest) -> OCRResponse:
        """Extract text using PaddleOCR"""
        start_time = time.time()
        
        try:
            # Get OCR engine
            ocr_engine = self._get_ocr_engine()
            
            # Prepare image data
            image_data = self.prepare_image_data(
                request.image_path,
                request.image_data,
                request.image_base64
            )
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            # Perform OCR
            results = ocr_engine.ocr(image_array, cls=self.use_angle_cls)
            
            # Process results
            text_blocks = []
            total_confidence = 0
            valid_blocks = 0
            
            if results and results[0]:  # PaddleOCR returns list of results
                for line in results[0]:
                    if line:
                        bbox_coords, (text, confidence) = line
                        
                        if text.strip():  # Only include non-empty text
                            # Convert bbox coordinates
                            bbox = OCRBoundingBox(
                                x1=float(min(coord[0] for coord in bbox_coords)),
                                y1=float(min(coord[1] for coord in bbox_coords)),
                                x2=float(max(coord[0] for coord in bbox_coords)),
                                y2=float(max(coord[1] for coord in bbox_coords)),
                                width=float(max(coord[0] for coord in bbox_coords) - min(coord[0] for coord in bbox_coords)),
                                height=float(max(coord[1] for coord in bbox_coords) - min(coord[1] for coord in bbox_coords))
                            )
                            
                            text_block = OCRTextBlock(
                                text=text,
                                confidence=float(confidence),
                                bbox=bbox if request.return_bboxes else None,
                                language=self.lang
                            )
                            
                            text_blocks.append(text_block)
                            total_confidence += confidence
                            valid_blocks += 1
            
            # Calculate average confidence
            avg_confidence = total_confidence / valid_blocks if valid_blocks > 0 else 0.0
            
            # Combine all text
            full_text = self.combine_text_blocks(text_blocks)
            
            processing_time = time.time() - start_time
            
            return OCRResponse(
                text_blocks=text_blocks,
                full_text=full_text,
                confidence_avg=avg_confidence,
                processing_time=processing_time,
                image_info={
                    "width": image.width,
                    "height": image.height,
                    "format": image.format,
                    "mode": image.mode
                }
            )
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            raise RuntimeError(f"OCR extraction failed: {e}")
    
    async def health_check(self) -> bool:
        """Check PaddleOCR health"""
        try:
            # Try to initialize the engine
            self._get_ocr_engine()
            return True
        except Exception as e:
            logger.error(f"PaddleOCR health check failed: {e}")
            return False
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages for PaddleOCR"""
        # PaddleOCR supported languages
        return [
            "en", "ch", "ta", "te", "ka", "ja", "ko", "hi", "ar", "cyrl", "devanagari",
            "fr", "de", "it", "es", "pt", "ru", "uk", "be", "bg", "ur", "fa", "uz", "az",
            "mn", "ne", "si", "my", "km", "th", "lo", "vi", "ms", "id", "tl", "hm", "chr"
        ]
