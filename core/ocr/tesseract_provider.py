"""
Tesseract OCR provider implementation
"""
import logging
import time
import os
from typing import Dict, Any, List
from PIL import Image
import io

from .base_provider import BaseOCRProvider, OCRRequest, OCRResponse, OCRTextBlock, OCRBoundingBox

logger = logging.getLogger(__name__)


class TesseractProvider(BaseOCRProvider):
    """Tesseract OCR provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tesseract_cmd = config.get("tesseract_cmd")  # Custom tesseract path
        self.default_config = config.get("tesseract_config", "--oem 3 --psm 6")
        
        # Set environment variables for Tesseract
        if not os.environ.get('TESSDATA_PREFIX'):
            # Try common locations
            possible_paths = [
                '/usr/share/tesseract-ocr/5/tessdata/',
                '/usr/share/tesseract-ocr/4.00/tessdata/',
                '/usr/share/tessdata/',
                '/opt/homebrew/share/tessdata/',
                '/usr/local/share/tessdata/'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    os.environ['TESSDATA_PREFIX'] = path
                    logger.info(f"Set TESSDATA_PREFIX to {path}")
                    break
        
    async def extract_text(self, request: OCRRequest) -> OCRResponse:
        """Extract text using Tesseract"""
        start_time = time.time()
        
        try:
            import pytesseract
            
            # Set custom tesseract command if provided
            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            
            # Prepare image data
            image_data = self.prepare_image_data(
                request.image_path,
                request.image_data,
                request.image_base64
            )
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Prepare language string - handle common language codes
            lang_codes = []
            for lang in request.languages:
                # Map common language codes to Tesseract codes
                lang_map = {
                    'en': 'eng',
                    'pt': 'por', 
                    'por': 'por',
                    'es': 'spa',
                    'fr': 'fra',
                    'de': 'deu',
                    'it': 'ita'
                }
                tesseract_lang = lang_map.get(lang.lower(), lang)
                lang_codes.append(tesseract_lang)
            
            lang_string = "+".join(lang_codes) if lang_codes else "eng"
            
            logger.info(f"Using Tesseract with languages: {lang_string}")
            
            # Extract text with bounding boxes if requested
            if request.return_bboxes:
                # Get detailed data including bounding boxes
                data = pytesseract.image_to_data(
                    image, 
                    lang=lang_string,
                    config=self.default_config,
                    output_type=pytesseract.Output.DICT
                )
                
                text_blocks = []
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    if text:  # Only include non-empty text
                        confidence = float(data['conf'][i]) / 100.0  # Convert to 0-1 scale
                        
                        # Skip very low confidence results
                        if confidence < 0.1:
                            continue
                        
                        bbox = OCRBoundingBox(
                            x1=float(data['left'][i]),
                            y1=float(data['top'][i]),
                            x2=float(data['left'][i] + data['width'][i]),
                            y2=float(data['top'][i] + data['height'][i]),
                            width=float(data['width'][i]),
                            height=float(data['height'][i])
                        )
                        
                        text_block = OCRTextBlock(
                            text=text,
                            confidence=confidence,
                            bbox=bbox,
                            language=lang_string
                        )
                        
                        text_blocks.append(text_block)
            else:
                # Simple text extraction
                full_text = pytesseract.image_to_string(
                    image,
                    lang=lang_string,
                    config=self.default_config
                )
                
                # Create single text block
                text_blocks = [OCRTextBlock(
                    text=full_text,
                    confidence=1.0,  # Tesseract doesn't provide overall confidence easily
                    bbox=None,
                    language=lang_string
                )]
            
            # Calculate average confidence
            avg_confidence = sum(block.confidence for block in text_blocks) / len(text_blocks) if text_blocks else 0.0
            
            # Combine all text
            full_text = self.combine_text_blocks(text_blocks)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Tesseract extracted {len(text_blocks)} text blocks with avg confidence {avg_confidence:.2f}")
            
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
            
        except ImportError:
            raise RuntimeError("pytesseract not installed. Install with: pip install pytesseract")
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            raise RuntimeError(f"OCR extraction failed: {e}")
    
    async def health_check(self) -> bool:
        """Check Tesseract health"""
        try:
            import pytesseract
            
            # Set custom tesseract command if provided
            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            
            # Try to get version (simple health check)
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            
            # Try to get available languages
            langs = pytesseract.get_languages(config='')
            logger.info(f"Available Tesseract languages: {langs}")
            
            return True
            
        except Exception as e:
            logger.error(f"Tesseract health check failed: {e}")
            return False
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages for Tesseract"""
        try:
            import pytesseract
            
            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            
            # Get available languages
            langs = pytesseract.get_languages(config='')
            return langs
            
        except Exception as e:
            logger.error(f"Failed to get Tesseract languages: {e}")
            # Return common languages as fallback
            return ["eng", "por", "fra", "deu", "spa", "ita", "rus", "chi_sim", "chi_tra", "jpn", "kor"]
