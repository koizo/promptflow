"""
Base OCR provider interface
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
import base64
from pathlib import Path


class OCRBoundingBox(BaseModel):
    """OCR bounding box coordinates"""
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float


class OCRTextBlock(BaseModel):
    """OCR detected text block"""
    text: str
    confidence: float
    bbox: Optional[OCRBoundingBox] = None
    language: Optional[str] = None


class OCRRequest(BaseModel):
    """OCR request structure"""
    image_path: Optional[str] = None
    image_data: Optional[bytes] = None
    image_base64: Optional[str] = None
    languages: List[str] = ["en"]  # Default to English
    detect_orientation: bool = True
    return_confidence: bool = True
    return_bboxes: bool = True


class OCRResponse(BaseModel):
    """OCR response structure"""
    text_blocks: List[OCRTextBlock]
    full_text: str
    confidence_avg: float
    processing_time: Optional[float] = None
    image_info: Optional[Dict[str, Any]] = None


class BaseOCRProvider(ABC):
    """Abstract base class for OCR providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_formats = config.get("supported_formats", [".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".bmp"])
        self.max_image_size = config.get("max_image_size", "10MB")
    
    @abstractmethod
    async def extract_text(self, request: OCRRequest) -> OCRResponse:
        """Extract text from image"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the OCR provider is healthy"""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        pass
    
    def validate_image_format(self, file_path: str) -> bool:
        """Validate if image format is supported"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_formats
    
    def prepare_image_data(self, image_path: Optional[str] = None, 
                          image_data: Optional[bytes] = None,
                          image_base64: Optional[str] = None) -> bytes:
        """Prepare image data from various input formats"""
        if image_data:
            return image_data
        elif image_base64:
            return base64.b64decode(image_base64)
        elif image_path:
            with open(image_path, 'rb') as f:
                return f.read()
        else:
            raise ValueError("No image data provided")
    
    def combine_text_blocks(self, text_blocks: List[OCRTextBlock]) -> str:
        """Combine text blocks into full text"""
        return "\n".join([block.text for block in text_blocks if block.text.strip()])
