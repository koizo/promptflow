"""
OCR integration module
"""
from .base_provider import (
    BaseOCRProvider, 
    OCRRequest, 
    OCRResponse, 
    OCRTextBlock, 
    OCRBoundingBox
)
from .huggingface_provider import HuggingFaceOCRProvider
from .tesseract_provider import TesseractProvider
from .ocr_manager import ocr_manager, OCRManager
from .ocr_executor import OCRExecutor

__all__ = [
    "BaseOCRProvider",
    "OCRRequest",
    "OCRResponse", 
    "OCRTextBlock",
    "OCRBoundingBox",
    "HuggingFaceOCRProvider",
    "TesseractProvider",
    "OCRManager",
    "ocr_manager",
    "OCRExecutor"
]
