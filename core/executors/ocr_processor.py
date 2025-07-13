"""
OCR Processor Executor - Extract text from images using OCR
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .base_executor import BaseExecutor, ExecutionResult, FlowContext
from ..ocr.ocr_manager import ocr_manager

logger = logging.getLogger(__name__)


class OCRProcessor(BaseExecutor):
    """
    Extract text from images using Optical Character Recognition.
    
    This executor uses the original working OCR implementation that was
    functioning before the executor-based architecture changes.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.logger = logging.getLogger(f"executor.{self.name.lower()}")
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Execute OCR text extraction from image.
        
        Config parameters:
        - image_path (required): Path to image file
        - provider (optional): OCR provider to use (default: "tesseract")
        - languages (optional): Languages for OCR (default: ["en"])
        - return_bboxes (optional): Return bounding boxes (default: True)
        - return_confidence (optional): Return confidence scores (default: True)
        - confidence_threshold (optional): Minimum confidence threshold (default: 0.0)
        """
        try:
            # Get image path
            image_path = config.get('image_path')
            if not image_path:
                return ExecutionResult(
                    success=False,
                    error="image_path is required"
                )
            
            # Validate image exists
            image_path = Path(image_path)
            if not image_path.exists():
                return ExecutionResult(
                    success=False,
                    error=f"Image file not found: {image_path}"
                )
            
            # Get configuration - using the original working parameters
            provider = config.get('provider', 'tesseract')
            languages = config.get('languages', ['en'])
            return_bboxes = config.get('return_bboxes', True)
            return_confidence = config.get('return_confidence', True)
            confidence_threshold = config.get('confidence_threshold', 0.0)
            
            # Ensure languages is a list
            if isinstance(languages, str):
                languages = [languages]
            
            self.logger.info(f"Processing image {image_path} with {provider} OCR, languages: {languages}")
            
            # Perform OCR using the original working method
            response = await ocr_manager.extract_text(
                image_path=str(image_path),
                provider=provider,
                languages=languages,
                return_bboxes=return_bboxes,
                return_confidence=return_confidence
            )
            
            # Filter by confidence if threshold is set
            filtered_text = response.full_text
            if confidence_threshold > 0.0 and response.text_blocks:
                # Filter text blocks by confidence
                filtered_blocks = [
                    block for block in response.text_blocks
                    if block.confidence >= confidence_threshold
                ]
                filtered_text = ' '.join([block.text for block in filtered_blocks])
            
            # Post-process the text to make it more coherent for LLM analysis
            cleaned_text = self._clean_ocr_text(filtered_text)
            
            # Calculate statistics
            total_words = len(response.full_text.split()) if response.full_text else 0
            filtered_word_count = len(cleaned_text.split()) if cleaned_text else 0
            
            self.logger.info(f"OCR completed: {len(response.text_blocks)} blocks, avg confidence: {response.confidence_avg:.2f}")
            
            return ExecutionResult(
                success=True,
                outputs={
                    "text": cleaned_text,
                    "original_text": response.full_text,
                    "raw_ocr_text": filtered_text,
                    "confidence": response.confidence_avg,
                    "word_count": filtered_word_count,
                    "total_words": total_words,
                    "provider": provider,
                    "languages": languages,
                    "image_file": image_path.name,
                    "confidence_threshold": confidence_threshold,
                    "text_blocks_count": len(response.text_blocks)
                },
                metadata={
                    "ocr_provider": provider,
                    "languages_used": languages,
                    "image_processed": str(image_path),
                    "extraction_confidence": response.confidence_avg,
                    "character_count": len(filtered_text),
                    "words_extracted": filtered_word_count,
                    "total_text_blocks": len(response.text_blocks),
                    "processing_time": response.processing_time
                }
            )
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {str(e)}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=f"OCR processing failed: {str(e)}"
            )
    
    def _clean_ocr_text(self, text: str) -> str:
        """
        Clean and post-process OCR text to make it more coherent for LLM analysis.
        
        This method provides generic text cleaning:
        1. Removes excessive line breaks and whitespace
        2. Filters out likely OCR artifacts (single characters, symbols)
        3. Joins meaningful text elements
        4. Preserves important structure while improving readability
        """
        if not text:
            return ""
        
        import re
        
        # Split into lines and clean each line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Filter out likely OCR artifacts while preserving meaningful content
        meaningful_lines = []
        for line in lines:
            # Keep lines that are:
            # - More than 2 characters (likely real words/phrases)
            # - Numbers (years, quantities, etc.)
            # - Important single characters with context (like "D." for titles)
            # - Copyright and other important symbols
            if (len(line) > 2 or 
                line.isdigit() or 
                re.match(r'^[A-Z]\.$', line) or  # Single letter with period (titles, initials)
                line in ['©', '®', '™', '&']):  # Important symbols
                meaningful_lines.append(line)
        
        # Join meaningful lines with spaces, preserving some structure
        cleaned_text = ' '.join(meaningful_lines)
        
        # Basic cleanup
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Multiple spaces to single
        cleaned_text = cleaned_text.strip()
        
        # If cleaning removed too much content, fall back to basic cleanup
        if not cleaned_text or len(cleaned_text) < len(text) * 0.3:
            # Basic cleanup only - just remove excessive line breaks
            basic_cleaned = re.sub(r'\n+', ' ', text)  # Replace line breaks with spaces
            basic_cleaned = re.sub(r'\s+', ' ', basic_cleaned)  # Multiple spaces to single
            return basic_cleaned.strip()
        
        return cleaned_text
