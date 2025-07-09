"""
OCR step executor for flows
"""
import logging
from typing import Dict, Any, Optional, List
from .ocr_manager import ocr_manager

logger = logging.getLogger(__name__)


class OCRExecutor:
    """Executor for OCR steps in flows"""
    
    @staticmethod
    async def execute_ocr_step(
        step_config: Dict[str, Any],
        context_variables: Dict[str, Any],
        file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute an OCR step
        
        Args:
            step_config: Step configuration from DSL
            context_variables: Variables available for processing
            file_path: Optional file path to process
            
        Returns:
            Dictionary containing the OCR results and metadata
        """
        try:
            # Extract step configuration
            provider = step_config.get("provider", "huggingface")
            model = step_config.get("model", "microsoft/trocr-base-printed")
            languages = step_config.get("languages", ["en"])
            return_bboxes = step_config.get("return_bboxes", True)
            return_confidence = step_config.get("return_confidence", True)
            
            # Get input file
            input_key = step_config.get("input")
            if input_key and input_key in context_variables:
                file_input = context_variables[input_key]
            elif file_path:
                file_input = file_path
            else:
                raise ValueError("No input file provided for OCR step")
            
            logger.info(f"Executing OCR step with provider: {provider}")
            
            # Determine input type and prepare accordingly
            image_path = None
            image_data = None
            image_base64 = None
            
            if isinstance(file_input, str):
                # Assume it's a file path
                image_path = file_input
            elif isinstance(file_input, bytes):
                # Raw image data
                image_data = file_input
            elif isinstance(file_input, dict) and "base64" in file_input:
                # Base64 encoded image
                image_base64 = file_input["base64"]
            else:
                raise ValueError(f"Unsupported input type for OCR: {type(file_input)}")
            
            # Perform OCR
            response = await ocr_manager.extract_text(
                image_path=image_path,
                image_data=image_data,
                image_base64=image_base64,
                provider=provider,
                languages=languages,
                return_bboxes=return_bboxes,
                return_confidence=return_confidence
            )
            
            # Format results
            result = {
                "full_text": response.full_text,
                "confidence_avg": response.confidence_avg,
                "processing_time": response.processing_time,
                "provider": provider,
                "text_blocks_count": len(response.text_blocks),
                "image_info": response.image_info
            }
            
            # Include detailed text blocks if requested
            if step_config.get("include_blocks", False):
                result["text_blocks"] = [
                    {
                        "text": block.text,
                        "confidence": block.confidence,
                        "bbox": block.bbox.model_dump() if block.bbox else None,
                        "language": block.language
                    }
                    for block in response.text_blocks
                ]
            
            logger.info(f"OCR completed: {len(response.text_blocks)} blocks, avg confidence: {response.confidence_avg:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"OCR step execution failed: {e}")
            raise RuntimeError(f"OCR step failed: {e}")
    
    @staticmethod
    async def health_check(provider: str = "huggingface") -> Dict[str, Any]:
        """Check OCR provider health"""
        try:
            is_healthy = await ocr_manager.health_check(provider)
            supported_languages = ocr_manager.get_supported_languages(provider)
            
            return {
                "provider": provider,
                "healthy": is_healthy,
                "supported_languages": supported_languages[:10],  # Limit for readability
                "total_languages": len(supported_languages)
            }
            
        except Exception as e:
            logger.error(f"OCR health check failed: {e}")
            return {
                "provider": provider,
                "healthy": False,
                "error": str(e)
            }
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get information about available OCR providers"""
        providers = ocr_manager.get_providers()
        provider_info = {}
        
        for provider in providers:
            try:
                languages = ocr_manager.get_supported_languages(provider)
                provider_info[provider] = {
                    "supported_languages": languages[:10],  # Show first 10
                    "total_languages": len(languages),
                    "description": {
                        "huggingface": "TrOCR - Transformer-based OCR model from Microsoft with high accuracy",
                        "tesseract": "Tesseract - Traditional OCR engine with wide language support"
                    }.get(provider, "OCR Provider")
                }
            except Exception as e:
                provider_info[provider] = {
                    "error": str(e)
                }
        
        return {
            "providers": provider_info,
            "default_provider": "huggingface"
        }
