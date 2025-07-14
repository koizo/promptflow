"""
Test helper utilities for executor testing.

Provides common utilities, mock factories, and helper functions
for testing executors and flow components.
"""

import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock

from core.executors.base_executor import FlowContext, ExecutionResult


class MockFactory:
    """Factory for creating mock objects for testing."""
    
    @staticmethod
    def create_mock_llm_response(content: str = "Mock LLM response", 
                                metadata: Dict[str, Any] = None) -> Mock:
        """Create a mock LLM response."""
        mock_response = Mock()
        mock_response.content = content
        mock_response.metadata = metadata or {"model": "test-model", "tokens": 10}
        return mock_response
    
    @staticmethod
    def create_mock_file_upload(filename: str = "test_file.txt", 
                               content: bytes = b"test content",
                               content_type: str = "text/plain") -> Mock:
        """Create a mock file upload object."""
        mock_file = Mock()
        mock_file.filename = filename
        mock_file.content_type = content_type
        mock_file.size = len(content)
        mock_file.read = Mock(return_value=content)
        return mock_file
    
    @staticmethod
    def create_mock_ocr_result(text: str = "Extracted text", 
                              confidence: float = 0.9) -> Dict[str, Any]:
        """Create a mock OCR result."""
        return {
            "text": text,
            "confidence": confidence,
            "word_count": len(text.split()),
            "bounding_boxes": [{"x": 0, "y": 0, "width": 100, "height": 20}]
        }
    
    @staticmethod
    def create_mock_vision_result(predictions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a mock vision classification result."""
        if predictions is None:
            predictions = [
                {"label": "document", "confidence": 0.95},
                {"label": "text", "confidence": 0.87}
            ]
        
        return {
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None,
            "processing_time": 1.23
        }


class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_sample_image_bytes() -> bytes:
        """Create minimal PNG image bytes for testing."""
        # Minimal 1x1 PNG image
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
    
    @staticmethod
    def create_sample_pdf_bytes() -> bytes:
        """Create minimal PDF bytes for testing."""
        return b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n174\n%%EOF'
    
    @staticmethod
    def create_sample_audio_bytes() -> bytes:
        """Create minimal WAV audio bytes for testing."""
        # Minimal WAV header for a 1-second silent audio
        return b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    
    @staticmethod
    def create_sample_text_content(length: int = 100) -> str:
        """Create sample text content of specified length."""
        base_text = "This is sample text content for testing purposes. "
        repetitions = (length // len(base_text)) + 1
        return (base_text * repetitions)[:length]


class FlowTestHelper:
    """Helper for testing flow execution scenarios."""
    
    @staticmethod
    def create_context_with_results(step_results: Dict[str, Dict[str, Any]]) -> FlowContext:
        """Create FlowContext with pre-populated step results."""
        context = FlowContext(flow_name="test_flow", inputs={}, flow_id=f"test-flow-{uuid.uuid4().hex[:8]}")
        
        for step_name, outputs in step_results.items():
            result = ExecutionResult(success=True, outputs=outputs)
            context.add_step_result(step_name, result)
        
        return context
    
    @staticmethod
    def assert_execution_result(result: ExecutionResult, 
                              success: bool = True,
                              required_outputs: List[str] = None,
                              error_contains: str = None):
        """Assert ExecutionResult has expected properties."""
        assert isinstance(result, ExecutionResult)
        assert result.success == success
        
        if success:
            assert result.error is None
            assert isinstance(result.outputs, dict)
            
            if required_outputs:
                for output_key in required_outputs:
                    assert output_key in result.outputs, f"Missing required output: {output_key}"
        else:
            assert result.error is not None
            assert isinstance(result.error, str)
            
            if error_contains:
                assert error_contains.lower() in result.error.lower()
    
    @staticmethod
    def create_temp_file(content: bytes, suffix: str = ".tmp") -> Path:
        """Create a temporary file with given content."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            return Path(temp_file.name)


class ConfigFactory:
    """Factory for creating test configurations."""
    
    @staticmethod
    def create_file_handler_config(filename: str = "test.txt",
                                  content: bytes = b"test content",
                                  **kwargs) -> Dict[str, Any]:
        """Create FileHandler configuration."""
        config = {
            "file_content": content,
            "filename": filename,
            "save_temp": True,
            "validate_format": True
        }
        config.update(kwargs)
        return config
    
    @staticmethod
    def create_llm_analyzer_config(text: str = "Sample text for analysis",
                                  prompt: str = "Analyze this text",
                                  **kwargs) -> Dict[str, Any]:
        """Create LLMAnalyzer configuration."""
        config = {
            "text": text,
            "prompt": prompt,
            "model": "test-model",
            "temperature": 0.7
        }
        config.update(kwargs)
        return config
    
    @staticmethod
    def create_response_formatter_config(data: Dict[str, Any] = None,
                                       template: str = "structured",
                                       **kwargs) -> Dict[str, Any]:
        """Create ResponseFormatter configuration."""
        if data is None:
            data = {"result": "test_result", "confidence": 0.9}
        
        config = {
            "data": data,
            "template": template,
            "format": "json"
        }
        config.update(kwargs)
        return config
    
    @staticmethod
    def create_vision_classifier_config(file_path: str = "test_image.jpg",
                                      provider: str = "huggingface",
                                      **kwargs) -> Dict[str, Any]:
        """Create VisionClassifier configuration."""
        config = {
            "file": file_path,
            "provider": provider,
            "model": "google/vit-base-patch16-224",
            "top_k": 5
        }
        config.update(kwargs)
        return config


# Convenience functions for common test patterns
def create_mock_context(flow_id: str = None) -> FlowContext:
    """Create a mock FlowContext with optional flow_id."""
    if flow_id is None:
        flow_id = f"test-flow-{uuid.uuid4().hex[:8]}"
    return FlowContext(flow_name="test_flow", inputs={}, flow_id=flow_id)


def create_successful_result(outputs: Dict[str, Any] = None,
                           metadata: Dict[str, Any] = None) -> ExecutionResult:
    """Create a successful ExecutionResult."""
    return ExecutionResult(
        success=True,
        outputs=outputs or {"result": "success"},
        metadata=metadata or {"test": True}
    )


def create_error_result(error: str = "Test error",
                       metadata: Dict[str, Any] = None) -> ExecutionResult:
    """Create an error ExecutionResult."""
    return ExecutionResult(
        success=False,
        error=error,
        metadata=metadata or {"test": True}
    )
