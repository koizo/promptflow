"""
Unit tests for OCRProcessor executor.

Tests OCR text extraction from images, text cleaning,
confidence filtering, and provider integration.
"""

import pytest
import uuid
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from core.executors.ocr_processor import OCRProcessor
from core.executors.base_executor import FlowContext, ExecutionResult


class MockOCRResponse:
    """Mock OCR response for testing."""
    
    def __init__(self, full_text="", text_blocks=None, confidence_avg=0.85, processing_time=1.5):
        self.full_text = full_text
        self.text_blocks = text_blocks or []
        self.confidence_avg = confidence_avg
        self.processing_time = processing_time


class MockTextBlock:
    """Mock text block for testing."""
    
    def __init__(self, text="", confidence=0.9):
        self.text = text
        self.confidence = confidence


class TestOCRProcessor:
    """Test suite for OCRProcessor executor."""
    
    @pytest.fixture
    def ocr_processor(self):
        """Create OCRProcessor instance for testing."""
        return OCRProcessor()
    
    @pytest.fixture
    def mock_context(self):
        """Create mock FlowContext for testing."""
        return FlowContext(
            flow_name="test_flow",
            inputs={"test_param": "test_value"},
            flow_id=f"test-flow-{uuid.uuid4().hex[:8]}"
        )
    
    @pytest.fixture
    def sample_image_file(self):
        """Create a temporary image file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            # Write minimal PNG header (simplified for testing)
            temp_file.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde')
            temp_file.flush()
            yield temp_file.name
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except:
            pass
    
    @pytest.fixture
    def sample_config(self, sample_image_file):
        """Sample configuration for OCR processing."""
        return {
            "image_path": sample_image_file,
            "provider": "tesseract",
            "languages": ["en"],
            "return_bboxes": True,
            "return_confidence": True,
            "confidence_threshold": 0.5
        }
    
    @pytest.fixture
    def mock_ocr_response(self):
        """Mock OCR response with sample text."""
        text_blocks = [
            MockTextBlock("This is a sample document", 0.95),
            MockTextBlock("with multiple text blocks", 0.88),
            MockTextBlock("and varying confidence scores", 0.72),
            MockTextBlock("x", 0.3)  # Low confidence artifact
        ]
        
        full_text = "This is a sample document\nwith multiple text blocks\nand varying confidence scores\nx"
        
        return MockOCRResponse(
            full_text=full_text,
            text_blocks=text_blocks,
            confidence_avg=0.76,
            processing_time=2.1
        )
    
    @pytest.mark.asyncio
    async def test_missing_image_path(self, ocr_processor, mock_context):
        """Test error when image_path is missing."""
        config = {"provider": "tesseract"}
        
        result = await ocr_processor.execute(mock_context, config)
        
        assert result.success is False
        assert "image_path is required" in result.error
    
    @pytest.mark.asyncio
    async def test_nonexistent_image_file(self, ocr_processor, mock_context):
        """Test error when image file doesn't exist."""
        config = {
            "image_path": "/nonexistent/image.png",
            "provider": "tesseract"
        }
        
        result = await ocr_processor.execute(mock_context, config)
        
        assert result.success is False
        assert "Image file not found" in result.error
    
    @pytest.mark.asyncio
    async def test_successful_ocr_processing(self, ocr_processor, mock_context, 
                                           sample_config, mock_ocr_response):
        """Test successful OCR text extraction."""
        with patch('core.executors.ocr_processor.ocr_manager.extract_text', 
                   return_value=mock_ocr_response) as mock_extract:
            
            result = await ocr_processor.execute(mock_context, sample_config)
            
            assert result.success is True
            assert "text" in result.outputs
            assert "original_text" in result.outputs
            assert "confidence" in result.outputs
            assert "word_count" in result.outputs
            assert "provider" in result.outputs
            assert "languages" in result.outputs
            assert "image_file" in result.outputs
            
            # Verify OCR manager was called with correct parameters
            mock_extract.assert_called_once()
            call_args = mock_extract.call_args[1]
            assert call_args["provider"] == "tesseract"
            assert call_args["languages"] == ["en"]
            assert call_args["return_bboxes"] is True
            assert call_args["return_confidence"] is True
    
    @pytest.mark.asyncio
    async def test_default_configuration(self, ocr_processor, mock_context, 
                                       sample_image_file, mock_ocr_response):
        """Test OCR processing with default configuration."""
        config = {"image_path": sample_image_file}
        
        with patch('core.executors.ocr_processor.ocr_manager.extract_text', 
                   return_value=mock_ocr_response) as mock_extract:
            
            result = await ocr_processor.execute(mock_context, config)
            
            assert result.success is True
            assert result.outputs["provider"] == "tesseract"  # Default provider
            assert result.outputs["languages"] == ["en"]  # Default language
            
            # Verify default parameters were used
            call_args = mock_extract.call_args[1]
            assert call_args["provider"] == "tesseract"
            assert call_args["languages"] == ["en"]
    
    @pytest.mark.asyncio
    async def test_multiple_languages(self, ocr_processor, mock_context, 
                                    sample_image_file, mock_ocr_response):
        """Test OCR processing with multiple languages."""
        config = {
            "image_path": sample_image_file,
            "languages": ["en", "es", "fr"]
        }
        
        with patch('core.executors.ocr_processor.ocr_manager.extract_text', 
                   return_value=mock_ocr_response) as mock_extract:
            
            result = await ocr_processor.execute(mock_context, config)
            
            assert result.success is True
            assert result.outputs["languages"] == ["en", "es", "fr"]
            
            call_args = mock_extract.call_args[1]
            assert call_args["languages"] == ["en", "es", "fr"]
    
    @pytest.mark.asyncio
    async def test_single_language_string(self, ocr_processor, mock_context, 
                                        sample_image_file, mock_ocr_response):
        """Test OCR processing with single language as string."""
        config = {
            "image_path": sample_image_file,
            "languages": "de"  # Single language as string
        }
        
        with patch('core.executors.ocr_processor.ocr_manager.extract_text', 
                   return_value=mock_ocr_response) as mock_extract:
            
            result = await ocr_processor.execute(mock_context, config)
            
            assert result.success is True
            assert result.outputs["languages"] == ["de"]  # Should be converted to list
            
            call_args = mock_extract.call_args[1]
            assert call_args["languages"] == ["de"]
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering(self, ocr_processor, mock_context, 
                                                sample_image_file):
        """Test confidence threshold filtering."""
        # Create mock response with varying confidence levels
        text_blocks = [
            MockTextBlock("High confidence text", 0.95),
            MockTextBlock("Medium confidence text", 0.75),
            MockTextBlock("Low confidence text", 0.45),  # Below threshold
            MockTextBlock("Very low confidence", 0.25)   # Below threshold
        ]
        
        mock_response = MockOCRResponse(
            full_text="High confidence text\nMedium confidence text\nLow confidence text\nVery low confidence",
            text_blocks=text_blocks,
            confidence_avg=0.60
        )
        
        config = {
            "image_path": sample_image_file,
            "confidence_threshold": 0.7
        }
        
        with patch('core.executors.ocr_processor.ocr_manager.extract_text', 
                   return_value=mock_response):
            
            result = await ocr_processor.execute(mock_context, config)
            
            assert result.success is True
            
            # Should only include text blocks above threshold
            filtered_text = result.outputs["raw_ocr_text"]
            assert "High confidence text" in filtered_text
            assert "Medium confidence text" in filtered_text
            assert "Low confidence text" not in filtered_text
            assert "Very low confidence" not in filtered_text
    
    @pytest.mark.asyncio
    async def test_no_confidence_threshold(self, ocr_processor, mock_context, 
                                         sample_config, mock_ocr_response):
        """Test OCR processing without confidence threshold."""
        config = sample_config.copy()
        config["confidence_threshold"] = 0.0  # No filtering
        
        with patch('core.executors.ocr_processor.ocr_manager.extract_text', 
                   return_value=mock_ocr_response):
            
            result = await ocr_processor.execute(mock_context, config)
            
            assert result.success is True
            
            # Should include all text (no filtering)
            assert result.outputs["raw_ocr_text"] == mock_ocr_response.full_text
    
    @pytest.mark.asyncio
    async def test_ocr_processing_error(self, ocr_processor, mock_context, sample_config):
        """Test handling of OCR processing errors."""
        with patch('core.executors.ocr_processor.ocr_manager.extract_text', 
                   side_effect=Exception("OCR service unavailable")):
            
            result = await ocr_processor.execute(mock_context, sample_config)
            
            assert result.success is False
            assert "OCR processing failed" in result.error
            assert "OCR service unavailable" in result.error
    
    def test_ocr_processor_name(self, ocr_processor):
        """Test OCRProcessor name property."""
        assert ocr_processor.name == "OCRProcessor"
    
    def test_clean_ocr_text_basic(self, ocr_processor):
        """Test basic OCR text cleaning."""
        raw_text = "This is a sample document\nwith multiple lines\nand some text"
        
        cleaned = ocr_processor._clean_ocr_text(raw_text)
        
        assert "This is a sample document" in cleaned
        assert "with multiple lines" in cleaned
        assert "and some text" in cleaned
        # Should join lines with spaces
        assert "\n" not in cleaned
    
    def test_clean_ocr_text_artifacts(self, ocr_processor):
        """Test OCR text cleaning with artifacts."""
        raw_text = "Real text here\nx\n|\nMore real text\n@\nFinal text"
        
        cleaned = ocr_processor._clean_ocr_text(raw_text)
        
        assert "Real text here" in cleaned
        assert "More real text" in cleaned
        assert "Final text" in cleaned
        # Should filter out single character artifacts
        assert " x " not in cleaned
        assert " | " not in cleaned
        assert " @ " not in cleaned
    
    def test_clean_ocr_text_preserve_important(self, ocr_processor):
        """Test OCR text cleaning preserves important single characters."""
        raw_text = "Dr. Smith\nA.\nCopyright ©\n2023\n&\nCompany"
        
        cleaned = ocr_processor._clean_ocr_text(raw_text)
        
        assert "Dr. Smith" in cleaned
        assert "A." in cleaned  # Should preserve single letter with period
        assert "©" in cleaned   # Should preserve copyright symbol
        assert "2023" in cleaned  # Should preserve numbers
        assert "&" in cleaned   # Should preserve important symbols
        assert "Company" in cleaned
    
    def test_clean_ocr_text_empty(self, ocr_processor):
        """Test OCR text cleaning with empty input."""
        assert ocr_processor._clean_ocr_text("") == ""
        assert ocr_processor._clean_ocr_text(None) == ""
    
    def test_clean_ocr_text_fallback(self, ocr_processor):
        """Test OCR text cleaning fallback when too much content is removed."""
        # Text that would be heavily filtered
        raw_text = "a\nb\nc\nd\ne\nf"  # All single characters
        
        cleaned = ocr_processor._clean_ocr_text(raw_text)
        
        # Should fallback to basic cleanup (just removing line breaks)
        assert cleaned == "a b c d e f"
    
    def test_clean_ocr_text_whitespace_normalization(self, ocr_processor):
        """Test OCR text cleaning normalizes whitespace."""
        raw_text = "Text   with    excessive     spaces\n\n\nand   line   breaks"
        
        cleaned = ocr_processor._clean_ocr_text(raw_text)
        
        # Should normalize multiple spaces to single spaces
        assert "   " not in cleaned
        assert "Text with excessive spaces" in cleaned
        assert "and line breaks" in cleaned


class TestOCRProcessorIntegration:
    """Integration tests for OCRProcessor with flow context."""
    
    @pytest.mark.asyncio
    async def test_ocr_processor_in_flow_context(self):
        """Test OCRProcessor as part of a flow context."""
        ocr_processor = OCRProcessor()
        context = FlowContext(
            flow_name="document_analysis_flow",
            inputs={"document_image": "scan.png"},
            flow_id="integration-test"
        )
        
        # Add previous step result (e.g., from file handler)
        file_result = ExecutionResult(
            success=True,
            outputs={
                "temp_path": "/tmp/uploaded_document.png",
                "filename": "document_scan.png",
                "file_extension": ".png"
            }
        )
        context.add_step_result("file_upload", file_result)
        
        config = {
            "image_path": context.step_results["file_upload"].outputs["temp_path"],
            "provider": "tesseract",
            "languages": ["en"],
            "confidence_threshold": 0.6
        }
        
        # Mock OCR response
        mock_response = MockOCRResponse(
            full_text="INVOICE\nCompany Name: ABC Corp\nDate: 2023-12-01\nAmount: $1,234.56\nThank you for your business!",
            text_blocks=[
                MockTextBlock("INVOICE", 0.98),
                MockTextBlock("Company Name: ABC Corp", 0.95),
                MockTextBlock("Date: 2023-12-01", 0.92),
                MockTextBlock("Amount: $1,234.56", 0.89),
                MockTextBlock("Thank you for your business!", 0.85)
            ],
            confidence_avg=0.92
        )
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('core.executors.ocr_processor.ocr_manager.extract_text', 
                   return_value=mock_response):
            
            result = await ocr_processor.execute(context, config)
            
            # Add result to context
            context.add_step_result("ocr_extraction", result)
            
            # Verify integration
            assert result.success is True
            assert context.step_results["ocr_extraction"] is not None
            
            # Verify the result can be used by subsequent steps
            ocr_result = context.step_results["ocr_extraction"]
            assert "INVOICE" in ocr_result.outputs["text"]
            assert "ABC Corp" in ocr_result.outputs["text"]
            assert ocr_result.outputs["confidence"] == 0.92
            assert ocr_result.outputs["provider"] == "tesseract"
            assert ocr_result.outputs["word_count"] > 0
            
            # Verify metadata
            assert "ocr_provider" in ocr_result.metadata
            assert "extraction_confidence" in ocr_result.metadata
            assert ocr_result.metadata["ocr_provider"] == "tesseract"
    
    @pytest.mark.asyncio
    async def test_ocr_to_llm_analysis_pipeline(self):
        """Test OCR processor feeding into LLM analysis."""
        ocr_processor = OCRProcessor()
        context = FlowContext(
            flow_name="document_understanding_flow",
            inputs={"receipt_image": "receipt.jpg"},
            flow_id="pipeline-test"
        )
        
        config = {
            "image_path": "/tmp/receipt.jpg",
            "provider": "tesseract",
            "languages": ["en"],
            "confidence_threshold": 0.7
        }
        
        # Mock OCR response with receipt data
        mock_response = MockOCRResponse(
            full_text="GROCERY STORE\nMilk $3.99\nBread $2.49\nEggs $4.99\nTotal: $11.47\nThank you!",
            text_blocks=[
                MockTextBlock("GROCERY STORE", 0.95),
                MockTextBlock("Milk $3.99", 0.88),
                MockTextBlock("Bread $2.49", 0.91),
                MockTextBlock("Eggs $4.99", 0.87),
                MockTextBlock("Total: $11.47", 0.93),
                MockTextBlock("Thank you!", 0.82)
            ],
            confidence_avg=0.89
        )
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('core.executors.ocr_processor.ocr_manager.extract_text', 
                   return_value=mock_response):
            
            result = await ocr_processor.execute(context, config)
            
            # Add OCR result to context
            context.add_step_result("text_extraction", result)
            
            # Verify OCR result is ready for LLM analysis
            assert result.success is True
            extracted_text = result.outputs["text"]
            
            # Text should be clean and ready for LLM processing
            assert "GROCERY STORE" in extracted_text
            assert "$3.99" in extracted_text
            assert "$11.47" in extracted_text
            
            # Verify structure is preserved for analysis
            assert len(extracted_text.split()) >= 8  # Should have meaningful word count
            assert result.outputs["word_count"] >= 8
            
            # The extracted text should be suitable for further analysis
            # (e.g., by LLMAnalyzer or SentimentAnalyzer)
            assert len(extracted_text) > 20  # Should have substantial content
            assert result.outputs["confidence"] > 0.8  # Should have good confidence
