"""
Unit tests for ImageHandler executor.

Tests image processing, validation, resizing, enhancement,
format conversion, and temporary file management.
"""

import pytest
import uuid
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image as PILImage

from core.executors.image_handler import ImageHandler
from core.executors.base_executor import FlowContext, ExecutionResult


class TestImageHandler:
    """Test suite for ImageHandler executor."""
    
    @pytest.fixture
    def image_handler(self):
        """Create ImageHandler instance for testing."""
        return ImageHandler()
    
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
        # Create a simple test image
        image = PILImage.new('RGB', (100, 100), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            image.save(temp_file.name, 'PNG')
            yield temp_file.name
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except:
            pass
    
    @pytest.fixture
    def sample_jpeg_file(self):
        """Create a temporary JPEG file for testing."""
        # Create a simple test image
        image = PILImage.new('RGB', (200, 150), color='blue')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image.save(temp_file.name, 'JPEG')
            yield temp_file.name
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except:
            pass
    
    @pytest.fixture
    def sample_config(self, sample_image_file):
        """Sample configuration for image processing."""
        return {
            "image_path": sample_image_file,
            "save_temp": True,
            "validate_format": True
        }
    
    @pytest.mark.asyncio
    async def test_pil_not_available(self, mock_context):
        """Test error when PIL is not available."""
        with patch('core.executors.image_handler.PIL_AVAILABLE', False):
            image_handler = ImageHandler()
            config = {"image_path": "/fake/image.png"}
            
            result = await image_handler.execute(mock_context, config)
            
            assert result.success is False
            assert "PIL/Pillow is required" in result.error
    
    @pytest.mark.asyncio
    async def test_missing_image_path(self, image_handler, mock_context):
        """Test error when image_path is missing."""
        config = {"save_temp": True}
        
        result = await image_handler.execute(mock_context, config)
        
        assert result.success is False
        assert "image_path is required" in result.error
    
    @pytest.mark.asyncio
    async def test_nonexistent_image_file(self, image_handler, mock_context):
        """Test error when image file doesn't exist."""
        config = {
            "image_path": "/nonexistent/image.png"
        }
        
        result = await image_handler.execute(mock_context, config)
        
        assert result.success is False
        assert "Image file not found" in result.error
    
    @pytest.mark.asyncio
    async def test_invalid_image_file(self, image_handler, mock_context):
        """Test error when file is not a valid image."""
        # Create a text file with image extension
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False, mode='w') as temp_file:
            temp_file.write("This is not an image")
            temp_file.flush()
            
            config = {"image_path": temp_file.name}
            
            result = await image_handler.execute(mock_context, config)
            
            assert result.success is False
            assert "Failed to load image" in result.error
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except:
            pass
    
    @pytest.mark.asyncio
    async def test_unsupported_format_validation(self, image_handler, mock_context):
        """Test unsupported format validation."""
        # Create a mock image with unsupported format
        mock_image = Mock()
        mock_image.format = 'WEBP'  # Unsupported format
        mock_image.size = (100, 100)
        mock_image.mode = 'RGB'
        
        with patch('PIL.Image.open', return_value=mock_image):
            config = {
                "image_path": "/fake/image.webp",
                "validate_format": True
            }
            
            with patch('pathlib.Path.exists', return_value=True):
                result = await image_handler.execute(mock_context, config)
                
                assert result.success is False
                assert "Unsupported image format: WEBP" in result.error
    
    @pytest.mark.asyncio
    async def test_successful_basic_processing(self, image_handler, mock_context, sample_config):
        """Test successful basic image processing."""
        result = await image_handler.execute(mock_context, sample_config)
        
        assert result.success is True
        assert "original_path" in result.outputs
        assert "original_format" in result.outputs
        assert "original_size" in result.outputs
        assert "processed_format" in result.outputs
        assert "processed_size" in result.outputs
        assert "processing_steps" in result.outputs
        assert "temp_path" in result.outputs
        
        # Verify original image properties
        assert result.outputs["original_format"] == "PNG"
        assert result.outputs["original_size"] == (100, 100)
        
        # Verify temp file was created
        temp_path = result.outputs["temp_path"]
        assert os.path.exists(temp_path)
        
        # Cleanup
        image_handler.cleanup_temp_files()
    
    @pytest.mark.asyncio
    async def test_image_resizing(self, image_handler, mock_context, sample_image_file):
        """Test image resizing functionality."""
        config = {
            "image_path": sample_image_file,
            "resize": (50, 50),  # Resize to 50x50
            "save_temp": True
        }
        
        result = await image_handler.execute(mock_context, config)
        
        assert result.success is True
        assert result.outputs["processed_size"] == (50, 50)
        assert "resized_to_(50, 50)" in result.outputs["processing_steps"]
        assert result.outputs["image_processed"] is True
        
        # Cleanup
        image_handler.cleanup_temp_files()
    
    @pytest.mark.asyncio
    async def test_aspect_ratio_preservation(self, image_handler, mock_context):
        """Test that aspect ratio is preserved during resizing."""
        # Create a rectangular image (200x100)
        image = PILImage.new('RGB', (200, 100), color='green')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            image.save(temp_file.name, 'PNG')
            
            config = {
                "image_path": temp_file.name,
                "resize": (100, 100),  # Square target
                "save_temp": True
            }
            
            result = await image_handler.execute(mock_context, config)
            
            assert result.success is True
            # Should be resized to (100, 50) to maintain 2:1 aspect ratio
            assert result.outputs["processed_size"] == (100, 50)
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except:
            pass
        image_handler.cleanup_temp_files()
    
    @pytest.mark.asyncio
    async def test_no_upscaling(self, image_handler, mock_context, sample_image_file):
        """Test that images are not upscaled."""
        config = {
            "image_path": sample_image_file,
            "resize": (200, 200),  # Larger than original (100x100)
            "save_temp": True
        }
        
        result = await image_handler.execute(mock_context, config)
        
        assert result.success is True
        # Should remain original size (100x100)
        assert result.outputs["processed_size"] == (100, 100)
        # The resize step is recorded even if no actual resizing occurred
        processing_steps = result.outputs["processing_steps"]
        assert "resized_to_(100, 100)" in processing_steps  # Records the "resize" to original size
        
        # Cleanup
        image_handler.cleanup_temp_files()
    
    @pytest.mark.asyncio
    async def test_grayscale_conversion(self, image_handler, mock_context, sample_image_file):
        """Test grayscale conversion."""
        config = {
            "image_path": sample_image_file,
            "convert_grayscale": True,
            "save_temp": True
        }
        
        result = await image_handler.execute(mock_context, config)
        
        assert result.success is True
        assert result.outputs["processed_mode"] == "L"  # Grayscale mode
        assert "converted_to_grayscale" in result.outputs["processing_steps"]
        assert result.outputs["image_processed"] is True
        
        # Cleanup
        image_handler.cleanup_temp_files()
    
    @pytest.mark.asyncio
    async def test_contrast_enhancement(self, image_handler, mock_context, sample_image_file):
        """Test contrast enhancement."""
        config = {
            "image_path": sample_image_file,
            "enhance_contrast": True,
            "save_temp": True
        }
        
        result = await image_handler.execute(mock_context, config)
        
        assert result.success is True
        assert "enhanced_contrast" in result.outputs["processing_steps"]
        assert result.outputs["image_processed"] is True
        
        # Cleanup
        image_handler.cleanup_temp_files()
    
    @pytest.mark.asyncio
    async def test_format_conversion(self, image_handler, mock_context, sample_image_file):
        """Test format conversion."""
        config = {
            "image_path": sample_image_file,
            "output_format": "JPEG",
            "quality": 85,
            "save_temp": True
        }
        
        result = await image_handler.execute(mock_context, config)
        
        assert result.success is True
        assert result.outputs["processed_format"] == "JPEG"
        
        # Check if RGB conversion step was applied (depends on original image mode)
        processing_steps = result.outputs["processing_steps"]
        # PNG images are typically RGB already, so conversion may not be needed
        # Just verify the format conversion worked
        
        # Verify temp file has correct extension
        temp_path = result.outputs["temp_path"]
        assert temp_path.endswith('.jpg')
        
        # Cleanup
        image_handler.cleanup_temp_files()
    
    @pytest.mark.asyncio
    async def test_multiple_processing_steps(self, image_handler, mock_context, sample_image_file):
        """Test multiple processing steps combined."""
        config = {
            "image_path": sample_image_file,
            "resize": (75, 75),
            "convert_grayscale": True,
            "enhance_contrast": True,
            "output_format": "PNG",
            "save_temp": True
        }
        
        result = await image_handler.execute(mock_context, config)
        
        assert result.success is True
        assert result.outputs["processed_size"] == (75, 75)
        assert result.outputs["processed_mode"] == "L"  # Grayscale
        assert result.outputs["processed_format"] == "PNG"
        
        processing_steps = result.outputs["processing_steps"]
        assert "resized_to_(75, 75)" in processing_steps
        assert "converted_to_grayscale" in processing_steps
        assert "enhanced_contrast" in processing_steps
        assert len(processing_steps) == 3
        
        # Cleanup
        image_handler.cleanup_temp_files()
    
    @pytest.mark.asyncio
    async def test_no_temp_file_save(self, image_handler, mock_context, sample_image_file):
        """Test processing without saving temp file."""
        config = {
            "image_path": sample_image_file,
            "save_temp": False,
            "resize": (50, 50)
        }
        
        result = await image_handler.execute(mock_context, config)
        
        assert result.success is True
        assert "temp_path" not in result.outputs
        assert result.outputs.get("temp_file_created", False) is False
        assert result.outputs["processed_size"] == (50, 50)
    
    @pytest.mark.asyncio
    async def test_skip_format_validation(self, image_handler, mock_context):
        """Test skipping format validation."""
        # Create a mock image with unsupported format
        mock_image = Mock()
        mock_image.format = 'WEBP'
        mock_image.size = (100, 100)
        mock_image.mode = 'RGB'
        mock_image.copy.return_value = mock_image
        mock_image.close = Mock()
        
        with patch('PIL.Image.open', return_value=mock_image), \
             patch('pathlib.Path.exists', return_value=True):
            
            config = {
                "image_path": "/fake/image.webp",
                "validate_format": False,  # Skip validation
                "save_temp": False
            }
            
            result = await image_handler.execute(mock_context, config)
            
            assert result.success is True
            assert result.outputs["original_format"] == "WEBP"
    
    def test_image_handler_name(self, image_handler):
        """Test ImageHandler name property."""
        assert image_handler.name == "ImageHandler"
    
    def test_required_config_keys(self, image_handler):
        """Test required configuration keys."""
        required_keys = image_handler.get_required_config_keys()
        assert "image_path" in required_keys
    
    def test_optional_config_keys(self, image_handler):
        """Test optional configuration keys."""
        optional_keys = image_handler.get_optional_config_keys()
        expected_keys = [
            "save_temp", "validate_format", "resize", "enhance_contrast",
            "convert_grayscale", "output_format", "quality"
        ]
        
        for key in expected_keys:
            assert key in optional_keys
    
    def test_config_validation_success(self, image_handler):
        """Test successful configuration validation."""
        valid_configs = [
            {"image_path": "/path/to/image.png"},
            {"image_path": "/path/to/image.png", "save_temp": True},
            {"image_path": "/path/to/image.png", "resize": (100, 100)},
            {"image_path": "/path/to/image.png", "output_format": "JPEG", "quality": 85}
        ]
        
        for config in valid_configs:
            # Should not raise exception
            image_handler.validate_config(config)
    
    def test_config_validation_failures(self, image_handler):
        """Test configuration validation failures."""
        invalid_configs = [
            {"image_path": "/path/to/image.png", "save_temp": "not_boolean"},
            {"image_path": "/path/to/image.png", "resize": (100,)},  # Wrong tuple size
            {"image_path": "/path/to/image.png", "resize": (-100, 100)},  # Negative dimension
            {"image_path": "/path/to/image.png", "output_format": "INVALID"},
            {"image_path": "/path/to/image.png", "quality": 150},  # Quality > 100
            {"image_path": "/path/to/image.png", "quality": 0}  # Quality < 1
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                image_handler.validate_config(config)
    
    def test_get_info(self, image_handler):
        """Test executor information."""
        info = image_handler.get_info()
        
        assert "capabilities" in info
        assert "supported_formats" in info
        assert "processing_options" in info
        assert "pil_available" in info
        
        # Check capabilities
        capabilities = info["capabilities"]
        assert "Image format validation" in capabilities
        assert "Image resizing with aspect ratio preservation" in capabilities
        assert "Contrast enhancement for OCR" in capabilities
        
        # Check supported formats
        formats = info["supported_formats"]
        assert "JPEG" in formats
        assert "PNG" in formats
        assert "TIFF" in formats
    
    def test_cleanup_temp_files(self, image_handler):
        """Test temporary file cleanup."""
        # Create some fake temp files
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_files.append(temp_file.name)
        
        # Add to image handler's temp files list
        image_handler.temp_files = temp_files.copy()
        
        # Verify files exist
        for temp_file in temp_files:
            assert os.path.exists(temp_file)
        
        # Cleanup
        image_handler.cleanup_temp_files()
        
        # Verify files are deleted
        for temp_file in temp_files:
            assert not os.path.exists(temp_file)
        
        # Verify temp files list is cleared
        assert len(image_handler.temp_files) == 0


class TestImageHandlerProcessingMethods:
    """Test suite for image processing helper methods."""
    
    @pytest.fixture
    def image_handler(self):
        return ImageHandler()
    
    def test_resize_image_downscale(self, image_handler):
        """Test image resizing with downscaling."""
        # Create test image
        image = PILImage.new('RGB', (200, 100), color='red')
        
        # Resize to smaller dimensions
        resized = image_handler._resize_image(image, 100, 50)
        
        assert resized.size == (100, 50)
        
        # Cleanup
        image.close()
        resized.close()
    
    def test_resize_image_aspect_ratio(self, image_handler):
        """Test aspect ratio preservation during resize."""
        # Create test image (2:1 aspect ratio)
        image = PILImage.new('RGB', (200, 100), color='blue')
        
        # Resize to square target (should maintain aspect ratio)
        resized = image_handler._resize_image(image, 80, 80)
        
        # Should be 80x40 to maintain 2:1 ratio
        assert resized.size == (80, 40)
        
        # Cleanup
        image.close()
        resized.close()
    
    def test_resize_image_no_upscale(self, image_handler):
        """Test that resize doesn't upscale images."""
        # Create small test image
        image = PILImage.new('RGB', (50, 50), color='green')
        
        # Try to resize to larger dimensions
        resized = image_handler._resize_image(image, 100, 100)
        
        # Should remain original size
        assert resized.size == (50, 50)
        assert resized is image  # Should return original image
        
        # Cleanup
        image.close()
    
    def test_enhance_contrast(self, image_handler):
        """Test contrast enhancement."""
        # Create test image
        image = PILImage.new('RGB', (100, 100), color='gray')
        
        # Enhance contrast
        enhanced = image_handler._enhance_contrast(image)
        
        # Should return a different image object
        assert enhanced is not image
        assert enhanced.size == image.size
        assert enhanced.mode == image.mode
        
        # Cleanup
        image.close()
        enhanced.close()
    
    def test_save_processed_image_png(self, image_handler):
        """Test saving processed image as PNG."""
        # Create test image
        image = PILImage.new('RGB', (100, 100), color='yellow')
        
        # Save as PNG
        temp_path = image_handler._save_processed_image(image, 'PNG', 95, 'test')
        
        # Verify file was created
        assert temp_path.exists()
        assert temp_path.suffix == '.png'
        assert str(temp_path) in image_handler.temp_files
        
        # Verify image can be loaded
        saved_image = PILImage.open(temp_path)
        assert saved_image.size == (100, 100)
        
        # Cleanup
        image.close()
        saved_image.close()
        image_handler.cleanup_temp_files()
    
    def test_save_processed_image_jpeg(self, image_handler):
        """Test saving processed image as JPEG."""
        # Create test image
        image = PILImage.new('RGB', (100, 100), color='orange')
        
        # Save as JPEG with specific quality
        temp_path = image_handler._save_processed_image(image, 'JPEG', 80, 'test')
        
        # Verify file was created
        assert temp_path.exists()
        assert temp_path.suffix == '.jpg'
        
        # Cleanup
        image.close()
        image_handler.cleanup_temp_files()


class TestImageHandlerIntegration:
    """Integration tests for ImageHandler with flow context."""
    
    @pytest.mark.asyncio
    async def test_image_handler_in_flow_context(self):
        """Test ImageHandler as part of a flow context."""
        image_handler = ImageHandler()
        context = FlowContext(
            flow_name="image_processing_flow",
            inputs={"image_file": "document.png"},
            flow_id="integration-test"
        )
        
        # Add previous step result (e.g., from file handler)
        file_result = ExecutionResult(
            success=True,
            outputs={
                "temp_path": "/tmp/uploaded_document.png",
                "filename": "scanned_document.png",
                "file_extension": ".png"
            }
        )
        context.add_step_result("file_upload", file_result)
        
        # Create actual test image
        image = PILImage.new('RGB', (300, 400), color='white')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            image.save(temp_file.name, 'PNG')
            
            config = {
                "image_path": temp_file.name,
                "resize": (200, 300),  # Resize for OCR optimization
                "enhance_contrast": True,  # Enhance for better OCR
                "convert_grayscale": True,  # Convert to grayscale
                "output_format": "PNG",
                "save_temp": True
            }
            
            result = await image_handler.execute(context, config)
            
            # Add result to context
            context.add_step_result("image_preprocessing", result)
            
            # Verify integration
            assert result.success is True
            assert context.step_results["image_preprocessing"] is not None
            
            # Verify the result can be used by subsequent steps (e.g., OCR)
            preprocessing_result = context.step_results["image_preprocessing"]
            # Allow for small rounding differences in aspect ratio calculation
            processed_size = preprocessing_result.outputs["processed_size"]
            assert processed_size[0] == 200  # Width should be exact
            assert 266 <= processed_size[1] <= 267  # Height may have rounding difference
            assert preprocessing_result.outputs["processed_mode"] == "L"  # Grayscale
            assert preprocessing_result.outputs["image_processed"] is True
            
            # Verify processing steps
            steps = preprocessing_result.outputs["processing_steps"]
            assert any("resized_to_" in step for step in steps)  # Some resize step
            assert "converted_to_grayscale" in steps
            assert "enhanced_contrast" in steps
            
            # Verify temp file is ready for OCR processing
            temp_path = preprocessing_result.outputs["temp_path"]
            assert os.path.exists(temp_path)
            
            # Verify processed image can be loaded
            processed_image = PILImage.open(temp_path)
            processed_size = processed_image.size
            assert processed_size[0] == 200
            assert 266 <= processed_size[1] <= 267  # Allow for rounding
            assert processed_image.mode == "L"  # Grayscale
            
            # Cleanup
            processed_image.close()
        
        # Cleanup
        image.close()
        try:
            os.unlink(temp_file.name)
        except:
            pass
        image_handler.cleanup_temp_files()
    
    @pytest.mark.asyncio
    async def test_image_to_ocr_pipeline(self):
        """Test image processing feeding into OCR processing."""
        image_handler = ImageHandler()
        context = FlowContext(
            flow_name="document_ocr_flow",
            inputs={"document_scan": "receipt.jpg"},
            flow_id="pipeline-test"
        )
        
        # Create a test image that simulates a scanned document
        # Use a larger image with some text-like patterns
        image = PILImage.new('RGB', (800, 600), color='white')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image.save(temp_file.name, 'JPEG')
            
            config = {
                "image_path": temp_file.name,
                "resize": (1200, 900),  # Don't upscale (should remain 800x600)
                "enhance_contrast": True,  # Optimize for OCR
                "convert_grayscale": True,  # OCR works better with grayscale
                "output_format": "PNG",  # PNG is better for OCR than JPEG
                "save_temp": True
            }
            
            result = await image_handler.execute(context, config)
            
            # Add preprocessing result to context
            context.add_step_result("image_preprocessing", result)
            
            # Verify preprocessing result is ready for OCR
            assert result.success is True
            preprocessing_result = context.step_results["image_preprocessing"]
            
            # Image should be optimized for OCR
            assert preprocessing_result.outputs["processed_mode"] == "L"  # Grayscale
            assert preprocessing_result.outputs["processed_format"] == "PNG"  # Better for OCR
            assert "enhanced_contrast" in preprocessing_result.outputs["processing_steps"]
            
            # Verify the processed image is suitable for OCR processing
            temp_path = preprocessing_result.outputs["temp_path"]
            assert os.path.exists(temp_path)
            assert temp_path.endswith('.png')
            
            # The processed image should be ready for OCRProcessor
            processed_image = PILImage.open(temp_path)
            assert processed_image.mode == "L"  # Grayscale is optimal for OCR
            assert processed_image.size == (800, 600)  # Original size (no upscaling)
            
            # Cleanup
            processed_image.close()
        
        # Cleanup
        image.close()
        try:
            os.unlink(temp_file.name)
        except:
            pass
        image_handler.cleanup_temp_files()
