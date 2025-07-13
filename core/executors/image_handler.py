"""
Image Handler Executor

Reusable executor for handling image uploads, validation, and preprocessing.
Supports format conversion, resizing, and optimization for OCR processing.
"""

import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

from .base_executor import BaseExecutor, ExecutionResult, FlowContext

# Import PIL for image processing
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageHandler(BaseExecutor):
    """
    Handle image uploads, validation, and preprocessing.
    
    Supports format validation, resizing, enhancement, and optimization
    for better OCR results. Can convert between formats and apply filters.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.temp_files: List[str] = []
        
        if not PIL_AVAILABLE:
            self.logger.warning("PIL/Pillow not available. Image processing capabilities limited.")
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Handle image upload and preprocessing.
        
        Config parameters:
        - image_path (required): Path to image file or image content
        - save_temp (optional): Whether to save processed image (default: True)
        - validate_format (optional): Whether to validate image format (default: True)
        - resize (optional): Resize image to max dimensions (width, height)
        - enhance_contrast (optional): Enhance contrast for better OCR (default: False)
        - convert_grayscale (optional): Convert to grayscale (default: False)
        - output_format (optional): Output format ('PNG', 'JPEG', etc.)
        - quality (optional): JPEG quality (1-100, default: 95)
        """
        try:
            if not PIL_AVAILABLE:
                return ExecutionResult(
                    success=False,
                    error="PIL/Pillow is required for image processing. Please install: pip install Pillow"
                )
            
            image_path = config.get('image_path')
            if not image_path:
                return ExecutionResult(
                    success=False,
                    error="image_path is required"
                )
            
            image_path = Path(image_path)
            if not image_path.exists():
                return ExecutionResult(
                    success=False,
                    error=f"Image file not found: {image_path}"
                )
            
            # Get configuration
            save_temp = config.get('save_temp', True)
            validate_format = config.get('validate_format', True)
            resize = config.get('resize')
            enhance_contrast = config.get('enhance_contrast', False)
            convert_grayscale = config.get('convert_grayscale', False)
            output_format = config.get('output_format')
            quality = config.get('quality', 95)
            
            self.logger.info(f"Processing image: {image_path}")
            
            # Load and validate image
            try:
                image = Image.open(image_path)
                original_format = image.format
                original_size = image.size
                original_mode = image.mode
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    error=f"Failed to load image: {str(e)}"
                )
            
            # Validate format if requested
            if validate_format:
                supported_formats = ['JPEG', 'PNG', 'TIFF', 'BMP', 'GIF']
                if original_format not in supported_formats:
                    return ExecutionResult(
                        success=False,
                        error=f"Unsupported image format: {original_format}. Supported: {supported_formats}"
                    )
            
            # Process image
            processed_image = image.copy()
            processing_steps = []
            
            # Convert to RGB if needed (for JPEG output)
            if output_format == 'JPEG' and processed_image.mode in ('RGBA', 'LA', 'P'):
                processed_image = processed_image.convert('RGB')
                processing_steps.append("converted_to_rgb")
            
            # Resize if requested
            if resize:
                max_width, max_height = resize
                processed_image = self._resize_image(processed_image, max_width, max_height)
                processing_steps.append(f"resized_to_{processed_image.size}")
            
            # Convert to grayscale if requested
            if convert_grayscale:
                processed_image = processed_image.convert('L')
                processing_steps.append("converted_to_grayscale")
            
            # Enhance contrast if requested
            if enhance_contrast:
                processed_image = self._enhance_contrast(processed_image)
                processing_steps.append("enhanced_contrast")
            
            # Prepare output information
            final_format = output_format or original_format
            final_size = processed_image.size
            final_mode = processed_image.mode
            
            outputs = {
                "original_path": str(image_path),
                "original_format": original_format,
                "original_size": original_size,
                "original_mode": original_mode,
                "processed_format": final_format,
                "processed_size": final_size,
                "processed_mode": final_mode,
                "processing_steps": processing_steps,
                "image_processed": len(processing_steps) > 0
            }
            
            # Save processed image if requested
            if save_temp:
                temp_path = self._save_processed_image(
                    processed_image, 
                    final_format, 
                    quality,
                    image_path.stem
                )
                outputs["temp_path"] = str(temp_path)
                outputs["temp_file_created"] = True
            
            # Close images to free memory
            image.close()
            processed_image.close()
            
            return ExecutionResult(
                success=True,
                outputs=outputs,
                metadata={
                    "operation": "image_processing",
                    "original_file": str(image_path),
                    "processing_applied": len(processing_steps) > 0,
                    "size_change": {
                        "original": original_size,
                        "processed": final_size,
                        "reduction_ratio": (original_size[0] * original_size[1]) / (final_size[0] * final_size[1])
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {str(e)}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=f"Image processing failed: {str(e)}"
            )
    
    def _resize_image(self, image: Image.Image, max_width: int, max_height: int) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        original_width, original_height = image.size
        
        # Calculate scaling factor
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        scale_factor = min(width_ratio, height_ratio, 1.0)  # Don't upscale
        
        if scale_factor < 1.0:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Enhance image contrast for better OCR results."""
        # Apply contrast enhancement
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.2)  # 20% contrast increase
        
        # Apply sharpening filter
        sharpened = enhanced.filter(ImageFilter.SHARPEN)
        
        return sharpened
    
    def _save_processed_image(self, image: Image.Image, format: str, 
                            quality: int, base_name: str) -> Path:
        """Save processed image to temporary file."""
        # Determine file extension
        ext_map = {
            'JPEG': '.jpg',
            'PNG': '.png',
            'TIFF': '.tiff',
            'BMP': '.bmp',
            'GIF': '.gif'
        }
        extension = ext_map.get(format, '.png')
        
        # Create temporary file
        temp_fd, temp_path_str = tempfile.mkstemp(
            suffix=extension,
            prefix=f"processed_{base_name}_"
        )
        temp_path = Path(temp_path_str)
        os.close(temp_fd)
        
        # Save image
        save_kwargs = {}
        if format == 'JPEG':
            save_kwargs['quality'] = quality
            save_kwargs['optimize'] = True
        elif format == 'PNG':
            save_kwargs['optimize'] = True
        
        image.save(temp_path, format=format, **save_kwargs)
        
        # Track temp file for cleanup
        self.temp_files.append(str(temp_path))
        
        self.logger.info(f"Saved processed image: {temp_path}")
        return temp_path
    
    def cleanup_temp_files(self):
        """Clean up all temporary files created by this executor."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    self.logger.info(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        
        self.temp_files.clear()
    
    def get_required_config_keys(self) -> List[str]:
        """Required configuration keys."""
        return ["image_path"]
    
    def get_optional_config_keys(self) -> List[str]:
        """Optional configuration keys."""
        return [
            "save_temp",
            "validate_format",
            "resize",
            "enhance_contrast",
            "convert_grayscale",
            "output_format",
            "quality"
        ]
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate executor configuration."""
        super().validate_config(config)
        
        # Validate boolean flags
        for bool_key in ['save_temp', 'validate_format', 'enhance_contrast', 'convert_grayscale']:
            if bool_key in config and not isinstance(config[bool_key], bool):
                raise ValueError(f"{bool_key} must be a boolean")
        
        # Validate resize dimensions
        if 'resize' in config:
            resize = config['resize']
            if not isinstance(resize, (list, tuple)) or len(resize) != 2:
                raise ValueError("resize must be a tuple/list of (width, height)")
            
            width, height = resize
            if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
                raise ValueError("resize dimensions must be positive integers")
        
        # Validate output format
        if 'output_format' in config:
            format = config['output_format']
            valid_formats = ['JPEG', 'PNG', 'TIFF', 'BMP', 'GIF']
            if format not in valid_formats:
                raise ValueError(f"output_format must be one of: {valid_formats}")
        
        # Validate quality
        if 'quality' in config:
            quality = config['quality']
            if not isinstance(quality, int) or quality < 1 or quality > 100:
                raise ValueError("quality must be an integer between 1 and 100")
    
    def get_info(self) -> Dict[str, Any]:
        """Get executor information."""
        info = super().get_info()
        info.update({
            "capabilities": [
                "Image format validation",
                "Image resizing with aspect ratio preservation",
                "Contrast enhancement for OCR",
                "Grayscale conversion",
                "Format conversion",
                "Quality optimization",
                "Temporary file management"
            ],
            "supported_formats": ["JPEG", "PNG", "TIFF", "BMP", "GIF"],
            "processing_options": [
                "Resize to max dimensions",
                "Enhance contrast",
                "Convert to grayscale",
                "Format conversion",
                "Quality adjustment"
            ],
            "pil_available": PIL_AVAILABLE
        })
        return info
    
    def __del__(self):
        """Cleanup temp files when executor is destroyed."""
        self.cleanup_temp_files()
