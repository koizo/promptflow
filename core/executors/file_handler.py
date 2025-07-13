"""
File Handler Executor

Reusable executor for handling file uploads, validation, and temporary storage.
Manages file lifecycle, format validation, and cleanup operations.
"""

import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import mimetypes
import hashlib

from .base_executor import BaseExecutor, ExecutionResult, FlowContext

logger = logging.getLogger(__name__)


class FileHandler(BaseExecutor):
    """
    Handle file uploads, validation, and temporary storage.
    
    Supports file format validation, size limits, and automatic cleanup.
    Creates temporary files for processing and manages their lifecycle.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.temp_files: List[str] = []  # Track temp files for cleanup
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Handle file upload and validation.
        
        Config parameters:
        - file_content (required): File content (bytes or file-like object)
        - filename (required): Original filename
        - save_temp (optional): Whether to save to temporary file (default: True)
        - validate_format (optional): Whether to validate file format (default: True)
        - allowed_formats (optional): List of allowed file extensions
        - max_size (optional): Maximum file size in bytes
        - preserve_name (optional): Whether to preserve original filename (default: False)
        """
        try:
            # Get file information
            file_content = config.get('file_content')
            filename = config.get('filename')
            
            if not file_content:
                return ExecutionResult(
                    success=False,
                    error="file_content is required"
                )
            
            if not filename:
                return ExecutionResult(
                    success=False,
                    error="filename is required"
                )
            
            # Get configuration
            save_temp = config.get('save_temp', True)
            validate_format = config.get('validate_format', True)
            allowed_formats = config.get('allowed_formats', [])
            max_size = config.get('max_size')
            preserve_name = config.get('preserve_name', False)
            
            # Get file info
            file_path = Path(filename)
            file_extension = file_path.suffix.lower()
            file_size = len(file_content) if isinstance(file_content, bytes) else None
            
            self.logger.info(f"Processing file: {filename} ({file_extension})")
            
            # Validate file format
            if validate_format and allowed_formats:
                if file_extension not in allowed_formats:
                    return ExecutionResult(
                        success=False,
                        error=f"File format {file_extension} not allowed. Allowed: {allowed_formats}"
                    )
            
            # Validate file size
            if max_size and file_size and file_size > max_size:
                return ExecutionResult(
                    success=False,
                    error=f"File size {file_size} bytes exceeds maximum {max_size} bytes"
                )
            
            # Calculate file hash for integrity
            file_hash = None
            if isinstance(file_content, bytes):
                file_hash = hashlib.md5(file_content).hexdigest()
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(filename)
            
            outputs = {
                "filename": filename,
                "file_extension": file_extension,
                "file_size": file_size,
                "mime_type": mime_type,
                "file_hash": file_hash
            }
            
            # Save to temporary file if requested
            temp_path = None
            if save_temp:
                temp_path = self._save_temp_file(
                    file_content, 
                    file_extension, 
                    preserve_name, 
                    filename
                )
                outputs["temp_path"] = str(temp_path)
                outputs["temp_file_created"] = True
            
            return ExecutionResult(
                success=True,
                outputs=outputs,
                metadata={
                    "operation": "file_upload",
                    "validation_performed": validate_format,
                    "temp_file_created": save_temp,
                    "file_info": {
                        "name": filename,
                        "extension": file_extension,
                        "size": file_size,
                        "mime_type": mime_type,
                        "hash": file_hash
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"File handling failed: {str(e)}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=f"File handling failed: {str(e)}"
            )
    
    def _save_temp_file(self, file_content: bytes, file_extension: str, 
                       preserve_name: bool, original_filename: str) -> Path:
        """Save file content to temporary file."""
        if preserve_name:
            # Use original filename in temp directory
            temp_dir = tempfile.gettempdir()
            temp_path = Path(temp_dir) / original_filename
            
            # Ensure unique filename if file exists
            counter = 1
            while temp_path.exists():
                stem = Path(original_filename).stem
                temp_path = Path(temp_dir) / f"{stem}_{counter}{file_extension}"
                counter += 1
        else:
            # Create temporary file with proper extension
            temp_fd, temp_path_str = tempfile.mkstemp(suffix=file_extension)
            temp_path = Path(temp_path_str)
            os.close(temp_fd)  # Close the file descriptor
        
        # Write content to file
        if isinstance(file_content, bytes):
            temp_path.write_bytes(file_content)
        else:
            # Handle file-like objects
            with open(temp_path, 'wb') as f:
                if hasattr(file_content, 'read'):
                    f.write(file_content.read())
                else:
                    f.write(file_content)
        
        # Track temp file for cleanup
        self.temp_files.append(str(temp_path))
        
        self.logger.info(f"Saved temporary file: {temp_path}")
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
        return ["file_content", "filename"]
    
    def get_optional_config_keys(self) -> List[str]:
        """Optional configuration keys."""
        return [
            "save_temp",
            "validate_format",
            "allowed_formats",
            "max_size",
            "preserve_name"
        ]
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate executor configuration."""
        super().validate_config(config)
        
        # Validate boolean flags
        for bool_key in ['save_temp', 'validate_format', 'preserve_name']:
            if bool_key in config and not isinstance(config[bool_key], bool):
                raise ValueError(f"{bool_key} must be a boolean")
        
        # Validate allowed_formats is list
        if 'allowed_formats' in config:
            allowed_formats = config['allowed_formats']
            if not isinstance(allowed_formats, list):
                raise ValueError("allowed_formats must be a list")
            
            # Ensure all formats start with dot
            for fmt in allowed_formats:
                if not isinstance(fmt, str) or not fmt.startswith('.'):
                    raise ValueError("File formats must be strings starting with '.' (e.g., '.pdf')")
        
        # Validate max_size is positive integer
        if 'max_size' in config:
            max_size = config['max_size']
            if not isinstance(max_size, int) or max_size <= 0:
                raise ValueError("max_size must be a positive integer")
    
    def get_info(self) -> Dict[str, Any]:
        """Get executor information."""
        info = super().get_info()
        info.update({
            "capabilities": [
                "File upload handling",
                "File format validation",
                "File size validation",
                "Temporary file management",
                "File integrity checking (MD5 hash)",
                "MIME type detection",
                "Automatic cleanup"
            ],
            "supported_operations": [
                "File validation",
                "Temporary file creation",
                "File metadata extraction",
                "Format checking"
            ]
        })
        return info
    
    def __del__(self):
        """Cleanup temp files when executor is destroyed."""
        self.cleanup_temp_files()
