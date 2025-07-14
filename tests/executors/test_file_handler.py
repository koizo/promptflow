"""
Unit tests for FileHandler executor.

Tests file upload handling, validation, temporary file management,
and configuration options.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import uuid

from core.executors.file_handler import FileHandler
from core.executors.base_executor import FlowContext, ExecutionResult


class TestFileHandler:
    """Test suite for FileHandler executor."""
    
    @pytest.fixture
    def file_handler(self):
        """Create FileHandler instance for testing."""
        return FileHandler()
    
    @pytest.fixture
    def mock_context(self):
        """Create mock FlowContext for testing."""
        return FlowContext(
            flow_name="test_flow",
            inputs={"test_param": "test_value"},
            flow_id=f"test-flow-{uuid.uuid4().hex[:8]}"
        )
    
    @pytest.fixture
    def sample_file_config(self):
        """Sample file configuration for testing."""
        return {
            "file_content": b"Sample file content for testing",
            "filename": "test_file.txt",
            "save_temp": True,
            "validate_format": True,
            "allowed_formats": [".txt", ".pdf", ".jpg"],
            "max_size": 1048576  # 1MB
        }
    
    @pytest.fixture
    def mock_temp_file_creation(self):
        """Mock temporary file creation for testing."""
        with patch('core.executors.file_handler.Path.mkdir'), \
             patch('tempfile.mkstemp', return_value=(1, '/tmp/test_file.txt')), \
             patch('os.close'), \
             patch('os.chmod'), \
             patch('core.executors.file_handler.Path.write_bytes'), \
             patch('core.executors.file_handler.Path.exists', return_value=True):
            yield
    
    @pytest.mark.asyncio
    async def test_successful_file_handling(self, file_handler, mock_context, sample_file_config, mock_temp_file_creation):
        """Test successful file upload and processing."""
        result = await file_handler.execute(mock_context, sample_file_config)
        
        assert result.success is True
        assert "temp_path" in result.outputs
        assert "filename" in result.outputs
        assert "file_extension" in result.outputs
        assert result.outputs["filename"] == "test_file.txt"
        assert result.outputs["file_extension"] == ".txt"
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_file_size_validation(self, file_handler, mock_context):
        """Test file size validation."""
        config = {
            "file_content": b"x" * 2000000,  # 2MB content
            "filename": "large_file.txt",
            "max_size": 1048576  # 1MB limit
        }
        
        result = await file_handler.execute(mock_context, config)
        
        assert result.success is False
        assert "size" in result.error.lower()
        assert "exceeds" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_file_format_validation(self, file_handler, mock_context):
        """Test file format validation."""
        config = {
            "file_content": b"test content",
            "filename": "test_file.xyz",  # Unsupported format
            "validate_format": True,
            "allowed_formats": [".txt", ".pdf", ".jpg"]
        }
        
        result = await file_handler.execute(mock_context, config)
        
        assert result.success is False
        assert "format" in result.error.lower()
        assert "not allowed" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_missing_file_content(self, file_handler, mock_context):
        """Test handling of missing file content."""
        config = {
            "filename": "test_file.txt"
            # Missing file_content
        }
        
        result = await file_handler.execute(mock_context, config)
        
        assert result.success is False
        assert "file_content" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_missing_filename(self, file_handler, mock_context):
        """Test handling of missing filename."""
        config = {
            "file_content": b"test content"
            # Missing filename
        }
        
        result = await file_handler.execute(mock_context, config)
        
        assert result.success is False
        assert "filename" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_default_configuration(self, file_handler, mock_context, mock_temp_file_creation):
        """Test file handler with default configuration."""
        config = {
            "file_content": b"simple test content",
            "filename": "simple.txt"
        }
        
        result = await file_handler.execute(mock_context, config)
        
        assert result.success is True
        assert result.outputs["filename"] == "simple.txt"
        assert result.outputs["file_extension"] == ".txt"
    
    @pytest.mark.asyncio
    async def test_no_temp_file_creation(self, file_handler, mock_context):
        """Test file handling without creating temporary file."""
        config = {
            "file_content": b"test content",
            "filename": "no_temp.txt",
            "save_temp": False
        }
        
        result = await file_handler.execute(mock_context, config)
        
        assert result.success is True
        assert "temp_path" not in result.outputs or result.outputs["temp_path"] is None
        assert result.outputs["filename"] == "no_temp.txt"
    
    @pytest.mark.asyncio
    async def test_empty_file_handling(self, file_handler, mock_context):
        """Test handling of empty file."""
        config = {
            "file_content": b"",  # Empty content
            "filename": "empty_file.txt"
        }
        
        result = await file_handler.execute(mock_context, config)
        
        # Empty files are rejected by the implementation
        assert result.success is False
        assert "file_content" in result.error.lower()
    
    def test_file_handler_name(self, file_handler):
        """Test FileHandler name property."""
        assert file_handler.name == "FileHandler"


class TestFileHandlerIntegration:
    """Integration tests for FileHandler with other components."""
    
    @pytest.mark.asyncio
    async def test_file_handler_in_flow_context(self):
        """Test FileHandler as part of a flow context."""
        file_handler = FileHandler()
        context = FlowContext(flow_name="test_flow", inputs={}, flow_id="integration-test")
        
        config = {
            "file_content": b"Integration test content",
            "filename": "integration_test.txt"
        }
        
        with patch('core.executors.file_handler.Path.mkdir'), \
             patch('tempfile.mkstemp', return_value=(1, '/tmp/integration_test.txt')), \
             patch('os.close'), \
             patch('os.chmod'), \
             patch('core.executors.file_handler.Path.write_bytes'), \
             patch('core.executors.file_handler.Path.exists', return_value=True):
            
            result = await file_handler.execute(context, config)
            
            # Add result to context (simulating flow execution)
            context.add_step_result("file_handling", result)
            
            # Verify integration
            assert result.success is True
            assert context.step_results["file_handling"] is not None
            
            # Verify the result can be used by subsequent steps
            file_result = context.step_results["file_handling"]
            assert "temp_path" in file_result.outputs
            assert file_result.outputs["filename"] == "integration_test.txt"
