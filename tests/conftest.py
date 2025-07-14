"""
Shared pytest configuration and fixtures for executor testing.

This module provides common fixtures, test utilities, and configuration
that can be used across all executor test modules.
"""

import pytest
import tempfile
import uuid
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.executors.base_executor import FlowContext, ExecutionResult


@pytest.fixture
def mock_flow_context():
    """Create a mock FlowContext for testing."""
    context = FlowContext(
        flow_name="test_flow",
        inputs={"test_param": "test_value"},
        flow_id=f"test-flow-{uuid.uuid4().hex[:8]}"
    )
    return context


@pytest.fixture
def sample_config():
    """Basic configuration for testing."""
    return {
        "test_param": "test_value",
        "timeout": 30,
        "retry_count": 3
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_text_content():
    """Sample text content for testing."""
    return "This is a sample text for testing purposes. It contains multiple sentences and should be sufficient for basic testing scenarios."


@pytest.fixture
def mock_successful_result():
    """Mock successful ExecutionResult."""
    return ExecutionResult(
        success=True,
        outputs={"result": "test_output", "metadata": {"processed": True}},
        metadata={"execution_time": 1.23, "test_mode": True}
    )


@pytest.fixture
def mock_error_result():
    """Mock error ExecutionResult."""
    return ExecutionResult(
        success=False,
        error="Test error message",
        metadata={"execution_time": 0.5, "test_mode": True}
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    mock_response = Mock()
    mock_response.content = "This is a mocked LLM response for testing."
    mock_response.metadata = {"model": "test-model", "tokens": 10}
    return mock_response


@pytest.fixture
def sample_image_bytes():
    """Sample image bytes for testing (minimal PNG)."""
    # Minimal 1x1 PNG image bytes
    return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'


@pytest.fixture
def sample_pdf_bytes():
    """Sample PDF bytes for testing (minimal PDF)."""
    return b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n174\n%%EOF'


@pytest.fixture
def mock_file_upload():
    """Mock file upload object."""
    mock_file = Mock()
    mock_file.filename = "test_file.txt"
    mock_file.content_type = "text/plain"
    mock_file.size = 1024
    mock_file.read = Mock(return_value=b"Test file content")
    return mock_file


# Test utilities
class TestHelpers:
    """Helper methods for testing."""
    
    @staticmethod
    def create_temp_file(temp_dir: Path, filename: str, content: bytes) -> Path:
        """Create a temporary file with given content."""
        file_path = temp_dir / filename
        file_path.write_bytes(content)
        return file_path
    
    @staticmethod
    def assert_execution_result(result: ExecutionResult, success: bool = True, 
                              required_outputs: list = None):
        """Assert ExecutionResult has expected structure."""
        assert isinstance(result, ExecutionResult)
        assert result.success == success
        
        if success:
            assert result.error is None
            assert isinstance(result.outputs, dict)
            if required_outputs:
                for output in required_outputs:
                    assert output in result.outputs
        else:
            assert result.error is not None
            assert isinstance(result.error, str)
    
    @staticmethod
    def create_mock_context_with_results(step_results: Dict[str, Any]) -> FlowContext:
        """Create FlowContext with pre-populated step results."""
        context = FlowContext(
            flow_name="test_flow",
            inputs={"test": "value"},
            flow_id="test-flow-123"
        )
        for step_name, result in step_results.items():
            context.add_step_result(step_name, ExecutionResult(
                success=True,
                outputs=result
            ))
        return context


@pytest.fixture
def test_helpers():
    """Provide TestHelpers instance."""
    return TestHelpers()


# Async test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
