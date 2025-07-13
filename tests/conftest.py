"""
Pytest configuration and shared fixtures for all tests.
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.executors.base_executor import FlowContext, ExecutionResult
from core.flow_engine.yaml_loader import FlowDefinition, FlowStep, FlowInput, FlowOutput
from core.state_store import StateStore


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_flow_context():
    """Create a sample FlowContext for testing."""
    return FlowContext(
        flow_name="test_flow_123",
        inputs={"input_text": "Hello World", "param1": "value1"},
        flow_id="test_flow_123"
    )


@pytest.fixture
def sample_execution_result():
    """Create a sample ExecutionResult for testing."""
    return ExecutionResult(
        success=True,
        outputs={"result": "processed_data", "count": 42},
        metadata={"execution_time": 1.5, "model_used": "test_model"}
    )


@pytest.fixture
def sample_flow_definition():
    """Create a sample FlowDefinition for testing."""
    return FlowDefinition(
        name="test_flow",
        description="A test flow for unit testing",
        inputs=[
            FlowInput(name="input_text", type="string", required=True),
            FlowInput(name="optional_param", type="string", required=False, default="default_value")
        ],
        steps=[
            FlowStep(
                name="step1",
                executor="test_executor",
                config={"message": "{{ inputs.input_text }}"}
            ),
            FlowStep(
                name="step2", 
                executor="another_executor",
                config={"data": "{{ steps.step1.result }}"},
                depends_on=["step1"]
            )
        ],
        outputs=[
            FlowOutput(name="final_result", value="{{ steps.step2.output }}")
        ]
    )


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client for testing."""
    mock_client = AsyncMock()
    mock_client.ping.return_value = True
    mock_client.set.return_value = True
    mock_client.get.return_value = '{"test": "data"}'
    mock_client.delete.return_value = 1
    mock_client.exists.return_value = 1
    return mock_client


@pytest.fixture
def mock_state_store(mock_redis_client):
    """Create a mock StateStore for testing."""
    store = StateStore()
    store.redis_client = mock_redis_client
    return store


@pytest.fixture
def sample_yaml_flow():
    """Sample YAML flow definition for testing."""
    return {
        "name": "test_flow",
        "description": "Test flow for unit testing",
        "inputs": [
            {
                "name": "input_text",
                "type": "string", 
                "required": True,
                "description": "Input text to process"
            }
        ],
        "steps": [
            {
                "name": "process_text",
                "executor": "llm_analyzer",
                "config": {
                    "text": "{{ inputs.input_text }}",
                    "prompt": "Analyze this text: {{ inputs.input_text }}"
                }
            }
        ],
        "outputs": [
            {
                "name": "analysis_result",
                "value": "{{ steps.process_text.analysis }}"
            }
        ]
    }


@pytest.fixture
def sample_image_file(temp_dir):
    """Create a sample image file for testing."""
    from PIL import Image
    
    # Create a simple test image
    img = Image.new('RGB', (100, 100), color='white')
    img_path = temp_dir / "test_image.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n174\n%%EOF"


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "content": "This is a test response from the LLM",
        "model": "test_model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        "finish_reason": "stop"
    }


@pytest.fixture
def mock_ocr_response():
    """Mock OCR response for testing."""
    return {
        "text": "This is extracted text from the image",
        "confidence": 0.95,
        "bounding_boxes": [
            {"text": "This", "bbox": [10, 10, 50, 30], "confidence": 0.98},
            {"text": "is", "bbox": [55, 10, 75, 30], "confidence": 0.97}
        ]
    }


# Test data constants
TEST_FLOW_ID = "test_flow_12345"
TEST_USER_ID = "test_user_123"
TEST_EXECUTION_ID = "exec_12345"

# Mock configuration for testing
TEST_CONFIG = {
    "redis_url": "redis://localhost:6379/0",
    "redis_ttl": 3600,
    "ollama_base_url": "http://localhost:11434",
    "max_file_size": 10485760,
    "supported_formats": [".pdf", ".docx", ".txt", ".jpg", ".png"]
}
