# Creating Custom Executors

Complete guide to developing custom AI executors for the PromptFlow platform, enabling you to extend the system with new AI capabilities and integrations.

## Overview

Executors are the core processing units of the PromptFlow platform. They encapsulate specific AI capabilities and can be combined in flows to create complex processing pipelines. This guide covers everything from basic executor creation to advanced patterns and best practices.

## Executor Architecture

### Base Executor Structure
All executors inherit from the `BaseExecutor` class and implement a standardized interface:

```python
from core.executors.base_executor import BaseExecutor, ExecutionResult, FlowContext
from typing import Dict, Any
import logging

class MyCustomExecutor(BaseExecutor):
    """
    Custom executor for specific AI processing task.
    
    This executor demonstrates the basic structure and patterns
    for creating new AI capabilities in the platform.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.name = name or "my_custom_executor"
        self.logger = logging.getLogger(__name__)
        
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Main execution method - implement your AI logic here.
        
        Args:
            context: Flow execution context with metadata
            config: Configuration parameters from the flow
            
        Returns:
            ExecutionResult with success status, outputs, and metadata
        """
        try:
            # Your AI processing logic here
            result = await self._process_data(config)
            
            return ExecutionResult(
                success=True,
                outputs=result,
                metadata={
                    "executor": self.name,
                    "processing_time": 1.23
                }
            )
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            return ExecutionResult(
                success=False,
                error=str(e)
            )
    
    async def _process_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Private method for actual processing logic."""
        # Implement your specific AI processing here
        return {"result": "processed_data"}
```

### ExecutionResult Structure
The `ExecutionResult` class provides a standardized response format:

```python
@dataclass
class ExecutionResult:
    success: bool                                    # Execution success status
    outputs: Dict[str, Any] = field(default_factory=dict)  # Processing results
    error: Optional[str] = None                      # Error message if failed
    metadata: Dict[str, Any] = field(default_factory=dict) # Additional metadata
    execution_time: Optional[float] = None           # Processing time in seconds
```

### FlowContext Structure
The `FlowContext` provides execution context and utilities:

```python
@dataclass
class FlowContext:
    flow_id: str                    # Unique flow execution ID
    step_name: str                  # Current step name
    execution_id: str               # Execution instance ID
    config: Dict[str, Any]          # Flow configuration
    metadata: Dict[str, Any]        # Execution metadata
    temp_dir: str                   # Temporary directory for files
```

## Development Patterns

### Simple Processing Executor
For basic data transformation and processing:

```python
class TextProcessorExecutor(BaseExecutor):
    """Simple text processing executor."""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.name = name or "text_processor"
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        try:
            text = config.get('text', '')
            operation = config.get('operation', 'uppercase')
            
            if not text:
                return ExecutionResult(
                    success=False,
                    error="Text input is required"
                )
            
            # Process text based on operation
            if operation == 'uppercase':
                result = text.upper()
            elif operation == 'lowercase':
                result = text.lower()
            elif operation == 'word_count':
                result = len(text.split())
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unsupported operation: {operation}"
                )
            
            return ExecutionResult(
                success=True,
                outputs={
                    "processed_text": result,
                    "original_length": len(text),
                    "operation_used": operation
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Text processing failed: {str(e)}"
            )
```

### AI Model Executor
For integrating machine learning models:

```python
import torch
from transformers import pipeline
from typing import Optional

class ModelBasedExecutor(BaseExecutor):
    """Executor that uses AI models for processing."""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.name = name or "model_executor"
        self.model = None
        self.model_cache = {}
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        try:
            # Extract configuration
            text = config.get('text')
            model_name = config.get('model_name', 'default-model')
            device = config.get('device', 'auto')
            
            if not text:
                return ExecutionResult(
                    success=False,
                    error="Text input is required"
                )
            
            # Load or get cached model
            model = await self._get_model(model_name, device)
            
            # Process with model
            start_time = time.time()
            results = model(text)
            processing_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                outputs={
                    "predictions": results,
                    "model_used": model_name,
                    "processing_time": processing_time
                },
                metadata={
                    "model_name": model_name,
                    "device": device,
                    "input_length": len(text)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Model execution failed: {str(e)}")
            return ExecutionResult(
                success=False,
                error=f"Model processing failed: {str(e)}"
            )
    
    async def _get_model(self, model_name: str, device: str):
        """Load or retrieve cached model."""
        cache_key = f"{model_name}_{device}"
        
        if cache_key not in self.model_cache:
            self.logger.info(f"Loading model: {model_name}")
            
            # Determine device
            if device == 'auto':
                device_id = 0 if torch.cuda.is_available() else -1
            elif device == 'cuda':
                device_id = 0
            else:
                device_id = -1
            
            # Load model
            model = pipeline(
                "text-classification",  # or your specific task
                model=model_name,
                device=device_id
            )
            
            self.model_cache[cache_key] = model
            self.logger.info(f"Model loaded successfully: {model_name}")
        
        return self.model_cache[cache_key]
```

### File Processing Executor
For handling file inputs and outputs:

```python
import tempfile
import os
from pathlib import Path

class FileProcessorExecutor(BaseExecutor):
    """Executor for processing file inputs."""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.name = name or "file_processor"
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        try:
            file_data = config.get('file')
            processing_type = config.get('processing_type', 'analyze')
            
            if not file_data:
                return ExecutionResult(
                    success=False,
                    error="File input is required"
                )
            
            # Handle different file input formats
            file_path = await self._handle_file_input(file_data, context.temp_dir)
            
            try:
                # Process the file
                result = await self._process_file(file_path, processing_type)
                
                return ExecutionResult(
                    success=True,
                    outputs=result,
                    metadata={
                        "file_processed": Path(file_path).name,
                        "processing_type": processing_type,
                        "file_size": os.path.getsize(file_path)
                    }
                )
                
            finally:
                # Clean up temporary file
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    
        except Exception as e:
            self.logger.error(f"File processing failed: {str(e)}")
            return ExecutionResult(
                success=False,
                error=f"File processing failed: {str(e)}"
            )
    
    async def _handle_file_input(self, file_data, temp_dir: str) -> str:
        """Handle different file input formats."""
        if isinstance(file_data, dict) and 'content' in file_data:
            # File uploaded as dictionary with content
            filename = file_data.get('filename', 'uploaded_file')
            file_ext = Path(filename).suffix or '.tmp'
            
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext, dir=temp_dir)
            try:
                with os.fdopen(temp_fd, 'wb') as tmp_file:
                    tmp_file.write(file_data['content'])
                return temp_path
            except Exception as e:
                os.close(temp_fd)
                raise e
                
        elif isinstance(file_data, (str, Path)):
            # File path provided
            file_path = str(file_data)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            return file_path
            
        else:
            raise ValueError("Invalid file input format")
    
    async def _process_file(self, file_path: str, processing_type: str) -> Dict[str, Any]:
        """Process the file based on type."""
        file_info = {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "file_extension": Path(file_path).suffix
        }
        
        if processing_type == 'analyze':
            # Analyze file content
            with open(file_path, 'rb') as f:
                content = f.read()
                file_info.update({
                    "content_length": len(content),
                    "content_type": self._detect_content_type(content)
                })
        
        return file_info
    
    def _detect_content_type(self, content: bytes) -> str:
        """Simple content type detection."""
        if content.startswith(b'\xff\xd8\xff'):
            return "image/jpeg"
        elif content.startswith(b'\x89PNG'):
            return "image/png"
        elif content.startswith(b'%PDF'):
            return "application/pdf"
        else:
            return "application/octet-stream"
```

### Multi-Provider Executor
For supporting multiple AI providers:

```python
from abc import ABC, abstractmethod
from typing import Union

class MultiProviderExecutor(BaseExecutor):
    """Executor supporting multiple AI providers."""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.name = name or "multi_provider_executor"
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available providers."""
        self.providers = {
            'openai': OpenAIProvider(),
            'huggingface': HuggingFaceProvider(),
            'local': LocalProvider()
        }
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        try:
            provider_name = config.get('provider', 'local').lower()
            
            if provider_name not in self.providers:
                return ExecutionResult(
                    success=False,
                    error=f"Unsupported provider: {provider_name}"
                )
            
            provider = self.providers[provider_name]
            
            # Process with selected provider
            result = await provider.process(config)
            
            return ExecutionResult(
                success=True,
                outputs=result,
                metadata={
                    "provider_used": provider_name,
                    "provider_version": provider.get_version()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Multi-provider execution failed: {str(e)}")
            return ExecutionResult(
                success=False,
                error=f"Processing failed: {str(e)}"
            )

class BaseProvider(ABC):
    """Base class for AI providers."""
    
    @abstractmethod
    async def process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process data with this provider."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get provider version."""
        pass

class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""
    
    async def process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Implement OpenAI API processing
        return {"result": "openai_processed", "provider": "openai"}
    
    def get_version(self) -> str:
        return "openai-1.0"

class HuggingFaceProvider(BaseProvider):
    """HuggingFace models provider."""
    
    async def process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Implement HuggingFace processing
        return {"result": "huggingface_processed", "provider": "huggingface"}
    
    def get_version(self) -> str:
        return "transformers-4.21.0"

class LocalProvider(BaseProvider):
    """Local processing provider."""
    
    async def process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Implement local processing
        return {"result": "local_processed", "provider": "local"}
    
    def get_version(self) -> str:
        return "local-1.0"
```

## Advanced Patterns

### Async Processing with Queues
For long-running or resource-intensive tasks:

```python
import asyncio
from asyncio import Queue

class AsyncQueueExecutor(BaseExecutor):
    """Executor with async queue processing."""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.name = name or "async_queue_executor"
        self.processing_queue = Queue(maxsize=100)
        self.workers_started = False
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        try:
            # Start workers if not already started
            if not self.workers_started:
                await self._start_workers()
            
            # Add task to queue
            task_id = f"{context.execution_id}_{context.step_name}"
            await self.processing_queue.put({
                'task_id': task_id,
                'config': config,
                'context': context
            })
            
            # For async processing, return immediately with task ID
            return ExecutionResult(
                success=True,
                outputs={
                    "task_id": task_id,
                    "status": "queued",
                    "queue_size": self.processing_queue.qsize()
                },
                metadata={
                    "async_processing": True,
                    "executor": self.name
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Queue processing failed: {str(e)}"
            )
    
    async def _start_workers(self, num_workers: int = 3):
        """Start background worker tasks."""
        for i in range(num_workers):
            asyncio.create_task(self._worker(f"worker-{i}"))
        self.workers_started = True
    
    async def _worker(self, worker_name: str):
        """Background worker for processing queued tasks."""
        while True:
            try:
                # Get task from queue
                task = await self.processing_queue.get()
                
                # Process task
                result = await self._process_task(task)
                
                # Mark task as done
                self.processing_queue.task_done()
                
                self.logger.info(f"{worker_name} completed task {task['task_id']}")
                
            except Exception as e:
                self.logger.error(f"{worker_name} error: {str(e)}")
    
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual task."""
        # Implement your actual processing logic here
        await asyncio.sleep(1)  # Simulate processing time
        return {"processed": True, "task_id": task['task_id']}
```

### Batch Processing Executor
For processing multiple items efficiently:

```python
class BatchProcessorExecutor(BaseExecutor):
    """Executor for batch processing multiple items."""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.name = name or "batch_processor"
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        try:
            items = config.get('items', [])
            batch_size = config.get('batch_size', 10)
            parallel_processing = config.get('parallel_processing', True)
            
            if not items:
                return ExecutionResult(
                    success=False,
                    error="Items list is required for batch processing"
                )
            
            # Process in batches
            results = []
            total_items = len(items)
            
            for i in range(0, total_items, batch_size):
                batch = items[i:i + batch_size]
                
                if parallel_processing:
                    batch_results = await self._process_batch_parallel(batch, config)
                else:
                    batch_results = await self._process_batch_sequential(batch, config)
                
                results.extend(batch_results)
                
                # Progress reporting
                processed_count = min(i + batch_size, total_items)
                self.logger.info(f"Processed {processed_count}/{total_items} items")
            
            return ExecutionResult(
                success=True,
                outputs={
                    "results": results,
                    "total_processed": len(results),
                    "batch_size": batch_size,
                    "parallel_processing": parallel_processing
                },
                metadata={
                    "total_items": total_items,
                    "batches_processed": (total_items + batch_size - 1) // batch_size
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Batch processing failed: {str(e)}"
            )
    
    async def _process_batch_parallel(self, batch: list, config: Dict[str, Any]) -> list:
        """Process batch items in parallel."""
        tasks = [self._process_item(item, config) for item in batch]
        return await asyncio.gather(*tasks)
    
    async def _process_batch_sequential(self, batch: list, config: Dict[str, Any]) -> list:
        """Process batch items sequentially."""
        results = []
        for item in batch:
            result = await self._process_item(item, config)
            results.append(result)
        return results
    
    async def _process_item(self, item: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual item."""
        # Implement your item processing logic here
        return {
            "item": item,
            "processed": True,
            "timestamp": time.time()
        }
```

## Integration and Registration

### Registering Your Executor
Add your executor to the system registry:

```python
# In core/executors/__init__.py
from .my_custom_executor import MyCustomExecutor

# Register in the executor registry
def register_executors(registry):
    """Register all available executors."""
    # ... existing registrations ...
    registry.register("my_custom_executor", MyCustomExecutor)
```

### Creating Flow Integration
Create a flow that uses your executor:

```yaml
# flows/my_custom_flow/flow.yaml
name: "my_custom_flow"
version: "1.0.0"
description: "Flow using my custom executor"

inputs:
  - name: "input_data"
    type: "string"
    required: true
    description: "Data to process with custom executor"

steps:
  - name: "custom_processing"
    executor: "my_custom_executor"
    config:
      data: "{{ inputs.input_data }}"
      processing_options:
        mode: "advanced"
        quality: "high"

config:
  execution:
    mode: "async"
    timeout: 300
```

### Docker Integration
Update Docker configuration for new dependencies:

```dockerfile
# Add to requirements.txt
your-new-dependency==1.0.0
another-package>=2.0.0

# Update Dockerfile if needed
RUN pip install your-special-package
```

## Testing Your Executor

### Unit Tests
Create comprehensive unit tests:

```python
import pytest
from unittest.mock import Mock, patch
from core.executors.my_custom_executor import MyCustomExecutor
from core.executors.base_executor import FlowContext, ExecutionResult

class TestMyCustomExecutor:
    
    @pytest.fixture
    def executor(self):
        return MyCustomExecutor()
    
    @pytest.fixture
    def mock_context(self):
        return FlowContext(
            flow_id="test-flow-123",
            step_name="test_step",
            execution_id="exec-456",
            config={},
            metadata={},
            temp_dir="/tmp/test"
        )
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, executor, mock_context):
        """Test successful executor execution."""
        config = {
            "text": "test input",
            "operation": "uppercase"
        }
        
        result = await executor.execute(mock_context, config)
        
        assert result.success is True
        assert "processed_text" in result.outputs
        assert result.outputs["processed_text"] == "TEST INPUT"
    
    @pytest.mark.asyncio
    async def test_missing_input_error(self, executor, mock_context):
        """Test error handling for missing input."""
        config = {}  # Missing required text input
        
        result = await executor.execute(mock_context, config)
        
        assert result.success is False
        assert "required" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_operation_error(self, executor, mock_context):
        """Test error handling for invalid operation."""
        config = {
            "text": "test input",
            "operation": "invalid_operation"
        }
        
        result = await executor.execute(mock_context, config)
        
        assert result.success is False
        assert "unsupported operation" in result.error.lower()
    
    @pytest.mark.asyncio
    @patch('your_module.external_api_call')
    async def test_external_api_integration(self, mock_api, executor, mock_context):
        """Test integration with external APIs."""
        mock_api.return_value = {"result": "api_response"}
        
        config = {"api_endpoint": "https://api.example.com/process"}
        
        result = await executor.execute(mock_context, config)
        
        assert result.success is True
        mock_api.assert_called_once()
```

### Integration Tests
Test your executor in real flow scenarios:

```python
import pytest
from core.flow_engine.flow_runner import FlowRunner
from core.executors.executor_registry import ExecutorRegistry

class TestMyCustomExecutorIntegration:
    
    @pytest.fixture
    def flow_runner(self):
        registry = ExecutorRegistry()
        registry.register("my_custom_executor", MyCustomExecutor)
        return FlowRunner(registry)
    
    @pytest.mark.asyncio
    async def test_flow_execution(self, flow_runner):
        """Test executor within a complete flow."""
        flow_config = {
            "name": "test_flow",
            "inputs": [
                {"name": "text", "type": "string", "required": True}
            ],
            "steps": [
                {
                    "name": "process",
                    "executor": "my_custom_executor",
                    "config": {
                        "text": "{{ inputs.text }}",
                        "operation": "uppercase"
                    }
                }
            ]
        }
        
        inputs = {"text": "hello world"}
        
        result = await flow_runner.execute_flow(flow_config, inputs)
        
        assert result.success is True
        assert result.outputs["process"]["processed_text"] == "HELLO WORLD"
```

## Best Practices

### Error Handling
1. **Comprehensive Error Handling**: Always wrap execution in try-catch blocks
2. **Meaningful Error Messages**: Provide clear, actionable error messages
3. **Error Classification**: Distinguish between user errors and system errors
4. **Graceful Degradation**: Provide fallback behavior when possible

### Performance Optimization
1. **Resource Management**: Properly manage memory and file resources
2. **Caching**: Cache expensive operations and model loading
3. **Async Operations**: Use async/await for I/O operations
4. **Batch Processing**: Support batch processing for efficiency

### Security Considerations
1. **Input Validation**: Validate all inputs thoroughly
2. **File Handling**: Safely handle file uploads and temporary files
3. **API Security**: Secure external API integrations
4. **Resource Limits**: Implement appropriate resource limits

### Code Quality
1. **Documentation**: Provide comprehensive docstrings and comments
2. **Type Hints**: Use type hints for better code clarity
3. **Logging**: Implement appropriate logging for debugging
4. **Testing**: Write comprehensive unit and integration tests

### Configuration Management
1. **Flexible Configuration**: Support various configuration options
2. **Default Values**: Provide sensible defaults for optional parameters
3. **Validation**: Validate configuration parameters
4. **Documentation**: Document all configuration options

## Deployment Considerations

### Container Updates
When adding new executors, update the container configuration:

```yaml
# docker-compose.yml
services:
  celery-worker-custom:
    build: .
    command: celery -A celery_app worker --loglevel=info --queues=custom_processing
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis
    volumes:
      - ./uploads:/app/uploads
```

### Resource Requirements
Document resource requirements for your executor:

```yaml
# Resource requirements documentation
resources:
  memory: "2GB"        # Minimum memory requirement
  cpu: "1 core"        # CPU requirements
  gpu: "optional"      # GPU requirements
  disk: "1GB"          # Temporary disk space
  
dependencies:
  - "torch>=1.9.0"
  - "transformers>=4.20.0"
  - "custom-package==1.0.0"
```

### Monitoring and Observability
Add monitoring capabilities:

```python
import time
from typing import Dict, Any

class MonitoredExecutor(BaseExecutor):
    """Executor with built-in monitoring."""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.metrics = {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0.0
        }
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        start_time = time.time()
        self.metrics["executions"] += 1
        
        try:
            result = await self._execute_impl(context, config)
            
            if result.success:
                self.metrics["successes"] += 1
            else:
                self.metrics["failures"] += 1
                
            return result
            
        except Exception as e:
            self.metrics["failures"] += 1
            raise e
            
        finally:
            execution_time = time.time() - start_time
            self.metrics["total_time"] += execution_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get executor performance metrics."""
        return {
            **self.metrics,
            "average_time": self.metrics["total_time"] / max(self.metrics["executions"], 1),
            "success_rate": self.metrics["successes"] / max(self.metrics["executions"], 1)
        }
```

This comprehensive guide provides everything needed to create custom executors for the PromptFlow platform. Follow these patterns and best practices to build robust, scalable AI capabilities that integrate seamlessly with the existing system.
