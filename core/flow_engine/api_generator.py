"""
API Generator

Automatically generates FastAPI endpoints from YAML flow definitions.
Creates POST endpoints for flow execution, GET endpoints for flow information,
and handles input validation, file uploads, and response formatting.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING
from pydantic import BaseModel, Field, create_model
import logging
import tempfile
import uuid
from pathlib import Path
from datetime import datetime, timezone

from .yaml_loader import FlowDefinition, FlowInput
from .celery_config import should_execute_async

if TYPE_CHECKING:
    from .flow_runner import FlowRunner

logger = logging.getLogger(__name__)


class FlowAPIGenerator:
    """
    Generate FastAPI endpoints from YAML flow definitions.
    
    Creates standardized API endpoints with proper validation,
    file handling, and response formatting for each flow.
    """
    
    def __init__(self, flow_runner: "FlowRunner"):
        self.flow_runner = flow_runner
        self.logger = logging.getLogger(__name__)
    
    def generate_router_for_flow(self, flow_name: str) -> APIRouter:
        """
        Generate a complete FastAPI router for a specific flow.
        
        Args:
            flow_name: Name of the flow to generate router for
            
        Returns:
            FastAPI router with all endpoints for the flow
        """
        flow_def = self.flow_runner.get_flow(flow_name)
        if not flow_def:
            raise ValueError(f"Flow '{flow_name}' not found")
        
        # Create router with flow-specific prefix
        router = APIRouter(
            prefix=f"/api/v1/{flow_name.replace('_', '-')}",
            tags=[f"{flow_name.replace('_', ' ').title()} Flow"]
        )
        
        # Generate endpoints
        self._add_execution_endpoint(router, flow_def)
        self._add_info_endpoint(router, flow_def)
        self._add_health_endpoint(router, flow_def)
        self._add_supported_formats_endpoint(router, flow_def)
        
        # Add async-specific endpoints if flow supports async execution
        execution_config = flow_def.config.get('execution', {})
        if execution_config.get('mode') in ['async', 'auto']:
            self._add_async_management_endpoints(router, flow_def)
        
        return router
    
    def generate_all_routers(self) -> List[APIRouter]:
        """
        Generate routers for all available flows.
        
        Returns:
            List of FastAPI routers for all flows
        """
        routers = []
        
        for flow_name in self.flow_runner.list_flows():
            try:
                router = self.generate_router_for_flow(flow_name)
                routers.append(router)
                self.logger.info(f"Generated API router for flow: {flow_name}")
            except Exception as e:
                self.logger.error(f"Failed to generate router for flow {flow_name}: {e}")
        
        return routers
    
    def _add_execution_endpoint(self, router: APIRouter, flow_def: FlowDefinition):
        """Add main flow execution endpoint with sync/async support based on YAML config."""
        
        # Determine if flow needs file upload
        has_file_input = any(inp.type == "file" for inp in flow_def.inputs)
        
        if has_file_input:
            self._add_file_upload_endpoint(router, flow_def)
        else:
            self._add_json_endpoint(router, flow_def)
    
    def _add_file_upload_endpoint(self, router: APIRouter, flow_def: FlowDefinition):
        """Add file upload endpoint for flows that accept files with sync/async support."""
        
        # Get non-file inputs for form parameters
        form_inputs = [inp for inp in flow_def.inputs if inp.type != "file"]
        
        from fastapi import Request
        
        @router.post("/execute", 
                    summary=f"Execute {flow_def.name} flow",
                    description=flow_def.description,
                    response_model=Dict[str, Any])
        async def execute_flow_with_file(
            request: Request,
            file: UploadFile = File(..., description="File to process"),
            callback_url: Optional[str] = Form(None, description="Callback URL for async execution")
        ):
            try:
                # Read file content as bytes
                file_content = await file.read()
                
                # Prepare inputs with proper file structure
                inputs = {
                    "file": {
                        "content": file_content,
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "size": len(file_content)
                    }
                }
                
                # Parse form data from the request
                form_data = {}
                try:
                    form = await request.form()
                    for key, value in form.items():
                        if key not in ['file', 'callback_url']:  # Skip special fields
                            form_data[key] = value
                except Exception as e:
                    self.logger.warning(f"Could not parse form data: {e}")
                
                # Add form parameters to inputs with proper type conversion
                for inp in form_inputs:
                    value = form_data.get(inp.name)
                    
                    if value is not None:
                        # Convert form data to proper types
                        if inp.type == "boolean":
                            inputs[inp.name] = value.lower() in ('true', '1', 'yes', 'on') if isinstance(value, str) else bool(value)
                        elif inp.type == "integer":
                            inputs[inp.name] = int(value) if isinstance(value, str) else value
                        elif inp.type == "number":
                            inputs[inp.name] = float(value) if isinstance(value, str) else value
                        elif inp.type == "array" and isinstance(value, str):
                            # Convert comma-separated string to array
                            inputs[inp.name] = [item.strip() for item in value.split(',') if item.strip()]
                        else:
                            inputs[inp.name] = value
                    elif inp.default is not None:
                        # Use default value if no form data provided
                        inputs[inp.name] = inp.default
                        self.logger.info(f"Using default value for '{inp.name}': {inp.default}")
                #FIXME: Activate again in the future
                #self.logger.info(f"Flow {flow_def.name} received inputs: {list(inputs.keys())}")
                for key, value in inputs.items():
                    if key != 'file':  # Don't log file content
                        self.logger.info(f"  {key}: {value}")
                
                # Validate inputs against flow definition
                validated_inputs = self._validate_inputs(inputs, flow_def)
                
                # Determine execution mode (sync vs async)
                should_async = should_execute_async(flow_def, validated_inputs)
                
                if should_async:
                    # Async execution
                    flow_id = str(uuid.uuid4())
                    
                    # Check if first step is file handler - execute it synchronously
                    first_step = flow_def.steps[0] if flow_def.steps else None
                    processed_inputs = validated_inputs.copy()
                    
                    if first_step and first_step.executor == "file_handler":
                        logger.info(f"Executing file handler step synchronously for flow {flow_id}")
                        
                        try:
                            # Execute file handler synchronously
                            file_outputs = await self._execute_file_handler_sync(first_step, validated_inputs)
                            
                            # COMPLETELY REMOVE binary content and replace with file metadata + temp path
                            if 'file' in processed_inputs:
                                processed_inputs['file'] = {
                                    'filename': processed_inputs['file']['filename'],
                                    'content_type': processed_inputs['file']['content_type'],
                                    'size': processed_inputs['file']['size'],
                                    'temp_path': file_outputs.get('temp_path')
                                    # NOTE: 'content' field is intentionally removed to eliminate binary data
                                }
                            
                            # Add file handler outputs to processed inputs for template resolution
                            processed_inputs['file_handler_outputs'] = file_outputs
                            
                            # Add step outputs for template resolution in remaining steps
                            processed_inputs['steps'] = {
                                'handle_file': file_outputs
                            }
                            
                            logger.info(f"File handler completed, temp file: {file_outputs.get('temp_path')}")
                            logger.info(f"Binary content removed from processed_inputs for Celery")
                            
                        except Exception as e:
                            logger.error(f"File handler failed: {e}")
                            raise HTTPException(status_code=400, detail=f"File processing failed: {str(e)}")
                    
                    # CREATE REDIS RECORD FIRST (before Celery submission) with processed inputs
                    print(f"DEBUG: About to initialize flow state for {flow_id} with status 'queued'")
                    await self.flow_runner._initialize_flow_state(flow_id, flow_def.name, processed_inputs, status="queued")
                    print(f"DEBUG: Flow state initialization completed for {flow_id}")
                    
                    # DEBUG: Verify binary content removal for file upload endpoint
                    if 'file' in processed_inputs:
                        file_data = processed_inputs['file']
                        has_content = 'content' in file_data
                        has_temp_path = 'temp_path' in file_data
                        logger.info(f"DEBUG: File upload - Processed inputs for Celery - has_content: {has_content}, has_temp_path: {has_temp_path}")
                        if has_temp_path:
                            logger.info(f"DEBUG: File upload - Temp path: {file_data['temp_path']}")
                    
                    # THEN submit to Celery with processed inputs and modified flow
                    from celery_app import execute_flow_async
                    
                    # Create modified flow definition without file handler step if it was processed
                    flow_for_celery = flow_def
                    if first_step and first_step.executor == "file_handler":
                        # Create a copy of flow definition without the first step
                        import copy
                        flow_for_celery = copy.deepcopy(flow_def)
                        flow_for_celery.steps = flow_def.steps[1:]  # Skip file handler step
                        logger.info(f"Modified flow for Celery: {len(flow_for_celery.steps)} steps (file handler excluded)")
                    
                    task = execute_flow_async.apply_async(
                        args=[flow_for_celery.name, processed_inputs, flow_id, callback_url],
                        queue=flow_def.name  # Route to flow-specific queue
                    )
                    
                    # Get execution config for response
                    execution_config = flow_def.config.get('execution', {})
                    
                    return {
                        "flow_id": flow_id,
                        "task_id": task.id,
                        "status": "queued",
                        "execution_mode": "async",
                        "queue": flow_def.name,
                        "worker": f"celery-worker-{flow_def.name}",
                        "callback_url": callback_url,
                        
                        "urls": {
                            "status": f"/api/v1/{flow_def.name.replace('_', '-')}/status/{flow_id}",
                            "cancel": f"/api/v1/{flow_def.name.replace('_', '-')}/cancel/{flow_id}"
                        },
                        
                        "config": {
                            "auto_resume": execution_config.get('auto_resume', True),
                            "max_retries": execution_config.get('retry_count', 3),
                            "timeout": execution_config.get('timeout', 300),
                            "callback_enabled": flow_def.config.get('callbacks', {}).get('enabled', False),
                            "max_concurrent": execution_config.get('max_concurrent', 2)
                        }
                    }
                else:
                    # Sync execution (existing behavior)
                    result = await self.flow_runner.run_flow(flow_def.name, validated_inputs)
                    return result
                
            except Exception as e:
                self.logger.error(f"Flow execution failed: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
    
    def _add_json_endpoint(self, router: APIRouter, flow_def: FlowDefinition):
        """Add JSON endpoint for flows that don't need file uploads with sync/async support."""
        
        # Create Pydantic model for request validation
        request_model = self._create_request_model(flow_def)
        
        @router.post(f"/execute",
                    summary=f"Execute {flow_def.name} flow", 
                    description=flow_def.description,
                    response_model=Dict[str, Any])
        async def execute_flow_json(
            request: request_model,
            callback_url: Optional[str] = None
        ):
            try:
                # Convert request to dict
                inputs = request.dict()
                
                # Determine execution mode (sync vs async)
                should_async = should_execute_async(flow_def, inputs)
                
                if should_async:
                    # Async execution
                    flow_id = str(uuid.uuid4())
                    
                    # Check if first step is file handler - execute it synchronously
                    first_step = flow_def.steps[0] if flow_def.steps else None
                    processed_inputs = inputs.copy()
                    
                    if first_step and first_step.executor == "file_handler":
                        logger.info(f"Executing file handler step synchronously for flow {flow_id}")
                        
                        try:
                            # Execute file handler synchronously
                            file_outputs = await self._execute_file_handler_sync(first_step, inputs)
                            
                            # COMPLETELY REMOVE binary content and replace with file metadata + temp path
                            if 'file' in processed_inputs:
                                processed_inputs['file'] = {
                                    'filename': processed_inputs['file']['filename'],
                                    'content_type': processed_inputs['file']['content_type'],
                                    'size': processed_inputs['file']['size'],
                                    'temp_path': file_outputs.get('temp_path')
                                    # NOTE: 'content' field is intentionally removed to eliminate binary data
                                }
                            
                            # Add file handler outputs to processed inputs for template resolution
                            processed_inputs['file_handler_outputs'] = file_outputs
                            
                            # Add step outputs for template resolution in remaining steps
                            processed_inputs['steps'] = {
                                'handle_file': file_outputs
                            }
                            
                            logger.info(f"File handler completed, temp file: {file_outputs.get('temp_path')}")
                            
                        except Exception as e:
                            logger.error(f"File handler failed: {e}")
                            raise HTTPException(status_code=400, detail=f"File processing failed: {str(e)}")
                    
                    # CREATE REDIS RECORD FIRST (before Celery submission) with processed inputs
                    print(f"DEBUG: About to initialize flow state for {flow_id} with status 'queued'")
                    await self.flow_runner._initialize_flow_state(flow_id, flow_def.name, processed_inputs, status="queued")
                    print(f"DEBUG: Flow state initialization completed for {flow_id}")
                    
                    # DEBUG: Verify binary content removal for JSON endpoint
                    if 'file' in processed_inputs:
                        file_data = processed_inputs['file']
                        has_content = 'content' in file_data
                        has_temp_path = 'temp_path' in file_data
                        logger.info(f"DEBUG: JSON endpoint - Processed inputs for Celery - has_content: {has_content}, has_temp_path: {has_temp_path}")
                        if has_temp_path:
                            logger.info(f"DEBUG: JSON endpoint - Temp path: {file_data['temp_path']}")
                    
                    # THEN submit to Celery with processed inputs and modified flow
                    from celery_app import execute_flow_async
                    
                    # Create modified flow definition without file handler step if it was processed
                    flow_for_celery = flow_def
                    if first_step and first_step.executor == "file_handler":
                        # Create a copy of flow definition without the first step
                        import copy
                        flow_for_celery = copy.deepcopy(flow_def)
                        flow_for_celery.steps = flow_def.steps[1:]  # Skip file handler step
                        logger.info(f"Modified flow for Celery: {len(flow_for_celery.steps)} steps (file handler excluded)")
                    
                    task = execute_flow_async.apply_async(
                        args=[flow_for_celery.name, processed_inputs, flow_id, callback_url],
                        queue=flow_def.name  # Route to flow-specific queue
                    )
                    
                    # Get execution config for response
                    execution_config = flow_def.config.get('execution', {})
                    
                    return {
                        "flow_id": flow_id,
                        "task_id": task.id,
                        "status": "queued",
                        "execution_mode": "async",
                        "queue": flow_def.name,
                        "worker": f"celery-worker-{flow_def.name}",
                        "callback_url": callback_url,
                        
                        "urls": {
                            "status": f"/api/v1/{flow_def.name.replace('_', '-')}/status/{flow_id}",
                            "cancel": f"/api/v1/{flow_def.name.replace('_', '-')}/cancel/{flow_id}"
                        },
                        
                        "config": {
                            "auto_resume": execution_config.get('auto_resume', True),
                            "max_retries": execution_config.get('retry_count', 3),
                            "timeout": execution_config.get('timeout', 300),
                            "callback_enabled": flow_def.config.get('callbacks', {}).get('enabled', False),
                            "max_concurrent": execution_config.get('max_concurrent', 2)
                        }
                    }
                else:
                    # Sync execution (existing behavior)
                    result = await self.flow_runner.run_flow(flow_def.name, inputs)
                    return result
                
            except Exception as e:
                self.logger.error(f"Flow execution failed: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
    
    def _add_async_management_endpoints(self, router: APIRouter, flow_def: FlowDefinition):
        """Add async flow management endpoints (status, cancel, resume)."""
        
        @router.get("/status/{flow_id}",
                   summary=f"Get {flow_def.name} flow status",
                   response_model=Dict[str, Any])
        async def get_flow_status(flow_id: str):
            """Get detailed status of an async flow execution."""
            try:
                # Get Celery task info
                from celery_app import celery_app
                task = celery_app.AsyncResult(flow_id)
                
                # Get detailed flow state from Redis
                flow_state = await self.flow_runner.state_store.get_flow_state(flow_id)
                
                if not flow_state:
                    raise HTTPException(status_code=404, detail=f"Flow {flow_id} not found")
                
                # Calculate progress
                total_steps = len(flow_state.steps)
                completed_steps = [s for s in flow_state.steps if s.status == "completed"]
                failed_steps = [s for s in flow_state.steps if s.status == "failed"]
                
                progress_percentage = (len(completed_steps) / total_steps * 100) if total_steps > 0 else 0
                
                return {
                    "flow_id": flow_id,
                    "status": flow_state.status if flow_state else "unknown",
                    
                    # Progress information
                    "progress": {
                        "percentage": progress_percentage,
                        "current_step": flow_state.current_step if flow_state else None,
                        "completed_steps": [s.step_name for s in completed_steps],
                        "failed_steps": [s.step_name for s in failed_steps]
                    },
                    
                    # Execution details
                    "execution": {
                        "started_at": flow_state.started_at.isoformat() if flow_state and flow_state.started_at else None,
                        "queue": flow_state.flow_name,
                        "worker": f"celery-worker-{flow_state.flow_name}",
                        "retry_count": flow_state.metadata.get("retry_count", 0) if flow_state and flow_state.metadata else 0
                    },
                    
                    # Callback information
                    "callback": {
                        "url": flow_state.callback_url if flow_state else None,
                        "status": flow_state.callback_status if flow_state else None,
                        "enabled": flow_def.config.get('callbacks', {}).get('enabled', False)
                    },
                    
                    # Available actions
                    "actions": {
                        "cancel_url": f"/api/v1/{flow_def.name.replace('_', '-')}/cancel/{flow_id}" if task.state in ['PENDING', 'STARTED'] else None,
                        "resume_url": f"/api/v1/{flow_def.name.replace('_', '-')}/resume/{flow_id}" if flow_state and flow_state.status == "failed" else None
                    }
                }
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get flow status: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.post("/cancel/{flow_id}",
                    summary=f"Cancel {flow_def.name} flow execution",
                    response_model=Dict[str, Any])
        async def cancel_flow_execution(flow_id: str):
            """Cancel a running async flow execution."""
            try:
                # Cancel Celery task
                from celery_app import celery_app, revoke_task
                revoke_task(flow_id, terminate=True)
                
                # Cancel flow in our system
                result = await self.flow_runner.cancel_flow(flow_id)
                
                return {
                    "flow_id": flow_id,
                    "status": "cancelled",
                    "message": f"Flow {flow_id} cancelled successfully",
                    "result": result
                }
                
            except Exception as e:
                self.logger.error(f"Failed to cancel flow: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.post("/resume/{flow_id}",
                    summary=f"Resume {flow_def.name} flow execution",
                    response_model=Dict[str, Any])
        async def resume_flow_execution(flow_id: str):
            """Resume a failed async flow execution."""
            try:
                # Submit resume task to Celery
                from celery_app import resume_flow_async
                task = resume_flow_async.delay(flow_id)
                
                return {
                    "flow_id": flow_id,
                    "task_id": task.id,
                    "status": "resuming",
                    "message": f"Flow {flow_id} resumption started"
                }
                
            except Exception as e:
                self.logger.error(f"Failed to resume flow: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
    
    def _add_info_endpoint(self, router: APIRouter, flow_def: FlowDefinition):
        """Add flow information endpoint."""
        
        @router.get("/info",
                   summary=f"Get {flow_def.name} flow information",
                   response_model=Dict[str, Any])
        async def get_flow_info():
            """Get detailed information about this flow."""
            try:
                return self.flow_runner.get_flow_info(flow_def.name)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def _add_health_endpoint(self, router: APIRouter, flow_def: FlowDefinition):
        """Add flow health check endpoint."""
        
        @router.get("/health",
                   summary=f"Check {flow_def.name} flow health",
                   response_model=Dict[str, Any])
        async def check_flow_health():
            """Check if flow and its dependencies are healthy."""
            try:
                health_status = {
                    "flow": flow_def.name,
                    "status": "healthy",
                    "version": flow_def.version,
                    "steps": len(flow_def.steps),
                    "executors": {}
                }
                
                # Check executor health
                for step in flow_def.steps:
                    executor_name = step.executor
                    try:
                        executor = self.flow_runner.executor_registry.get_executor(executor_name)
                        health_status["executors"][executor_name] = {
                            "status": "available",
                            "info": executor.get_info()
                        }
                    except Exception as e:
                        health_status["executors"][executor_name] = {
                            "status": "error",
                            "error": str(e)
                        }
                        health_status["status"] = "degraded"
                
                return health_status
                
            except Exception as e:
                return {
                    "flow": flow_def.name,
                    "status": "error",
                    "error": str(e)
                }
    
    def _add_supported_formats_endpoint(self, router: APIRouter, flow_def: FlowDefinition):
        """Add supported formats endpoint for flows that process files."""
        
        # Check if flow has file processing steps
        has_file_processing = any(
            step.executor in ["document_extractor", "ocr_processor", "image_handler", "file_handler"]
            for step in flow_def.steps
        )
        
        if has_file_processing:
            @router.get("/supported-formats",
                       summary=f"Get supported file formats for {flow_def.name}",
                       response_model=Dict[str, Any])
            async def get_supported_formats():
                """Get list of supported file formats for this flow."""
                try:
                    formats = {
                        "flow": flow_def.name,
                        "supported_formats": []
                    }
                    
                    # Get formats from relevant executors
                    for step in flow_def.steps:
                        if step.executor == "document_extractor":
                            executor = self.flow_runner.executor_registry.get_executor("document_extractor")
                            if hasattr(executor, 'get_supported_formats'):
                                formats["document_formats"] = executor.get_supported_formats()
                        
                        elif step.executor == "ocr_processor":
                            formats["image_formats"] = [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"]
                        
                        elif step.executor == "image_handler":
                            formats["image_processing_formats"] = ["JPEG", "PNG", "TIFF", "BMP", "GIF"]
                    
                    return formats
                    
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
    
    def _validate_inputs(self, inputs: Dict[str, Any], flow_def: FlowDefinition) -> Dict[str, Any]:
        """
        Validate flow inputs against the flow definition.
        
        Args:
            inputs: Input parameters to validate
            flow_def: Flow definition containing input schema
            
        Returns:
            Validated inputs dictionary
        """
        validated = {}
        
        for flow_input in flow_def.inputs:
            value = inputs.get(flow_input.name)
            
            # Check required inputs
            if flow_input.required and value is None:
                raise ValueError(f"Required input '{flow_input.name}' is missing")
            
            # Use default value if not provided
            if value is None and flow_input.default is not None:
                value = flow_input.default
            
            # Validate enum values
            if value is not None and flow_input.enum and value not in flow_input.enum:
                raise ValueError(f"Input '{flow_input.name}' must be one of {flow_input.enum}, got '{value}'")
            
            if value is not None:
                validated[flow_input.name] = value
        
        return validated

    async def _execute_file_handler_sync(self, step, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute file handler step synchronously in the API endpoint.
        
        Args:
            step: The file handler step definition
            inputs: Flow inputs containing file data
            
        Returns:
            Dictionary containing file handler outputs (temp_path, filename, etc.)
        """
        from ..executors.file_handler import FileHandler
        from ..executors.base_executor import FlowContext
        
        # Create a temporary flow context for the file handler
        context = FlowContext(
            flow_name="temp_file_handler",
            inputs=inputs,
            flow_id="temp_file_handler"
        )
        
        # Set current step for context
        context.current_step = step.name
        
        # Initialize file handler executor
        file_handler = FileHandler(step.name)
        
        # Preserve temp files for async processing (they'll be cleaned up by the worker)
        file_handler.preserve_temp_files = True
        
        # Prepare config by resolving template variables
        config = {}
        for key, value in step.config.items():
            if isinstance(value, str) and "{{" in value:
                # Simple template resolution for file handler
                if value == "{{ inputs.file.content }}":
                    file_data = inputs.get('file', {})
                    # Handle both binary content and temp path scenarios
                    if 'content' in file_data:
                        config[key] = file_data.get('content')
                    elif 'temp_path' in file_data:
                        # If we already have a temp path, read the file content
                        import pathlib
                        temp_path = pathlib.Path(file_data['temp_path'])
                        if temp_path.exists():
                            config[key] = temp_path.read_bytes()
                        else:
                            raise ValueError(f"Temp file not found: {temp_path}")
                    else:
                        raise ValueError("No file content or temp path available")
                elif value == "{{ inputs.file.filename }}":
                    config[key] = inputs.get('file', {}).get('filename')
                elif value == "{{ inputs.file.size }}":
                    config[key] = inputs.get('file', {}).get('size')
                elif value == "{{ inputs.file.content_type }}":
                    config[key] = inputs.get('file', {}).get('content_type')
                else:
                    # For other templates, try basic resolution
                    resolved_value = value
                    if "{{ inputs." in value:
                        # Extract the path and resolve it
                        import re
                        matches = re.findall(r'{{ inputs\.([^}]+) }}', value)
                        for match in matches:
                            keys = match.split('.')
                            current = inputs
                            for k in keys:
                                if isinstance(current, dict) and k in current:
                                    current = current[k]
                                else:
                                    current = None
                                    break
                            if current is not None:
                                resolved_value = resolved_value.replace(f'{{{{ inputs.{match} }}}}', str(current))
                    config[key] = resolved_value
            else:
                config[key] = value
        
        # Execute file handler
        result = await file_handler.execute(context, config)
        
        if not result.success:
            raise ValueError(f"File handler failed: {result.error}")
        
        return result.outputs
        """Validate inputs against flow definition."""
        validated = {}
        
        self.logger.info(f"Validating inputs for flow {flow_def.name}")
        #self.logger.info(f"Received inputs: {inputs}")
        
        for input_def in flow_def.inputs:
            value = inputs.get(input_def.name)
            
            # Check required inputs
            if input_def.required and value is None:
                if input_def.default is not None:
                    validated[input_def.name] = input_def.default
                else:
                    raise ValueError(f"Required input '{input_def.name}' is missing")
            elif value is not None:
                # Type validation and conversion
                if input_def.enum and value not in input_def.enum:
                    raise ValueError(f"Input '{input_def.name}' must be one of: {input_def.enum}")
                
                # Handle type conversion for form data
                if input_def.type == "boolean" and isinstance(value, str):
                    validated[input_def.name] = value.lower() in ('true', '1', 'yes', 'on')
                elif input_def.type == "array" and isinstance(value, str):
                    # Handle comma-separated strings for arrays
                    validated[input_def.name] = [item.strip() for item in value.split(',') if item.strip()]
                elif input_def.type == "array" and isinstance(value, list):
                    validated[input_def.name] = value
                else:
                    validated[input_def.name] = value
            elif input_def.default is not None:
                # Only use default if no value was provided
                validated[input_def.name] = input_def.default
        
        return validated
    
    def _create_request_model(self, flow_def: FlowDefinition) -> BaseModel:
        """Create Pydantic model for request validation."""
        fields = {}
        
        for input_def in flow_def.inputs:
            if input_def.type == "file":
                continue  # Skip file inputs for JSON endpoints
            
            # Map flow input types to Python types
            python_type = self._map_input_type(input_def.type)
            
            # Create field with validation
            field_kwargs = {
                "description": input_def.description,
            }
            
            if not input_def.required:
                field_kwargs["default"] = input_def.default
            
            if input_def.enum:
                # For enum types, we'll use the first enum value as default if not required
                if not input_def.required and input_def.default is None:
                    field_kwargs["default"] = input_def.enum[0]
            
            fields[input_def.name] = (python_type, Field(**field_kwargs))
        
        # Create dynamic model
        model_name = f"{flow_def.name.title()}Request"
        return create_model(model_name, **fields)
    
    def _map_input_type(self, input_type: str) -> type:
        """Map flow input type to Python type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": List[str],
            "object": Dict[str, Any]
        }
        return type_mapping.get(input_type, str)

    
    def get_openapi_schema_for_flow(self, flow_def: FlowDefinition) -> Dict[str, Any]:
        """Generate OpenAPI schema for a specific flow."""
        return {
            "flow_name": flow_def.name,
            "version": flow_def.version,
            "description": flow_def.description,
            "endpoints": {
                "execute": {
                    "method": "POST",
                    "path": f"/api/v1/{flow_def.name.replace('_', '-')}/execute",
                    "summary": f"Execute {flow_def.name} flow",
                    "description": flow_def.description
                },
                "info": {
                    "method": "GET", 
                    "path": f"/api/v1/{flow_def.name.replace('_', '-')}/info",
                    "summary": f"Get {flow_def.name} flow information"
                },
                "health": {
                    "method": "GET",
                    "path": f"/api/v1/{flow_def.name.replace('_', '-')}/health", 
                    "summary": f"Check {flow_def.name} flow health"
                }
            },
            "inputs": [
                {
                    "name": inp.name,
                    "type": inp.type,
                    "required": inp.required,
                    "default": inp.default,
                    "description": inp.description,
                    "enum": inp.enum
                }
                for inp in flow_def.inputs
            ],
            "tags": [f"{flow_def.name.replace('_', ' ').title()} Flow"]
        }
