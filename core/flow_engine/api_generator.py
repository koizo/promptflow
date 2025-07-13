"""
API Generator

Automatically generates FastAPI endpoints from YAML flow definitions.
Creates POST endpoints for flow execution, GET endpoints for flow information,
and handles input validation, file uploads, and response formatting.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING
from pydantic import BaseModel, Field, create_model
import logging
import tempfile
from pathlib import Path

from .yaml_loader import FlowDefinition, FlowInput

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
        """Add main flow execution endpoint."""
        
        # Determine if flow needs file upload
        has_file_input = any(inp.type == "file" for inp in flow_def.inputs)
        
        if has_file_input:
            self._add_file_upload_endpoint(router, flow_def)
        else:
            self._add_json_endpoint(router, flow_def)
    
    def _add_file_upload_endpoint(self, router: APIRouter, flow_def: FlowDefinition):
        """Add file upload endpoint for flows that accept files."""
        
        # Get non-file inputs for form parameters
        form_inputs = [inp for inp in flow_def.inputs if inp.type != "file"]
        
        from fastapi import Request
        
        @router.post("/execute", 
                    summary=f"Execute {flow_def.name} flow",
                    description=flow_def.description,
                    response_model=Dict[str, Any])
        async def execute_flow_with_file(
            request: Request,
            file: UploadFile = File(..., description="File to process")
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
                        if key != 'file':  # Skip the file field
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
                
                self.logger.info(f"Flow {flow_def.name} received inputs: {list(inputs.keys())}")
                for key, value in inputs.items():
                    if key != 'file':  # Don't log file content
                        self.logger.info(f"  {key}: {value}")
                
                # Validate inputs against flow definition
                validated_inputs = self._validate_inputs(inputs, flow_def)
                
                # Execute flow
                result = await self.flow_runner.run_flow(flow_def.name, validated_inputs)
                
                return result
                
            except Exception as e:
                self.logger.error(f"Flow execution failed: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
    
    def _add_json_endpoint(self, router: APIRouter, flow_def: FlowDefinition):
        """Add JSON endpoint for flows that don't need file uploads."""
        
        # Create Pydantic model for request validation
        request_model = self._create_request_model(flow_def)
        
        @router.post(f"/execute",
                    summary=f"Execute {flow_def.name} flow", 
                    description=flow_def.description,
                    response_model=Dict[str, Any])
        async def execute_flow_json(request: request_model):
            try:
                # Convert request to dict
                inputs = request.dict()
                
                # Execute flow
                result = await self.flow_runner.run_flow(flow_def.name, inputs)
                
                return result
                
            except Exception as e:
                self.logger.error(f"Flow execution failed: {str(e)}", exc_info=True)
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
        """Validate inputs against flow definition."""
        validated = {}
        
        self.logger.info(f"Validating inputs for flow {flow_def.name}")
        self.logger.info(f"Received inputs: {inputs}")
        
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
