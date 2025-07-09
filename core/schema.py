"""
Pydantic schemas for request/response validation
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class StepType(str, Enum):
    """Supported DSL step types"""
    OCR = "ocr"
    LLM = "llm"
    COMBINE = "combine"
    APPROVAL = "approval"


class InputType(str, Enum):
    """Supported input types"""
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    PDF_OR_IMAGE = "pdf_or_image"


class FlowInput(BaseModel):
    """Flow input definition"""
    name: str
    type: InputType
    required: bool = True
    description: Optional[str] = None


class FlowStep(BaseModel):
    """Flow step definition"""
    type: StepType
    model: Optional[str] = None
    input: Optional[str] = None
    output_key: str
    template: Optional[str] = None
    prompt_file: Optional[str] = None
    target_url: Optional[str] = None
    payload_template: Optional[str] = None
    resume_key: Optional[str] = None
    store_as: Optional[str] = None


class FlowOutput(BaseModel):
    """Flow output definition"""
    key: str
    format: str = "json"


class FlowDefinition(BaseModel):
    """Complete flow definition from YAML"""
    name: str
    description: Optional[str] = None
    inputs: List[FlowInput]
    steps: List[FlowStep]
    output: FlowOutput


class FlowMetadata(BaseModel):
    """Flow metadata from meta.yaml"""
    name: str
    description: str
    version: str
    author: Optional[str] = None
    tags: List[str] = []
    category: Optional[str] = None


class FlowExecutionRequest(BaseModel):
    """Request to execute a flow"""
    inputs: Dict[str, Any]
    callback_url: Optional[str] = None


class FlowExecutionResponse(BaseModel):
    """Response from flow execution"""
    flow_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class FlowState(BaseModel):
    """Flow execution state for persistence"""
    flow_id: str
    flow_name: str
    current_step: int
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    status: str
    created_at: str
    updated_at: str
    callback_url: Optional[str] = None


class CallbackPayload(BaseModel):
    """Callback payload for async steps"""
    flow_id: str
    step_result: Dict[str, Any]
    approved: Optional[bool] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    services: Dict[str, str]


class CatalogResponse(BaseModel):
    """Catalog response"""
    flows: List[Dict[str, Any]]
    total: int


class ErrorResponse(BaseModel):
    """Error response"""
    detail: str
    error_code: Optional[str] = None
