"""
Pydantic schemas for request/response validation
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


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


class FlowStatus(str, Enum):
    """Flow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    QUEUED = "queued"


class StepStatus(str, Enum):
    """Step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


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


class FlowStepState(BaseModel):
    """Individual step execution state for Redis persistence"""
    step_name: str
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    inputs: Dict[str, Any] = {}
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = {}


class FlowState(BaseModel):
    """Enhanced flow execution state for Redis persistence"""
    flow_id: str
    flow_name: str
    status: FlowStatus = FlowStatus.PENDING
    inputs: Dict[str, Any]
    steps: List[FlowStepState] = []
    current_step: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_execution_time: Optional[float] = None
    callback_url: Optional[str] = None
    callback_status: Optional[str] = None  # Status of callback execution (pending, sent, failed)
    metadata: Dict[str, Any] = {}
    
    # Legacy fields for backward compatibility
    current_step_index: int = 0  # For old current_step int field
    outputs: Dict[str, Any] = {}  # For old outputs field
    created_at: Optional[str] = None  # For old created_at string field
    updated_at: Optional[str] = None  # For old updated_at string field
    
    def get_step_state(self, step_name: str) -> Optional[FlowStepState]:
        """Get state for a specific step"""
        return next((step for step in self.steps if step.step_name == step_name), None)
    
    def update_step_state(self, step_name: str, **kwargs) -> None:
        """Update state for a specific step"""
        step_state = self.get_step_state(step_name)
        if step_state:
            for key, value in kwargs.items():
                if hasattr(step_state, key):
                    setattr(step_state, key, value)
        else:
            # Create new step state
            new_step = FlowStepState(step_name=step_name, **kwargs)
            self.steps.append(new_step)
    
    def get_completed_steps(self) -> List[str]:
        """Get list of completed step names"""
        return [step.step_name for step in self.steps if step.status == StepStatus.COMPLETED]
    
    def get_failed_steps(self) -> List[str]:
        """Get list of failed step names"""
        return [step.step_name for step in self.steps if step.status == StepStatus.FAILED]


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
