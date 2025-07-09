"""
Common utility functions
"""
import uuid
import hashlib
import mimetypes
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


def generate_flow_id() -> str:
    """Generate a unique flow ID"""
    return str(uuid.uuid4())


def generate_hash(data: str) -> str:
    """Generate SHA256 hash of data"""
    return hashlib.sha256(data.encode()).hexdigest()


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.utcnow().isoformat()


def is_image_file(file_path: str) -> bool:
    """Check if file is an image"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type is not None and mime_type.startswith('image/')


def is_pdf_file(file_path: str) -> bool:
    """Check if file is a PDF"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type == 'application/pdf'


def is_audio_file(file_path: str) -> bool:
    """Check if file is an audio file"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type is not None and mime_type.startswith('audio/')


def validate_file_type(file_path: str, expected_type: str) -> bool:
    """Validate file type against expected type"""
    if expected_type == "image":
        return is_image_file(file_path)
    elif expected_type == "pdf":
        return is_pdf_file(file_path)
    elif expected_type == "audio":
        return is_audio_file(file_path)
    elif expected_type == "pdf_or_image":
        return is_pdf_file(file_path) or is_image_file(file_path)
    else:
        return True  # Default to allowing any file type


def format_template(template: str, variables: Dict[str, Any]) -> str:
    """Format a template string with variables"""
    try:
        return template.format(**variables)
    except KeyError as e:
        raise ValueError(f"Missing template variable: {e}")


def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get value from dictionary"""
    return dictionary.get(key, default)


def ensure_directory(path: str) -> Path:
    """Ensure directory exists, create if it doesn't"""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    return Path(file_path).stat().st_size / (1024 * 1024)


def parse_size_string(size_str: str) -> int:
    """Parse size string like '50MB' to bytes"""
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)  # Assume bytes


class FlowExecutionContext:
    """Context object for flow execution"""
    
    def __init__(self, flow_id: str, flow_name: str):
        self.flow_id = flow_id
        self.flow_name = flow_name
        self.start_time = datetime.utcnow()
        self.variables = {}
        self.step_outputs = {}
        self.current_step = 0
    
    def set_variable(self, key: str, value: Any):
        """Set a context variable"""
        self.variables[key] = value
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable"""
        return self.variables.get(key, default)
    
    def set_step_output(self, step_index: int, output: Any):
        """Set output for a specific step"""
        self.step_outputs[step_index] = output
    
    def get_step_output(self, step_index: int) -> Any:
        """Get output from a specific step"""
        return self.step_outputs.get(step_index)
    
    def get_all_outputs(self) -> Dict[str, Any]:
        """Get all step outputs with their keys"""
        return self.step_outputs.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization"""
        return {
            "flow_id": self.flow_id,
            "flow_name": self.flow_name,
            "start_time": self.start_time.isoformat(),
            "variables": self.variables,
            "step_outputs": self.step_outputs,
            "current_step": self.current_step
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlowExecutionContext":
        """Create context from dictionary"""
        context = cls(data["flow_id"], data["flow_name"])
        context.start_time = datetime.fromisoformat(data["start_time"])
        context.variables = data.get("variables", {})
        context.step_outputs = data.get("step_outputs", {})
        context.current_step = data.get("current_step", 0)
        return context
