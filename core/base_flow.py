"""
Abstract base class for all flows
"""
import os
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from fastapi import APIRouter
from pathlib import Path

from .schema import FlowDefinition, FlowMetadata, FlowExecutionRequest, FlowExecutionResponse


class BaseFlow(ABC):
    """Abstract base class for all flows"""
    
    def __init__(self, flow_dir: str):
        """Initialize flow with its directory path"""
        self.flow_dir = Path(flow_dir)
        self.flow_name = self.flow_dir.name
        
        # Load metadata and DSL definition
        self.metadata = self._load_metadata()
        self.definition = self._load_definition()
        
    def _load_metadata(self) -> FlowMetadata:
        """Load flow metadata from meta.yaml"""
        meta_file = self.flow_dir / "meta.yaml"
        if not meta_file.exists():
            raise FileNotFoundError(f"meta.yaml not found in {self.flow_dir}")
        
        with open(meta_file, 'r') as f:
            meta_data = yaml.safe_load(f)
        
        return FlowMetadata(**meta_data)
    
    def _load_definition(self) -> FlowDefinition:
        """Load flow definition from dsl.yaml"""
        dsl_file = self.flow_dir / "dsl.yaml"
        if not dsl_file.exists():
            raise FileNotFoundError(f"dsl.yaml not found in {self.flow_dir}")
        
        with open(dsl_file, 'r') as f:
            dsl_data = yaml.safe_load(f)
        
        return FlowDefinition(**dsl_data)
    
    def _load_prompt(self, prompt_file: str) -> str:
        """Load prompt template from file"""
        prompt_path = self.flow_dir / prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file {prompt_file} not found in {self.flow_dir}")
        
        with open(prompt_path, 'r') as f:
            return f.read().strip()
    
    @abstractmethod
    async def execute(self, request: FlowExecutionRequest) -> FlowExecutionResponse:
        """Execute the flow with given inputs"""
        pass
    
    @abstractmethod
    def get_router(self) -> APIRouter:
        """Get FastAPI router for this flow"""
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate inputs against flow definition"""
        required_inputs = {inp.name for inp in self.definition.inputs if inp.required}
        provided_inputs = set(inputs.keys())
        
        missing_inputs = required_inputs - provided_inputs
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {missing_inputs}")
        
        return True
    
    def get_catalog_info(self) -> Dict[str, Any]:
        """Get catalog information for this flow"""
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "version": self.metadata.version,
            "author": self.metadata.author,
            "tags": self.metadata.tags,
            "category": self.metadata.category,
            "inputs": [
                {
                    "name": inp.name,
                    "type": inp.type.value,
                    "required": inp.required,
                    "description": inp.description
                }
                for inp in self.definition.inputs
            ],
            "endpoint": f"/{self.flow_name}"
        }
