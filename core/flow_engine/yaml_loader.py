"""
YAML Flow Loader

Handles loading, parsing, and validation of YAML flow definitions.
Provides structured access to flow configuration and validates
flow syntax and dependencies.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class FlowInput:
    """Represents a flow input parameter."""
    name: str
    type: str
    required: bool = True
    default: Any = None
    description: str = ""
    enum: Optional[List[Any]] = None


@dataclass
class FlowOutput:
    """Represents a flow output parameter."""
    name: str
    value: Any
    description: str = ""


@dataclass
class FlowStep:
    """Represents a single step in a flow."""
    name: str
    executor: str
    config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    parallel_group: Optional[str] = None
    outputs: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class FlowDefinition:
    """Complete flow definition loaded from YAML."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    inputs: List[FlowInput] = field(default_factory=list)
    outputs: List[FlowOutput] = field(default_factory=list)
    steps: List[FlowStep] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def get_step(self, step_name: str) -> Optional[FlowStep]:
        """Get step by name."""
        for step in self.steps:
            if step.name == step_name:
                return step
        return None
    
    def get_input(self, input_name: str) -> Optional[FlowInput]:
        """Get input by name."""
        for input_def in self.inputs:
            if input_def.name == input_name:
                return input_def
        return None
    
    def validate_dependencies(self) -> List[str]:
        """Validate step dependencies and return any errors."""
        errors = []
        step_names = {step.name for step in self.steps}
        
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_names:
                    errors.append(f"Step '{step.name}' depends on unknown step '{dep}'")
        
        return errors
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Get execution order considering dependencies.
        Returns list of lists where each inner list contains steps that can run in parallel.
        """
        # Simple topological sort
        remaining_steps = {step.name: step for step in self.steps}
        execution_order = []
        
        while remaining_steps:
            # Find steps with no unresolved dependencies
            ready_steps = []
            for step_name, step in remaining_steps.items():
                if all(dep not in remaining_steps for dep in step.depends_on):
                    ready_steps.append(step_name)
            
            if not ready_steps:
                # Circular dependency detected
                raise ValueError(f"Circular dependency detected in steps: {list(remaining_steps.keys())}")
            
            execution_order.append(ready_steps)
            
            # Remove ready steps
            for step_name in ready_steps:
                del remaining_steps[step_name]
        
        return execution_order


class YAMLFlowLoader:
    """
    Loads and validates YAML flow definitions.
    
    Provides methods to load flows from files or directories,
    validate their structure, and convert them to FlowDefinition objects.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_flow(self, flow_path: Path) -> FlowDefinition:
        """
        Load a single flow from a YAML file.
        
        Args:
            flow_path: Path to the flow.yaml file
            
        Returns:
            FlowDefinition object
            
        Raises:
            FileNotFoundError: If flow file doesn't exist
            yaml.YAMLError: If YAML is invalid
            ValueError: If flow structure is invalid
        """
        if not flow_path.exists():
            raise FileNotFoundError(f"Flow file not found: {flow_path}")
        
        try:
            with open(flow_path, 'r', encoding='utf-8') as f:
                flow_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {flow_path}: {e}")
        
        return self._parse_flow_data(flow_data, flow_path)
    
    def load_flows_from_directory(self, flows_dir: Path) -> Dict[str, FlowDefinition]:
        """
        Load all flows from a directory.
        
        Args:
            flows_dir: Path to directory containing flow subdirectories
            
        Returns:
            Dictionary mapping flow names to FlowDefinition objects
        """
        flows = {}
        
        if not flows_dir.exists():
            self.logger.warning(f"Flows directory not found: {flows_dir}")
            return flows
        
        for flow_dir in flows_dir.iterdir():
            if not flow_dir.is_dir():
                continue
            
            flow_file = flow_dir / "flow.yaml"
            if not flow_file.exists():
                self.logger.warning(f"No flow.yaml found in {flow_dir}")
                continue
            
            try:
                flow_def = self.load_flow(flow_file)
                flows[flow_def.name] = flow_def
                self.logger.info(f"Loaded flow: {flow_def.name}")
            except Exception as e:
                self.logger.error(f"Failed to load flow from {flow_dir}: {e}")
        
        return flows
    
    def _parse_flow_data(self, flow_data: Dict[str, Any], flow_path: Path) -> FlowDefinition:
        """Parse flow data dictionary into FlowDefinition object."""
        if not isinstance(flow_data, dict):
            raise ValueError(f"Flow data must be a dictionary in {flow_path}")
        
        # Required fields
        if 'name' not in flow_data:
            raise ValueError(f"Flow must have a 'name' field in {flow_path}")
        
        if 'steps' not in flow_data:
            raise ValueError(f"Flow must have 'steps' field in {flow_path}")
        
        # Parse inputs
        inputs = []
        for input_data in flow_data.get('inputs', []):
            inputs.append(self._parse_input(input_data))
        
        # Parse outputs
        outputs = []
        for output_data in flow_data.get('outputs', []):
            outputs.append(self._parse_output(output_data))
        
        # Parse steps
        steps = []
        for step_data in flow_data.get('steps', []):
            steps.append(self._parse_step(step_data))
        
        flow_def = FlowDefinition(
            name=flow_data['name'],
            version=flow_data.get('version', '1.0.0'),
            description=flow_data.get('description', ''),
            inputs=inputs,
            outputs=outputs,
            steps=steps,
            config=flow_data.get('config', {}),
            tags=flow_data.get('tags', [])
        )
        
        # Validate the flow
        self._validate_flow(flow_def)
        
        return flow_def
    
    def _parse_input(self, input_data: Dict[str, Any]) -> FlowInput:
        """Parse input definition."""
        if isinstance(input_data, str):
            # Simple string input
            return FlowInput(name=input_data, type="string")
        
        if not isinstance(input_data, dict):
            raise ValueError(f"Input must be string or dictionary: {input_data}")
        
        if 'name' not in input_data:
            raise ValueError(f"Input must have 'name' field: {input_data}")
        
        return FlowInput(
            name=input_data['name'],
            type=input_data.get('type', 'string'),
            required=input_data.get('required', True),
            default=input_data.get('default'),
            description=input_data.get('description', ''),
            enum=input_data.get('enum')
        )
    
    def _parse_output(self, output_data: Dict[str, Any]) -> FlowOutput:
        """Parse output definition."""
        if not isinstance(output_data, dict):
            raise ValueError(f"Output must be dictionary: {output_data}")
        
        if 'name' not in output_data:
            raise ValueError(f"Output must have 'name' field: {output_data}")
        
        return FlowOutput(
            name=output_data['name'],
            value=output_data.get('value'),
            description=output_data.get('description', '')
        )
    
    def _parse_step(self, step_data: Dict[str, Any]) -> FlowStep:
        """Parse step definition."""
        if not isinstance(step_data, dict):
            raise ValueError(f"Step must be dictionary: {step_data}")
        
        required_fields = ['name', 'executor']
        for field in required_fields:
            if field not in step_data:
                raise ValueError(f"Step must have '{field}' field: {step_data}")
        
        return FlowStep(
            name=step_data['name'],
            executor=step_data['executor'],
            config=step_data.get('config', {}),
            depends_on=step_data.get('depends_on', []),
            condition=step_data.get('condition'),
            parallel_group=step_data.get('parallel_group'),
            outputs=step_data.get('outputs', []),
            description=step_data.get('description', '')
        )
    
    def _validate_flow(self, flow_def: FlowDefinition) -> None:
        """Validate flow definition."""
        # Check for duplicate step names
        step_names = [step.name for step in flow_def.steps]
        if len(step_names) != len(set(step_names)):
            raise ValueError(f"Duplicate step names found in flow '{flow_def.name}'")
        
        # Check for duplicate input names
        input_names = [inp.name for inp in flow_def.inputs]
        if len(input_names) != len(set(input_names)):
            raise ValueError(f"Duplicate input names found in flow '{flow_def.name}'")
        
        # Validate dependencies
        dep_errors = flow_def.validate_dependencies()
        if dep_errors:
            raise ValueError(f"Dependency errors in flow '{flow_def.name}': {dep_errors}")
        
        self.logger.info(f"Flow '{flow_def.name}' validation passed")
