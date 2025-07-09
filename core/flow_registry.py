"""
Flow registry for loading and managing flows
"""
import os
import yaml
import logging
import importlib.util
from typing import Dict, List, Any, Optional
from pathlib import Path

from .config import settings
from .base_flow import BaseFlow
from .schema import FlowState, FlowExecutionResponse

logger = logging.getLogger(__name__)


class FlowRegistry:
    """Registry for managing and executing flows"""
    
    def __init__(self):
        self.flows: Dict[str, BaseFlow] = {}
        self.config: Dict[str, Any] = {}
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml"""
        config_path = Path(settings.config_file)
        
        if not config_path.exists():
            logger.warning(f"Config file {settings.config_file} not found, using defaults")
            return {
                "enabled_flows": [],
                "api_prefix": "/api/v1"
            }
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {settings.config_file}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {"enabled_flows": [], "api_prefix": "/api/v1"}
    
    def load_flows(self) -> Dict[str, BaseFlow]:
        """Load all flows from the flows directory"""
        self.config = self.load_config()
        flows_dir = Path(settings.flows_dir)
        
        if not flows_dir.exists():
            logger.warning(f"Flows directory {settings.flows_dir} not found")
            return {}
        
        enabled_flows = self.config.get("enabled_flows", [])
        
        for flow_dir in flows_dir.iterdir():
            if not flow_dir.is_dir():
                continue
            
            flow_name = flow_dir.name
            
            # Skip if not in enabled flows list (if list is specified)
            if enabled_flows and flow_name not in enabled_flows:
                logger.debug(f"Skipping disabled flow: {flow_name}")
                continue
            
            try:
                flow = self._load_single_flow(flow_dir)
                self.flows[flow_name] = flow
                logger.info(f"✅ Loaded flow: {flow_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to load flow {flow_name}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.flows)} flows total")
        return self.flows
    
    def _load_single_flow(self, flow_dir: Path) -> BaseFlow:
        """Load a single flow from its directory"""
        flow_name = flow_dir.name
        
        # Check required files
        required_files = ["flow.py", "meta.yaml", "dsl.yaml"]
        for file_name in required_files:
            file_path = flow_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required file {file_name} not found in {flow_dir}")
        
        # Dynamically import the flow module
        flow_module_path = flow_dir / "flow.py"
        spec = importlib.util.spec_from_file_location(f"flows.{flow_name}", flow_module_path)
        
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load flow module from {flow_module_path}")
        
        flow_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(flow_module)
        
        # Get the flow class (should be named after the flow or be the default export)
        flow_class = None
        for attr_name in dir(flow_module):
            attr = getattr(flow_module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, BaseFlow) and 
                attr != BaseFlow):
                flow_class = attr
                break
        
        if flow_class is None:
            raise ImportError(f"No BaseFlow subclass found in {flow_module_path}")
        
        # Instantiate the flow
        return flow_class(str(flow_dir))
    
    def get_flow(self, flow_name: str) -> Optional[BaseFlow]:
        """Get a specific flow by name"""
        return self.flows.get(flow_name)
    
    def get_flow_catalog(self) -> List[Dict[str, Any]]:
        """Get catalog information for all loaded flows"""
        catalog = []
        for flow_name, flow in self.flows.items():
            try:
                catalog.append(flow.get_catalog_info())
            except Exception as e:
                logger.error(f"Failed to get catalog info for {flow_name}: {e}")
                continue
        
        return catalog
    
    async def execute_flow(self, flow_name: str, inputs: Dict[str, Any], 
                          callback_url: Optional[str] = None) -> FlowExecutionResponse:
        """Execute a flow with given inputs"""
        flow = self.get_flow(flow_name)
        if not flow:
            raise ValueError(f"Flow '{flow_name}' not found")
        
        try:
            from .schema import FlowExecutionRequest
            request = FlowExecutionRequest(
                inputs=inputs,
                callback_url=callback_url
            )
            
            # Validate inputs
            flow.validate_inputs(inputs)
            
            # Execute flow
            response = await flow.execute(request)
            return response
            
        except Exception as e:
            logger.error(f"Flow execution failed for {flow_name}: {e}")
            raise
    
    async def resume_flow(self, flow_id: str, callback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resume a paused flow from callback"""
        # This would be implemented by the specific flow execution engine
        # For now, return a placeholder
        logger.info(f"Resuming flow {flow_id} with callback data")
        return {"status": "resumed", "flow_id": flow_id}
    
    def validate_all_flows(self) -> Dict[str, Any]:
        """Validate all loaded flows"""
        results = {}
        
        for flow_name, flow in self.flows.items():
            try:
                # Basic validation - check if required files exist and are valid
                validation_result = {
                    "valid": True,
                    "metadata": flow.metadata.dict(),
                    "definition": flow.definition.dict(),
                    "errors": []
                }
                
                # Additional validation logic can be added here
                
                results[flow_name] = validation_result
                
            except Exception as e:
                results[flow_name] = {
                    "valid": False,
                    "errors": [str(e)]
                }
        
        return results
