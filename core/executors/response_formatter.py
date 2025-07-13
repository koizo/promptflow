"""
Response Formatter Executor

Reusable executor for formatting final flow responses.
Provides standardized output formatting, data transformation,
and response structure for consistent API responses.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import logging

from .base_executor import BaseExecutor, ExecutionResult, FlowContext

logger = logging.getLogger(__name__)


class ResponseFormatter(BaseExecutor):
    """
    Format final flow responses with standardized structure.
    
    Creates consistent API responses with proper metadata,
    error handling, and data transformation capabilities.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Format flow response with standardized structure.
        
        Config parameters:
        - template (optional): Response template ('standard', 'minimal', 'detailed', 'custom')
        - include_metadata (optional): Whether to include execution metadata (default: True)
        - include_steps (optional): Whether to include step details (default: True)
        - custom_fields (optional): Additional custom fields to include
        - data_transformations (optional): Data transformation rules
        - success_message (optional): Custom success message
        - error_handling (optional): Error handling strategy
        """
        try:
            template = config.get('template', 'standard')
            include_metadata = config.get('include_metadata', True)
            include_steps = config.get('include_steps', True)
            custom_fields = config.get('custom_fields', {})
            data_transformations = config.get('data_transformations', {})
            success_message = config.get('success_message')
            error_handling = config.get('error_handling', 'include')
            
            self.logger.info(f"Formatting response using {template} template")
            
            # Determine overall success
            overall_success = len(context.failed_steps) == 0
            
            # Build base response
            if template == 'minimal':
                response = self._build_minimal_response(context, overall_success)
            elif template == 'detailed':
                response = self._build_detailed_response(context, overall_success)
            elif template == 'custom':
                response = self._build_custom_response(context, config, overall_success)
            else:  # standard
                response = self._build_standard_response(context, overall_success)
            
            # Add custom success message
            if success_message and overall_success:
                response['message'] = success_message
            
            # Include metadata if requested
            if include_metadata:
                response['metadata'] = self._build_metadata(context)
            
            # Include step details if requested
            if include_steps:
                response['steps'] = self._build_step_details(context, error_handling)
            
            # Add custom fields
            if custom_fields:
                for key, value in custom_fields.items():
                    # Resolve template values in custom fields
                    if isinstance(value, str) and '{{' in value:
                        # Simple template resolution
                        resolved_value = self._resolve_template_value(value, context)
                        response[key] = resolved_value
                    else:
                        response[key] = value
            
            # Apply data transformations
            if data_transformations:
                response = self._apply_transformations(response, data_transformations, context)
            
            return ExecutionResult(
                success=True,
                outputs=response,
                metadata={
                    "formatter_template": template,
                    "response_size": len(str(response)),
                    "fields_included": list(response.keys())
                }
            )
            
        except Exception as e:
            self.logger.error(f"Response formatting failed: {str(e)}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=f"Response formatting failed: {str(e)}"
            )
    
    def _build_standard_response(self, context: FlowContext, success: bool) -> Dict[str, Any]:
        """Build standard response format."""
        return {
            "success": success,
            "flow": context.flow_name,
            "completed_steps": context.completed_steps,
            "failed_steps": context.failed_steps,
            "total_steps": len(context.step_results),
            "execution_time": (datetime.now(timezone.utc) - context.started_at).total_seconds()
        }
    
    def _build_minimal_response(self, context: FlowContext, success: bool) -> Dict[str, Any]:
        """Build minimal response format."""
        response = {
            "success": success,
            "flow": context.flow_name
        }
        
        # Add main result if available
        if context.step_results:
            last_step = list(context.step_results.keys())[-1]
            last_result = context.step_results[last_step]
            if hasattr(last_result, 'outputs') and last_result.outputs:
                response["result"] = last_result.outputs
        
        return response
    
    def _build_detailed_response(self, context: FlowContext, success: bool) -> Dict[str, Any]:
        """Build detailed response format."""
        response = self._build_standard_response(context, success)
        
        # Add detailed step information
        response["step_details"] = {}
        for step_name, result in context.step_results.items():
            response["step_details"][step_name] = {
                "success": result.success,
                "execution_time": result.execution_time,
                "outputs": result.outputs if result.success else None,
                "error": result.error if not result.success else None
            }
        
        # Add execution summary
        response["execution_summary"] = context.get_execution_summary()
        
        return response
    
    def _build_custom_response(self, context: FlowContext, config: Dict[str, Any], 
                             success: bool) -> Dict[str, Any]:
        """Build custom response format."""
        custom_template = config.get('custom_template', {})
        
        # Start with base structure
        response = {
            "success": success,
            "flow": context.flow_name
        }
        
        # Apply custom template
        for key, value_template in custom_template.items():
            if isinstance(value_template, str):
                response[key] = self._resolve_template_value(value_template, context)
            else:
                response[key] = value_template
        
        return response
    
    def _build_metadata(self, context: FlowContext) -> Dict[str, Any]:
        """Build execution metadata."""
        return {
            "started_at": context.started_at.isoformat(),
            "execution_duration": (datetime.now(timezone.utc) - context.started_at).total_seconds(),
            "steps_executed": len(context.step_results),
            "success_rate": len(context.completed_steps) / len(context.step_results) if context.step_results else 0,
            "input_keys": list(context.inputs.keys()),
            "flow_context": {
                "flow_name": context.flow_name,
                "total_inputs": len(context.inputs)
            }
        }
    
    def _build_step_details(self, context: FlowContext, error_handling: str) -> Dict[str, Any]:
        """Build step execution details."""
        steps = {}
        
        for step_name, result in context.step_results.items():
            step_info = {
                "success": result.success,
                "execution_time": result.execution_time
            }
            
            if result.success:
                step_info["outputs"] = result.outputs
            else:
                if error_handling == 'include':
                    step_info["error"] = result.error
                elif error_handling == 'summary':
                    step_info["error"] = "Step failed"
                # 'exclude' option doesn't add error info
            
            steps[step_name] = step_info
        
        return steps
    
    def _resolve_template_value(self, template: str, context: FlowContext) -> Any:
        """Resolve simple template values."""
        # Simple template resolution for common patterns
        if template == "{{ flow_name }}":
            return context.flow_name
        elif template == "{{ completed_steps }}":
            return context.completed_steps
        elif template == "{{ failed_steps }}":
            return context.failed_steps
        elif template == "{{ total_steps }}":
            return len(context.step_results)
        elif template.startswith("{{ steps.") and template.endswith(" }}"):
            # Extract step reference
            step_ref = template[10:-3]  # Remove {{ steps. and }}
            parts = step_ref.split('.')
            
            if len(parts) >= 1:
                step_name = parts[0]
                if step_name in context.step_results:
                    result = context.step_results[step_name]
                    data = result.outputs if hasattr(result, 'outputs') else result
                    
                    # Navigate to nested key
                    for key in parts[1:]:
                        if isinstance(data, dict) and key in data:
                            data = data[key]
                        else:
                            return None
                    return data
        
        # Return template as-is if not resolved
        return template
    
    def _apply_transformations(self, response: Dict[str, Any], 
                             transformations: Dict[str, Any], 
                             context: FlowContext) -> Dict[str, Any]:
        """Apply data transformations to response."""
        for field, transformation in transformations.items():
            if field in response:
                if transformation == 'uppercase':
                    if isinstance(response[field], str):
                        response[field] = response[field].upper()
                elif transformation == 'lowercase':
                    if isinstance(response[field], str):
                        response[field] = response[field].lower()
                elif transformation == 'length':
                    response[f"{field}_length"] = len(str(response[field]))
                elif transformation == 'truncate':
                    if isinstance(response[field], str) and len(response[field]) > 100:
                        response[field] = response[field][:100] + "..."
        
        return response
    
    def get_required_config_keys(self) -> List[str]:
        """Required configuration keys."""
        return []  # No required keys
    
    def get_optional_config_keys(self) -> List[str]:
        """Optional configuration keys."""
        return [
            "template",
            "include_metadata",
            "include_steps",
            "custom_fields",
            "data_transformations",
            "success_message",
            "error_handling",
            "custom_template"
        ]
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate executor configuration."""
        super().validate_config(config)
        
        # Validate template
        if 'template' in config:
            template = config['template']
            valid_templates = ['standard', 'minimal', 'detailed', 'custom']
            if template not in valid_templates:
                raise ValueError(f"template must be one of: {valid_templates}")
        
        # Validate boolean flags
        for bool_key in ['include_metadata', 'include_steps']:
            if bool_key in config and not isinstance(config[bool_key], bool):
                raise ValueError(f"{bool_key} must be a boolean")
        
        # Validate custom_fields is dict
        if 'custom_fields' in config and not isinstance(config['custom_fields'], dict):
            raise ValueError("custom_fields must be a dictionary")
        
        # Validate error_handling
        if 'error_handling' in config:
            error_handling = config['error_handling']
            valid_strategies = ['include', 'summary', 'exclude']
            if error_handling not in valid_strategies:
                raise ValueError(f"error_handling must be one of: {valid_strategies}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get executor information."""
        info = super().get_info()
        info.update({
            "capabilities": [
                "Standardized response formatting",
                "Multiple response templates",
                "Metadata inclusion",
                "Step detail formatting",
                "Custom field addition",
                "Data transformations",
                "Error handling strategies"
            ],
            "response_templates": [
                "standard - Complete response with all details",
                "minimal - Minimal response with main result",
                "detailed - Detailed response with step information",
                "custom - Custom template-based response"
            ],
            "transformations": [
                "uppercase/lowercase - Text case conversion",
                "length - Add length information",
                "truncate - Truncate long text"
            ]
        })
        return info
