"""
Template Engine

Handles template processing for YAML flow definitions using Jinja2.
Supports variable substitution, conditional logic, and data transformation
within flow configurations.
"""

from jinja2 import Environment, BaseLoader, TemplateSyntaxError, UndefinedError, StrictUndefined
from typing import Dict, Any, Optional
import logging
import json
import re

logger = logging.getLogger(__name__)


class FlowTemplateLoader(BaseLoader):
    """Custom Jinja2 loader for flow templates."""
    
    def get_source(self, environment, template):
        # Not used for our string-based templates
        raise NotImplementedError()


class TemplateEngine:
    """
    Template engine for processing YAML flow configurations.
    
    Uses Jinja2 to process template expressions in flow definitions,
    allowing dynamic configuration based on inputs and step results.
    """
    
    def __init__(self):
        self.env = Environment(
            loader=FlowTemplateLoader(),
            # Use different delimiters to avoid conflicts with YAML
            variable_start_string='{{',
            variable_end_string='}}',
            block_start_string='{%',
            block_end_string='%}',
            comment_start_string='{#',
            comment_end_string='#}',
            # Enable auto-escaping for security
            autoescape=True,
            # Strict undefined variables
            undefined=StrictUndefined
        )
        
        # Add custom filters
        self.env.filters.update({
            'to_json': self._to_json_filter,
            'from_json': self._from_json_filter,
            'default': self._default_filter,
            'length': len,
            'upper': str.upper,
            'lower': str.lower,
            'strip': str.strip
        })
        
        # Add custom functions
        self.env.globals.update({
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict
        })
    
    def render_template(self, template_str: str, context: Dict[str, Any]) -> str:
        """
        Render a template string with given context.
        
        Args:
            template_str: Template string with Jinja2 syntax
            context: Variables available in template
            
        Returns:
            Rendered string
            
        Raises:
            TemplateSyntaxError: If template syntax is invalid
            UndefinedError: If template references undefined variables
        """
        try:
            template = self.env.from_string(template_str)
            return template.render(**context)
        except TemplateSyntaxError as e:
            raise TemplateSyntaxError(f"Template syntax error: {e}")
        except UndefinedError as e:
            raise UndefinedError(f"Undefined variable in template: {e}")
    
    def render_value(self, value: Any, context: Dict[str, Any]) -> Any:
        """
        Render a value that may contain templates.
        
        Handles strings, dictionaries, lists, and nested structures.
        Only processes strings that contain template syntax.
        Special handling for binary data and boolean values to preserve their types.
        """
        if isinstance(value, str):
            # Only process strings that contain template syntax
            if self._has_template_syntax(value):
                # Special handling for file content templates
                if value.strip() == "{{ inputs.file.content }}":
                    # Return the actual bytes object, not its string representation
                    try:
                        return context.get('inputs', {}).get('file', {}).get('content')
                    except (AttributeError, KeyError):
                        pass
                # Special handling for boolean templates
                elif value.strip().startswith("{{ inputs.") and value.strip().endswith(" }}"):
                    # Extract the variable path
                    var_path = value.strip()[2:-2].strip()  # Remove {{ and }}
                    try:
                        # Navigate the context to get the actual value
                        parts = var_path.split('.')
                        result = context
                        for part in parts:
                            result = result[part]
                        # Return the actual value (preserving type)
                        return result
                    except (KeyError, TypeError):
                        # Fall back to normal template rendering
                        return self.render_template(value, context)
                else:
                    return self.render_template(value, context)
            return value
        elif isinstance(value, dict):
            return {k: self.render_value(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.render_value(item, context) for item in value]
        else:
            return value
    
    def render_config(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render a configuration dictionary with template processing.
        
        Args:
            config: Configuration dictionary that may contain templates
            context: Template context with available variables
            
        Returns:
            Rendered configuration dictionary
        """
        return self.render_value(config, context)
    
    def build_context(self, 
                     inputs: Dict[str, Any], 
                     step_results: Dict[str, Any] = None,
                     flow_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Build template context from flow inputs and step results.
        
        Args:
            inputs: Flow input values
            step_results: Results from completed steps
            flow_metadata: Additional flow metadata
            
        Returns:
            Template context dictionary
        """
        context = {
            'inputs': inputs or {},
            'steps': {},
            'flow': flow_metadata or {}
        }
        
        # Add step results to context
        if step_results:
            for step_name, result in step_results.items():
                if hasattr(result, 'outputs') and result.outputs:
                    context['steps'][step_name] = result.outputs
                else:
                    context['steps'][step_name] = result
        
        # Add utility variables
        context.update({
            'completed_steps': list(step_results.keys()) if step_results else [],
            'total_steps': len(step_results) if step_results else 0
        })
        
        return context
    
    def _has_template_syntax(self, value: str) -> bool:
        """Check if string contains Jinja2 template syntax."""
        return '{{' in value and '}}' in value
    
    def _to_json_filter(self, value: Any) -> str:
        """Convert value to JSON string."""
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to convert to JSON: {e}")
            return str(value)
    
    def _from_json_filter(self, value: str) -> Any:
        """Parse JSON string to Python object."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return value
    
    def _default_filter(self, value: Any, default_value: Any) -> Any:
        """Return default value if input is None or empty."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return default_value
        return value
    
    def validate_template(self, template_str: str) -> Optional[str]:
        """
        Validate template syntax without rendering.
        
        Args:
            template_str: Template string to validate
            
        Returns:
            None if valid, error message if invalid
        """
        try:
            self.env.from_string(template_str)
            return None
        except TemplateSyntaxError as e:
            return str(e)
    
    def extract_variables(self, template_str: str) -> set:
        """
        Extract variable names used in template.
        
        Args:
            template_str: Template string to analyze
            
        Returns:
            Set of variable names found in template
        """
        try:
            # Parse template to AST
            ast = self.env.parse(template_str)
            
            # Extract variable names
            variables = set()
            for node in ast.find_all():
                if hasattr(node, 'name'):
                    variables.add(node.name)
            
            return variables
        except Exception as e:
            logger.warning(f"Failed to extract variables from template: {e}")
            return set()


# Global template engine instance
template_engine = TemplateEngine()
