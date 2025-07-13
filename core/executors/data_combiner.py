"""
Data Combiner Executor

Reusable executor for combining and merging results from multiple flow steps.
Supports various combination strategies and data transformation operations.
"""

from typing import Dict, Any, List, Union
import logging

from .base_executor import BaseExecutor, ExecutionResult, FlowContext

logger = logging.getLogger(__name__)


class DataCombiner(BaseExecutor):
    """
    Combine results from multiple flow steps.
    
    Supports various combination strategies including merging dictionaries,
    concatenating lists, joining text, and creating structured outputs.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Combine data from multiple sources.
        
        Config parameters:
        - sources (required): List of data sources to combine
        - strategy (optional): Combination strategy ('merge', 'concat', 'join', 'custom')
        - output_key (optional): Key name for combined output (default: 'combined')
        - separator (optional): Separator for join strategy (default: '\n\n')
        - merge_strategy (optional): How to handle key conflicts ('overwrite', 'keep_first', 'combine')
        """
        try:
            sources = config.get('sources', [])
            if not sources:
                return ExecutionResult(
                    success=False,
                    error="sources list is required for data combination"
                )
            
            strategy = config.get('strategy', 'merge')
            output_key = config.get('output_key', 'combined')
            separator = config.get('separator', '\n\n')
            merge_strategy = config.get('merge_strategy', 'overwrite')
            
            self.logger.info(f"Combining {len(sources)} sources using {strategy} strategy")
            
            # Resolve sources to actual data
            resolved_sources = []
            for source in sources:
                resolved_data = self._resolve_source(source, context)
                if resolved_data is not None:
                    resolved_sources.append(resolved_data)
            
            if not resolved_sources:
                return ExecutionResult(
                    success=False,
                    error="No valid data sources found to combine"
                )
            
            # Apply combination strategy
            if strategy == 'merge':
                combined = self._merge_dictionaries(resolved_sources, merge_strategy)
            elif strategy == 'concat':
                combined = self._concatenate_lists(resolved_sources)
            elif strategy == 'join':
                combined = self._join_text(resolved_sources, separator)
            elif strategy == 'custom':
                combined = self._custom_combination(resolved_sources, config)
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unknown combination strategy: {strategy}"
                )
            
            return ExecutionResult(
                success=True,
                outputs={
                    output_key: combined,
                    "sources_combined": len(resolved_sources),
                    "combination_strategy": strategy
                },
                metadata={
                    "combination_operation": strategy,
                    "sources_processed": len(resolved_sources),
                    "output_type": type(combined).__name__
                }
            )
            
        except Exception as e:
            self.logger.error(f"Data combination failed: {str(e)}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=f"Data combination failed: {str(e)}"
            )
    
    def _resolve_source(self, source: Any, context: FlowContext) -> Any:
        """Resolve a source reference to actual data."""
        if isinstance(source, str):
            # Handle template-like references
            if source.startswith('steps.'):
                # Extract step name and optional key
                parts = source.split('.')
                if len(parts) >= 2:
                    step_name = parts[1]
                    if step_name in context.step_results:
                        result = context.step_results[step_name]
                        if hasattr(result, 'outputs'):
                            data = result.outputs
                        else:
                            data = result
                        
                        # Navigate to nested key if specified
                        for key in parts[2:]:
                            if isinstance(data, dict) and key in data:
                                data = data[key]
                            else:
                                return None
                        return data
            elif source.startswith('inputs.'):
                # Extract input value
                parts = source.split('.')
                if len(parts) >= 2:
                    input_key = parts[1]
                    if input_key in context.inputs:
                        data = context.inputs[input_key]
                        
                        # Navigate to nested key if specified
                        for key in parts[2:]:
                            if isinstance(data, dict) and key in data:
                                data = data[key]
                            else:
                                return None
                        return data
            else:
                # Return string as-is
                return source
        
        # Return data as-is for non-string sources
        return source
    
    def _merge_dictionaries(self, sources: List[Any], merge_strategy: str) -> Dict[str, Any]:
        """Merge multiple dictionaries."""
        result = {}
        
        for source in sources:
            if not isinstance(source, dict):
                # Convert non-dict sources to dict
                if hasattr(source, '__dict__'):
                    source = source.__dict__
                else:
                    source = {"value": source}
            
            for key, value in source.items():
                if key in result:
                    # Handle key conflicts
                    if merge_strategy == 'overwrite':
                        result[key] = value
                    elif merge_strategy == 'keep_first':
                        pass  # Keep existing value
                    elif merge_strategy == 'combine':
                        # Combine values into list
                        if not isinstance(result[key], list):
                            result[key] = [result[key]]
                        if isinstance(value, list):
                            result[key].extend(value)
                        else:
                            result[key].append(value)
                else:
                    result[key] = value
        
        return result
    
    def _concatenate_lists(self, sources: List[Any]) -> List[Any]:
        """Concatenate multiple lists or convert to list."""
        result = []
        
        for source in sources:
            if isinstance(source, list):
                result.extend(source)
            elif isinstance(source, dict):
                # Add dictionary values to list
                result.extend(source.values())
            else:
                # Add single item to list
                result.append(source)
        
        return result
    
    def _join_text(self, sources: List[Any], separator: str) -> str:
        """Join multiple text sources."""
        text_parts = []
        
        for source in sources:
            if isinstance(source, str):
                text_parts.append(source)
            elif isinstance(source, dict):
                # Extract text from common keys
                for key in ['text', 'content', 'message', 'analysis', 'result']:
                    if key in source and isinstance(source[key], str):
                        text_parts.append(source[key])
                        break
                else:
                    # Convert dict to string representation
                    text_parts.append(str(source))
            else:
                # Convert to string
                text_parts.append(str(source))
        
        return separator.join(text_parts)
    
    def _custom_combination(self, sources: List[Any], config: Dict[str, Any]) -> Any:
        """Apply custom combination logic."""
        # This can be extended for specific custom combination needs
        custom_logic = config.get('custom_logic', 'default')
        
        if custom_logic == 'first_non_empty':
            # Return first non-empty source
            for source in sources:
                if source:
                    return source
            return None
        
        elif custom_logic == 'max_length':
            # Return source with maximum length
            max_source = None
            max_length = 0
            
            for source in sources:
                length = len(str(source))
                if length > max_length:
                    max_length = length
                    max_source = source
            
            return max_source
        
        elif custom_logic == 'structured':
            # Create structured output with all sources
            return {
                f"source_{i}": source 
                for i, source in enumerate(sources)
            }
        
        else:
            # Default: return all sources as list
            return sources
    
    def get_required_config_keys(self) -> List[str]:
        """Required configuration keys."""
        return ["sources"]
    
    def get_optional_config_keys(self) -> List[str]:
        """Optional configuration keys."""
        return [
            "strategy",
            "output_key",
            "separator",
            "merge_strategy",
            "custom_logic"
        ]
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate executor configuration."""
        super().validate_config(config)
        
        # Validate sources is list
        sources = config.get('sources', [])
        if not isinstance(sources, list):
            raise ValueError("sources must be a list")
        
        # Validate strategy
        if 'strategy' in config:
            strategy = config['strategy']
            valid_strategies = ['merge', 'concat', 'join', 'custom']
            if strategy not in valid_strategies:
                raise ValueError(f"strategy must be one of: {valid_strategies}")
        
        # Validate merge_strategy
        if 'merge_strategy' in config:
            merge_strategy = config['merge_strategy']
            valid_merge_strategies = ['overwrite', 'keep_first', 'combine']
            if merge_strategy not in valid_merge_strategies:
                raise ValueError(f"merge_strategy must be one of: {valid_merge_strategies}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get executor information."""
        info = super().get_info()
        info.update({
            "capabilities": [
                "Merge multiple dictionaries",
                "Concatenate lists and arrays",
                "Join text with custom separators",
                "Custom combination strategies",
                "Conflict resolution for duplicate keys",
                "Data type conversion and normalization"
            ],
            "combination_strategies": [
                "merge - Merge dictionaries",
                "concat - Concatenate lists",
                "join - Join text with separator",
                "custom - Apply custom logic"
            ],
            "merge_strategies": [
                "overwrite - Later values overwrite earlier ones",
                "keep_first - Keep first occurrence of each key",
                "combine - Combine conflicting values into lists"
            ]
        })
        return info
