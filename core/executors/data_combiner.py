"""
Data Combiner Executor

Reusable executor for combining and merging results from multiple flow steps.
Supports various combination strategies and data transformation operations.
"""

from typing import Dict, Any, List, Union, Optional
import logging
import json
from datetime import datetime, timezone

from .base_executor import BaseExecutor, ExecutionResult, FlowContext

logger = logging.getLogger(__name__)


class DataCombiner(BaseExecutor):
    """
    Combine results from multiple flow steps.
    
    Supports various combination strategies including merging dictionaries,
    concatenating lists, joining text, and creating structured outputs.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name or "data_combiner")
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Combine data from multiple sources.
        
        Args:
            context: Flow execution context
            config: Configuration containing:
                - sources: List of step names or direct values to combine
                - strategy: Combination strategy ('merge', 'concat', 'join', 'structured')
                - output_key: Key name for the combined result
                - join_separator: Separator for text joining (default: ' ')
                - merge_strategy: How to handle conflicts ('overwrite', 'keep_first', 'combine')
                - include_metadata: Whether to include source metadata
                - transform: Optional transformation rules
        
        Returns:
            ExecutionResult with combined data
        """
        try:
            # Validate required parameters
            sources = config.get('sources', [])
            if not sources:
                return ExecutionResult(
                    success=False,
                    error="No sources specified for data combination"
                )
            
            strategy = config.get('strategy', 'merge')
            output_key = config.get('output_key', 'combined_result')
            
            # Collect data from sources
            source_data = []
            metadata = {}
            
            for source in sources:
                if isinstance(source, str):
                    # Reference to a previous step
                    if source in context.step_results:
                        step_result = context.step_results[source]
                        if hasattr(step_result, 'outputs'):
                            source_data.append(step_result.outputs)
                            metadata[source] = {
                                'type': 'step_result',
                                'success': step_result.success,
                                'timestamp': getattr(step_result, 'timestamp', None)
                            }
                        else:
                            source_data.append(step_result)
                            metadata[source] = {'type': 'raw_result'}
                    else:
                        logger.warning(f"Step '{source}' not found in context")
                        continue
                else:
                    # Direct value
                    source_data.append(source)
                    metadata[f'direct_{len(metadata)}'] = {'type': 'direct_value'}
            
            if not source_data:
                return ExecutionResult(
                    success=False,
                    error="No valid source data found for combination"
                )
            
            # Apply combination strategy
            combined_result = await self._apply_strategy(
                source_data, strategy, config
            )
            
            # Prepare output
            outputs = {output_key: combined_result}
            
            # Include metadata if requested
            if config.get('include_metadata', False):
                outputs['combination_metadata'] = {
                    'sources_count': len(source_data),
                    'strategy': strategy,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source_info': metadata
                }
            
            # Apply transformations if specified
            if 'transform' in config:
                outputs = await self._apply_transformations(outputs, config['transform'])
            
            return ExecutionResult(
                success=True,
                outputs=outputs
            )
            
        except Exception as e:
            logger.error(f"Error in DataCombiner execution: {str(e)}")
            return ExecutionResult(
                success=False,
                error=f"Data combination failed: {str(e)}"
            )
    
    async def _apply_strategy(self, source_data: List[Any], strategy: str, config: Dict[str, Any]) -> Any:
        """Apply the specified combination strategy."""
        
        if strategy == 'merge':
            return await self._merge_strategy(source_data, config)
        elif strategy == 'concat':
            return await self._concat_strategy(source_data, config)
        elif strategy == 'join':
            return await self._join_strategy(source_data, config)
        elif strategy == 'structured':
            return await self._structured_strategy(source_data, config)
        elif strategy == 'aggregate':
            return await self._aggregate_strategy(source_data, config)
        else:
            raise ValueError(f"Unknown combination strategy: {strategy}")
    
    async def _merge_strategy(self, source_data: List[Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge dictionaries with conflict resolution."""
        merge_strategy = config.get('merge_strategy', 'overwrite')
        result = {}
        
        for data in source_data:
            if isinstance(data, dict):
                for key, value in data.items():
                    if key in result:
                        if merge_strategy == 'overwrite':
                            result[key] = value
                        elif merge_strategy == 'keep_first':
                            pass  # Keep existing value
                        elif merge_strategy == 'combine':
                            if isinstance(result[key], list) and isinstance(value, list):
                                result[key].extend(value)
                            elif isinstance(result[key], str) and isinstance(value, str):
                                result[key] = f"{result[key]} {value}"
                            else:
                                result[key] = [result[key], value]
                    else:
                        result[key] = value
            else:
                # Non-dict data, add with index
                result[f'source_{len([k for k in result.keys() if k.startswith("source_")])}'] = data
        
        return result
    
    async def _concat_strategy(self, source_data: List[Any], config: Dict[str, Any]) -> List[Any]:
        """Concatenate lists or convert to list and concatenate."""
        result = []
        
        for data in source_data:
            if isinstance(data, list):
                result.extend(data)
            else:
                result.append(data)
        
        return result
    
    async def _join_strategy(self, source_data: List[Any], config: Dict[str, Any]) -> str:
        """Join data as text with separator."""
        separator = config.get('join_separator', ' ')
        text_parts = []
        
        for data in source_data:
            if isinstance(data, str):
                text_parts.append(data)
            elif isinstance(data, dict):
                # Extract text fields or convert to JSON
                if 'text' in data:
                    text_parts.append(str(data['text']))
                elif 'content' in data:
                    text_parts.append(str(data['content']))
                else:
                    text_parts.append(json.dumps(data, ensure_ascii=False))
            else:
                text_parts.append(str(data))
        
        return separator.join(text_parts)
    
    async def _structured_strategy(self, source_data: List[Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured output with labeled sections."""
        structure_template = config.get('structure_template', {})
        result = {}
        
        # If template provided, use it
        if structure_template:
            for key, source_ref in structure_template.items():
                if isinstance(source_ref, int) and 0 <= source_ref < len(source_data):
                    result[key] = source_data[source_ref]
                elif isinstance(source_ref, str):
                    # Could be a path like "0.text" or "1.sentiment"
                    try:
                        parts = source_ref.split('.')
                        data = source_data[int(parts[0])]
                        for part in parts[1:]:
                            if isinstance(data, dict):
                                data = data.get(part)
                            else:
                                break
                        result[key] = data
                    except (ValueError, IndexError, KeyError):
                        result[key] = None
        else:
            # Default structure
            for i, data in enumerate(source_data):
                result[f'source_{i}'] = data
        
        return result
    
    async def _aggregate_strategy(self, source_data: List[Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate numeric data with statistics."""
        aggregations = config.get('aggregations', ['count', 'sum'])
        result = {}
        
        # Flatten numeric values
        numeric_values = []
        text_values = []
        
        for data in source_data:
            if isinstance(data, (int, float)):
                numeric_values.append(data)
            elif isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, (int, float)):
                        numeric_values.append(value)
                    elif isinstance(value, str):
                        text_values.append(value)
            elif isinstance(data, str):
                text_values.append(data)
        
        # Calculate aggregations
        if numeric_values and 'count' in aggregations:
            result['count'] = len(numeric_values)
        if numeric_values and 'sum' in aggregations:
            result['sum'] = sum(numeric_values)
        if numeric_values and 'avg' in aggregations:
            result['average'] = sum(numeric_values) / len(numeric_values)
        if numeric_values and 'min' in aggregations:
            result['minimum'] = min(numeric_values)
        if numeric_values and 'max' in aggregations:
            result['maximum'] = max(numeric_values)
        
        if text_values:
            result['text_count'] = len(text_values)
            if 'concat_text' in aggregations:
                result['combined_text'] = ' '.join(text_values)
        
        result['total_sources'] = len(source_data)
        
        return result
    
    async def _apply_transformations(self, outputs: Dict[str, Any], transform_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformation rules to the outputs."""
        
        # Rename keys
        if 'rename' in transform_config:
            for old_key, new_key in transform_config['rename'].items():
                if old_key in outputs:
                    outputs[new_key] = outputs.pop(old_key)
        
        # Filter keys
        if 'include_only' in transform_config:
            allowed_keys = transform_config['include_only']
            outputs = {k: v for k, v in outputs.items() if k in allowed_keys}
        
        if 'exclude' in transform_config:
            excluded_keys = transform_config['exclude']
            outputs = {k: v for k, v in outputs.items() if k not in excluded_keys}
        
        # Format values
        if 'format' in transform_config:
            for key, format_rule in transform_config['format'].items():
                if key in outputs:
                    if format_rule == 'json':
                        outputs[key] = json.dumps(outputs[key], ensure_ascii=False, indent=2)
                    elif format_rule == 'upper':
                        if isinstance(outputs[key], str):
                            outputs[key] = outputs[key].upper()
                    elif format_rule == 'lower':
                        if isinstance(outputs[key], str):
                            outputs[key] = outputs[key].lower()
        
        return outputs
    
    def get_required_config(self) -> List[str]:
        """Return list of required configuration keys."""
        return ['sources']
    
    def get_optional_config(self) -> Dict[str, Any]:
        """Return dictionary of optional configuration keys with defaults."""
        return {
            'strategy': 'merge',
            'output_key': 'combined_result',
            'join_separator': ' ',
            'merge_strategy': 'overwrite',
            'include_metadata': False,
            'transform': {},
            'structure_template': {},
            'aggregations': ['count', 'sum']
        }
