#!/usr/bin/env python3
"""
Minimal test for DataCombiner executor.
Tests core functionality without external dependencies.
"""

import asyncio
import sys
import traceback
from unittest.mock import Mock
from datetime import datetime
from typing import Dict, Any, List, Union, Optional
import json

# Minimal base executor implementation for testing
class ExecutionResult:
    def __init__(self, success: bool, outputs: Dict[str, Any] = None, error: str = None):
        self.success = success
        self.outputs = outputs or {}
        self.error = error

class FlowContext:
    def __init__(self):
        self.step_results = {}

class BaseExecutor:
    def __init__(self, name: str = None):
        self.name = name or "base_executor"
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        raise NotImplementedError

# Copy DataCombiner implementation directly
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
                    'timestamp': datetime.utcnow().isoformat(),
                    'source_info': metadata
                }
            
            return ExecutionResult(
                success=True,
                outputs=outputs
            )
            
        except Exception as e:
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


class TestRunner:
    """Simple test runner for DataCombiner."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def create_mock_context(self):
        """Create mock flow context with sample step results."""
        context = FlowContext()
        
        # Mock step results
        step1_result = Mock()
        step1_result.success = True
        step1_result.outputs = {"text": "Hello", "score": 0.8}
        step1_result.timestamp = "2024-01-01T00:00:00"
        
        step2_result = Mock()
        step2_result.success = True
        step2_result.outputs = {"text": "World", "score": 0.9}
        step2_result.timestamp = "2024-01-01T00:01:00"
        
        step3_result = Mock()
        step3_result.success = True
        step3_result.outputs = {"items": ["a", "b", "c"], "count": 3}
        
        context.step_results = {
            "step1": step1_result,
            "step2": step2_result,
            "step3": step3_result
        }
        
        return context
    
    async def run_test(self, test_name, test_func):
        """Run a single test."""
        self.tests_run += 1
        try:
            await test_func()
            self.tests_passed += 1
            print(f"✓ {test_name}")
        except Exception as e:
            self.tests_failed += 1
            self.failures.append((test_name, str(e), traceback.format_exc()))
            print(f"✗ {test_name}: {str(e)}")
    
    def assert_true(self, condition, message="Assertion failed"):
        """Simple assertion."""
        if not condition:
            raise AssertionError(message)
    
    def assert_equal(self, actual, expected, message=None):
        """Assert equality."""
        if actual != expected:
            msg = message or f"Expected {expected}, got {actual}"
            raise AssertionError(msg)
    
    def assert_in(self, item, container, message=None):
        """Assert item in container."""
        if item not in container:
            msg = message or f"{item} not found in {container}"
            raise AssertionError(msg)
    
    async def test_merge_strategy_basic(self):
        """Test basic merge strategy."""
        combiner = DataCombiner()
        context = self.create_mock_context()
        
        config = {
            "sources": ["step1", "step2"],
            "strategy": "merge",
            "output_key": "merged_data"
        }
        
        result = await combiner.execute(context, config)
        
        self.assert_true(result.success, "Execution should succeed")
        self.assert_in("merged_data", result.outputs, "Output should contain merged_data")
        merged = result.outputs["merged_data"]
        self.assert_equal(merged["text"], "World", "Last value should win")
        self.assert_equal(merged["score"], 0.9, "Last score should win")
    
    async def test_merge_strategy_keep_first(self):
        """Test merge strategy with keep_first."""
        combiner = DataCombiner()
        context = self.create_mock_context()
        
        config = {
            "sources": ["step1", "step2"],
            "strategy": "merge",
            "merge_strategy": "keep_first",
            "output_key": "merged_data"
        }
        
        result = await combiner.execute(context, config)
        
        self.assert_true(result.success, "Execution should succeed")
        merged = result.outputs["merged_data"]
        self.assert_equal(merged["text"], "Hello", "First value should be kept")
        self.assert_equal(merged["score"], 0.8, "First score should be kept")
    
    async def test_merge_strategy_combine(self):
        """Test merge strategy with combine."""
        combiner = DataCombiner()
        context = self.create_mock_context()
        
        config = {
            "sources": ["step1", "step2"],
            "strategy": "merge",
            "merge_strategy": "combine",
            "output_key": "merged_data"
        }
        
        result = await combiner.execute(context, config)
        
        self.assert_true(result.success, "Execution should succeed")
        merged = result.outputs["merged_data"]
        self.assert_equal(merged["text"], "Hello World", "Strings should be combined")
        self.assert_equal(merged["score"], [0.8, 0.9], "Values should be combined in list")
    
    async def test_concat_strategy(self):
        """Test concatenation strategy."""
        combiner = DataCombiner()
        context = self.create_mock_context()
        
        # Add list data to context
        step4_result = Mock()
        step4_result.outputs = ["x", "y", "z"]
        context.step_results["step4"] = step4_result
        
        config = {
            "sources": ["step3", "step4"],
            "strategy": "concat",
            "output_key": "concatenated_data"
        }
        
        result = await combiner.execute(context, config)
        
        self.assert_true(result.success, "Execution should succeed")
        concat_data = result.outputs["concatenated_data"]
        self.assert_equal(len(concat_data), 4, "Should have 4 items")
    
    async def test_join_strategy(self):
        """Test join strategy."""
        combiner = DataCombiner()
        context = self.create_mock_context()
        
        config = {
            "sources": ["step1", "step2"],
            "strategy": "join",
            "output_key": "joined_text"
        }
        
        result = await combiner.execute(context, config)
        
        self.assert_true(result.success, "Execution should succeed")
        joined = result.outputs["joined_text"]
        self.assert_in("Hello", joined, "Should contain Hello")
        self.assert_in("World", joined, "Should contain World")
    
    async def test_structured_strategy(self):
        """Test structured strategy."""
        combiner = DataCombiner()
        context = self.create_mock_context()
        
        config = {
            "sources": ["step1", "step2"],
            "strategy": "structured",
            "output_key": "structured_data"
        }
        
        result = await combiner.execute(context, config)
        
        self.assert_true(result.success, "Execution should succeed")
        structured = result.outputs["structured_data"]
        self.assert_in("source_0", structured, "Should have source_0")
        self.assert_in("source_1", structured, "Should have source_1")
    
    async def test_structured_strategy_with_template(self):
        """Test structured strategy with template."""
        combiner = DataCombiner()
        context = self.create_mock_context()
        
        config = {
            "sources": ["step1", "step2"],
            "strategy": "structured",
            "structure_template": {
                "first_text": "0.text",
                "second_score": "1.score"
            },
            "output_key": "structured_data"
        }
        
        result = await combiner.execute(context, config)
        
        self.assert_true(result.success, "Execution should succeed")
        structured = result.outputs["structured_data"]
        self.assert_equal(structured["first_text"], "Hello", "Should extract first text")
        self.assert_equal(structured["second_score"], 0.9, "Should extract second score")
    
    async def test_aggregate_strategy(self):
        """Test aggregate strategy."""
        combiner = DataCombiner()
        context = self.create_mock_context()
        
        config = {
            "sources": ["step1", "step2"],
            "strategy": "aggregate",
            "aggregations": ["count", "sum", "avg"],
            "output_key": "aggregated_data"
        }
        
        result = await combiner.execute(context, config)
        
        self.assert_true(result.success, "Execution should succeed")
        agg = result.outputs["aggregated_data"]
        self.assert_equal(agg["count"], 2, "Should count 2 numeric values")
        self.assert_equal(round(agg["sum"], 1), 1.7, "Should sum to 1.7")
        self.assert_equal(round(agg["average"], 2), 0.85, "Should average to 0.85")
    
    async def test_direct_values(self):
        """Test using direct values as sources."""
        combiner = DataCombiner()
        context = self.create_mock_context()
        
        config = {
            "sources": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ],
            "strategy": "merge",
            "output_key": "combined_data"
        }
        
        result = await combiner.execute(context, config)
        
        self.assert_true(result.success, "Execution should succeed")
        combined = result.outputs["combined_data"]
        self.assert_equal(combined["name"], "Bob", "Last name should win")
        self.assert_equal(combined["age"], 25, "Last age should win")
    
    async def test_include_metadata(self):
        """Test including metadata."""
        combiner = DataCombiner()
        context = self.create_mock_context()
        
        config = {
            "sources": ["step1", "step2"],
            "strategy": "merge",
            "include_metadata": True,
            "output_key": "combined_data"
        }
        
        result = await combiner.execute(context, config)
        
        self.assert_true(result.success, "Execution should succeed")
        self.assert_in("combination_metadata", result.outputs, "Should include metadata")
        metadata = result.outputs["combination_metadata"]
        self.assert_equal(metadata["sources_count"], 2, "Should count 2 sources")
        self.assert_equal(metadata["strategy"], "merge", "Should record strategy")
    
    async def test_error_no_sources(self):
        """Test error when no sources specified."""
        combiner = DataCombiner()
        context = self.create_mock_context()
        
        config = {"strategy": "merge"}
        
        result = await combiner.execute(context, config)
        
        self.assert_true(not result.success, "Should fail")
        self.assert_in("No sources specified", result.error, "Should have appropriate error")
    
    async def test_error_unknown_strategy(self):
        """Test error for unknown strategy."""
        combiner = DataCombiner()
        context = self.create_mock_context()
        
        config = {
            "sources": ["step1"],
            "strategy": "unknown_strategy"
        }
        
        result = await combiner.execute(context, config)
        
        self.assert_true(not result.success, "Should fail")
        self.assert_in("Unknown combination strategy", result.error, "Should have appropriate error")
    
    def test_config_methods(self):
        """Test configuration methods."""
        combiner = DataCombiner()
        
        required = combiner.get_required_config()
        self.assert_equal(required, ['sources'], "Should require sources")
        
        optional = combiner.get_optional_config()
        self.assert_equal(optional['strategy'], 'merge', "Default strategy should be merge")
        self.assert_equal(optional['output_key'], 'combined_result', "Default output key")
    
    async def run_all_tests(self):
        """Run all tests."""
        print("Running DataCombiner Tests...")
        print("=" * 50)
        
        # Async tests
        await self.run_test("test_merge_strategy_basic", self.test_merge_strategy_basic)
        await self.run_test("test_merge_strategy_keep_first", self.test_merge_strategy_keep_first)
        await self.run_test("test_merge_strategy_combine", self.test_merge_strategy_combine)
        await self.run_test("test_concat_strategy", self.test_concat_strategy)
        await self.run_test("test_join_strategy", self.test_join_strategy)
        await self.run_test("test_structured_strategy", self.test_structured_strategy)
        await self.run_test("test_structured_strategy_with_template", self.test_structured_strategy_with_template)
        await self.run_test("test_aggregate_strategy", self.test_aggregate_strategy)
        await self.run_test("test_direct_values", self.test_direct_values)
        await self.run_test("test_include_metadata", self.test_include_metadata)
        await self.run_test("test_error_no_sources", self.test_error_no_sources)
        await self.run_test("test_error_unknown_strategy", self.test_error_unknown_strategy)
        
        # Sync tests
        try:
            self.test_config_methods()
            self.tests_run += 1
            self.tests_passed += 1
            print("✓ test_config_methods")
        except Exception as e:
            self.tests_run += 1
            self.tests_failed += 1
            self.failures.append(("test_config_methods", str(e), traceback.format_exc()))
            print(f"✗ test_config_methods: {str(e)}")
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"Tests run: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.failures:
            print("\nFailures:")
            for test_name, error, tb in self.failures:
                print(f"\n{test_name}:")
                print(f"  Error: {error}")
        
        return self.tests_failed == 0


async def main():
    """Main test runner."""
    runner = TestRunner()
    success = await runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
