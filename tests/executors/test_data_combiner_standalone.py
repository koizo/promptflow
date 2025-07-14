#!/usr/bin/env python3
"""
Standalone test runner for DataCombiner executor.
Runs tests without pytest dependencies.
"""

import asyncio
import sys
import traceback
from unittest.mock import Mock
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, '/Users/hugo/Projects/Development/promptflow')

from core.executors.data_combiner import DataCombiner
from core.executors.base_executor import ExecutionResult, FlowContext


class TestRunner:
    """Simple test runner for DataCombiner."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def create_mock_context(self):
        """Create mock flow context with sample step results."""
        context = Mock(spec=FlowContext)
        
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
        self.assert_equal(agg["sum"], 1.7, "Should sum to 1.7")
        self.assert_equal(agg["average"], 0.85, "Should average to 0.85")
    
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
        await self.run_test("test_concat_strategy", self.test_concat_strategy)
        await self.run_test("test_join_strategy", self.test_join_strategy)
        await self.run_test("test_structured_strategy", self.test_structured_strategy)
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
                if "--verbose" in sys.argv:
                    print(f"  Traceback:\n{tb}")
        
        return self.tests_failed == 0


async def main():
    """Main test runner."""
    runner = TestRunner()
    success = await runner.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
