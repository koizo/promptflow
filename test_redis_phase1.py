#!/usr/bin/env python3
"""
Test Redis Integration - Phase 1
Test StateStore integration with FlowRunner and enhanced FlowContext
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.flow_engine.flow_runner import FlowRunner
from core.executors.base_executor import FlowContext
from core.schema import FlowState, FlowStatus, FlowStepState, StepStatus
from datetime import datetime


async def test_flow_runner_redis_integration():
    """Test FlowRunner Redis StateStore integration"""
    print("🧪 Testing FlowRunner Redis Integration...")
    
    # Initialize FlowRunner
    flow_runner = FlowRunner()
    
    try:
        # Test initialization
        await flow_runner.initialize()
        print("✅ FlowRunner initialized successfully")
        
        # Test Redis connection
        if flow_runner.redis_enabled:
            ping_result = await flow_runner.state_store.ping()
            print(f"✅ Redis connection test: {ping_result}")
        else:
            print("⚠️  Redis disabled - running in memory mode")
        
        # Test StateStore operations
        if flow_runner.redis_enabled:
            await test_state_store_operations(flow_runner.state_store)
        
    except Exception as e:
        print(f"❌ FlowRunner initialization failed: {e}")
        return False
    
    finally:
        # Cleanup
        await flow_runner.stop()
        print("🧹 FlowRunner stopped")
    
    return True


async def test_state_store_operations(state_store):
    """Test basic StateStore operations"""
    print("\n🧪 Testing StateStore Operations...")
    
    # Create test flow state
    flow_state = FlowState(
        flow_id="test-flow-123",
        flow_name="test_flow",
        status=FlowStatus.RUNNING,
        inputs={"test_input": "test_value"},
        started_at=datetime.utcnow(),
        steps=[
            FlowStepState(
                step_name="step1",
                status=StepStatus.COMPLETED,
                inputs={"input": "value"},
                outputs={"output": "result"},
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                execution_time=1.5
            ),
            FlowStepState(
                step_name="step2",
                status=StepStatus.RUNNING,
                inputs={"input": "value2"},
                started_at=datetime.utcnow()
            )
        ],
        current_step="step2"
    )
    
    try:
        # Test save flow state
        save_result = await state_store.save_flow_state(flow_state)
        print(f"✅ Save flow state: {save_result}")
        
        # Test retrieve flow state
        retrieved_state = await state_store.get_flow_state("test-flow-123")
        if retrieved_state:
            print(f"✅ Retrieved flow state: {retrieved_state.flow_name}")
            print(f"   Status: {retrieved_state.status}")
            print(f"   Steps: {len(retrieved_state.steps)}")
            print(f"   Current step: {retrieved_state.current_step}")
        else:
            print("❌ Failed to retrieve flow state")
        
        # Test list active flows
        active_flows = await state_store.list_active_flows()
        print(f"✅ Active flows: {active_flows}")
        
        # Test flow data operations
        data_set = await state_store.set_flow_data("test-flow-123", "metadata", {"test": "data"})
        print(f"✅ Set flow data: {data_set}")
        
        data_get = await state_store.get_flow_data("test-flow-123", "metadata")
        print(f"✅ Get flow data: {data_get}")
        
        # Test cleanup
        delete_result = await state_store.delete_flow_state("test-flow-123")
        print(f"✅ Delete flow state: {delete_result}")
        
    except Exception as e:
        print(f"❌ StateStore operation failed: {e}")
        return False
    
    return True


def test_flow_context_enhancements():
    """Test FlowContext enhancements"""
    print("\n🧪 Testing FlowContext Enhancements...")
    
    try:
        # Test FlowContext with flow_id
        context1 = FlowContext("test_flow", {"input": "value"})
        print(f"✅ FlowContext created with auto-generated flow_id: {context1.flow_id}")
        
        # Test FlowContext with explicit flow_id
        context2 = FlowContext("test_flow", {"input": "value"}, flow_id="explicit-flow-id")
        print(f"✅ FlowContext created with explicit flow_id: {context2.flow_id}")
        
        # Test Redis settings
        print(f"✅ Redis enabled: {context1.redis_enabled}")
        print(f"✅ Persist steps: {context1.persist_steps}")
        
        return True
        
    except Exception as e:
        print(f"❌ FlowContext test failed: {e}")
        return False


def test_flow_state_schema():
    """Test enhanced FlowState schema"""
    print("\n🧪 Testing FlowState Schema...")
    
    try:
        # Create FlowState with steps
        flow_state = FlowState(
            flow_id="schema-test-123",
            flow_name="schema_test",
            status=FlowStatus.RUNNING,
            inputs={"test": "input"},
            started_at=datetime.utcnow()
        )
        
        # Test step state operations
        flow_state.update_step_state("step1", status=StepStatus.COMPLETED, outputs={"result": "success"})
        flow_state.update_step_state("step2", status=StepStatus.RUNNING, inputs={"data": "test"})
        
        print(f"✅ FlowState created with {len(flow_state.steps)} steps")
        
        # Test helper methods
        completed_steps = flow_state.get_completed_steps()
        print(f"✅ Completed steps: {completed_steps}")
        
        failed_steps = flow_state.get_failed_steps()
        print(f"✅ Failed steps: {failed_steps}")
        
        # Test step retrieval
        step1_state = flow_state.get_step_state("step1")
        if step1_state:
            print(f"✅ Retrieved step1 state: {step1_state.status}")
        
        return True
        
    except Exception as e:
        print(f"❌ FlowState schema test failed: {e}")
        return False


async def main():
    """Run all Phase 1 tests"""
    print("🚀 Redis Integration Phase 1 Tests")
    print("=" * 50)
    
    # Test 1: FlowContext enhancements
    test1_result = test_flow_context_enhancements()
    
    # Test 2: FlowState schema
    test2_result = test_flow_state_schema()
    
    # Test 3: FlowRunner Redis integration
    test3_result = await test_flow_runner_redis_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"FlowContext Enhancements: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"FlowState Schema: {'✅ PASS' if test2_result else '❌ FAIL'}")
    print(f"FlowRunner Redis Integration: {'✅ PASS' if test3_result else '❌ FAIL'}")
    
    all_passed = test1_result and test2_result and test3_result
    print(f"\nOverall Result: {'🎉 ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
