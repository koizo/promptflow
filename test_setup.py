#!/usr/bin/env python3
"""
Test script to verify the core application structure
"""
import asyncio
import sys
from pathlib import Path

from core.flow_registry import FlowRegistry
from core.state_store import StateStore
from core.config import settings


async def test_core_components():
    """Test core components"""
    print("🧪 Testing AI Inference Platform Core Components")
    print("=" * 50)
    
    # Test 1: Configuration
    print("1. Testing configuration...")
    print(f"   ✅ Redis URL: {settings.redis_url}")
    print(f"   ✅ Flows directory: {settings.flows_dir}")
    print(f"   ✅ API host: {settings.api_host}:{settings.api_port}")
    
    # Test 2: Flow Registry
    print("\n2. Testing flow registry...")
    registry = FlowRegistry()
    
    try:
        flows = registry.load_flows()
        print(f"   ✅ Loaded {len(flows)} flows")
        
        for flow_name, flow in flows.items():
            print(f"   • {flow_name}: {flow.metadata.description}")
        
        # Test catalog
        catalog = registry.get_flow_catalog()
        print(f"   ✅ Generated catalog with {len(catalog)} entries")
        
    except Exception as e:
        print(f"   ❌ Flow registry error: {e}")
        return False
    
    # Test 3: State Store (Redis)
    print("\n3. Testing state store...")
    state_store = StateStore()
    
    try:
        await state_store.initialize()
        ping_result = await state_store.ping()
        print(f"   ✅ Redis connection: {ping_result}")
        
        # Test basic operations
        from core.schema import FlowState
        from core.utils import generate_flow_id, get_current_timestamp
        
        test_flow_id = generate_flow_id()
        test_state = FlowState(
            flow_id=test_flow_id,
            flow_name="test_flow",
            current_step=0,
            inputs={"test": "data"},
            outputs={},
            status="running",
            created_at=get_current_timestamp(),
            updated_at=get_current_timestamp()
        )
        
        # Save and retrieve
        save_result = await state_store.save_flow_state(test_state)
        print(f"   ✅ Save state: {save_result}")
        
        retrieved_state = await state_store.get_flow_state(test_flow_id)
        print(f"   ✅ Retrieve state: {retrieved_state is not None}")
        
        # Cleanup
        await state_store.delete_flow_state(test_flow_id)
        await state_store.close()
        
    except Exception as e:
        print(f"   ❌ State store error: {e}")
        print(f"   💡 Make sure Redis is running: docker-compose up redis")
        return False
    
    # Test 4: Flow Validation
    print("\n4. Testing flow validation...")
    try:
        validation_results = registry.validate_all_flows()
        
        all_valid = all(result["valid"] for result in validation_results.values())
        print(f"   ✅ All flows valid: {all_valid}")
        
        for flow_name, result in validation_results.items():
            status = "✅" if result["valid"] else "❌"
            print(f"   {status} {flow_name}")
            
            if not result["valid"]:
                for error in result.get("errors", []):
                    print(f"      Error: {error}")
        
    except Exception as e:
        print(f"   ❌ Validation error: {e}")
        return False
    
    print("\n🎉 All core components tested successfully!")
    return True


async def test_sample_flow():
    """Test the sample flow execution"""
    print("\n🧪 Testing Sample Flow Execution")
    print("=" * 50)
    
    registry = FlowRegistry()
    flows = registry.load_flows()
    
    if "sample_flow" not in flows:
        print("❌ Sample flow not found")
        return False
    
    try:
        # Test flow execution
        test_inputs = {
            "text_input": "This is a test input for the sample flow.",
            "user_note": "Testing the flow execution"
        }
        
        print(f"📥 Test inputs: {test_inputs}")
        
        response = await registry.execute_flow("sample_flow", test_inputs)
        
        print(f"✅ Flow execution completed")
        print(f"📤 Response: {response.model_dump()}")
        
        return response.status == "completed"
        
    except Exception as e:
        print(f"❌ Sample flow execution failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 AI Inference Platform - Core Structure Test")
    print("=" * 60)
    
    # Run tests
    success = asyncio.run(test_core_components())
    
    if success:
        success = asyncio.run(test_sample_flow())
    
    if success:
        print("\n🎉 All tests passed! Core structure is working correctly.")
        print("\n📋 Next steps:")
        print("   1. Start the application: uvicorn main:app --reload")
        print("   2. Or use Docker: docker-compose up")
        print("   3. Visit: http://localhost:8000")
        print("   4. Check catalog: http://localhost:8000/catalog")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
