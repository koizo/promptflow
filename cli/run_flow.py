#!/usr/bin/env python3
"""
CLI tool for running flows locally
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.flow_registry import FlowRegistry
from core.state_store import StateStore
from core.config import settings


async def run_flow(flow_name: str, inputs: dict, callback_url: str = None):
    """Run a flow with given inputs"""
    
    # Initialize components
    registry = FlowRegistry()
    state_store = StateStore()
    
    try:
        # Initialize state store
        await state_store.initialize()
        
        # Load flows
        flows = registry.load_flows()
        
        if flow_name not in flows:
            print(f"❌ Flow '{flow_name}' not found")
            print(f"Available flows: {list(flows.keys())}")
            return False
        
        print(f"🚀 Running flow: {flow_name}")
        print(f"📥 Inputs: {json.dumps(inputs, indent=2)}")
        
        # Execute flow
        response = await registry.execute_flow(flow_name, inputs, callback_url)
        
        print(f"✅ Flow completed successfully")
        print(f"📤 Response: {json.dumps(response.model_dump(), indent=2)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flow execution failed: {e}")
        return False
        
    finally:
        await state_store.close()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Run AI Inference Platform flows locally")
    
    parser.add_argument("flow_name", help="Name of the flow to run")
    parser.add_argument("--input", "-i", action="append", 
                       help="Input in key=value format (can be used multiple times)")
    parser.add_argument("--input-file", "-f", 
                       help="JSON file containing inputs")
    parser.add_argument("--callback-url", 
                       help="Callback URL for async flows")
    parser.add_argument("--list-flows", "-l", action="store_true",
                       help="List available flows")
    
    args = parser.parse_args()
    
    if args.list_flows:
        # List available flows
        registry = FlowRegistry()
        flows = registry.load_flows()
        
        print("📋 Available flows:")
        for flow_name, flow in flows.items():
            print(f"  • {flow_name}: {flow.metadata.description}")
        
        return
    
    # Parse inputs
    inputs = {}
    
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                inputs.update(json.load(f))
        except Exception as e:
            print(f"❌ Failed to load input file: {e}")
            sys.exit(1)
    
    if args.input:
        for input_str in args.input:
            try:
                key, value = input_str.split('=', 1)
                inputs[key.strip()] = value.strip()
            except ValueError:
                print(f"❌ Invalid input format: {input_str}")
                print("Use format: key=value")
                sys.exit(1)
    
    if not inputs:
        print("❌ No inputs provided. Use --input or --input-file")
        sys.exit(1)
    
    # Run the flow
    success = asyncio.run(run_flow(args.flow_name, inputs, args.callback_url))
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
