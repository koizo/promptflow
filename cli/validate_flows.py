#!/usr/bin/env python3
"""
CLI tool for validating flows and metadata
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.flow_registry import FlowRegistry


def validate_flows(flow_name: str = None):
    """Validate flows"""
    
    registry = FlowRegistry()
    
    try:
        # Load flows
        flows = registry.load_flows()
        
        if not flows:
            print("❌ No flows found")
            return False
        
        # Validate all flows
        validation_results = registry.validate_all_flows()
        
        # Filter by specific flow if requested
        if flow_name:
            if flow_name not in validation_results:
                print(f"❌ Flow '{flow_name}' not found")
                return False
            validation_results = {flow_name: validation_results[flow_name]}
        
        # Display results
        all_valid = True
        
        for flow_name, result in validation_results.items():
            if result["valid"]:
                print(f"✅ {flow_name}: Valid")
                
                # Show metadata
                metadata = result.get("metadata", {})
                print(f"   Description: {metadata.get('description', 'N/A')}")
                print(f"   Version: {metadata.get('version', 'N/A')}")
                print(f"   Author: {metadata.get('author', 'N/A')}")
                
                # Show inputs
                definition = result.get("definition", {})
                inputs = definition.get("inputs", [])
                if inputs:
                    print(f"   Inputs: {len(inputs)}")
                    for inp in inputs:
                        required = "required" if inp.get("required", True) else "optional"
                        print(f"     • {inp['name']} ({inp['type']}) - {required}")
                
                # Show steps
                steps = definition.get("steps", [])
                print(f"   Steps: {len(steps)}")
                for i, step in enumerate(steps):
                    print(f"     {i+1}. {step['type']} -> {step['output_key']}")
                
            else:
                print(f"❌ {flow_name}: Invalid")
                for error in result.get("errors", []):
                    print(f"   Error: {error}")
                all_valid = False
            
            print()  # Empty line between flows
        
        # Summary
        total_flows = len(validation_results)
        valid_flows = sum(1 for r in validation_results.values() if r["valid"])
        
        print(f"📊 Summary: {valid_flows}/{total_flows} flows are valid")
        
        return all_valid
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Validate AI Inference Platform flows")
    
    parser.add_argument("flow_name", nargs="?", 
                       help="Name of specific flow to validate (optional)")
    parser.add_argument("--json", action="store_true",
                       help="Output results in JSON format")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.json:
        # JSON output mode
        registry = FlowRegistry()
        flows = registry.load_flows()
        validation_results = registry.validate_all_flows()
        
        if args.flow_name:
            if args.flow_name in validation_results:
                validation_results = {args.flow_name: validation_results[args.flow_name]}
            else:
                validation_results = {}
        
        print(json.dumps(validation_results, indent=2))
        return
    
    # Regular validation
    success = validate_flows(args.flow_name)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
