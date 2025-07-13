#!/usr/bin/env python3
"""
Flow Generator CLI Tool

Generates YAML flow definitions from natural language descriptions using LLM.
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm.llm_manager import LLMManager
from core.flow_engine.yaml_loader import YAMLFlowLoader


class FlowGenerator:
    """Generates YAML flows from natural language descriptions."""
    
    def __init__(self):
        self.llm_manager = LLMManager()
        self.yaml_loader = YAMLFlowLoader()
        
        # Available executors with descriptions
        self.available_executors = {
            "file_handler": {
                "description": "Handles file uploads, validation, and temporary storage",
                "inputs": ["file_content", "file_path"],
                "outputs": ["temp_path", "file_type", "file_size"]
            },
            "document_extractor": {
                "description": "Extracts text from documents (PDF, Word, Excel, PowerPoint)",
                "inputs": ["file_path", "extract_images"],
                "outputs": ["text", "images", "page_count", "word_count"]
            },
            "ocr_processor": {
                "description": "Performs OCR on images to extract text",
                "inputs": ["image_path", "images", "provider"],
                "outputs": ["text", "confidence", "bounding_boxes"]
            },
            "vision_classifier": {
                "description": "Classifies images using computer vision models (HuggingFace or OpenAI Vision)",
                "inputs": ["file", "provider", "model", "top_k"],
                "outputs": ["predictions", "confidence_scores", "processing_time"]
            },
            "whisper_processor": {
                "description": "Converts speech to text using Whisper models (local or OpenAI API)",
                "inputs": ["file", "provider", "model_size", "language", "temperature"],
                "outputs": ["transcription", "language_detected", "confidence", "segments"]
            },
            "sentiment_analyzer": {
                "description": "Analyzes sentiment and emotions in text (HuggingFace or LLM)",
                "inputs": ["text", "provider", "analysis_type", "model"],
                "outputs": ["sentiment", "confidence", "emotions", "aspects"]
            },
            "llm_analyzer": {
                "description": "Analyzes text using Large Language Models",
                "inputs": ["text", "prompt", "model", "temperature"],
                "outputs": ["analysis", "summary", "key_points"]
            },
            "image_handler": {
                "description": "Processes and optimizes images",
                "inputs": ["image_data", "resize", "format"],
                "outputs": ["processed_image", "metadata"]
            },
            "data_combiner": {
                "description": "Combines data from multiple sources",
                "inputs": ["sources", "combine_method"],
                "outputs": ["combined_data", "source_count"]
            },
            "response_formatter": {
                "description": "Formats responses in various templates",
                "inputs": ["data", "template", "format"],
                "outputs": ["formatted_response"]
            }
        }
        
    @staticmethod
    def to_snake_case(text: str) -> str:
        """Convert text to snake_case."""
        # Replace spaces and special characters with underscores
        text = re.sub(r'[^\w\s]', '', text)
        # Replace spaces with underscores
        text = re.sub(r'\s+', '_', text)
        # Convert to lowercase
        text = text.lower()
        # Remove multiple consecutive underscores
        text = re.sub(r'_+', '_', text)
        # Remove leading/trailing underscores
        text = text.strip('_')
        return text
        
        # Available executors with descriptions
        self.available_executors = {
            "file_handler": {
                "description": "Handles file uploads, validation, and temporary storage",
                "inputs": ["file_content", "file_path"],
                "outputs": ["temp_path", "file_type", "file_size"]
            },
            "document_extractor": {
                "description": "Extracts text from documents (PDF, Word, Excel, PowerPoint)",
                "inputs": ["file_path", "extract_images"],
                "outputs": ["text", "images", "page_count", "word_count"]
            },
            "ocr_processor": {
                "description": "Performs OCR on images to extract text",
                "inputs": ["image_path", "images", "provider"],
                "outputs": ["text", "confidence", "bounding_boxes"]
            },
            "vision_classifier": {
                "description": "Classifies images using computer vision models (HuggingFace or OpenAI Vision)",
                "inputs": ["file", "provider", "model", "top_k"],
                "outputs": ["predictions", "confidence_scores", "processing_time"]
            },
            "whisper_processor": {
                "description": "Converts speech to text using Whisper models (local or OpenAI API)",
                "inputs": ["file", "provider", "model_size", "language", "temperature"],
                "outputs": ["transcription", "language_detected", "confidence", "segments"]
            },
            "sentiment_analyzer": {
                "description": "Analyzes sentiment and emotions in text (HuggingFace or LLM)",
                "inputs": ["text", "provider", "analysis_type", "model"],
                "outputs": ["sentiment", "confidence", "emotions", "aspects"]
            },
            "llm_analyzer": {
                "description": "Analyzes text using Large Language Models",
                "inputs": ["text", "prompt", "model", "temperature"],
                "outputs": ["analysis", "summary", "key_points"]
            },
            "image_handler": {
                "description": "Processes and optimizes images",
                "inputs": ["image_data", "resize", "format"],
                "outputs": ["processed_image", "metadata"]
            },
            "data_combiner": {
                "description": "Combines data from multiple sources",
                "inputs": ["sources", "combine_method"],
                "outputs": ["combined_data", "source_count"]
            },
            "response_formatter": {
                "description": "Formats responses in various templates",
                "inputs": ["data", "template", "format"],
                "outputs": ["formatted_response"]
            }
        }
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for flow generation."""
        executor_docs = "\n".join([
            f"- {name}: {info['description']}\n  Inputs: {info['inputs']}\n  Outputs: {info['outputs']}"
            for name, info in self.available_executors.items()
        ])
        
        return f"""You are an AI workflow generator that creates YAML flow definitions from natural language descriptions.

Available Executors:
{executor_docs}

Your task is to generate a valid YAML flow definition that accomplishes the user's described workflow.

YAML Flow Structure:
```yaml
name: "flow_name_in_snake_case"
version: "1.0.0"
description: "Brief description"
inputs:
  - name: "input_name"
    type: "string|file|integer|boolean|float"
    required: true|false
    default: "default_value"  # optional
    description: "Input description"

steps:
  - name: "step_name"
    executor: "executor_name"
    config:
      parameter: "{{{{ inputs.input_name }}}}"  # Use Jinja2 templates
      another_param: "{{{{ steps.previous_step.output_name }}}}"
    depends_on: ["previous_step"]  # optional
    condition: "{{{{ inputs.enable_feature }}}}"  # optional

  # ALWAYS include response_formatter as the final step
  - name: "format_response"
    executor: "response_formatter"
    config:
      data: "{{{{ steps.previous_final_step.output }}}}"
      template: "structured"
      format: "json"
    depends_on: ["previous_final_step"]

outputs:
  - name: "output_name"
    value: "{{{{ steps.format_response.formatted_response }}}}"
    description: "Output description"

config:
  execution:
    mode: "async"  # or "sync" for simple flows
    timeout: 300
    max_retries: 3
    queue: "flow_name_queue"
    worker: "celery-worker-flow_name"
  
  validation:
    required_inputs: ["input1", "input2"]
    
  metadata:
    category: "category_name"  # e.g., "computer_vision", "text_analysis", "audio_processing"
    tags: ["tag1", "tag2", "tag3"]
    author: "AI Inference Platform"
```

Common AI Workflow Examples:

1. Image Classification:
   - Use vision_classifier with provider="huggingface" for fast local processing
   - Use vision_classifier with provider="openai" for detailed natural language descriptions
   - Common models: "google/vit-base-patch16-224", "microsoft/resnet-50"
   - Category: "computer_vision", Tags: ["image", "classification", "vision"]
   - ALWAYS end with response_formatter to structure the classification results

2. Speech Transcription:
   - Use whisper_processor with provider="local" for privacy
   - Use whisper_processor with provider="openai" for best accuracy
   - Model sizes: "tiny", "base", "small", "medium", "large"
   - Category: "audio_processing", Tags: ["speech", "transcription", "whisper"]
   - ALWAYS end with response_formatter to structure the transcription results

3. Sentiment Analysis:
   - Use sentiment_analyzer with provider="huggingface" for specialized models
   - Use sentiment_analyzer with provider="llm" for custom analysis
   - Analysis types: "basic", "emotions", "aspects", "comprehensive"
   - Category: "text_analysis", Tags: ["sentiment", "nlp", "analysis"]
   - ALWAYS end with response_formatter to structure the sentiment results

4. Document Processing:
   - Chain file_handler → document_extractor → ocr_processor → llm_analyzer → response_formatter
   - Use ocr_processor for images within documents
   - Use sentiment_analyzer for customer feedback analysis
   - Category: "document_processing", Tags: ["ocr", "document", "extraction"]
   - ALWAYS end with response_formatter to structure the analysis results

Response Formatter Configuration:
- template: "structured" (default), "summary", "detailed", "custom"
- format: "json" (default), "yaml", "text", "html"
- The response_formatter step is CRITICAL for callbacks and API responses

Rules:
1. ALWAYS use snake_case for flow names (e.g., "image_classification", "speech_transcription")
2. Use Jinja2 template syntax for dynamic values: {{{{ inputs.name }}}}, {{{{ steps.step_name.output }}}}
3. Chain steps logically using depends_on
4. Choose appropriate executors based on the task
5. Include meaningful input validation and descriptions
6. Provide clear step names and descriptions
7. Always include the config section with execution, validation, and metadata
8. Set appropriate queue names matching the flow name
9. Choose relevant category and tags for metadata
10. ALWAYS include response_formatter as the final step - this is MANDATORY for callbacks
11. The final output should reference steps.format_response.formatted_response
12. Only use the available executors listed above
13. Return ONLY the YAML content, no explanations or markdown formatting"""

    async def generate_flow(self, description: str, interactive: bool = False) -> str:
        """Generate a flow from natural language description."""
        
        if interactive:
            description = await self._interactive_refinement(description)
        
        try:
            # Generate the flow using LLM
            response = await self.llm_manager.generate(
                prompt=f"Generate a YAML flow for: {description}",
                system_prompt=self.get_system_prompt(),
                model="mistral",
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2000
            )
            
            yaml_content = response.content.strip()
            
            # Clean up the response (remove any markdown formatting)
            if yaml_content.startswith("```yaml"):
                yaml_content = yaml_content.split("```yaml")[1]
            if yaml_content.endswith("```"):
                yaml_content = yaml_content.rsplit("```", 1)[0]
            
            yaml_content = yaml_content.strip()
            
            # Validate the generated YAML
            await self._validate_generated_flow(yaml_content)
            
            return yaml_content
            
        except Exception as e:
            raise Exception(f"Failed to generate flow: {str(e)}")
    
    async def _interactive_refinement(self, initial_description: str) -> str:
        """Interactive mode to refine the flow description."""
        print(f"\n🤖 Initial description: {initial_description}")
        
        # Ask clarifying questions
        questions = [
            "What type of input will this flow process? (file, text, image, etc.)",
            "What should be the main output of this flow?",
            "Are there any specific requirements or constraints?",
            "Should this flow run synchronously or asynchronously?"
        ]
        
        answers = []
        for question in questions:
            answer = input(f"\n❓ {question}\n   Answer (or press Enter to skip): ").strip()
            if answer:
                answers.append(f"{question} {answer}")
        
        if answers:
            enhanced_description = f"{initial_description}\n\nAdditional details:\n" + "\n".join(answers)
            return enhanced_description
        
        return initial_description
    
    async def _validate_generated_flow(self, yaml_content: str) -> None:
        """Validate the generated YAML flow."""
        try:
            # Parse YAML
            flow_data = yaml.safe_load(yaml_content)
            
            # Basic structure validation
            required_fields = ["name", "steps", "config"]
            for field in required_fields:
                if field not in flow_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate flow name is snake_case
            flow_name = flow_data.get("name", "")
            if not re.match(r'^[a-z][a-z0-9_]*$', flow_name):
                # Try to fix the name automatically
                corrected_name = self.to_snake_case(flow_name)
                print(f"⚠️  Flow name '{flow_name}' converted to snake_case: '{corrected_name}'")
                # Update the YAML content
                yaml_content = yaml_content.replace(f'name: "{flow_name}"', f'name: "{corrected_name}"')
                flow_data["name"] = corrected_name
            
            # Validate config section structure
            config = flow_data.get("config", {})
            required_config_sections = ["execution", "validation", "metadata"]
            for section in required_config_sections:
                if section not in config:
                    raise ValueError(f"Missing config section: {section}")
            
            # Validate execution config
            execution = config.get("execution", {})
            required_execution_fields = ["mode", "timeout", "max_retries"]
            for field in required_execution_fields:
                if field not in execution:
                    raise ValueError(f"Missing execution config field: {field}")
            
            # Validate executors exist
            steps = flow_data.get("steps", [])
            for step in steps:
                executor = step.get("executor")
                if executor not in self.available_executors:
                    raise ValueError(f"Unknown executor: {executor}")
            
            # Validate response_formatter is the final step
            if len(steps) > 0:
                final_step = steps[-1]
                if final_step.get("executor") != "response_formatter":
                    print("⚠️  Warning: Final step should be 'response_formatter' for proper callback handling")
            
            # Validate outputs reference response_formatter
            outputs = flow_data.get("outputs", [])
            for output in outputs:
                output_value = output.get("value", "")
                if "steps.format_response" not in output_value and "response_formatter" not in output_value:
                    print("⚠️  Warning: Outputs should reference response_formatter step for proper formatting")
            
            print("✅ Generated flow validation passed")
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax: {e}")
        except Exception as e:
            raise ValueError(f"Flow validation failed: {e}")
    
    def save_flow(self, yaml_content: str, output_path: Path) -> None:
        """Save the generated flow to a file."""
        try:
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the YAML content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            
            print(f"✅ Flow saved to: {output_path}")
            
        except Exception as e:
            raise Exception(f"Failed to save flow: {str(e)}")
    
    def preview_flow(self, yaml_content: str) -> None:
        """Preview the generated flow structure."""
        try:
            flow_data = yaml.safe_load(yaml_content)
            
            print("\n" + "="*60)
            print(f"📋 FLOW PREVIEW: {flow_data.get('name', 'Unnamed Flow')}")
            print("="*60)
            
            if flow_data.get('description'):
                print(f"📝 Description: {flow_data['description']}")
            
            # Show inputs
            inputs = flow_data.get('inputs', [])
            if inputs:
                print(f"\n📥 Inputs ({len(inputs)}):")
                for inp in inputs:
                    required = "required" if inp.get('required', True) else "optional"
                    print(f"  • {inp['name']} ({inp.get('type', 'string')}) - {required}")
            
            # Show steps
            steps = flow_data.get('steps', [])
            print(f"\n⚙️  Steps ({len(steps)}):")
            for i, step in enumerate(steps, 1):
                executor = step.get('executor', 'unknown')
                depends = step.get('depends_on', [])
                dep_info = f" (depends on: {', '.join(depends)})" if depends else ""
                print(f"  {i}. {step['name']} → {executor}{dep_info}")
            
            # Show outputs
            outputs = flow_data.get('outputs', [])
            if outputs:
                print(f"\n📤 Outputs ({len(outputs)}):")
                for out in outputs:
                    print(f"  • {out['name']}: {out.get('value', 'N/A')}")
            
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error previewing flow: {e}")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate YAML flows from natural language descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Extract text from PDF and analyze sentiment"
  %(prog)s "Process image with OCR then summarize" --output my_flow.yaml
  %(prog)s --interactive "Analyze documents"
  %(prog)s "OCR processing workflow" --preview-only
        """
    )
    
    parser.add_argument(
        "description",
        help="Natural language description of the desired flow"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path (default: generated based on flow name)"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode with clarifying questions"
    )
    
    parser.add_argument(
        "-p", "--preview-only",
        action="store_true",
        help="Only preview the flow, don't save to file"
    )
    
    parser.add_argument(
        "--flows-dir",
        type=Path,
        default=Path("flows"),
        help="Directory to save flows (default: ./flows)"
    )
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting Flow Generator...")
        
        generator = FlowGenerator()
        
        # Generate the flow
        print(f"🧠 Generating flow for: {args.description}")
        yaml_content = await generator.generate_flow(
            args.description, 
            interactive=args.interactive
        )
        
        # Preview the flow
        generator.preview_flow(yaml_content)
        
        if not args.preview_only:
            # Determine output path
            if args.output:
                output_path = args.output
            else:
                # Generate filename from flow name
                flow_data = yaml.safe_load(yaml_content)
                flow_name = flow_data.get('name', 'generated_flow')
                safe_name = "".join(c for c in flow_name if c.isalnum() or c in ('_', '-')).lower()
                output_path = args.flows_dir / safe_name / "flow.yaml"
            
            # Save the flow
            generator.save_flow(yaml_content, output_path)
            
            print(f"\n🎉 Flow generation completed!")
            print(f"📁 Saved to: {output_path}")
            print(f"\n💡 To test your flow:")
            print(f"   cd {output_path.parent}")
            print(f"   # Add any required prompt files or configurations")
            print(f"   # Then test with the main application")
        else:
            print("\n👀 Preview mode - flow not saved")
            print("\n📋 Generated YAML:")
            print("-" * 40)
            print(yaml_content)
            print("-" * 40)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
