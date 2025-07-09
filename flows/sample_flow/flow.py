"""
Sample Flow Implementation with Real LLM Integration
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from core.base_flow import BaseFlow
from core.schema import FlowExecutionRequest, FlowExecutionResponse
from core.utils import generate_flow_id, format_template
from core.llm import LLMExecutor

logger = logging.getLogger(__name__)


class SampleFlow(BaseFlow):
    """Sample flow with real LLM integration"""
    
    async def execute(self, request: FlowExecutionRequest) -> FlowExecutionResponse:
        """Execute the sample flow with real LLM"""
        flow_id = generate_flow_id()
        
        try:
            # Validate inputs
            self.validate_inputs(request.inputs)
            
            # Get inputs
            text_input = request.inputs.get("text_input", "")
            user_note = request.inputs.get("user_note", "No note provided")
            
            # Execute steps according to DSL
            outputs = {}
            context_variables = {
                "text_input": text_input,
                "user_note": user_note,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Step 1: Combine template
            combined_step = next(step for step in self.definition.steps if step.type == "combine")
            combined_text = format_template(combined_step.template, context_variables)
            outputs["combined_text"] = combined_text
            context_variables["combined_text"] = combined_text
            
            # Step 2: Real LLM processing
            llm_step = next(step for step in self.definition.steps if step.type == "llm")
            
            # Load prompt template
            prompt_template = self._load_prompt(llm_step.prompt_file)
            
            # Prepare LLM step configuration
            step_config = {
                "model": llm_step.model,
                "provider": getattr(llm_step, 'provider', 'ollama'),
                "temperature": getattr(llm_step, 'temperature', 0.7),
                "max_tokens": getattr(llm_step, 'max_tokens', None),
                "system_prompt": getattr(llm_step, 'system_prompt', None),
                "input": llm_step.input
            }
            
            # Execute LLM step
            llm_result = await LLMExecutor.execute_llm_step(
                step_config=step_config,
                context_variables=context_variables,
                prompt_template=prompt_template
            )
            
            # Try to parse JSON response, fallback to raw content
            try:
                processed_result = json.loads(llm_result["content"])
            except json.JSONDecodeError:
                # If not valid JSON, create a structured response
                processed_result = {
                    "raw_response": llm_result["content"],
                    "model_info": {
                        "model": llm_result["model"],
                        "provider": llm_result["provider"],
                        "usage": llm_result["usage"]
                    }
                }
            
            outputs["processed_result"] = processed_result
            
            # Return final result
            final_output_key = self.definition.output.key
            final_result = outputs.get(final_output_key)
            
            return FlowExecutionResponse(
                flow_id=flow_id,
                status="completed",
                result=final_result
            )
            
        except Exception as e:
            logger.error(f"Sample flow execution failed: {e}")
            return FlowExecutionResponse(
                flow_id=flow_id,
                status="failed",
                error=str(e)
            )
    
    def get_router(self) -> APIRouter:
        """Get FastAPI router for this flow"""
        router = APIRouter()
        
        @router.post("/execute")
        async def execute_flow(request: FlowExecutionRequest):
            """Execute the sample flow"""
            try:
                response = await self.execute(request)
                return response
            except Exception as e:
                logger.error(f"Flow execution error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.get("/info")
        async def get_flow_info():
            """Get flow information"""
            return self.get_catalog_info()
        
        @router.get("/schema")
        async def get_flow_schema():
            """Get flow input/output schema"""
            return {
                "inputs": [inp.model_dump() for inp in self.definition.inputs],
                "steps": [step.model_dump() for step in self.definition.steps],
                "output": self.definition.output.model_dump()
            }
        
        @router.get("/llm-health")
        async def check_llm_health():
            """Check LLM provider health"""
            try:
                health_info = await LLMExecutor.health_check("ollama")
                return health_info
            except Exception as e:
                logger.error(f"LLM health check error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return router
