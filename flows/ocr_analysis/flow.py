"""
OCR Analysis Flow Implementation
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from core.base_flow import BaseFlow
from core.schema import FlowExecutionRequest, FlowExecutionResponse
from core.utils import generate_flow_id, format_template
from core.llm import LLMExecutor
from core.ocr import OCRExecutor

logger = logging.getLogger(__name__)


class OCRAnalysisFlow(BaseFlow):
    """OCR Analysis flow with configurable OCR providers"""
    
    async def execute(self, request: FlowExecutionRequest) -> FlowExecutionResponse:
        """Execute the OCR analysis flow"""
        flow_id = generate_flow_id()
        
        try:
            # Validate inputs
            self.validate_inputs(request.inputs)
            
            # Get inputs with defaults
            image_file = request.inputs.get("image_file")
            analysis_type = request.inputs.get("analysis_type", "comprehensive")
            ocr_provider = request.inputs.get("ocr_provider", "huggingface")
            ocr_model = request.inputs.get("ocr_model", "microsoft/trocr-base-printed")
            languages_str = request.inputs.get("languages", "en")
            
            # Parse languages
            languages = [lang.strip() for lang in languages_str.split(",")]
            
            # Execute steps according to DSL
            outputs = {}
            context_variables = {
                "analysis_type": analysis_type,
                "ocr_provider": ocr_provider,
                "ocr_model": ocr_model,
                "languages": languages_str
            }
            
            # Step 1: OCR processing
            ocr_step = next(step for step in self.definition.steps if step.type == "ocr")
            
            # Prepare OCR step configuration
            ocr_config = {
                "provider": ocr_provider,
                "model": ocr_model,
                "languages": languages,
                "return_bboxes": getattr(ocr_step, 'return_bboxes', True),
                "return_confidence": getattr(ocr_step, 'return_confidence', True),
                "include_blocks": getattr(ocr_step, 'include_blocks', False),
                "input": ocr_step.input
            }
            
            # Execute OCR step
            ocr_result = await OCRExecutor.execute_ocr_step(
                step_config=ocr_config,
                context_variables={"image_file": image_file},
                file_path=image_file if isinstance(image_file, str) else None
            )
            
            outputs["ocr_result"] = ocr_result["full_text"]
            context_variables.update({
                "ocr_result": ocr_result["full_text"],
                "ocr_confidence": ocr_result["confidence_avg"],
                "text_blocks_count": ocr_result["text_blocks_count"],
                "ocr_provider": ocr_result["provider"],
                "ocr_model": ocr_model
            })
            
            # Step 2: Combine template
            combine_step = next(step for step in self.definition.steps if step.type == "combine")
            combined_text = format_template(combine_step.template, context_variables)
            outputs["combined_analysis_input"] = combined_text
            context_variables["combined_analysis_input"] = combined_text
            
            # Step 3: LLM analysis
            llm_step = next(step for step in self.definition.steps if step.type == "llm")
            
            # Load prompt template
            prompt_template = self._load_prompt(llm_step.prompt_file)
            
            # Prepare LLM step configuration
            llm_config = {
                "model": llm_step.model,
                "provider": getattr(llm_step, 'provider', 'ollama'),
                "temperature": getattr(llm_step, 'temperature', 0.3),
                "max_tokens": getattr(llm_step, 'max_tokens', None),
                "system_prompt": getattr(llm_step, 'system_prompt', None),
                "input": llm_step.input
            }
            
            # Execute LLM step
            llm_result = await LLMExecutor.execute_llm_step(
                step_config=llm_config,
                context_variables=context_variables,
                prompt_template=prompt_template
            )
            
            # Try to parse JSON response
            try:
                analysis_result = json.loads(llm_result["content"])
            except json.JSONDecodeError:
                # If not valid JSON, create structured response
                analysis_result = {
                    "raw_analysis": llm_result["content"],
                    "ocr_metadata": {
                        "provider": ocr_result["provider"],
                        "confidence": ocr_result["confidence_avg"],
                        "processing_time": ocr_result["processing_time"],
                        "text_blocks": ocr_result["text_blocks_count"]
                    },
                    "llm_metadata": {
                        "model": llm_result["model"],
                        "provider": llm_result["provider"],
                        "usage": llm_result["usage"]
                    }
                }
            
            # Add OCR metadata to the result
            if isinstance(analysis_result, dict):
                analysis_result["ocr_metadata"] = {
                    "provider": ocr_result["provider"],
                    "confidence_avg": ocr_result["confidence_avg"],
                    "processing_time": ocr_result["processing_time"],
                    "text_blocks_count": ocr_result["text_blocks_count"],
                    "image_info": ocr_result.get("image_info", {})
                }
            
            outputs["analysis_result"] = analysis_result
            
            # Return final result
            final_output_key = self.definition.output.key
            final_result = outputs.get(final_output_key)
            
            return FlowExecutionResponse(
                flow_id=flow_id,
                status="completed",
                result=final_result
            )
            
        except Exception as e:
            logger.error(f"OCR analysis flow execution failed: {e}")
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
            """Execute the OCR analysis flow"""
            try:
                response = await self.execute(request)
                return response
            except Exception as e:
                logger.error(f"Flow execution error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.post("/upload")
        async def upload_and_analyze(
            file: UploadFile = File(...),
            analysis_type: str = Form("comprehensive"),
            ocr_provider: str = Form("huggingface"),
            ocr_model: str = Form("microsoft/trocr-base-printed"),
            languages: str = Form("en")
        ):
            """Upload file and analyze with OCR + LLM"""
            try:
                # Save uploaded file temporarily
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Create request
                    request = FlowExecutionRequest(inputs={
                        "image_file": tmp_file_path,
                        "analysis_type": analysis_type,
                        "ocr_provider": ocr_provider,
                        "ocr_model": ocr_model,
                        "languages": languages
                    })
                    
                    # Execute flow
                    response = await self.execute(request)
                    return response
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                        
            except Exception as e:
                logger.error(f"Upload and analyze error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.get("/info")
        async def get_flow_info():
            """Get flow information"""
            return self.get_catalog_info()
        
        @router.get("/ocr-providers")
        async def get_ocr_providers():
            """Get available OCR providers"""
            try:
                return OCRExecutor.get_provider_info()
            except Exception as e:
                logger.error(f"OCR providers info error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.get("/ocr-health")
        async def check_ocr_health(provider: str = "huggingface"):
            """Check OCR provider health"""
            try:
                health_info = await OCRExecutor.health_check(provider)
                return health_info
            except Exception as e:
                logger.error(f"OCR health check error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return router
