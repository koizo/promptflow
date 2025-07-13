"""API router for document analysis flow endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from typing import Dict, Any, Optional
import tempfile
import os
from pathlib import Path
import logging

from core.document_extraction.document_manager import DocumentExtractionManager
from core.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/document-analysis", tags=["Document Analysis Flow"])

# Global manager instances
_doc_manager: Optional[DocumentExtractionManager] = None
_llm_manager: Optional[LLMManager] = None


def get_document_manager() -> DocumentExtractionManager:
    """Get or create document extraction manager instance."""
    global _doc_manager
    if _doc_manager is None:
        # Use default configuration for now
        doc_config = {
            'default_provider': 'langchain',
            'providers': {
                'langchain': {
                    'chunk_size': 1000,
                    'chunk_overlap': 200,
                    'preserve_metadata': True
                }
            }
        }
        _doc_manager = DocumentExtractionManager(doc_config)
    return _doc_manager


def get_llm_manager() -> LLMManager:
    """Get or create LLM manager instance."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


@router.post("/analyze-document")
async def analyze_document_with_llm(
    file: UploadFile = File(...),
    analysis_prompt: str = Form("Analyze this document and provide a summary", description="Analysis prompt for LLM"),
    provider: Optional[str] = Form(None, description="Document extraction provider to use"),
    llm_model: Optional[str] = Form(None, description="LLM model to use for analysis"),
    chunk_text: bool = Form(False, description="Whether to split text into chunks for analysis"),
    doc_manager: DocumentExtractionManager = Depends(get_document_manager),
    llm_manager: LLMManager = Depends(get_llm_manager)
) -> Dict[str, Any]:
    """
    Extract text from document and analyze it with LLM.
    
    This endpoint combines document extraction with LLM analysis in a complete flow.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Get file extension
    file_extension = Path(file.filename).suffix.lower()
    
    # Check if format is supported
    if not doc_manager.supports_format(file_extension, provider):
        supported_formats = doc_manager.get_supported_formats(provider)
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {file_extension}. Supported formats: {supported_formats}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        try:
            # Write uploaded content to temp file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            temp_path = Path(temp_file.name)
            
            # Step 1: Extract text from document
            if chunk_text:
                results = doc_manager.extract_with_chunking(temp_path, provider)
                extraction_result = {
                    "success": True,
                    "file_name": file.filename,
                    "provider": provider or doc_manager.default_provider,
                    "chunked": True,
                    "total_chunks": len(results),
                    "chunks": [result.to_dict() for result in results],
                    "text": "\n\n".join([r.text for r in results])
                }
            else:
                result = doc_manager.extract_text(temp_path, provider)
                extraction_result = {
                    "success": True,
                    "file_name": file.filename,
                    "provider": provider or doc_manager.default_provider,
                    "chunked": False,
                    "text": result.text,
                    **result.to_dict()
                }
            
            # Step 2: Perform LLM analysis
            try:
                # Prepare text for analysis
                if extraction_result["chunked"]:
                    # Analyze each chunk separately
                    chunk_analyses = []
                    for chunk in extraction_result["chunks"]:
                        response = await llm_manager.generate(
                            prompt=f"{analysis_prompt}\n\nDocument text:\n{chunk['text']}",
                            model=llm_model or "mistral"
                        )
                        chunk_analyses.append({
                            "chunk_index": chunk["metadata"].get("chunk_index", 0),
                            "analysis": response.content
                        })
                    
                    return {
                        "flow": "document_analysis",
                        "steps_completed": ["document_extraction", "llm_analysis"],
                        **extraction_result,
                        "llm_analysis": {
                            "prompt": analysis_prompt,
                            "model": llm_model or "mistral",
                            "chunk_analyses": chunk_analyses
                        }
                    }
                else:
                    # Analyze full document
                    response = await llm_manager.generate(
                        prompt=f"{analysis_prompt}\n\nDocument text:\n{extraction_result['text']}",
                        model=llm_model or "mistral"
                    )
                    
                    return {
                        "flow": "document_analysis",
                        "steps_completed": ["document_extraction", "llm_analysis"],
                        **extraction_result,
                        "llm_analysis": {
                            "prompt": analysis_prompt,
                            "model": llm_model or "mistral",
                            "analysis": response.content
                        }
                    }
                    
            except Exception as e:
                logger.error(f"Error in LLM analysis: {str(e)}")
                # Return extraction result even if LLM analysis fails
                return {
                    "flow": "document_analysis",
                    "steps_completed": ["document_extraction"],
                    "steps_failed": ["llm_analysis"],
                    **extraction_result,
                    "llm_analysis": {
                        "error": f"LLM analysis failed: {str(e)}"
                    }
                }
                
        except Exception as e:
            logger.error(f"Error extracting text from {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Document analysis flow failed: {str(e)}")
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file.name}: {str(e)}")


@router.get("/info")
async def get_flow_info() -> Dict[str, Any]:
    """Get information about the document analysis flow."""
    return {
        "flow_name": "document_analysis",
        "description": "Complete document analysis flow combining text extraction and LLM analysis",
        "steps": [
            {
                "name": "document_extraction",
                "description": "Extract text from uploaded document using LangChain providers"
            },
            {
                "name": "llm_analysis", 
                "description": "Analyze extracted text using LLM with custom prompt"
            }
        ],
        "supported_formats": [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".txt", ".csv", ".md", ".rtf"],
        "available_providers": {
            "document_extraction": ["langchain"],
            "llm": ["ollama", "openai"]
        }
    }


@router.get("/health")
async def health_check(
    doc_manager: DocumentExtractionManager = Depends(get_document_manager),
    llm_manager: LLMManager = Depends(get_llm_manager)
) -> Dict[str, Any]:
    """Health check for document analysis flow components."""
    try:
        # Check document extraction health
        doc_health = doc_manager.health_check()
        
        # Check LLM health
        llm_health = await llm_manager.health_check("ollama")
        
        # Determine if document extraction is healthy
        doc_healthy = False
        if isinstance(doc_health, dict):
            # Check if any provider is healthy
            for provider_name, provider_status in doc_health.items():
                if isinstance(provider_status, dict) and provider_status.get('status') == 'healthy':
                    doc_healthy = True
                    break
        
        # Determine if LLM is healthy
        llm_healthy = llm_health if isinstance(llm_health, bool) else False
        
        return {
            "flow": "document_analysis",
            "status": "healthy" if doc_healthy and llm_healthy else "unhealthy",
            "components": {
                "document_extraction": {
                    "status": "healthy" if doc_healthy else "unhealthy",
                    "providers": doc_health
                },
                "llm": {
                    "status": "healthy" if llm_healthy else "unhealthy",
                    "details": llm_health
                }
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "flow": "document_analysis",
            "status": "unhealthy",
            "error": str(e)
        }
