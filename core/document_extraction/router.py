"""API router for core document text extraction endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
import tempfile
import os
from pathlib import Path
import logging

from .document_manager import DocumentExtractionManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/document-extraction", tags=["Document Extraction Core"])

# Global manager instance
_manager: Optional[DocumentExtractionManager] = None


def get_document_manager() -> DocumentExtractionManager:
    """Get or create document extraction manager instance."""
    global _manager
    if _manager is None:
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
        _manager = DocumentExtractionManager(doc_config)
    return _manager


@router.post("/extract")
async def extract_text_from_file(
    file: UploadFile = File(...),
    provider: Optional[str] = Query(None, description="Document extraction provider to use"),
    chunk_text: bool = Query(False, description="Whether to split text into chunks"),
    chunk_size: Optional[int] = Query(None, description="Size of text chunks"),
    chunk_overlap: Optional[int] = Query(None, description="Overlap between chunks"),
    manager: DocumentExtractionManager = Depends(get_document_manager)
) -> Dict[str, Any]:
    """
    Extract text from uploaded document file.
    
    Supports: PDF, Word, PowerPoint, Excel, Text, CSV, and more.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Get file extension
    file_extension = Path(file.filename).suffix.lower()
    
    # Check if format is supported
    if not manager.supports_format(file_extension, provider):
        supported_formats = manager.get_supported_formats(provider)
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
            
            # Extract text
            if chunk_text:
                results = manager.extract_with_chunking(
                    temp_path, 
                    provider, 
                    chunk_size, 
                    chunk_overlap
                )
                return {
                    "success": True,
                    "file_name": file.filename,
                    "provider": provider or manager.default_provider,
                    "chunked": True,
                    "total_chunks": len(results),
                    "chunks": [result.to_dict() for result in results]
                }
            else:
                result = manager.extract_text(temp_path, provider)
                return {
                    "success": True,
                    "file_name": file.filename,
                    "provider": provider or manager.default_provider,
                    "chunked": False,
                    **result.to_dict()
                }
                
        except Exception as e:
            logger.error(f"Error extracting text from {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file.name}: {str(e)}")


@router.get("/info")
async def get_extraction_info(
    provider: Optional[str] = Query(None, description="Specific provider to get info for"),
    manager: DocumentExtractionManager = Depends(get_document_manager)
) -> Dict[str, Any]:
    """Get information about document extraction providers and supported formats."""
    return {
        "available_providers": list(manager.providers.keys()),
        "default_provider": manager.default_provider,
        "provider_info": manager.get_provider_info(provider)
    }


@router.get("/supported-formats")
async def get_supported_formats(
    provider: Optional[str] = Query(None, description="Specific provider to check"),
    manager: DocumentExtractionManager = Depends(get_document_manager)
) -> Dict[str, Any]:
    """Get supported file formats for document extraction."""
    if provider:
        return {
            "provider": provider,
            "supported_formats": manager.get_supported_formats(provider)
        }
    else:
        return {
            "default_provider": manager.default_provider,
            "all_providers": {
                name: manager.get_supported_formats(name)
                for name in manager.providers.keys()
            }
        }


@router.get("/health")
async def health_check(
    provider: Optional[str] = Query(None, description="Specific provider to check"),
    manager: DocumentExtractionManager = Depends(get_document_manager)
) -> Dict[str, Any]:
    """Health check for document extraction providers."""
    return {
        "status": "healthy",
        "providers": manager.health_check(provider)
    }


# analyze-document endpoint moved to flows/document_analysis/router.py
