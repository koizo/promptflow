"""
Document Extractor Executor

Reusable executor for extracting text from various document formats.
Uses existing DocumentExtractionManager and LangChain providers.
Supports chunking, multiple formats, and metadata preservation.
"""

from typing import Dict, Any, List
from pathlib import Path
import logging

from .base_executor import BaseExecutor, ExecutionResult, FlowContext
from ..document_extraction.document_manager import DocumentExtractionManager

logger = logging.getLogger(__name__)


class DocumentExtractor(BaseExecutor):
    """
    Extract text from documents using LangChain providers.
    
    Supports PDF, Word, Excel, PowerPoint, text files, and more.
    Can optionally chunk text for optimal LLM processing.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.manager = None
    
    def _get_manager(self) -> DocumentExtractionManager:
        """Get or create document extraction manager."""
        if self.manager is None:
            # Use default configuration
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
            self.manager = DocumentExtractionManager(doc_config)
        return self.manager
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Extract text from document file.
        
        Config parameters:
        - file_path (required): Path to document file
        - provider (optional): Document extraction provider ('langchain')
        - chunk_text (optional): Whether to split text into chunks (default: False)
        - chunk_size (optional): Size of text chunks (default: 1000)
        - chunk_overlap (optional): Overlap between chunks (default: 200)
        """
        try:
            file_path = config.get('file_path')
            if not file_path:
                return ExecutionResult(
                    success=False,
                    error="file_path is required for document extraction"
                )
            
            file_path = Path(file_path)
            if not file_path.exists():
                return ExecutionResult(
                    success=False,
                    error=f"File not found: {file_path}"
                )
            
            # Get configuration
            provider = config.get('provider')
            chunk_text = config.get('chunk_text', False)
            chunk_size = config.get('chunk_size')
            chunk_overlap = config.get('chunk_overlap')
            
            manager = self._get_manager()
            
            # Check if format is supported
            file_extension = file_path.suffix.lower()
            if not manager.supports_format(file_extension, provider):
                supported_formats = manager.get_supported_formats(provider)
                return ExecutionResult(
                    success=False,
                    error=f"Unsupported file format: {file_extension}. Supported: {supported_formats}"
                )
            
            self.logger.info(f"Extracting text from {file_path} (chunk_text={chunk_text})")
            
            # Extract text
            if chunk_text:
                results = manager.extract_with_chunking(
                    file_path, 
                    provider, 
                    chunk_size, 
                    chunk_overlap
                )
                
                # Convert results to dictionaries
                chunks = [result.to_dict() for result in results]
                full_text = "\n\n".join([r.text for r in results])
                
                return ExecutionResult(
                    success=True,
                    outputs={
                        "text": full_text,
                        "chunks": chunks,
                        "chunked": True,
                        "total_chunks": len(results),
                        "file_name": file_path.name,
                        "file_extension": file_extension,
                        "provider": provider or manager.default_provider
                    },
                    metadata={
                        "extraction_method": "chunked",
                        "chunk_count": len(results),
                        "total_characters": len(full_text)
                    }
                )
            else:
                result = manager.extract_text(file_path, provider)
                
                return ExecutionResult(
                    success=True,
                    outputs={
                        "text": result.text,
                        "chunked": False,
                        "file_name": file_path.name,
                        "file_extension": file_extension,
                        "provider": provider or manager.default_provider,
                        "metadata": result.metadata
                    },
                    metadata={
                        "extraction_method": "full_document",
                        "total_characters": len(result.text),
                        "document_metadata": result.metadata
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Document extraction failed: {str(e)}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=f"Document extraction failed: {str(e)}"
            )
    
    def get_required_config_keys(self) -> List[str]:
        """Required configuration keys."""
        return ["file_path"]
    
    def get_optional_config_keys(self) -> List[str]:
        """Optional configuration keys."""
        return [
            "provider",
            "chunk_text", 
            "chunk_size",
            "chunk_overlap"
        ]
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate executor configuration."""
        super().validate_config(config)
        
        # Validate chunk_text is boolean
        if 'chunk_text' in config and not isinstance(config['chunk_text'], bool):
            raise ValueError("chunk_text must be a boolean")
        
        # Validate chunk_size is positive integer
        if 'chunk_size' in config:
            chunk_size = config['chunk_size']
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                raise ValueError("chunk_size must be a positive integer")
        
        # Validate chunk_overlap is non-negative integer
        if 'chunk_overlap' in config:
            chunk_overlap = config['chunk_overlap']
            if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
                raise ValueError("chunk_overlap must be a non-negative integer")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        manager = self._get_manager()
        return manager.get_supported_formats()
    
    def get_info(self) -> Dict[str, Any]:
        """Get executor information."""
        info = super().get_info()
        info.update({
            "supported_formats": self.get_supported_formats(),
            "capabilities": [
                "Text extraction from multiple document formats",
                "Chunking support for large documents", 
                "Metadata preservation",
                "LangChain provider integration"
            ],
            "providers": ["langchain"]
        })
        return info
