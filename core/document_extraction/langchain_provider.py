"""LangChain-based document text extraction provider."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    TextLoader,
    CSVLoader,
    UnstructuredFileLoader  # Generic unstructured loader for fallback
)

from .base_provider import BaseDocumentProvider, DocumentExtractionResult

logger = logging.getLogger(__name__)


class LangChainDocumentProvider(BaseDocumentProvider):
    """Document text extraction provider using LangChain loaders."""
    
    # Mapping of file extensions to loader classes
    LOADER_MAPPING = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.doc': UnstructuredFileLoader,  # Fallback for older Word docs
        '.pptx': UnstructuredPowerPointLoader,
        '.ppt': UnstructuredPowerPointLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.xls': UnstructuredExcelLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader,
        '.md': TextLoader,
        '.rtf': UnstructuredFileLoader,
        '.odt': UnstructuredFileLoader,
        '.ods': UnstructuredFileLoader,
        '.odp': UnstructuredFileLoader,
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        self.preserve_metadata = self.config.get('preserve_metadata', True)
        
    def extract_text(self, file_path: Path) -> DocumentExtractionResult:
        """
        Extract text from document using appropriate LangChain loader.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentExtractionResult with extracted text and metadata
        """
        if not self.validate_file(file_path):
            raise ValueError(f"Invalid or unsupported file: {file_path}")
        
        file_extension = file_path.suffix.lower()
        loader_class = self.LOADER_MAPPING.get(file_extension)
        
        if not loader_class:
            raise ValueError(f"No loader available for file type: {file_extension}")
        
        try:
            # Initialize the appropriate loader
            if file_extension == '.csv':
                loader = loader_class(str(file_path), encoding='utf-8')
            else:
                loader = loader_class(str(file_path))
            
            # Load documents
            documents = loader.load()
            
            # Combine all document content
            combined_text = ""
            combined_metadata = {}
            page_count = len(documents)
            
            for i, doc in enumerate(documents):
                combined_text += doc.page_content
                if i < len(documents) - 1:
                    combined_text += "\n\n"  # Separate pages/sections
                
                # Merge metadata from all documents
                if self.preserve_metadata and doc.metadata:
                    for key, value in doc.metadata.items():
                        if key not in combined_metadata:
                            combined_metadata[key] = value
                        elif isinstance(value, (list, tuple)):
                            if not isinstance(combined_metadata[key], list):
                                combined_metadata[key] = [combined_metadata[key]]
                            combined_metadata[key].extend(value)
            
            # Add extraction metadata
            combined_metadata.update({
                'extraction_method': 'langchain',
                'loader_type': loader_class.__name__,
                'file_info': self.get_file_info(file_path)
            })
            
            return DocumentExtractionResult(
                text=combined_text,
                metadata=combined_metadata,
                page_count=page_count,
                file_type=file_extension,
                source=str(file_path)
            )
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise RuntimeError(f"Failed to extract text: {str(e)}")
    
    def supports_format(self, file_extension: str) -> bool:
        """Check if file format is supported."""
        return file_extension.lower() in self.LOADER_MAPPING
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.LOADER_MAPPING.keys())
    
    def extract_with_chunking(
        self, 
        file_path: Path, 
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[DocumentExtractionResult]:
        """
        Extract text and split into chunks for LLM processing.
        
        Args:
            file_path: Path to the document file
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of DocumentExtractionResult objects, one per chunk
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Extract full document first
        full_result = self.extract_text(file_path)
        
        # Initialize text splitter
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Split text into chunks
        chunks = text_splitter.split_text(full_result.text)
        
        # Create result objects for each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = full_result.metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk)
            })
            
            chunk_result = DocumentExtractionResult(
                text=chunk,
                metadata=chunk_metadata,
                page_count=full_result.page_count,
                file_type=full_result.file_type,
                source=full_result.source
            )
            chunk_results.append(chunk_result)
        
        return chunk_results
    
    def get_loader_info(self) -> Dict[str, Any]:
        """Get information about available loaders."""
        return {
            'provider_name': self.name,
            'supported_formats': self.get_supported_formats(),
            'loader_mapping': {
                ext: loader.__name__ 
                for ext, loader in self.LOADER_MAPPING.items()
            },
            'config': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'preserve_metadata': self.preserve_metadata
            }
        }
