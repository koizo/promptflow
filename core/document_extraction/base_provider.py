"""Base provider interface for document text extraction."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DocumentExtractionResult:
    """Result object for document text extraction."""
    
    def __init__(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
        page_count: Optional[int] = None,
        file_type: Optional[str] = None,
        source: Optional[str] = None
    ):
        self.text = text
        self.metadata = metadata or {}
        self.page_count = page_count
        self.file_type = file_type
        self.source = source
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "page_count": self.page_count,
            "file_type": self.file_type,
            "source": self.source,
            "text_length": len(self.text)
        }


class BaseDocumentProvider(ABC):
    """Base class for document text extraction providers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
        
    @abstractmethod
    def extract_text(self, file_path: Path) -> DocumentExtractionResult:
        """
        Extract text from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentExtractionResult containing extracted text and metadata
        """
        pass
    
    @abstractmethod
    def supports_format(self, file_extension: str) -> bool:
        """
        Check if this provider supports the given file format.
        
        Args:
            file_extension: File extension (e.g., '.pdf', '.docx')
            
        Returns:
            True if format is supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        pass
    
    def validate_file(self, file_path: Path) -> bool:
        """
        Validate if file exists and is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is valid and supported
        """
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
            
        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return False
            
        file_extension = file_path.suffix.lower()
        if not self.supports_format(file_extension):
            logger.error(f"Unsupported file format: {file_extension}")
            return False
            
        return True
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get basic file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        return {
            "name": file_path.name,
            "size": file_path.stat().st_size,
            "extension": file_path.suffix.lower(),
            "absolute_path": str(file_path.absolute())
        }
