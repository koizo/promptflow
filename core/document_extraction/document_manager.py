"""Document extraction manager for handling multiple providers."""

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from .base_provider import BaseDocumentProvider, DocumentExtractionResult
from .langchain_provider import LangChainDocumentProvider

logger = logging.getLogger(__name__)


class DocumentExtractionManager:
    """Manager for document text extraction providers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.providers: Dict[str, BaseDocumentProvider] = {}
        self.default_provider = self.config.get('default_provider', 'langchain')
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available document extraction providers."""
        try:
            # Initialize LangChain provider
            langchain_config = self.config.get('providers', {}).get('langchain', {})
            self.providers['langchain'] = LangChainDocumentProvider(langchain_config)
            logger.info("Initialized LangChain document provider")
            
        except Exception as e:
            logger.error(f"Failed to initialize document providers: {str(e)}")
    
    def extract_text(
        self, 
        file_path: Path, 
        provider_name: Optional[str] = None
    ) -> DocumentExtractionResult:
        """
        Extract text from document using specified or default provider.
        
        Args:
            file_path: Path to the document file
            provider_name: Name of provider to use (optional)
            
        Returns:
            DocumentExtractionResult with extracted text and metadata
        """
        provider_name = provider_name or self.default_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        
        provider = self.providers[provider_name]
        
        if not provider.validate_file(file_path):
            raise ValueError(f"File validation failed: {file_path}")
        
        logger.info(f"Extracting text from {file_path} using {provider_name}")
        return provider.extract_text(file_path)
    
    def extract_with_chunking(
        self,
        file_path: Path,
        provider_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[DocumentExtractionResult]:
        """
        Extract text and split into chunks for LLM processing.
        
        Args:
            file_path: Path to the document file
            provider_name: Name of provider to use (optional)
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of DocumentExtractionResult objects, one per chunk
        """
        provider_name = provider_name or self.default_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        
        provider = self.providers[provider_name]
        
        # Check if provider supports chunking
        if hasattr(provider, 'extract_with_chunking'):
            return provider.extract_with_chunking(file_path, chunk_size, chunk_overlap)
        else:
            # Fallback: extract full text and chunk manually
            result = provider.extract_text(file_path)
            return [result]  # Return as single chunk
    
    def get_supported_formats(self, provider_name: Optional[str] = None) -> List[str]:
        """
        Get supported file formats for a provider.
        
        Args:
            provider_name: Name of provider (optional, uses default if not specified)
            
        Returns:
            List of supported file extensions
        """
        provider_name = provider_name or self.default_provider
        
        if provider_name not in self.providers:
            return []
        
        return self.providers[provider_name].get_supported_formats()
    
    def supports_format(self, file_extension: str, provider_name: Optional[str] = None) -> bool:
        """
        Check if a file format is supported.
        
        Args:
            file_extension: File extension to check
            provider_name: Name of provider (optional)
            
        Returns:
            True if format is supported
        """
        provider_name = provider_name or self.default_provider
        
        if provider_name not in self.providers:
            return False
        
        return self.providers[provider_name].supports_format(file_extension)
    
    def get_best_provider_for_file(self, file_path: Path) -> Optional[str]:
        """
        Get the best provider for a specific file type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Name of the best provider, or None if no provider supports the format
        """
        file_extension = file_path.suffix.lower()
        
        # For now, return default provider if it supports the format
        if self.supports_format(file_extension):
            return self.default_provider
        
        # Check other providers
        for provider_name, provider in self.providers.items():
            if provider.supports_format(file_extension):
                return provider_name
        
        return None
    
    def get_provider_info(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a provider.
        
        Args:
            provider_name: Name of provider (optional)
            
        Returns:
            Dictionary with provider information
        """
        if provider_name:
            if provider_name not in self.providers:
                return {}
            provider = self.providers[provider_name]
            info = {
                'name': provider_name,
                'supported_formats': provider.get_supported_formats(),
                'config': provider.config
            }
            # Add provider-specific info if available
            if hasattr(provider, 'get_loader_info'):
                info.update(provider.get_loader_info())
            return info
        else:
            # Return info for all providers
            return {
                name: self.get_provider_info(name) 
                for name in self.providers.keys()
            }
    
    def health_check(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform health check on providers.
        
        Args:
            provider_name: Name of specific provider to check (optional)
            
        Returns:
            Dictionary with health status
        """
        if provider_name:
            if provider_name not in self.providers:
                return {
                    'provider': provider_name,
                    'status': 'not_found',
                    'message': f"Provider '{provider_name}' not available"
                }
            
            try:
                provider = self.providers[provider_name]
                return {
                    'provider': provider_name,
                    'status': 'healthy',
                    'supported_formats': provider.get_supported_formats(),
                    'config': provider.config
                }
            except Exception as e:
                return {
                    'provider': provider_name,
                    'status': 'error',
                    'message': str(e)
                }
        else:
            # Check all providers
            results = {}
            for name in self.providers.keys():
                results[name] = self.health_check(name)
            return results
