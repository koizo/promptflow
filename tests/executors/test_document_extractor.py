"""
Unit tests for DocumentExtractor executor.

Tests document text extraction, chunking, format support,
and integration with DocumentExtractionManager.
"""

import pytest
import uuid
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from core.executors.document_extractor import DocumentExtractor
from core.executors.base_executor import FlowContext, ExecutionResult


class MockExtractionResult:
    """Mock extraction result for testing."""
    
    def __init__(self, text="", metadata=None):
        self.text = text
        self.metadata = metadata or {}
    
    def to_dict(self):
        """Convert to dictionary format."""
        return {
            "text": self.text,
            "metadata": self.metadata
        }


class TestDocumentExtractor:
    """Test suite for DocumentExtractor executor."""
    
    @pytest.fixture
    def document_extractor(self):
        """Create DocumentExtractor instance for testing."""
        return DocumentExtractor()
    
    @pytest.fixture
    def mock_context(self):
        """Create mock FlowContext for testing."""
        return FlowContext(
            flow_name="test_flow",
            inputs={"test_param": "test_value"},
            flow_id=f"test-flow-{uuid.uuid4().hex[:8]}"
        )
    
    @pytest.fixture
    def sample_pdf_file(self):
        """Create a temporary PDF file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            # Write minimal PDF content (simplified for testing)
            temp_file.write(b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n%%EOF')
            temp_file.flush()
            yield temp_file.name
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except:
            pass
    
    @pytest.fixture
    def sample_txt_file(self):
        """Create a temporary text file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as temp_file:
            temp_file.write("This is a sample text document.\nIt has multiple lines.\nFor testing purposes.")
            temp_file.flush()
            yield temp_file.name
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except:
            pass
    
    @pytest.fixture
    def sample_config(self, sample_pdf_file):
        """Sample configuration for document extraction."""
        return {
            "file_path": sample_pdf_file,
            "provider": "langchain",
            "chunk_text": False
        }
    
    @pytest.fixture
    def sample_chunked_config(self, sample_pdf_file):
        """Sample configuration for chunked document extraction."""
        return {
            "file_path": sample_pdf_file,
            "provider": "langchain",
            "chunk_text": True,
            "chunk_size": 500,
            "chunk_overlap": 100
        }
    
    @pytest.fixture
    def mock_manager(self):
        """Mock DocumentExtractionManager for testing."""
        manager = Mock()
        manager.default_provider = "langchain"
        manager.supports_format.return_value = True
        manager.get_supported_formats.return_value = [".pdf", ".docx", ".txt"]
        
        # Mock extraction result
        mock_result = MockExtractionResult(
            text="This is extracted text from the document. It contains multiple sentences and paragraphs.",
            metadata={"pages": 1, "author": "Test Author"}
        )
        manager.extract_text.return_value = mock_result
        
        # Mock chunked extraction results
        chunk_results = [
            MockExtractionResult("This is extracted text from", {"chunk": 1}),
            MockExtractionResult("the document. It contains", {"chunk": 2}),
            MockExtractionResult("multiple sentences and paragraphs.", {"chunk": 3})
        ]
        manager.extract_with_chunking.return_value = chunk_results
        
        return manager
    
    @pytest.mark.asyncio
    async def test_missing_file_path(self, document_extractor, mock_context):
        """Test error when file_path is missing."""
        config = {"provider": "langchain"}
        
        result = await document_extractor.execute(mock_context, config)
        
        assert result.success is False
        assert "file_path is required" in result.error
    
    @pytest.mark.asyncio
    async def test_nonexistent_file(self, document_extractor, mock_context):
        """Test error when file doesn't exist."""
        config = {
            "file_path": "/nonexistent/document.pdf",
            "provider": "langchain"
        }
        
        result = await document_extractor.execute(mock_context, config)
        
        assert result.success is False
        assert "File not found" in result.error
    
    @pytest.mark.asyncio
    async def test_unsupported_format(self, document_extractor, mock_context, sample_txt_file):
        """Test error with unsupported file format."""
        config = {
            "file_path": sample_txt_file,
            "provider": "langchain"
        }
        
        mock_manager = Mock()
        mock_manager.supports_format.return_value = False
        mock_manager.get_supported_formats.return_value = [".pdf", ".docx"]
        
        with patch.object(document_extractor, '_get_manager', return_value=mock_manager):
            result = await document_extractor.execute(mock_context, config)
            
            assert result.success is False
            assert "Unsupported file format" in result.error
            assert ".txt" in result.error
    
    @pytest.mark.asyncio
    async def test_successful_document_extraction(self, document_extractor, mock_context, 
                                                sample_config, mock_manager):
        """Test successful document text extraction."""
        with patch.object(document_extractor, '_get_manager', return_value=mock_manager):
            result = await document_extractor.execute(mock_context, sample_config)
            
            assert result.success is True
            assert "text" in result.outputs
            assert "chunked" in result.outputs
            assert "file_name" in result.outputs
            assert "file_extension" in result.outputs
            assert "provider" in result.outputs
            assert "metadata" in result.outputs
            
            assert result.outputs["chunked"] is False
            assert result.outputs["file_extension"] == ".pdf"
            assert result.outputs["provider"] == "langchain"
            assert "This is extracted text" in result.outputs["text"]
            
            # Verify manager was called correctly
            mock_manager.supports_format.assert_called_once_with(".pdf", "langchain")
            mock_manager.extract_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_successful_chunked_extraction(self, document_extractor, mock_context, 
                                               sample_chunked_config, mock_manager):
        """Test successful chunked document extraction."""
        with patch.object(document_extractor, '_get_manager', return_value=mock_manager):
            result = await document_extractor.execute(mock_context, sample_chunked_config)
            
            assert result.success is True
            assert "text" in result.outputs
            assert "chunks" in result.outputs
            assert "chunked" in result.outputs
            assert "total_chunks" in result.outputs
            
            assert result.outputs["chunked"] is True
            assert result.outputs["total_chunks"] == 3
            assert len(result.outputs["chunks"]) == 3
            
            # Verify full text is concatenated from chunks
            full_text = result.outputs["text"]
            assert "This is extracted text from" in full_text
            assert "the document. It contains" in full_text
            assert "multiple sentences and paragraphs." in full_text
            
            # Verify manager was called correctly
            mock_manager.extract_with_chunking.assert_called_once()
            call_args = mock_manager.extract_with_chunking.call_args[0]
            assert call_args[1] == "langchain"  # provider
            assert call_args[2] == 500  # chunk_size
            assert call_args[3] == 100  # chunk_overlap
    
    @pytest.mark.asyncio
    async def test_default_configuration(self, document_extractor, mock_context, 
                                       sample_pdf_file, mock_manager):
        """Test document extraction with default configuration."""
        config = {"file_path": sample_pdf_file}
        
        with patch.object(document_extractor, '_get_manager', return_value=mock_manager):
            result = await document_extractor.execute(mock_context, config)
            
            assert result.success is True
            assert result.outputs["chunked"] is False  # Default is no chunking
            assert result.outputs["provider"] == "langchain"  # Default provider
    
    @pytest.mark.asyncio
    async def test_extraction_error(self, document_extractor, mock_context, sample_config):
        """Test handling of extraction errors."""
        mock_manager = Mock()
        mock_manager.supports_format.return_value = True
        mock_manager.extract_text.side_effect = Exception("Extraction failed")
        
        with patch.object(document_extractor, '_get_manager', return_value=mock_manager):
            result = await document_extractor.execute(mock_context, sample_config)
            
            assert result.success is False
            assert "Document extraction failed" in result.error
            assert "Extraction failed" in result.error
    
    def test_document_extractor_name(self, document_extractor):
        """Test DocumentExtractor name property."""
        assert document_extractor.name == "DocumentExtractor"
    
    def test_get_manager_singleton(self, document_extractor):
        """Test that manager is created as singleton."""
        with patch('core.executors.document_extractor.DocumentExtractionManager') as mock_manager_class:
            mock_instance = Mock()
            mock_manager_class.return_value = mock_instance
            
            # First call should create manager
            manager1 = document_extractor._get_manager()
            assert manager1 == mock_instance
            
            # Second call should return same instance
            manager2 = document_extractor._get_manager()
            assert manager2 == mock_instance
            assert manager1 is manager2
            
            # Manager should only be created once
            mock_manager_class.assert_called_once()
    
    def test_get_manager_configuration(self, document_extractor):
        """Test manager configuration."""
        with patch('core.executors.document_extractor.DocumentExtractionManager') as mock_manager_class:
            mock_instance = Mock()
            mock_manager_class.return_value = mock_instance
            
            document_extractor._get_manager()
            
            # Verify configuration was passed correctly
            call_args = mock_manager_class.call_args[0][0]
            assert call_args['default_provider'] == 'langchain'
            assert 'providers' in call_args
            assert 'langchain' in call_args['providers']
            
            langchain_config = call_args['providers']['langchain']
            assert langchain_config['chunk_size'] == 1000
            assert langchain_config['chunk_overlap'] == 200
            assert langchain_config['preserve_metadata'] is True
    
    def test_required_config_keys(self, document_extractor):
        """Test required configuration keys."""
        required_keys = document_extractor.get_required_config_keys()
        assert "file_path" in required_keys
    
    def test_optional_config_keys(self, document_extractor):
        """Test optional configuration keys."""
        optional_keys = document_extractor.get_optional_config_keys()
        expected_keys = ["provider", "chunk_text", "chunk_size", "chunk_overlap"]
        
        for key in expected_keys:
            assert key in optional_keys
    
    def test_config_validation_success(self, document_extractor):
        """Test successful configuration validation."""
        valid_configs = [
            {"file_path": "/path/to/doc.pdf"},
            {"file_path": "/path/to/doc.pdf", "chunk_text": True},
            {"file_path": "/path/to/doc.pdf", "chunk_size": 500},
            {"file_path": "/path/to/doc.pdf", "chunk_overlap": 100}
        ]
        
        for config in valid_configs:
            # Should not raise exception
            document_extractor.validate_config(config)
    
    def test_config_validation_failures(self, document_extractor):
        """Test configuration validation failures."""
        invalid_configs = [
            {"file_path": "/path/to/doc.pdf", "chunk_text": "not_boolean"},
            {"file_path": "/path/to/doc.pdf", "chunk_size": -1},
            {"file_path": "/path/to/doc.pdf", "chunk_size": "not_integer"},
            {"file_path": "/path/to/doc.pdf", "chunk_overlap": -1},
            {"file_path": "/path/to/doc.pdf", "chunk_overlap": "not_integer"}
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                document_extractor.validate_config(config)
    
    def test_get_supported_formats(self, document_extractor):
        """Test getting supported file formats."""
        mock_manager = Mock()
        mock_manager.get_supported_formats.return_value = [".pdf", ".docx", ".txt", ".xlsx"]
        
        with patch.object(document_extractor, '_get_manager', return_value=mock_manager):
            formats = document_extractor.get_supported_formats()
            
            assert ".pdf" in formats
            assert ".docx" in formats
            assert ".txt" in formats
            assert ".xlsx" in formats
    
    def test_get_info(self, document_extractor):
        """Test executor information."""
        mock_manager = Mock()
        mock_manager.get_supported_formats.return_value = [".pdf", ".docx"]
        
        with patch.object(document_extractor, '_get_manager', return_value=mock_manager):
            info = document_extractor.get_info()
            
            assert "supported_formats" in info
            assert "capabilities" in info
            assert "providers" in info
            
            # Check supported formats
            assert ".pdf" in info["supported_formats"]
            assert ".docx" in info["supported_formats"]
            
            # Check capabilities
            capabilities = info["capabilities"]
            assert "Text extraction from multiple document formats" in capabilities
            assert "Chunking support for large documents" in capabilities
            assert "Metadata preservation" in capabilities
            
            # Check providers
            assert "langchain" in info["providers"]


class TestDocumentExtractorChunking:
    """Test suite for document chunking functionality."""
    
    @pytest.fixture
    def document_extractor(self):
        return DocumentExtractor()
    
    @pytest.fixture
    def mock_context(self):
        return FlowContext(
            flow_name="test_flow",
            inputs={},
            flow_id="chunking-test"
        )
    
    @pytest.mark.asyncio
    async def test_chunking_with_custom_parameters(self, document_extractor, mock_context):
        """Test chunking with custom chunk size and overlap."""
        config = {
            "file_path": "/fake/document.pdf",
            "chunk_text": True,
            "chunk_size": 800,
            "chunk_overlap": 150
        }
        
        mock_manager = Mock()
        mock_manager.supports_format.return_value = True
        
        # Create mock chunks with different sizes
        chunk_results = [
            MockExtractionResult("First chunk of text with exactly 800 characters" + "x" * 750, {"chunk": 1}),
            MockExtractionResult("Second chunk with overlap from previous" + "y" * 750, {"chunk": 2}),
            MockExtractionResult("Final chunk of the document" + "z" * 750, {"chunk": 3})
        ]
        mock_manager.extract_with_chunking.return_value = chunk_results
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch.object(document_extractor, '_get_manager', return_value=mock_manager):
            
            result = await document_extractor.execute(mock_context, config)
            
            assert result.success is True
            assert result.outputs["chunked"] is True
            assert result.outputs["total_chunks"] == 3
            
            # Verify custom parameters were passed
            mock_manager.extract_with_chunking.assert_called_once()
            call_args = mock_manager.extract_with_chunking.call_args[0]
            assert call_args[2] == 800  # chunk_size
            assert call_args[3] == 150  # chunk_overlap
    
    @pytest.mark.asyncio
    async def test_chunking_metadata_preservation(self, document_extractor, mock_context):
        """Test that chunk metadata is preserved."""
        config = {
            "file_path": "/fake/document.pdf",
            "chunk_text": True
        }
        
        mock_manager = Mock()
        mock_manager.supports_format.return_value = True
        
        # Create chunks with rich metadata
        chunk_results = [
            MockExtractionResult("First chunk", {"page": 1, "section": "intro", "chunk_id": "chunk_1"}),
            MockExtractionResult("Second chunk", {"page": 2, "section": "body", "chunk_id": "chunk_2"})
        ]
        mock_manager.extract_with_chunking.return_value = chunk_results
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch.object(document_extractor, '_get_manager', return_value=mock_manager):
            
            result = await document_extractor.execute(mock_context, config)
            
            assert result.success is True
            chunks = result.outputs["chunks"]
            
            # Verify metadata is preserved in chunks
            assert chunks[0]["metadata"]["page"] == 1
            assert chunks[0]["metadata"]["section"] == "intro"
            assert chunks[1]["metadata"]["page"] == 2
            assert chunks[1]["metadata"]["section"] == "body"
    
    @pytest.mark.asyncio
    async def test_empty_chunks_handling(self, document_extractor, mock_context):
        """Test handling of empty chunks."""
        config = {
            "file_path": "/fake/document.pdf",
            "chunk_text": True
        }
        
        mock_manager = Mock()
        mock_manager.supports_format.return_value = True
        mock_manager.extract_with_chunking.return_value = []  # No chunks
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch.object(document_extractor, '_get_manager', return_value=mock_manager):
            
            result = await document_extractor.execute(mock_context, config)
            
            assert result.success is True
            assert result.outputs["chunked"] is True
            assert result.outputs["total_chunks"] == 0
            assert result.outputs["chunks"] == []
            assert result.outputs["text"] == ""


class TestDocumentExtractorIntegration:
    """Integration tests for DocumentExtractor with flow context."""
    
    @pytest.mark.asyncio
    async def test_document_extractor_in_flow_context(self):
        """Test DocumentExtractor as part of a flow context."""
        document_extractor = DocumentExtractor()
        context = FlowContext(
            flow_name="document_analysis_flow",
            inputs={"document_file": "report.pdf"},
            flow_id="integration-test"
        )
        
        # Add previous step result (e.g., from file handler)
        file_result = ExecutionResult(
            success=True,
            outputs={
                "temp_path": "/tmp/uploaded_report.pdf",
                "filename": "quarterly_report.pdf",
                "file_extension": ".pdf"
            }
        )
        context.add_step_result("file_upload", file_result)
        
        config = {
            "file_path": context.step_results["file_upload"].outputs["temp_path"],
            "provider": "langchain",
            "chunk_text": True,
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
        
        # Mock extraction results
        mock_manager = Mock()
        mock_manager.supports_format.return_value = True
        
        chunk_results = [
            MockExtractionResult(
                "Executive Summary: This quarterly report shows strong performance across all business units.",
                {"page": 1, "section": "executive_summary"}
            ),
            MockExtractionResult(
                "Financial Results: Revenue increased by 15% compared to the previous quarter.",
                {"page": 2, "section": "financial_results"}
            ),
            MockExtractionResult(
                "Future Outlook: We expect continued growth in the next quarter based on market trends.",
                {"page": 3, "section": "outlook"}
            )
        ]
        mock_manager.extract_with_chunking.return_value = chunk_results
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch.object(document_extractor, '_get_manager', return_value=mock_manager):
            
            result = await document_extractor.execute(context, config)
            
            # Add result to context
            context.add_step_result("document_extraction", result)
            
            # Verify integration
            assert result.success is True
            assert context.step_results["document_extraction"] is not None
            
            # Verify the result can be used by subsequent steps
            extraction_result = context.step_results["document_extraction"]
            assert extraction_result.outputs["chunked"] is True
            assert extraction_result.outputs["total_chunks"] == 3
            
            # Verify content is suitable for LLM analysis
            full_text = extraction_result.outputs["text"]
            assert "Executive Summary" in full_text
            assert "Financial Results" in full_text
            assert "Future Outlook" in full_text
            
            # Verify chunks are available for individual processing
            chunks = extraction_result.outputs["chunks"]
            assert len(chunks) == 3
            assert chunks[0]["metadata"]["section"] == "executive_summary"
            assert chunks[1]["metadata"]["section"] == "financial_results"
            assert chunks[2]["metadata"]["section"] == "outlook"
    
    @pytest.mark.asyncio
    async def test_document_to_llm_analysis_pipeline(self):
        """Test document extraction feeding into LLM analysis."""
        document_extractor = DocumentExtractor()
        context = FlowContext(
            flow_name="document_understanding_flow",
            inputs={"contract_file": "contract.pdf"},
            flow_id="pipeline-test"
        )
        
        config = {
            "file_path": "/tmp/contract.pdf",
            "provider": "langchain",
            "chunk_text": False  # Extract as single document for analysis
        }
        
        # Mock extraction result with contract content
        mock_manager = Mock()
        mock_manager.supports_format.return_value = True
        mock_manager.default_provider = "langchain"
        
        contract_result = MockExtractionResult(
            text="EMPLOYMENT CONTRACT\n\nThis agreement is between Company ABC and John Doe.\n\nTerms: Full-time employment starting January 1, 2024.\nSalary: $75,000 per year.\nBenefits: Health insurance, 401k matching.\n\nBoth parties agree to the terms outlined above.",
            metadata={"pages": 2, "document_type": "contract"}
        )
        mock_manager.extract_text.return_value = contract_result
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch.object(document_extractor, '_get_manager', return_value=mock_manager):
            
            result = await document_extractor.execute(context, config)
            
            # Add extraction result to context
            context.add_step_result("contract_extraction", result)
            
            # Verify extraction result is ready for LLM analysis
            assert result.success is True
            extracted_text = result.outputs["text"]
            
            # Text should be clean and ready for LLM processing
            assert "EMPLOYMENT CONTRACT" in extracted_text
            assert "Company ABC" in extracted_text
            assert "$75,000" in extracted_text
            
            # Verify structure is preserved for analysis
            assert len(extracted_text.split()) >= 20  # Should have meaningful word count
            assert result.outputs["chunked"] is False  # Single document
            
            # The extracted text should be suitable for contract analysis
            # (e.g., by LLMAnalyzer for key terms extraction)
            assert len(extracted_text) > 100  # Should have substantial content
            assert "metadata" in result.outputs
            assert result.outputs["metadata"]["document_type"] == "contract"
