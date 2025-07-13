#!/usr/bin/env python3
"""Test script for document extraction functionality."""

import asyncio
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_document_extraction():
    """Test document extraction functionality."""
    try:
        from core.document_extraction import DocumentExtractionManager
        from core.config import get_config
        
        # Get configuration
        config = get_config()
        doc_config = config.get('document_extraction', {})
        
        # Initialize manager
        manager = DocumentExtractionManager(doc_config)
        
        # Test 1: Check supported formats
        logger.info("=== Testing Supported Formats ===")
        formats = manager.get_supported_formats()
        logger.info(f"Supported formats: {formats}")
        
        # Test 2: Health check
        logger.info("=== Testing Health Check ===")
        health = manager.health_check()
        logger.info(f"Health status: {health}")
        
        # Test 3: Provider info
        logger.info("=== Testing Provider Info ===")
        info = manager.get_provider_info()
        logger.info(f"Provider info: {info}")
        
        # Test 4: Create a simple text file and test extraction
        logger.info("=== Testing Text File Extraction ===")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = """This is a test document for the AI Inference Platform.

Key Points:
1. Document extraction using LangChain
2. Support for multiple file formats
3. Integration with LLM analysis

The platform supports PDF, Word, PowerPoint, Excel, and text files.
It can extract text and analyze it using various LLM models."""
            
            f.write(test_content)
            f.flush()
            
            test_file = Path(f.name)
            
            try:
                # Test basic extraction
                result = manager.extract_text(test_file)
                logger.info(f"Extraction successful!")
                logger.info(f"Text length: {len(result.text)}")
                logger.info(f"Metadata: {result.metadata}")
                
                # Test chunked extraction
                chunks = manager.extract_with_chunking(test_file, chunk_size=100, chunk_overlap=20)
                logger.info(f"Chunked extraction: {len(chunks)} chunks")
                
                return True
                
            finally:
                # Clean up
                test_file.unlink()
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

async def test_document_analysis_flow():
    """Test the document analysis flow."""
    try:
        from flows.document_analysis.flow import DocumentAnalysisFlow
        
        logger.info("=== Testing Document Analysis Flow ===")
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            test_content = """Project Proposal: AI Document Processing System

Executive Summary:
This proposal outlines the development of an AI-powered document processing system that combines OCR and LLM technologies.

Key Features:
- Multi-format document support (PDF, Word, Excel, PowerPoint)
- Advanced text extraction using LangChain
- Intelligent analysis using large language models
- RESTful API for easy integration

Budget: $50,000
Timeline: 3 months
Team: 4 developers

Next Steps:
1. Approve budget and timeline
2. Assemble development team
3. Begin system architecture design
4. Implement core functionality"""
            
            f.write(test_content)
            f.flush()
            
            test_file = Path(f.name)
            
            try:
                # Initialize flow
                flow = DocumentAnalysisFlow()
                
                # Test different analysis types
                analysis_types = ["summary", "key_points", "entities", "comprehensive"]
                
                for analysis_type in analysis_types:
                    logger.info(f"Testing {analysis_type} analysis...")
                    
                    inputs = {
                        "file_path": str(test_file),
                        "analysis_type": analysis_type,
                        "chunk_text": False,  # Keep simple for testing
                        "document_provider": "langchain",
                        "llm_model": "mistral"
                    }
                    
                    try:
                        result = await flow.execute(inputs)
                        logger.info(f"✅ {analysis_type} analysis completed")
                        logger.info(f"Analysis result keys: {list(result.keys())}")
                        
                        if "analysis" in result:
                            analysis = result["analysis"]
                            if isinstance(analysis, dict) and "analysis" in analysis:
                                logger.info(f"Analysis preview: {analysis['analysis'][:200]}...")
                            else:
                                logger.info(f"Analysis preview: {str(analysis)[:200]}...")
                    
                    except Exception as e:
                        logger.error(f"❌ {analysis_type} analysis failed: {str(e)}")
                
                return True
                
            finally:
                # Clean up
                test_file.unlink()
        
    except Exception as e:
        logger.error(f"Flow test failed: {str(e)}")
        return False

async def test_api_endpoints():
    """Test API endpoints (requires running server)."""
    try:
        import httpx
        
        logger.info("=== Testing API Endpoints ===")
        
        base_url = "http://localhost:8000"
        
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                logger.info("✅ Health endpoint working")
            else:
                logger.warning(f"⚠️ Health endpoint returned {response.status_code}")
            
            # Test document extraction info
            response = await client.get(f"{base_url}/api/v1/document-extraction/info")
            if response.status_code == 200:
                info = response.json()
                logger.info("✅ Document extraction info endpoint working")
                logger.info(f"Available providers: {info.get('available_providers', [])}")
            else:
                logger.warning(f"⚠️ Document extraction info returned {response.status_code}")
            
            # Test supported formats
            response = await client.get(f"{base_url}/api/v1/document-extraction/supported-formats")
            if response.status_code == 200:
                formats = response.json()
                logger.info("✅ Supported formats endpoint working")
                logger.info(f"Supported formats: {formats}")
            else:
                logger.warning(f"⚠️ Supported formats returned {response.status_code}")
        
        return True
        
    except Exception as e:
        logger.error(f"API test failed: {str(e)}")
        logger.info("Note: Make sure the server is running with 'uvicorn main:app --reload'")
        return False

async def main():
    """Run all tests."""
    logger.info("🚀 Starting Document Extraction Tests")
    
    tests = [
        ("Document Extraction Manager", test_document_extraction),
        ("Document Analysis Flow", test_document_analysis_flow),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = await test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - PASSED")
            else:
                logger.error(f"❌ {test_name} - FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERROR: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed")
        return False

if __name__ == "__main__":
    asyncio.run(main())
