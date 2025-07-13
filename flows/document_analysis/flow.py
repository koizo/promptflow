"""Document Analysis Flow - Extract text from documents and analyze with LLM."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.base_flow import BaseFlow
from core.document_extraction import DocumentExtractionManager
from core.llm import LLMManager

logger = logging.getLogger(__name__)


class DocumentAnalysisFlow(BaseFlow):
    """Flow for extracting text from documents and analyzing with LLM."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize managers with default configurations
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
        self.doc_manager = DocumentExtractionManager(doc_config)
        self.llm_manager = LLMManager()
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the document analysis flow."""
        try:
            # Validate inputs
            file_path = inputs.get('file_path')
            if not file_path:
                raise ValueError("file_path is required")
            
            file_path = Path(file_path)
            if not file_path.exists():
                raise ValueError(f"File does not exist: {file_path}")
            
            # Get parameters
            analysis_type = inputs.get('analysis_type', 'comprehensive')
            custom_prompt = inputs.get('custom_prompt', '')
            chunk_text = inputs.get('chunk_text', True)
            document_provider = inputs.get('document_provider', 'langchain')
            llm_model = inputs.get('llm_model', 'mistral')
            
            logger.info(f"Starting document analysis for: {file_path}")
            
            # Step 1: Extract text from document
            extraction_result = await self._extract_text(
                file_path, document_provider, chunk_text
            )
            
            # Step 2: Prepare analysis prompt
            analysis_prompt = self._prepare_analysis_prompt(analysis_type, custom_prompt)
            
            # Step 3: Analyze with LLM
            analysis_result = await self._analyze_with_llm(
                extraction_result, analysis_prompt, llm_model
            )
            
            # Step 4: Format results
            final_result = self._format_results(
                inputs, extraction_result, analysis_result
            )
            
            logger.info(f"Document analysis completed for: {file_path}")
            return final_result
            
        except Exception as e:
            logger.error(f"Document analysis failed: {str(e)}")
            raise
    
    async def _extract_text(
        self, 
        file_path: Path, 
        provider: str, 
        chunk_text: bool
    ) -> Dict[str, Any]:
        """Extract text from document."""
        try:
            if chunk_text:
                results = self.doc_manager.extract_with_chunking(file_path, provider)
                return {
                    "extracted_text": "\n\n".join([r.text for r in results]),
                    "chunks": [r.to_dict() for r in results],
                    "metadata": results[0].metadata if results else {},
                    "chunked": True,
                    "chunk_count": len(results)
                }
            else:
                result = self.doc_manager.extract_text(file_path, provider)
                return {
                    "extracted_text": result.text,
                    "chunks": [],
                    "metadata": result.metadata,
                    "chunked": False,
                    "chunk_count": 1
                }
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise RuntimeError(f"Failed to extract text: {str(e)}")
    
    def _prepare_analysis_prompt(self, analysis_type: str, custom_prompt: str) -> str:
        """Prepare analysis prompt based on analysis type."""
        prompts = {
            "summary": "Please provide a concise summary of the following document:",
            "comprehensive": """Please provide a comprehensive analysis of the following document including:
1. Summary
2. Key points
3. Main themes
4. Important entities (people, organizations, dates, locations)
5. Conclusions or recommendations

Document:""",
            "key_points": "Please extract and list the key points from the following document:",
            "entities": "Please identify and extract important entities (people, organizations, dates, locations, etc.) from the following document:",
            "custom": custom_prompt if custom_prompt else "Please analyze the following document:"
        }
        
        return prompts.get(analysis_type, prompts["comprehensive"])
    
    async def _analyze_with_llm(
        self, 
        extraction_result: Dict[str, Any], 
        analysis_prompt: str, 
        model: str
    ) -> Dict[str, Any]:
        """Analyze extracted text with LLM."""
        try:
            if extraction_result["chunked"]:
                # Analyze each chunk separately
                chunk_analyses = []
                for i, chunk in enumerate(extraction_result["chunks"]):
                    chunk_text = chunk["text"]
                    full_prompt = f"{analysis_prompt}\n\n{chunk_text}"
                    
                    analysis = await self.llm_manager.generate_response(
                        prompt=full_prompt,
                        model=model
                    )
                    
                    chunk_analyses.append({
                        "chunk_index": i,
                        "chunk_text_length": len(chunk_text),
                        "analysis": analysis
                    })
                
                # Generate overall summary if multiple chunks
                if len(chunk_analyses) > 1:
                    summary_prompt = f"""Based on the following analyses of different sections of a document, 
provide an overall summary and key insights:

{chr(10).join([f"Section {i+1}: {analysis['analysis']}" for i, analysis in enumerate(chunk_analyses)])}

Overall Summary:"""
                    
                    overall_summary = await self.llm_manager.generate_response(
                        prompt=summary_prompt,
                        model=model
                    )
                    
                    return {
                        "chunk_analyses": chunk_analyses,
                        "overall_summary": overall_summary,
                        "analysis_type": "chunked",
                        "total_chunks": len(chunk_analyses)
                    }
                else:
                    return {
                        "analysis": chunk_analyses[0]["analysis"],
                        "analysis_type": "single_chunk",
                        "total_chunks": 1
                    }
            else:
                # Analyze full document
                full_prompt = f"{analysis_prompt}\n\n{extraction_result['extracted_text']}"
                
                analysis = await self.llm_manager.generate_response(
                    prompt=full_prompt,
                    model=model
                )
                
                return {
                    "analysis": analysis,
                    "analysis_type": "full_document",
                    "total_chunks": 1
                }
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            raise RuntimeError(f"Failed to analyze with LLM: {str(e)}")
    
    def _format_results(
        self, 
        inputs: Dict[str, Any], 
        extraction_result: Dict[str, Any], 
        analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format final results."""
        return {
            "success": True,
            "document_info": {
                "file_path": str(inputs["file_path"]),
                "extraction_provider": inputs.get("document_provider", "langchain"),
                "analysis_type": inputs.get("analysis_type", "comprehensive"),
                "llm_model": inputs.get("llm_model", "mistral"),
                "chunked": extraction_result["chunked"],
                "chunk_count": extraction_result["chunk_count"]
            },
            "extracted_text": extraction_result["extracted_text"],
            "text_length": len(extraction_result["extracted_text"]),
            "metadata": extraction_result["metadata"],
            "analysis": analysis_result,
            "chunks": extraction_result["chunks"] if extraction_result["chunked"] else []
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get flow input/output schema."""
        return {
            "inputs": {
                "file_path": {"type": "string", "required": True, "description": "Path to document file"},
                "analysis_type": {"type": "string", "default": "comprehensive", "enum": ["summary", "comprehensive", "key_points", "entities", "custom"]},
                "custom_prompt": {"type": "string", "required": False, "description": "Custom analysis prompt"},
                "chunk_text": {"type": "boolean", "default": True, "description": "Whether to split document into chunks"},
                "document_provider": {"type": "string", "default": "langchain", "description": "Document extraction provider"},
                "llm_model": {"type": "string", "default": "mistral", "description": "LLM model to use"}
            },
            "outputs": {
                "success": {"type": "boolean", "description": "Whether the analysis was successful"},
                "document_info": {"type": "object", "description": "Document processing information"},
                "extracted_text": {"type": "string", "description": "Extracted text from document"},
                "text_length": {"type": "integer", "description": "Length of extracted text"},
                "metadata": {"type": "object", "description": "Document metadata"},
                "analysis": {"type": "object", "description": "LLM analysis results"},
                "chunks": {"type": "array", "description": "Text chunks (if chunking enabled)"}
            }
        }
