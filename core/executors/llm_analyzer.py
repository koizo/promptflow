"""
LLM Analyzer Executor

Reusable executor for analyzing text using Large Language Models.
Uses existing LLMManager and supports multiple providers and models.
Handles both single text analysis and chunked text processing.
"""

from typing import Dict, Any, List
import logging

from .base_executor import BaseExecutor, ExecutionResult, FlowContext
from ..llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class LLMAnalyzer(BaseExecutor):
    """
    Analyze text using Large Language Models.
    
    Supports custom prompts, different models, and chunked text processing.
    Can work with any text input from previous steps or direct configuration.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.llm_manager = None
    
    def _get_llm_manager(self) -> LLMManager:
        """Get or create LLM manager."""
        if self.llm_manager is None:
            self.llm_manager = LLMManager()
        return self.llm_manager
    
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Analyze text using LLM.
        
        Config parameters:
        - text (required): Text to analyze (can be template reference)
        - prompt (required): Analysis prompt for the LLM
        - model (optional): LLM model to use (default: 'mistral')
        - provider (optional): LLM provider to use (default: 'ollama')
        - chunks (optional): List of text chunks to analyze separately
        - combine_chunks (optional): Whether to combine chunk analyses (default: True)
        """
        try:
            # Get text to analyze
            text = config.get('text')
            chunks = config.get('chunks')
            
            if not text and not chunks:
                return ExecutionResult(
                    success=False,
                    error="Either 'text' or 'chunks' must be provided for analysis"
                )
            
            # Get analysis prompt
            prompt = config.get('prompt')
            if not prompt:
                return ExecutionResult(
                    success=False,
                    error="Analysis prompt is required"
                )
            
            # Get LLM configuration
            model = config.get('model', 'mistral')
            provider = config.get('provider', 'ollama')
            combine_chunks = config.get('combine_chunks', True)
            
            llm_manager = self._get_llm_manager()
            
            self.logger.info(f"Analyzing text with {provider}/{model}")
            
            # Handle chunked analysis
            if chunks:
                return await self._analyze_chunks(
                    chunks, prompt, model, provider, combine_chunks, llm_manager
                )
            
            # Handle single text analysis
            return await self._analyze_text(
                text, prompt, model, provider, llm_manager
            )
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {str(e)}", exc_info=True)
            return ExecutionResult(
                success=False,
                error=f"LLM analysis failed: {str(e)}"
            )
    
    async def _analyze_text(self, text: str, prompt: str, model: str, 
                          provider: str, llm_manager: LLMManager) -> ExecutionResult:
        """Analyze single text with LLM."""
        # Construct full prompt
        full_prompt = f"{prompt}\n\nText to analyze:\n{text}"
        
        # Generate analysis
        response = await llm_manager.generate(
            prompt=full_prompt,
            model=model
        )
        
        return ExecutionResult(
            success=True,
            outputs={
                "analysis": response.content,
                "prompt": prompt,
                "model": model,
                "provider": provider,
                "text_length": len(text),
                "analysis_type": "single_text"
            },
            metadata={
                "input_characters": len(text),
                "output_characters": len(response.content),
                "model_used": model,
                "provider_used": provider
            }
        )
    
    async def _analyze_chunks(self, chunks: List[Dict[str, Any]], prompt: str, 
                            model: str, provider: str, combine_chunks: bool,
                            llm_manager: LLMManager) -> ExecutionResult:
        """Analyze multiple text chunks with LLM."""
        chunk_analyses = []
        total_input_chars = 0
        total_output_chars = 0
        
        # Analyze each chunk
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)
            
            if not chunk_text.strip():
                continue
            
            # Construct prompt for this chunk
            full_prompt = f"{prompt}\n\nText chunk {i+1} to analyze:\n{chunk_text}"
            
            try:
                # Generate analysis for this chunk
                response = await llm_manager.generate(
                    prompt=full_prompt,
                    model=model
                )
                
                chunk_analysis = {
                    "chunk_index": i,
                    "analysis": response.content,
                    "chunk_text_length": len(chunk_text),
                    "analysis_length": len(response.content)
                }
                
                # Include original chunk metadata if available
                if isinstance(chunk, dict) and 'metadata' in chunk:
                    chunk_analysis['chunk_metadata'] = chunk['metadata']
                
                chunk_analyses.append(chunk_analysis)
                total_input_chars += len(chunk_text)
                total_output_chars += len(response.content)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze chunk {i}: {str(e)}")
                chunk_analyses.append({
                    "chunk_index": i,
                    "analysis": f"Analysis failed: {str(e)}",
                    "error": True
                })
        
        # Combine analyses if requested
        combined_analysis = None
        if combine_chunks and chunk_analyses:
            try:
                # Create a summary prompt
                all_analyses = "\n\n".join([
                    f"Chunk {ca['chunk_index'] + 1}: {ca['analysis']}" 
                    for ca in chunk_analyses 
                    if not ca.get('error', False)
                ])
                
                summary_prompt = f"""Based on the following analyses of different text chunks, provide a comprehensive summary:

{all_analyses}

Please provide a unified analysis that combines insights from all chunks."""
                
                summary_response = await llm_manager.generate(
                    prompt=summary_prompt,
                    model=model
                )
                
                combined_analysis = summary_response.content
                total_output_chars += len(combined_analysis)
                
            except Exception as e:
                self.logger.warning(f"Failed to combine chunk analyses: {str(e)}")
                combined_analysis = f"Failed to combine analyses: {str(e)}"
        
        return ExecutionResult(
            success=True,
            outputs={
                "chunk_analyses": chunk_analyses,
                "combined_analysis": combined_analysis,
                "prompt": prompt,
                "model": model,
                "provider": provider,
                "total_chunks": len(chunks),
                "successful_analyses": len([ca for ca in chunk_analyses if not ca.get('error', False)]),
                "analysis_type": "chunked_text"
            },
            metadata={
                "total_input_characters": total_input_chars,
                "total_output_characters": total_output_chars,
                "chunks_processed": len(chunk_analyses),
                "model_used": model,
                "provider_used": provider,
                "combined_analysis_generated": combined_analysis is not None
            }
        )
    
    def get_required_config_keys(self) -> List[str]:
        """Required configuration keys."""
        return ["prompt"]
    
    def get_optional_config_keys(self) -> List[str]:
        """Optional configuration keys."""
        return [
            "text",
            "chunks", 
            "model",
            "provider",
            "combine_chunks"
        ]
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate executor configuration."""
        super().validate_config(config)
        
        # Must have either text or chunks
        if not config.get('text') and not config.get('chunks'):
            raise ValueError("Either 'text' or 'chunks' must be provided")
        
        # Validate chunks format if provided
        if 'chunks' in config:
            chunks = config['chunks']
            if not isinstance(chunks, list):
                raise ValueError("chunks must be a list")
        
        # Validate combine_chunks is boolean
        if 'combine_chunks' in config and not isinstance(config['combine_chunks'], bool):
            raise ValueError("combine_chunks must be a boolean")
    
    def get_info(self) -> Dict[str, Any]:
        """Get executor information."""
        info = super().get_info()
        info.update({
            "capabilities": [
                "Text analysis using Large Language Models",
                "Custom prompt support",
                "Multiple model support",
                "Chunked text processing",
                "Analysis combination and summarization"
            ],
            "supported_providers": ["ollama", "openai"],
            "supported_models": {
                "ollama": ["mistral", "llama3.2", "llama3.1", "codellama"],
                "openai": ["gpt-3.5-turbo", "gpt-4"]
            }
        })
        return info
