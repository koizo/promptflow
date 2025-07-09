"""
LLM step executor for flows
"""
import logging
from typing import Dict, Any, Optional
from .llm_manager import llm_manager
from ..utils import format_template

logger = logging.getLogger(__name__)


class LLMExecutor:
    """Executor for LLM steps in flows"""
    
    @staticmethod
    async def execute_llm_step(
        step_config: Dict[str, Any],
        context_variables: Dict[str, Any],
        prompt_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute an LLM step
        
        Args:
            step_config: Step configuration from DSL
            context_variables: Variables available for template formatting
            prompt_template: Optional prompt template content
            
        Returns:
            Dictionary containing the LLM response and metadata
        """
        try:
            # Extract step configuration
            model = step_config.get("model", "mistral")
            provider = step_config.get("provider", "ollama")
            temperature = step_config.get("temperature", 0.7)
            max_tokens = step_config.get("max_tokens")
            system_prompt = step_config.get("system_prompt")
            
            # Get input content
            input_key = step_config.get("input")
            if input_key and input_key in context_variables:
                input_content = context_variables[input_key]
            else:
                input_content = context_variables.get("combined_text", "")
            
            # Format prompt if template is provided
            if prompt_template:
                # Add input content to context variables for template formatting
                template_vars = context_variables.copy()
                template_vars["input_content"] = input_content
                
                formatted_prompt = format_template(prompt_template, template_vars)
            else:
                formatted_prompt = str(input_content)
            
            logger.info(f"Executing LLM step with model: {model}")
            
            # Generate response
            response = await llm_manager.generate(
                prompt=formatted_prompt,
                model=model,
                provider=provider,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Return structured result
            return {
                "content": response.content,
                "model": response.model,
                "usage": response.usage,
                "finish_reason": response.finish_reason,
                "provider": provider
            }
            
        except Exception as e:
            logger.error(f"LLM step execution failed: {e}")
            raise RuntimeError(f"LLM step failed: {e}")
    
    @staticmethod
    async def health_check(provider: str = "ollama") -> Dict[str, Any]:
        """Check LLM provider health"""
        try:
            is_healthy = await llm_manager.health_check(provider)
            available_models = llm_manager.get_available_models(provider)
            
            return {
                "provider": provider,
                "healthy": is_healthy,
                "available_models": available_models
            }
            
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return {
                "provider": provider,
                "healthy": False,
                "error": str(e)
            }
