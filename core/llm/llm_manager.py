"""
LLM Manager for handling different LLM providers
"""
import logging
from typing import Dict, Any, Optional
from .base_provider import BaseLLMProvider, LLMRequest, LLMResponse, LLMMessage
from .ollama_provider import OllamaProvider
from ..config import settings

logger = logging.getLogger(__name__)


class LLMManager:
    """Manager for LLM providers"""
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize LLM providers based on configuration"""
        # Initialize Ollama provider
        ollama_config = {
            "base_url": settings.ollama_base_url,
            "timeout": 60,
            "available_models": ["mistral", "llama3.2", "llama3.1", "codellama"]
        }
        self.providers["ollama"] = OllamaProvider(ollama_config)
        
        # Add other providers here in the future (OpenAI, etc.)
        logger.info(f"Initialized {len(self.providers)} LLM providers")
    
    def get_provider(self, provider_name: str = "ollama") -> BaseLLMProvider:
        """Get LLM provider by name"""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
        return self.providers[provider_name]
    
    async def generate(
        self, 
        prompt: str, 
        model: str = "mistral",
        provider: str = "ollama",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """Generate response using specified provider and model"""
        try:
            llm_provider = self.get_provider(provider)
            
            # Create messages
            messages = llm_provider.create_messages(prompt, system_prompt)
            
            # Create request
            request = LLMRequest(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Generate response
            response = await llm_provider.generate(request)
            logger.info(f"Generated response using {provider}/{model}")
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    async def health_check(self, provider: str = "ollama") -> bool:
        """Check health of specified provider"""
        try:
            llm_provider = self.get_provider(provider)
            return await llm_provider.health_check()
        except Exception as e:
            logger.error(f"Health check failed for {provider}: {e}")
            return False
    
    def get_available_models(self, provider: str = "ollama") -> list[str]:
        """Get available models for specified provider"""
        try:
            llm_provider = self.get_provider(provider)
            return llm_provider.get_available_models()
        except Exception as e:
            logger.error(f"Failed to get models for {provider}: {e}")
            return []
    
    def get_providers(self) -> list[str]:
        """Get list of available providers"""
        return list(self.providers.keys())


# Global LLM manager instance
llm_manager = LLMManager()
