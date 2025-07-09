"""
Ollama LLM provider implementation
"""
import logging
import httpx
from typing import Dict, Any, List
from .base_provider import BaseLLMProvider, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout", 60)
        self.available_models = config.get("available_models", ["mistral"])
        
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Ollama"""
        try:
            # Convert messages to Ollama format
            ollama_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ]
            
            # Prepare request payload
            payload = {
                "model": request.model,
                "messages": ollama_messages,
                "stream": request.stream,
                "options": {
                    "temperature": request.temperature,
                }
            }
            
            if request.max_tokens:
                payload["options"]["num_predict"] = request.max_tokens
            
            # Make request to Ollama
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                return LLMResponse(
                    content=result["message"]["content"],
                    model=request.model,
                    usage={
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                    },
                    finish_reason=result.get("done_reason", "stop")
                )
                
        except httpx.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise RuntimeError(f"Ollama request failed: {e}")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise RuntimeError(f"LLM generation failed: {e}")
    
    async def health_check(self) -> bool:
        """Check Ollama health"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get available Ollama models from configuration"""
        return self.available_models
