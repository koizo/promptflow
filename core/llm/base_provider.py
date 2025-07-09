"""
Base LLM provider interface
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel


class LLMMessage(BaseModel):
    """LLM message structure"""
    role: str  # system, user, assistant
    content: str


class LLMRequest(BaseModel):
    """LLM request structure"""
    messages: List[LLMMessage]
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False


class LLMResponse(BaseModel):
    """LLM response structure"""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM provider is healthy"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        pass
    
    def format_prompt(self, template: str, variables: Dict[str, Any]) -> str:
        """Format prompt template with variables"""
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")
    
    def create_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[LLMMessage]:
        """Create message list from prompt"""
        messages = []
        
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        
        messages.append(LLMMessage(role="user", content=prompt))
        
        return messages
