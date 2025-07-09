"""
Configuration management using Pydantic Settings
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=True, description="API reload")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    redis_ttl: int = Field(default=3600, description="Redis TTL in seconds")
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    
    # Application Settings
    environment: str = Field(default="development", description="Environment")
    log_level: str = Field(default="INFO", description="Log level")
    max_file_size: str = Field(default="50MB", description="Max file size")
    upload_dir: str = Field(default="./uploads", description="Upload directory")
    
    # Flow Configuration
    flows_dir: str = Field(default="./flows", description="Flows directory")
    config_file: str = Field(default="./config.yaml", description="Config file path")
    
    # Callback Configuration
    callback_base_url: str = Field(default="http://localhost:8000", description="Callback base URL")
    approval_timeout: int = Field(default=3600, description="Approval timeout")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }


# Global settings instance
settings = Settings()
