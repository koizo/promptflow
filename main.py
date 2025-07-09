"""
AI Inference Platform - Main FastAPI Application
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.flow_registry import FlowRegistry
from core.router_factory import RouterFactory
from core.state_store import StateStore
from core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting AI Inference Platform...")
    
    # Initialize core components
    try:
        # Initialize state store (Redis)
        await app.state.state_store.initialize()
        logger.info("✅ State store initialized")
        
        # Load and register flows
        flows = app.state.flow_registry.load_flows()
        logger.info(f"✅ Loaded {len(flows)} flows")
        
        # Register dynamic routes
        app.state.router_factory.register_flow_routes(app, flows)
        logger.info("✅ Flow routes registered")
        
        logger.info("🎉 Application startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down AI Inference Platform...")
    await app.state.state_store.close()


# Create FastAPI app
app = FastAPI(
    title="AI Inference Platform",
    description="Modular, pluggable AI inference platform with declarative flow definitions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
app.state.flow_registry = FlowRegistry()
app.state.router_factory = RouterFactory()
app.state.state_store = StateStore()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Inference Platform",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        await app.state.state_store.ping()
        
        # Check LLM provider health
        from core.llm import LLMExecutor
        llm_health = await LLMExecutor.health_check("ollama")
        
        return {
            "status": "healthy",
            "services": {
                "api": "up",
                "redis": "up",
                "llm": "up" if llm_health["healthy"] else "down"
            },
            "llm_info": llm_health
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/catalog")
async def get_catalog():
    """Get catalog of available flows"""
    flows = app.state.flow_registry.get_flow_catalog()
    return {
        "flows": flows,
        "total": len(flows)
    }


@app.get("/llm/info")
async def get_llm_info():
    """Get LLM provider information"""
    try:
        from core.llm import llm_manager
        
        providers = llm_manager.get_providers()
        provider_info = {}
        
        for provider in providers:
            health = await llm_manager.health_check(provider)
            models = llm_manager.get_available_models(provider)
            
            provider_info[provider] = {
                "healthy": health,
                "available_models": models
            }
        
        return {
            "providers": provider_info,
            "default_provider": "ollama",
            "default_model": "mistral"
        }
    except Exception as e:
        logger.error(f"LLM info request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/info")
async def get_llm_info():
    """Get LLM provider information"""
    try:
        from core.llm import llm_manager
        
        providers = llm_manager.get_providers()
        provider_info = {}
        
        for provider in providers:
            health = await llm_manager.health_check(provider)
            models = llm_manager.get_available_models(provider)
            
            provider_info[provider] = {
                "healthy": health,
                "available_models": models
            }
        
        return {
            "providers": provider_info,
            "default_provider": "ollama",
            "default_model": "mistral"
        }
    except Exception as e:
        logger.error(f"LLM info request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/callback/{flow_id}")
async def callback_handler(flow_id: str, payload: dict):
    """Handle async callbacks for paused flows"""
    try:
        # Resume flow execution
        result = await app.state.flow_registry.resume_flow(flow_id, payload)
        return {"status": "resumed", "result": result}
    except Exception as e:
        logger.error(f"Callback failed for flow {flow_id}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )

@app.get("/ocr/info")
async def get_ocr_info():
    """Get OCR provider information"""
    try:
        from core.ocr import OCRExecutor
        
        provider_info = OCRExecutor.get_provider_info()
        
        # Add health checks
        for provider in provider_info["providers"]:
            try:
                health = await OCRExecutor.health_check(provider)
                provider_info["providers"][provider]["healthy"] = health["healthy"]
            except Exception as e:
                provider_info["providers"][provider]["healthy"] = False
                provider_info["providers"][provider]["error"] = str(e)
        
        return provider_info
    except Exception as e:
        logger.error(f"OCR info request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
