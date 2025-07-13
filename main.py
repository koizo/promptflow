"""
AI Inference Platform with OCR + LLM Integration

Production-ready platform with executor-based flows and auto-generated APIs.
All flows are defined in YAML and automatically generate REST API endpoints.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.flow_engine.flow_runner import FlowRunner
from core.router_factory import RouterFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flow runner instance
flow_runner: FlowRunner = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global flow_runner
    
    # Startup
    logger.info("Starting AI Inference Platform...")
    
    # Initialize flow runner
    flows_dir = Path("flows")
    flow_runner = FlowRunner(flows_dir)
    
    # Start flow runner
    await flow_runner.start()
    
    # Generate and register API routers for all flows
    try:
        api_routers = flow_runner.generate_api_routers()
        for router in api_routers:
            app.include_router(router)
        
        logger.info(f"Generated API endpoints for {len(api_routers)} flows")
        
        # List available flows
        flows = flow_runner.list_flows()
        logger.info(f"Available flows: {flows}")
        
    except Exception as e:
        logger.error(f"Failed to generate API routers: {e}")
    
    # Add core document extraction router (for backward compatibility)
    try:
        router_factory = RouterFactory()
        core_routers = router_factory.get_all_routers()
        for router in core_routers:
            app.include_router(router)
        logger.info("Added core API routers")
    except Exception as e:
        logger.warning(f"Failed to add core routers: {e}")
    
    logger.info("AI Inference Platform started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Inference Platform...")
    if flow_runner:
        await flow_runner.stop()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="AI Inference Platform",
    description="Production-ready AI platform with OCR, LLM, and document processing capabilities. All flows are defined in YAML and automatically generate REST API endpoints.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with platform information."""
    return {
        "name": "AI Inference Platform",
        "version": "2.0.0",
        "description": "Production-ready AI platform with executor-based flows",
        "architecture": "YAML-defined flows with auto-generated APIs",
        "features": [
            "Document text extraction",
            "OCR image processing", 
            "LLM analysis and generation",
            "Multi-format support",
            "Reusable executor components",
            "Template-based configuration"
        ],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "catalog": "/catalog",
            "flows": "/flows"
        }
    }


@app.get("/health")
async def health_check():
    """System health check."""
    global flow_runner
    
    if not flow_runner:
        raise HTTPException(status_code=503, detail="Flow runner not initialized")
    
    try:
        # Get flow runner status
        flows = flow_runner.list_flows()
        context_stats = flow_runner.get_context_manager().get_execution_stats()
        
        return {
            "status": "healthy",
            "platform": "AI Inference Platform",
            "version": "2.0.0",
            "flows": {
                "total": len(flows),
                "available": flows
            },
            "execution_stats": context_stats,
            "architecture": "executor-based"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.get("/catalog")
async def get_flow_catalog():
    """Get catalog of all available flows."""
    global flow_runner
    
    if not flow_runner:
        raise HTTPException(status_code=503, detail="Flow runner not initialized")
    
    try:
        flows = flow_runner.list_flows()
        catalog = {}
        
        for flow_name in flows:
            flow_info = flow_runner.get_flow_info(flow_name)
            catalog[flow_name] = {
                "name": flow_info["name"],
                "version": flow_info["version"],
                "description": flow_info["description"],
                "inputs": flow_info["inputs"],
                "steps": len(flow_info["steps"]),
                "tags": flow_info.get("tags", []),
                "endpoints": {
                    "execute": f"/api/v1/{flow_name.replace('_', '-')}/execute",
                    "info": f"/api/v1/{flow_name.replace('_', '-')}/info",
                    "health": f"/api/v1/{flow_name.replace('_', '-')}/health"
                }
            }
        
        return {
            "platform": "AI Inference Platform",
            "total_flows": len(catalog),
            "flows": catalog
        }
        
    except Exception as e:
        logger.error(f"Failed to get flow catalog: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/flows")
async def list_flows():
    """List all available flows with basic information."""
    global flow_runner
    
    if not flow_runner:
        raise HTTPException(status_code=503, detail="Flow runner not initialized")
    
    try:
        flows = flow_runner.list_flows()
        return {
            "flows": flows,
            "total": len(flows),
            "architecture": "YAML-defined with auto-generated APIs"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/flows/{flow_name}")
async def get_flow_details(flow_name: str):
    """Get detailed information about a specific flow."""
    global flow_runner
    
    if not flow_runner:
        raise HTTPException(status_code=503, detail="Flow runner not initialized")
    
    try:
        flow_info = flow_runner.get_flow_info(flow_name)
        return flow_info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
