"""
Router factory for dynamic route registration
"""
import logging
from typing import Dict
from fastapi import FastAPI

from .base_flow import BaseFlow

logger = logging.getLogger(__name__)


class RouterFactory:
    """Factory for creating and registering flow routers"""
    
    def __init__(self):
        self.registered_routes = set()
    
    def register_flow_routes(self, app: FastAPI, flows: Dict[str, BaseFlow]):
        """Register routes for all flows"""
        api_prefix = getattr(app.state.flow_registry.config, 'api_prefix', '/api/v1')
        
        for flow_name, flow in flows.items():
            try:
                router = flow.get_router()
                
                # Add prefix to router
                route_prefix = f"{api_prefix}/{flow_name}"
                
                # Include router in the app
                app.include_router(
                    router,
                    prefix=route_prefix,
                    tags=[flow_name]
                )
                
                self.registered_routes.add(route_prefix)
                logger.info(f"✅ Registered routes for flow: {flow_name} at {route_prefix}")
                
            except Exception as e:
                logger.error(f"❌ Failed to register routes for flow {flow_name}: {e}")
                continue
        
        logger.info(f"Registered {len(self.registered_routes)} flow route prefixes")
    
    def get_registered_routes(self) -> list:
        """Get list of registered route prefixes"""
        return list(self.registered_routes)
