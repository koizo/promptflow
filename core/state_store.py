"""
Flow state management using Redis for persistence
"""
import json
import logging
import redis.asyncio as redis
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from .schema import FlowState
from .config import settings

logger = logging.getLogger(__name__)

class StateStore:
    """Redis-based state store for flow execution state"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.ttl = settings.redis_ttl
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
    
    async def ping(self) -> bool:
        """Test Redis connection"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        
        result = await self.redis_client.ping()
        return result
    
    async def save_flow_state(self, flow_state: FlowState) -> bool:
        """Save flow execution state"""
        try:
            key = f"flow:{flow_state.flow_id}"
            
            # Serialize flow state
            try:
                data = flow_state.model_dump_json()
            except Exception as e:
                logger.error(f"Error serializing to JSON: {e}")
                return False
            
            # Save with TTL
            await self.redis_client.setex(key, self.ttl, data)
            logger.debug(f"Saved flow state: {flow_state.flow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save flow state: {e}")
            return False
    
    async def get_flow_state(self, flow_id: str) -> Optional[FlowState]:
        """Get flow execution state"""
        try:
            key = f"flow:{flow_id}"
            data = await self.redis_client.get(key)
            
            if data:
                flow_state = FlowState.model_validate_json(data)
                logger.debug(f"Retrieved flow state: {flow_id}")
                return flow_state
            else:
                logger.debug(f"Flow state not found: {flow_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to get flow state: {e}")
            return None
    
    async def delete_flow_state(self, flow_id: str) -> bool:
        """Delete flow execution state"""
        try:
            key = f"flow:{flow_id}"
            result = await self.redis_client.delete(key)
            logger.debug(f"Deleted flow state for {flow_id}")
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete flow state: {e}")
            return False
