"""
Redis-based state store for flow persistence
"""
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import redis.asyncio as redis

from .config import settings
from .schema import FlowState

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
            value = flow_state.json()
            
            await self.redis_client.setex(key, self.ttl, value)
            logger.debug(f"Saved flow state for {flow_state.flow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save flow state: {e}")
            return False
    
    async def get_flow_state(self, flow_id: str) -> Optional[FlowState]:
        """Retrieve flow execution state"""
        try:
            key = f"flow:{flow_id}"
            value = await self.redis_client.get(key)
            
            if value:
                data = json.loads(value)
                return FlowState(**data)
            
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
    
    async def list_active_flows(self) -> list[str]:
        """List all active flow IDs"""
        try:
            keys = await self.redis_client.keys("flow:*")
            flow_ids = [key.replace("flow:", "") for key in keys]
            return flow_ids
            
        except Exception as e:
            logger.error(f"Failed to list active flows: {e}")
            return []
    
    async def extend_flow_ttl(self, flow_id: str, ttl: Optional[int] = None) -> bool:
        """Extend TTL for a flow state"""
        try:
            key = f"flow:{flow_id}"
            ttl = ttl or self.ttl
            result = await self.redis_client.expire(key, ttl)
            return result
            
        except Exception as e:
            logger.error(f"Failed to extend TTL for flow {flow_id}: {e}")
            return False
    
    async def set_flow_data(self, flow_id: str, data_key: str, data_value: Any) -> bool:
        """Set additional data for a flow"""
        try:
            key = f"flow:{flow_id}:data:{data_key}"
            value = json.dumps(data_value) if not isinstance(data_value, str) else data_value
            
            await self.redis_client.setex(key, self.ttl, value)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set flow data: {e}")
            return False
    
    async def get_flow_data(self, flow_id: str, data_key: str) -> Optional[Any]:
        """Get additional data for a flow"""
        try:
            key = f"flow:{flow_id}:data:{data_key}"
            value = await self.redis_client.get(key)
            
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get flow data: {e}")
            return None
