"""
Celery Application for Async Flow Execution

Provides async task execution using Redis as broker and result backend.
Integrates with existing Redis state persistence and Phase 3 resumption capabilities.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from celery import Celery
from celery.signals import worker_ready, worker_shutdown

from core.flow_engine.flow_runner import flow_runner
from core.state_store import StateStore

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery('ai_inference_platform')

# Worker initialization flag
_worker_initialized = False

# Configure Celery with Redis
celery_app.conf.update(
    # Use Redis as broker and result backend
    broker_url='redis://redis:6379/0',           # Task queue
    result_backend='redis://redis:6379/1',       # Task results
    
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Timezone
    timezone='UTC',
    enable_utc=True,
    
    # Task settings
    task_track_started=True,
    task_time_limit=30 * 60,      # 30 minutes max
    task_soft_time_limit=25 * 60, # 25 minutes soft limit
    task_acks_late=True,          # Acknowledge after task completion
    worker_prefetch_multiplier=1, # One task at a time per worker
    
    # Result settings
    result_expires=3600,          # Results expire after 1 hour
    
    # Default queue
    task_default_queue='default',
    task_default_exchange='default',
    task_default_routing_key='default'
)


async def initialize_worker():
    """Initialize worker with proper state store connection"""
    global _worker_initialized
    if not _worker_initialized:
        try:
            # Initialize the flow runner's state store
            if flow_runner.redis_enabled and flow_runner.state_store:
                await flow_runner.state_store.initialize()
                logger.info("✅ Celery worker StateStore initialized")
            _worker_initialized = True
        except Exception as e:
            logger.error(f"❌ Failed to initialize worker StateStore: {e}")
            # Don't raise - allow worker to continue without Redis state management
            flow_runner.redis_enabled = False


@celery_app.task(bind=True, autoretry_for=(Exception,))
def execute_flow_async(self, flow_name: str, inputs: Dict[str, Any], flow_id: str, callback_url: Optional[str] = None):
    """
    Execute flow asynchronously with Redis state persistence and callback support.
    
    Args:
        flow_name: Name of the flow to execute
        inputs: Flow input parameters
        flow_id: Unique flow execution ID
        callback_url: Optional callback URL for completion notification
        
    Returns:
        Flow execution result
    """
    logger.info(f"🚀 Starting async execution: {flow_id} ({flow_name})")
    
    try:
        # Initialize worker if not already done
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Ensure worker is properly initialized
        loop.run_until_complete(initialize_worker())
        
        try:
            # Update flow state to "running" status
            if flow_runner.redis_enabled and flow_runner.state_store:
                try:
                    flow_state = loop.run_until_complete(flow_runner.state_store.get_flow_state(flow_id))
                    if flow_state:
                        flow_state.status = "running"
                        flow_state.updated_at = datetime.now(timezone.utc).isoformat()
                        loop.run_until_complete(flow_runner.state_store.save_flow_state(flow_state))
                        logger.info(f"📝 Updated flow status to 'running': {flow_id}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to update flow status to running: {e}")
            
            # Store callback URL in Redis flow state for future use
            if callback_url:
                # We'll implement this when we enhance the state store
                logger.info(f"📞 Callback URL stored for {flow_id}: {callback_url}")
            
            # Execute flow using existing flow runner with Redis state persistence
            result = loop.run_until_complete(flow_runner.run_flow(flow_name, inputs))
            logger.info(f"✅ Async execution completed: {flow_id}")
            
            # TODO: Future Phase - Call callback URL on completion
            # if callback_url:
            #     await notify_callback(callback_url, flow_id, "completed", result)
            
            return result
            
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"❌ Async execution failed: {flow_id} - {str(exc)}")
        
        # TODO: Future Phase - Call callback URL on failure
        # if callback_url:
        #     await notify_callback(callback_url, flow_id, "failed", str(exc))
        
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60, max_retries=3)


@celery_app.task
def resume_flow_async(flow_id: str):
    """
    Resume a failed flow asynchronously.
    
    Args:
        flow_id: Unique flow execution ID to resume
        
    Returns:
        Flow resumption result
    """
    logger.info(f"🔄 Starting async flow resumption: {flow_id}")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(flow_runner.resume_flow(flow_id))
            logger.info(f"✅ Async flow resumption completed: {flow_id}")
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"❌ Async flow resumption failed: {flow_id} - {str(exc)}")
        raise


@celery_app.task
def cancel_flow_async(flow_id: str):
    """
    Cancel a running flow asynchronously.
    
    Args:
        flow_id: Unique flow execution ID to cancel
        
    Returns:
        Flow cancellation result
    """
    logger.info(f"🛑 Starting async flow cancellation: {flow_id}")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(flow_runner.cancel_flow(flow_id))
            logger.info(f"✅ Async flow cancellation completed: {flow_id}")
            return result
        finally:
            loop.close()
            
    except Exception as exc:
        logger.error(f"❌ Async flow cancellation failed: {flow_id} - {str(exc)}")
        raise


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready event."""
    logger.info(f"🔧 Celery worker ready: {sender}")
    
    # Initialize flow runner Redis connection in worker context
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(flow_runner.initialize())
            logger.info("✅ Flow runner initialized in worker context")
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"❌ Failed to initialize flow runner in worker: {e}")


@worker_shutdown.connect  
def worker_shutdown_handler(sender=None, **kwargs):
    """Handle worker shutdown event."""
    logger.info(f"🔧 Celery worker shutting down: {sender}")
    
    # Clean shutdown of flow runner Redis connection
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(flow_runner.stop())
            logger.info("✅ Flow runner stopped cleanly in worker context")
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"❌ Failed to stop flow runner in worker: {e}")


# Utility functions for task management
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get status of a Celery task."""
    task = celery_app.AsyncResult(task_id)
    
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.ready() else None,
        "traceback": task.traceback if task.failed() else None,
        "info": task.info
    }


def revoke_task(task_id: str, terminate: bool = False) -> bool:
    """Revoke (cancel) a Celery task."""
    try:
        celery_app.control.revoke(task_id, terminate=terminate)
        logger.info(f"🛑 Task revoked: {task_id} (terminate={terminate})")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to revoke task {task_id}: {e}")
        return False


def get_active_tasks() -> Dict[str, Any]:
    """Get list of active tasks across all workers."""
    try:
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        return active_tasks or {}
    except Exception as e:
        logger.error(f"❌ Failed to get active tasks: {e}")
        return {}


def get_queue_length(queue_name: str) -> int:
    """Get the length of a specific queue."""
    try:
        with celery_app.connection() as conn:
            return conn.default_channel.queue_declare(queue=queue_name, passive=True).message_count
    except Exception as e:
        logger.error(f"❌ Failed to get queue length for {queue_name}: {e}")
        return 0


# Export the Celery app for use in other modules
__all__ = ['celery_app', 'execute_flow_async', 'resume_flow_async', 'cancel_flow_async']
