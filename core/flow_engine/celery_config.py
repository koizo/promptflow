"""
Dynamic Celery Configuration

Generates Celery queues and routing dynamically based on existing flows.
Maintains the YAML-driven philosophy by auto-discovering async flows.
"""

import logging
from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .flow_runner import FlowRunner

logger = logging.getLogger(__name__)


def generate_dynamic_queues(flow_runner: 'FlowRunner') -> Dict[str, Dict[str, Any]]:
    """
    Generate Celery queues dynamically based on existing flows.
    
    Args:
        flow_runner: FlowRunner instance to discover flows from
        
    Returns:
        Dictionary of queue configurations for Celery
    """
    queues = {}
    
    # Discover all flows dynamically
    for flow_name in flow_runner.list_flows():
        flow_def = flow_runner.get_flow(flow_name)
        if not flow_def:
            continue
            
        # Only create queue for flows that support async execution
        execution_config = flow_def.config.get('execution', {})
        execution_mode = execution_config.get('mode', 'sync')
        
        if execution_mode in ['async', 'auto']:
            # One queue per async flow
            queue_name = flow_name
            queues[queue_name] = {
                'routing_key': flow_name,
                'queue_arguments': {
                    'x-message-ttl': 3600000,  # Messages expire after 1 hour
                }
            }
            
            logger.info(f"📋 Created queue for async flow: {queue_name}")
    
    logger.info(f"🎯 Generated {len(queues)} dynamic queues")
    return queues


def generate_task_routes(flow_runner: 'FlowRunner') -> Dict[str, Dict[str, Any]]:
    """
    Generate Celery task routing based on flows.
    
    Args:
        flow_runner: FlowRunner instance to discover flows from
        
    Returns:
        Dictionary of task routing configurations
    """
    routes = {
        'execute_flow_async': {
            'queue': lambda flow_name, **kwargs: flow_name
        },
        'resume_flow_async': {
            'queue': lambda flow_id, **kwargs: _get_flow_queue_from_id(flow_runner, flow_id)
        },
        'cancel_flow_async': {
            'queue': lambda flow_id, **kwargs: _get_flow_queue_from_id(flow_runner, flow_id)
        }
    }
    
    return routes


def _get_flow_queue_from_id(flow_runner: 'FlowRunner', flow_id: str) -> str:
    """
    Get the appropriate queue for a flow based on flow_id.
    
    Args:
        flow_runner: FlowRunner instance
        flow_id: Flow execution ID
        
    Returns:
        Queue name for the flow
    """
    try:
        # Try to get flow state from Redis to determine flow name
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            flow_state = loop.run_until_complete(flow_runner.state_store.get_flow_state(flow_id))
            if flow_state and flow_state.flow_name:
                return flow_state.flow_name
        finally:
            loop.close()
            
    except Exception as e:
        logger.warning(f"Could not determine queue for flow_id {flow_id}: {e}")
    
    # Fallback to default queue
    return 'default'


def configure_celery_with_flows(celery_app, flow_runner: 'FlowRunner'):
    """
    Configure Celery with dynamically discovered flows.
    
    Args:
        celery_app: Celery application instance
        flow_runner: FlowRunner instance to discover flows from
    """
    logger.info("🔧 Configuring Celery with dynamic flows...")
    
    # Generate queues based on existing flows
    dynamic_queues = generate_dynamic_queues(flow_runner)
    
    # Generate task routes
    task_routes = generate_task_routes(flow_runner)
    
    # Update Celery configuration
    celery_app.conf.update(
        task_routes=task_routes,
        task_queues=dynamic_queues
    )
    
    # Log configuration
    async_flows = []
    for flow_name in flow_runner.list_flows():
        flow_def = flow_runner.get_flow(flow_name)
        if flow_def:
            execution_config = flow_def.config.get('execution', {})
            if execution_config.get('mode') in ['async', 'auto']:
                async_flows.append({
                    'name': flow_name,
                    'queue': flow_name,
                    'max_concurrent': execution_config.get('max_concurrent', 2)
                })
    
    logger.info(f"✅ Configured Celery with {len(async_flows)} async flows:")
    for flow in async_flows:
        logger.info(f"  - {flow['name']}: queue={flow['queue']}, max_concurrent={flow['max_concurrent']}")


def get_async_flows(flow_runner: 'FlowRunner') -> List[Dict[str, Any]]:
    """
    Get list of flows that support async execution.
    
    Args:
        flow_runner: FlowRunner instance to discover flows from
        
    Returns:
        List of async flow configurations
    """
    async_flows = []
    
    for flow_name in flow_runner.list_flows():
        flow_def = flow_runner.get_flow(flow_name)
        if not flow_def:
            continue
            
        execution_config = flow_def.config.get('execution', {})
        execution_mode = execution_config.get('mode', 'sync')
        
        if execution_mode in ['async', 'auto']:
            async_flows.append({
                'name': flow_name,
                'mode': execution_mode,
                'queue': flow_name,
                'max_concurrent': execution_config.get('max_concurrent', 2),
                'timeout': execution_config.get('timeout', 300),
                'retry_count': execution_config.get('retry_count', 3),
                'auto_resume': execution_config.get('auto_resume', True),
                'callback_enabled': flow_def.config.get('callbacks', {}).get('enabled', False)
            })
    
    return async_flows


def should_execute_async(flow_def, inputs: Dict[str, Any]) -> bool:
    """
    Determine if a flow should execute asynchronously.
    
    Args:
        flow_def: Flow definition
        inputs: Flow input parameters
        
    Returns:
        True if flow should execute async, False otherwise
    """
    execution_config = flow_def.config.get('execution', {})
    mode = execution_config.get('mode', 'sync')
    
    if mode == 'async':
        return True
    elif mode == 'sync':
        return False
    elif mode == 'auto':
        # Auto-detection logic
        return _auto_detect_async_need(flow_def, inputs)
    
    return False


def _auto_detect_async_need(flow_def, inputs: Dict[str, Any]) -> bool:
    """
    Auto-detect if flow should run async based on complexity and inputs.
    
    Args:
        flow_def: Flow definition
        inputs: Flow input parameters
        
    Returns:
        True if async execution is recommended
    """
    # Check for large file uploads
    for input_name, input_value in inputs.items():
        if hasattr(input_value, 'size') and input_value.size > 10 * 1024 * 1024:  # 10MB
            logger.info(f"🔍 Auto-detect: Large file detected ({input_value.size} bytes) -> async")
            return True
    
    # Check flow complexity (number of steps)
    if len(flow_def.steps) > 5:
        logger.info(f"🔍 Auto-detect: Complex flow detected ({len(flow_def.steps)} steps) -> async")
        return True
    
    # Check for LLM steps (typically slower)
    llm_steps = [step for step in flow_def.steps if 'llm' in step.executor.lower()]
    if len(llm_steps) > 1:
        logger.info(f"🔍 Auto-detect: Multiple LLM steps detected ({len(llm_steps)}) -> async")
        return True
    
    # Default to sync for simple flows
    logger.info("🔍 Auto-detect: Simple flow -> sync")
    return False
