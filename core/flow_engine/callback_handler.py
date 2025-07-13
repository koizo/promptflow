"""
Callback Handler for Flow Execution Results

Centralized callback system that can be used by both sync and async flows.
Supports configurable payloads based on flow YAML configuration.
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CallbackHandler:
    """Handles HTTP callback notifications for flow execution results."""
    
    def __init__(self, state_store=None):
        self.session = None
        self.state_store = state_store
    
    async def send_callback(
        self,
        callback_url: str,
        flow_id: str,
        status: str,
        result: Dict[str, Any],
        flow_config: Dict[str, Any] = None,
        execution_time: float = None
    ) -> bool:
        """
        Send HTTP POST notification to callback URL with configurable payload.
        
        Args:
            callback_url: URL to send callback to
            flow_id: Flow execution ID
            status: Flow status (completed, failed, etc.)
            result: Flow execution result
            flow_config: Flow configuration for callback customization
            execution_time: Total execution time in seconds
            
        Returns:
            bool: True if callback was sent successfully
        """
        try:
            # Build callback payload based on flow configuration
            payload = self._build_payload(
                flow_id=flow_id,
                status=status,
                result=result,
                flow_config=flow_config,
                execution_time=execution_time
            )
            
            # Extract ResponseFormatter result from flow state if available
            if hasattr(self, 'state_store') and self.state_store:
                try:
                    flow_state = await self.state_store.get_flow_state(flow_id)
                    if flow_state and flow_state.metadata and "response_formatter_result" in flow_state.metadata:
                        response_formatter_result = flow_state.metadata["response_formatter_result"]
                        
                        # Add key fields from ResponseFormatter
                        key_fields = [
                            "transcript", "analysis", "audio_file", "audio_format", 
                            "audio_duration", "language_detected", "whisper_model", 
                            "llm_model", "analysis_type", "transcript_length"
                        ]
                        
                        for field in key_fields:
                            if field in response_formatter_result:
                                payload[field] = response_formatter_result[field]
                except Exception as e:
                    logger.warning(f"Failed to get ResponseFormatter result from flow state: {str(e)}")
            
            # Get retry configuration
            callback_config = flow_config.get('callbacks', {}) if flow_config else {}
            max_retries = callback_config.get('max_retries', 3)
            retry_delay = callback_config.get('retry_delay', 3)
            
            # Send HTTP POST with retries
            for attempt in range(max_retries + 1):
                try:
                    success = await self._send_http_request(callback_url, payload, attempt + 1)
                    if success:
                        return True
                        
                except Exception as e:
                    logger.warning(f"⚠️ Callback attempt {attempt + 1} failed: {str(e)}")
                    
                    if attempt < max_retries:
                        logger.info(f"🔄 Retrying callback in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(f"❌ All callback attempts failed for {callback_url}")
                        return False
                        
            return False
            
        except Exception as e:
            logger.error(f"❌ Callback notification error: {str(e)}")
            return False
    
    async def _send_http_request(self, callback_url: str, payload: Dict[str, Any], attempt: int) -> bool:
        """Send single HTTP request to callback URL."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                callback_url,
                json=payload,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'AI-Inference-Platform/1.0',
                    'X-Flow-Callback': 'true'
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status < 400:
                    logger.info(f"✅ Callback sent successfully to {callback_url} (attempt {attempt})")
                    return True
                else:
                    logger.warning(f"⚠️ Callback failed with status {response.status} (attempt {attempt})")
                    return False
    
    def _build_payload(
        self,
        flow_id: str,
        status: str,
        result: Dict[str, Any],
        flow_config: Dict[str, Any] = None,
        execution_time: float = None
    ) -> Dict[str, Any]:
        """
        Build callback payload using ResponseFormatter output directly.
        
        Args:
            flow_id: Flow execution ID
            status: Flow status
            result: Flow execution result
            flow_config: Flow configuration (simplified)
            execution_time: Execution time in seconds
            
        Returns:
            Dict: Callback payload
        """
        # Base payload with essential fields
        payload = {
            "flow_id": flow_id,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": result.get("success", False)
        }
        
        if execution_time is not None:
            payload["execution_time"] = execution_time
            
        logger.info(f"📤 CALLBACK FINAL PAYLOAD: {payload}")
        return payload
    
    def _resolve_flow_fields(
        self, 
        flow_fields: Dict[str, str], 
        result: Dict[str, Any],
        flow_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve flow fields using template expressions from YAML.
        
        Args:
            flow_fields: Field definitions from YAML
            result: Flow execution result
            flow_config: Flow configuration
            
        Returns:
            Dict: Resolved field values
        """
        resolved_fields = {}
        
        logger.info(f"🔍 RESOLVING FIELDS - Available result keys: {list(result.keys())}")
        logger.info(f"🔍 RESOLVING FIELDS - Flow fields to resolve: {flow_fields}")
        
        for field_name, field_template in flow_fields.items():
            try:
                # Simple template resolution (can be enhanced with Jinja2 if needed)
                resolved_value = self._resolve_template(field_template, result)
                logger.info(f"🔍 FIELD RESOLUTION - {field_name}: '{field_template}' -> {resolved_value}")
                if resolved_value is not None:
                    resolved_fields[field_name] = resolved_value
            except Exception as e:
                logger.warning(f"⚠️ Failed to resolve field '{field_name}': {e}")
        
        logger.info(f"🔍 RESOLVED FIELDS: {resolved_fields}")
        return resolved_fields
    
    def _resolve_template(self, template: str, result: Dict[str, Any]) -> Any:
        """
        Simple template resolution for field values.
        
        Args:
            template: Template string (e.g., "{{ steps.transcribe_speech.transcript }}")
            result: Flow execution result
            
        Returns:
            Resolved value or None if not found
        """
        if not isinstance(template, str) or not template.startswith('{{'):
            return template  # Return as-is if not a template
        
        # Extract path from template (e.g., "steps.transcribe_speech.transcript")
        path = template.strip('{{ }')
        logger.info(f"🔍 TEMPLATE RESOLUTION - Path: '{path}'")
        
        # Navigate through nested dictionary
        current = result
        path_parts = path.split('.')
        
        for i, key in enumerate(path_parts):
            logger.info(f"🔍 TEMPLATE RESOLUTION - Step {i+1}: Looking for key '{key}' in {type(current)}")
            if isinstance(current, dict):
                logger.info(f"🔍 TEMPLATE RESOLUTION - Available keys: {list(current.keys())}")
                if key in current:
                    current = current[key]
                    logger.info(f"🔍 TEMPLATE RESOLUTION - Found '{key}': {type(current)} = {current}")
                else:
                    logger.warning(f"⚠️ TEMPLATE RESOLUTION - Key '{key}' not found in {list(current.keys())}")
                    return None
            else:
                logger.warning(f"⚠️ TEMPLATE RESOLUTION - Cannot access '{key}' on {type(current)}")
                return None
        
        logger.info(f"✅ TEMPLATE RESOLUTION - Final value: {current}")
        return current
    
    def _apply_field_filter(
        self, 
        payload_data: Dict[str, Any], 
        field_filter: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply field filtering based on include/exclude configuration.
        
        Args:
            payload_data: Original payload data
            field_filter: Filter configuration
            
        Returns:
            Filtered payload data
        """
        mode = field_filter.get('mode', 'include')
        fields = field_filter.get('fields', [])
        
        if mode == 'include':
            # Only include specified fields
            return {k: v for k, v in payload_data.items() if k in fields}
        elif mode == 'exclude':
            # Exclude specified fields
            return {k: v for k, v in payload_data.items() if k not in fields}
        else:
            return payload_data
    
    def _build_metadata(
        self, 
        metadata_config: Dict[str, Any], 
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build metadata section based on configuration.
        
        Args:
            metadata_config: Metadata configuration
            result: Flow execution result
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        metadata_fields = metadata_config.get('fields', {})
        
        for field_name, field_template in metadata_fields.items():
            try:
                resolved_value = self._resolve_template(field_template, result)
                if resolved_value is not None:
                    metadata[field_name] = resolved_value
            except Exception as e:
                logger.warning(f"⚠️ Failed to resolve metadata field '{field_name}': {e}")
        
        return metadata


# Global callback handler instance
callback_handler = CallbackHandler()


async def send_flow_callback(
    callback_url: str,
    flow_id: str,
    status: str,
    result: Dict[str, Any],
    flow_config: Dict[str, Any] = None,
    execution_time: float = None,
    state_store = None
) -> bool:
    """
    Convenience function to send flow callback.
    Can be called from both sync and async contexts.
    """
    # Create callback handler with state_store if provided
    handler = CallbackHandler(state_store) if state_store else callback_handler
    
    return await handler.send_callback(
        callback_url=callback_url,
        flow_id=flow_id,
        status=status,
        result=result,
        flow_config=flow_config,
        execution_time=execution_time
    )


def send_flow_callback_sync(
    callback_url: str,
    flow_id: str,
    status: str,
    result: Dict[str, Any],
    flow_config: Dict[str, Any] = None,
    execution_time: float = None,
    state_store = None
) -> bool:
    """
    Synchronous wrapper for callback sending.
    Creates event loop if needed for sync contexts.
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, schedule as task
            task = asyncio.create_task(send_flow_callback(
                callback_url, flow_id, status, result, flow_config, execution_time, state_store
            ))
            return True  # Return immediately, callback will be sent asynchronously
        else:
            # Run in existing loop
            return loop.run_until_complete(send_flow_callback(
                callback_url, flow_id, status, result, flow_config, execution_time, state_store
            ))
    except RuntimeError:
        # No event loop, create new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(send_flow_callback(
                callback_url, flow_id, status, result, flow_config, execution_time, state_store
            ))
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"❌ Sync callback execution failed: {str(e)}")
        return False
