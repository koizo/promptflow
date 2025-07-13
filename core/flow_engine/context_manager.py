"""
Context Manager

Manages flow execution context, state persistence, and step coordination.
Handles context creation, state tracking, and cleanup for flow execution.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass, field
import uuid

from ..executors.base_executor import FlowContext, ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class FlowExecution:
    """Represents a flow execution instance."""
    execution_id: str
    flow_name: str
    context: FlowContext
    status: str = "running"  # running, completed, failed, paused
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    def update_status(self, status: str, error: str = None):
        """Update execution status."""
        self.status = status
        self.updated_at = datetime.utcnow()
        if error:
            self.error = error
        if status in ["completed", "failed"]:
            self.completed_at = datetime.utcnow()


class ContextManager:
    """
    Manages flow execution contexts and state.
    
    Provides context creation, state persistence, cleanup,
    and coordination between flow executions.
    """
    
    def __init__(self, cleanup_interval: int = 3600):
        self.executions: Dict[str, FlowExecution] = {}
        self.cleanup_interval = cleanup_interval  # seconds
        self.logger = logging.getLogger(__name__)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the context manager and cleanup task."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Context manager started")
    
    async def stop(self):
        """Stop the context manager and cleanup task."""
        if not self._running:
            return
        
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Context manager stopped")
    
    def create_context(self, flow_name: str, inputs: Dict[str, Any]) -> FlowExecution:
        """
        Create a new flow execution context.
        
        Args:
            flow_name: Name of the flow to execute
            inputs: Input parameters for the flow
            
        Returns:
            FlowExecution instance
        """
        execution_id = str(uuid.uuid4())
        context = FlowContext(flow_name, inputs)
        
        execution = FlowExecution(
            execution_id=execution_id,
            flow_name=flow_name,
            context=context
        )
        
        self.executions[execution_id] = execution
        self.logger.info(f"Created execution context: {execution_id} for flow: {flow_name}")
        
        return execution
    
    def get_execution(self, execution_id: str) -> Optional[FlowExecution]:
        """Get execution by ID."""
        return self.executions.get(execution_id)
    
    def update_execution(self, execution_id: str, status: str, error: str = None):
        """Update execution status."""
        if execution_id in self.executions:
            self.executions[execution_id].update_status(status, error)
            self.logger.info(f"Updated execution {execution_id} status to: {status}")
    
    def add_step_result(self, execution_id: str, step_name: str, result: ExecutionResult):
        """Add step result to execution context."""
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            execution.context.add_step_result(step_name, result)
            execution.updated_at = datetime.utcnow()
    
    def get_active_executions(self) -> List[FlowExecution]:
        """Get all active (running or paused) executions."""
        return [
            execution for execution in self.executions.values()
            if execution.status in ["running", "paused"]
        ]
    
    def get_completed_executions(self) -> List[FlowExecution]:
        """Get all completed executions."""
        return [
            execution for execution in self.executions.values()
            if execution.status in ["completed", "failed"]
        ]
    
    def cleanup_execution(self, execution_id: str):
        """Remove execution from memory."""
        if execution_id in self.executions:
            del self.executions[execution_id]
            self.logger.info(f"Cleaned up execution: {execution_id}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about executions."""
        total = len(self.executions)
        running = len([e for e in self.executions.values() if e.status == "running"])
        completed = len([e for e in self.executions.values() if e.status == "completed"])
        failed = len([e for e in self.executions.values() if e.status == "failed"])
        paused = len([e for e in self.executions.values() if e.status == "paused"])
        
        return {
            "total_executions": total,
            "running": running,
            "completed": completed,
            "failed": failed,
            "paused": paused,
            "success_rate": completed / total if total > 0 else 0
        }
    
    async def _cleanup_loop(self):
        """Background task to cleanup old executions."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_old_executions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_old_executions(self):
        """Clean up old completed executions."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)  # Keep for 24 hours
        
        to_cleanup = []
        for execution_id, execution in self.executions.items():
            if (execution.status in ["completed", "failed"] and 
                execution.completed_at and 
                execution.completed_at < cutoff_time):
                to_cleanup.append(execution_id)
        
        for execution_id in to_cleanup:
            self.cleanup_execution(execution_id)
        
        if to_cleanup:
            self.logger.info(f"Cleaned up {len(to_cleanup)} old executions")
    
    def pause_execution(self, execution_id: str) -> bool:
        """
        Pause a running execution.
        
        Args:
            execution_id: ID of execution to pause
            
        Returns:
            True if paused successfully, False otherwise
        """
        if execution_id not in self.executions:
            return False
        
        execution = self.executions[execution_id]
        if execution.status != "running":
            return False
        
        execution.update_status("paused")
        self.logger.info(f"Paused execution: {execution_id}")
        return True
    
    def resume_execution(self, execution_id: str) -> bool:
        """
        Resume a paused execution.
        
        Args:
            execution_id: ID of execution to resume
            
        Returns:
            True if resumed successfully, False otherwise
        """
        if execution_id not in self.executions:
            return False
        
        execution = self.executions[execution_id]
        if execution.status != "paused":
            return False
        
        execution.update_status("running")
        self.logger.info(f"Resumed execution: {execution_id}")
        return True
    
    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a running or paused execution.
        
        Args:
            execution_id: ID of execution to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        if execution_id not in self.executions:
            return False
        
        execution = self.executions[execution_id]
        if execution.status in ["completed", "failed"]:
            return False
        
        execution.update_status("failed", "Execution cancelled by user")
        self.logger.info(f"Cancelled execution: {execution_id}")
        return True


# Global context manager instance
context_manager = ContextManager()
