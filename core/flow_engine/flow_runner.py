"""
Flow Runner

Main orchestration engine for executing YAML-defined flows.
Coordinates executors, manages dependencies, handles templating,
and provides the primary interface for flow execution.

Now integrated with Redis StateStore for persistent flow and step management.
"""

import asyncio
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
import logging
import importlib
import inspect
import uuid
from datetime import datetime, timezone

from .yaml_loader import YAMLFlowLoader, FlowDefinition, FlowStep
from .template_engine import TemplateEngine
from .context_manager import ContextManager, FlowExecution
from ..executors.base_executor import BaseExecutor, ExecutionResult, FlowContext
from .api_generator import FlowAPIGenerator
from ..state_store import StateStore
from ..schema import FlowState

logger = logging.getLogger(__name__)


class ExecutorRegistry:
    """Registry for managing available executors."""
    
    def __init__(self):
        self.executors: Dict[str, Type[BaseExecutor]] = {}
        self.instances: Dict[str, BaseExecutor] = {}
    
    def register_executor(self, name: str, executor_class: Type[BaseExecutor]):
        """Register an executor class."""
        if not issubclass(executor_class, BaseExecutor):
            raise ValueError(f"Executor must inherit from BaseExecutor: {executor_class}")
        
        self.executors[name] = executor_class
        logger.info(f"Registered executor: {name}")
    
    def get_executor(self, name: str) -> BaseExecutor:
        """Get executor instance by name."""
        if name not in self.instances:
            if name not in self.executors:
                raise ValueError(f"Unknown executor: {name}")
            
            executor_class = self.executors[name]
            self.instances[name] = executor_class(name=name)
        
        return self.instances[name]
    
    def list_executors(self) -> List[str]:
        """List all registered executor names."""
        return list(self.executors.keys())
    
    def get_executor_info(self, name: str) -> Dict[str, Any]:
        """Get information about an executor."""
        if name not in self.executors:
            raise ValueError(f"Unknown executor: {name}")
        
        executor = self.get_executor(name)
        return executor.get_info()
    
    def auto_discover_executors(self, package_path: str = "core.executors"):
        """Auto-discover and register executors from a package."""
        try:
            package = importlib.import_module(package_path)
            package_dir = Path(package.__file__).parent
            
            for file_path in package_dir.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue
                
                module_name = f"{package_path}.{file_path.stem}"
                try:
                    module = importlib.import_module(module_name)
                    
                    # Find executor classes in module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseExecutor) and 
                            obj != BaseExecutor):
                            
                            # Use class name as executor name (convert CamelCase to snake_case)
                            executor_name = self._camel_to_snake(obj.__name__)
                            if executor_name.endswith("_executor"):
                                executor_name = executor_name[:-9]  # Remove _executor suffix
                            
                            self.register_executor(executor_name, obj)
                
                except Exception as e:
                    logger.warning(f"Failed to load executors from {module_name}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to auto-discover executors: {e}")
    
    def _camel_to_snake(self, name: str) -> str:
        """Convert CamelCase to snake_case."""
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class FlowRunner:
    """
    Main flow execution engine.
    
    Orchestrates the execution of YAML-defined flows using registered executors.
    Handles dependency management, templating, error handling, and state tracking.
    Now integrated with Redis StateStore for persistent flow and step management.
    """
    
    def __init__(self, flows_dir: Path = None):
        self.flows_dir = flows_dir or Path("flows")
        self.yaml_loader = YAMLFlowLoader()
        self.template_engine = TemplateEngine()
        self.context_manager = ContextManager()
        self.executor_registry = ExecutorRegistry()
        self.api_generator = FlowAPIGenerator(self)
        
        # Redis StateStore for flow persistence
        self.state_store = StateStore()
        self.redis_enabled = True  # Flag to enable/disable Redis persistence
        
        # Load flows
        self.flows: Dict[str, FlowDefinition] = {}
        self.load_flows()
        
        # Auto-discover executors
        self.executor_registry.auto_discover_executors()
    
    async def initialize(self):
        """Initialize Redis connection, Celery configuration, and flow runner components."""
        try:
            if self.redis_enabled:
                await self.state_store.initialize()
                logger.info("✅ Redis StateStore initialized successfully")
                
                # Configure Celery with discovered flows
                try:
                    from .celery_config import configure_celery_with_flows
                    from celery_app import celery_app
                    configure_celery_with_flows(celery_app, self)
                    logger.info("✅ Celery configured with dynamic flows")
                except ImportError as e:
                    logger.warning(f"⚠️  Celery not available: {e}")
                except Exception as e:
                    logger.error(f"❌ Failed to configure Celery: {e}")
            else:
                logger.info("⚠️  Redis persistence disabled - running in memory-only mode")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis StateStore: {e}")
            logger.warning("🔄 Falling back to memory-only execution")
            self.redis_enabled = False
    
    async def start(self):
        """Start the flow runner."""
        await self.initialize()
        await self.context_manager.start()
        logger.info("Flow runner started")
    
    async def stop(self):
        """Stop the flow runner."""
        if self.redis_enabled and self.state_store:
            await self.state_store.close()
            logger.info("Redis StateStore connection closed")
        await self.context_manager.stop()
        logger.info("Flow runner stopped")
    
    def load_flows(self):
        """Load all flows from the flows directory."""
        if not self.flows_dir.exists():
            logger.warning(f"Flows directory not found: {self.flows_dir}")
            return
        
        self.flows = self.yaml_loader.load_flows_from_directory(self.flows_dir)
        logger.info(f"Loaded {len(self.flows)} flows")
    
    def reload_flows(self):
        """Reload flows from directory."""
        self.load_flows()
    
    def get_flow(self, flow_name: str) -> Optional[FlowDefinition]:
        """Get flow definition by name."""
        return self.flows.get(flow_name)
    
    def list_flows(self) -> List[str]:
        """List all available flow names."""
        return list(self.flows.keys())
    
    def get_flow_info(self, flow_name: str) -> Dict[str, Any]:
        """Get information about a flow."""
        flow = self.get_flow(flow_name)
        if not flow:
            raise ValueError(f"Unknown flow: {flow_name}")
        
        return {
            "name": flow.name,
            "version": flow.version,
            "description": flow.description,
            "inputs": [
                {
                    "name": inp.name,
                    "type": inp.type,
                    "required": inp.required,
                    "default": inp.default,
                    "description": inp.description
                }
                for inp in flow.inputs
            ],
            "steps": [
                {
                    "name": step.name,
                    "executor": step.executor,
                    "depends_on": step.depends_on,
                    "description": step.description
                }
                for step in flow.steps
            ],
            "tags": flow.tags,
            "config": flow.config
        }
    
    async def run_flow(self, flow_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a flow with given inputs and persistent state tracking.
        
        Args:
            flow_name: Name of the flow to execute
            inputs: Input parameters for the flow
            
        Returns:
            Flow execution result
        """
        flow = self.get_flow(flow_name)
        if not flow:
            raise ValueError(f"Unknown flow: {flow_name}")
        
        # Create execution context
        execution = self.context_manager.create_context(flow_name, inputs)
        flow_id = execution.context.flow_id
        
        # Phase 2: Initialize flow state in Redis
        await self._initialize_flow_state(flow_id, flow_name, inputs)
        
        try:
            # Validate inputs
            self._validate_inputs(flow, inputs)
            
            # Execute flow steps
            await self._execute_flow(flow, execution)
            
            # Build final output
            result = await self._build_output(flow, execution)
            
            # Mark as completed
            self.context_manager.update_execution(execution.execution_id, "completed")
            
            # Phase 2: Update flow completion state
            await self._update_flow_progress(flow_id, "completed", "completed")
            await self._finalize_flow_state(flow_id, "completed", result)
            
            return result
            
        except Exception as e:
            error_msg = f"Flow execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Mark as failed
            self.context_manager.update_execution(execution.execution_id, "failed", error_msg)
            
            # Phase 2: Update flow failure state
            await self._update_flow_progress(flow_id, "failed", "failed")
            await self._finalize_flow_state(flow_id, "failed", None, error_msg)
            
            raise
    
    async def _execute_flow(self, flow: FlowDefinition, execution: FlowExecution):
        """Execute all steps in a flow."""
        context = execution.context
        
        # Get execution order (topological sort)
        execution_order = flow.get_execution_order()
        
        for step_batch in execution_order:
            # Execute steps in parallel if they're in the same batch
            if len(step_batch) == 1:
                # Single step
                step_name = step_batch[0]
                step = flow.get_step(step_name)
                await self._execute_step(step, context, flow)
            else:
                # Multiple steps - execute in parallel
                tasks = []
                for step_name in step_batch:
                    step = flow.get_step(step_name)
                    task = asyncio.create_task(self._execute_step(step, context, flow))
                    tasks.append(task)
                
                # Wait for all parallel steps to complete
                await asyncio.gather(*tasks)
    
    async def _execute_step(self, step: FlowStep, context: FlowContext, flow: FlowDefinition):
        """Execute a single step with state persistence."""
        logger.info(f"Executing step: {step.name}")
        
        # Check condition if specified
        if step.condition:
            template_context = self.template_engine.build_context(
                context.inputs, 
                context.step_results,
                {"flow_name": flow.name}
            )
            
            condition_result = self.template_engine.render_template(step.condition, template_context)
            if not self._evaluate_condition(condition_result):
                logger.info(f"Step {step.name} skipped due to condition: {step.condition}")
                # Save skipped step state
                await self._save_step_state(context.flow_id, step.name, "skipped", 
                                          metadata={"condition": step.condition, "reason": "condition_false"})
                return
        
        # Get executor
        executor = self.executor_registry.get_executor(step.executor)
        
        # Render step configuration with templates
        template_context = self.template_engine.build_context(
            context.inputs,
            context.step_results,
            {"flow_name": flow.name}
        )
        
        rendered_config = self.template_engine.render_config(step.config, template_context)
        
        # Save step start state (Phase 2: Pre-step state saving)
        await self._save_step_state(context.flow_id, step.name, "running", 
                                   inputs=rendered_config, started_at=datetime.now(timezone.utc))
        
        # Update flow progress
        await self._update_flow_progress(context.flow_id, step.name, "running")
        
        try:
            # Execute step
            context.current_step = step.name
            result = await executor._safe_execute(context, rendered_config)
            
            # Add result to context
            context.add_step_result(step.name, result)
            
            if result.success:
                # Save step completion state (Phase 2: Post-step state saving)
                await self._save_step_state(context.flow_id, step.name, "completed",
                                          outputs=result.outputs, completed_at=datetime.now(timezone.utc),
                                          metadata={"execution_time": result.execution_time})
                logger.info(f"✅ Step {step.name} completed successfully")
                
                # Special handling for ResponseFormatter: save result to flow state for callbacks
                if step.executor == "response_formatter" and result.outputs:
                    await self._save_response_formatter_result(context.flow_id, result.outputs)
            else:
                # Save step failure state
                await self._save_step_state(context.flow_id, step.name, "failed",
                                          error=result.error, completed_at=datetime.now(timezone.utc),
                                          metadata={"execution_time": result.execution_time})
                logger.error(f"❌ Step {step.name} failed: {result.error}")
                
        except Exception as e:
            # Save step failure state for unexpected errors
            await self._save_step_state(context.flow_id, step.name, "failed",
                                      error=str(e), completed_at=datetime.now(timezone.utc))
            logger.error(f"❌ Step {step.name} failed with exception: {str(e)}")
            raise
        
        if not result.success:
            raise RuntimeError(f"Step {step.name} failed: {result.error}")
    
    def _validate_inputs(self, flow: FlowDefinition, inputs: Dict[str, Any]):
        """Validate flow inputs."""
        for input_def in flow.inputs:
            # Apply default values for missing inputs (both required and optional)
            if input_def.name not in inputs and input_def.default is not None:
                inputs[input_def.name] = input_def.default
            
            # Check for required inputs that are still missing
            if input_def.required and input_def.name not in inputs:
                raise ValueError(f"Required input missing: {input_def.name}")
            
            # Validate enum values
            if input_def.enum and input_def.name in inputs:
                if inputs[input_def.name] not in input_def.enum:
                    raise ValueError(f"Input {input_def.name} must be one of: {input_def.enum}")
    
    async def _build_output(self, flow: FlowDefinition, execution: FlowExecution) -> Dict[str, Any]:
        """Build final flow output."""
        context = execution.context
        
        # Default output structure
        result = {
            "flow": flow.name,
            "execution_id": execution.execution_id,
            "success": len(context.failed_steps) == 0,
            "steps_completed": context.completed_steps,
            "steps_failed": context.failed_steps,
            "execution_summary": context.get_execution_summary()
        }
        
        # Process custom outputs if defined
        if flow.outputs:
            template_context = self.template_engine.build_context(
                context.inputs,
                context.step_results,
                {"flow_name": flow.name}
            )
            
            for output_def in flow.outputs:
                try:
                    output_value = self.template_engine.render_value(output_def.value, template_context)
                    result[output_def.name] = output_value
                except Exception as e:
                    logger.warning(f"Failed to render output {output_def.name}: {e}")
                    result[output_def.name] = None
        
        return result
    
    def _evaluate_condition(self, condition_result: str) -> bool:
        """Evaluate a condition result."""
        if isinstance(condition_result, bool):
            return condition_result
        
        # Convert string to boolean
        if isinstance(condition_result, str):
            return condition_result.lower() in ('true', '1', 'yes', 'on')
        
        return bool(condition_result)
    
    def get_executor_registry(self) -> ExecutorRegistry:
        """Get the executor registry."""
        return self.executor_registry
    
    def get_context_manager(self) -> ContextManager:
        """Get the context manager."""
        return self.context_manager
    
    def get_api_generator(self) -> FlowAPIGenerator:
        """Get the API generator."""
        return self.api_generator
    
    def generate_api_routers(self) -> List:
        """Generate FastAPI routers for all flows."""
        return self.api_generator.generate_all_routers()
    
    def generate_router_for_flow(self, flow_name: str):
        """Generate FastAPI router for a specific flow."""
        return self.api_generator.generate_router_for_flow(flow_name)
    
    # ========================================
    # PHASE 2: Step-by-Step State Persistence
    # ========================================
    
    async def _save_step_state(self, flow_id: str, step_name: str, status: str, 
                              inputs: Optional[Dict[str, Any]] = None,
                              outputs: Optional[Dict[str, Any]] = None,
                              error: Optional[str] = None,
                              started_at: Optional[datetime] = None,
                              completed_at: Optional[datetime] = None,
                              metadata: Optional[Dict[str, Any]] = None):
        """
        Save step state to Redis for persistence and resumption.
        
        Args:
            flow_id: Unique flow execution ID
            step_name: Name of the step
            status: Step status (running, completed, failed, skipped)
            inputs: Step input parameters
            outputs: Step output results
            error: Error message if step failed
            started_at: Step start timestamp
            completed_at: Step completion timestamp
            metadata: Additional metadata (execution_time, etc.)
        """
        if not self.redis_enabled or not self.state_store:
            return  # Skip if Redis not available
        
        try:
            # Get existing flow state or create new one
            flow_state = await self.state_store.get_flow_state(flow_id)
            if not flow_state:
                # Create new flow state if it doesn't exist
                flow_state = FlowState(
                    flow_id=flow_id,
                    flow_name="unknown",  # Will be updated by _update_flow_progress
                    status="running",
                    inputs={},  # Required field
                    started_at=datetime.now(timezone.utc),  # Required field
                    steps=[],
                    created_at=datetime.now(timezone.utc).isoformat(),  # Legacy field as string
                    updated_at=datetime.now(timezone.utc).isoformat()   # Legacy field as string
                )
            
            # Find existing step state or create new one
            step_state = None
            for step in flow_state.steps:
                if step.step_name == step_name:
                    step_state = step
                    break
            
            if not step_state:
                # Import FlowStepState here to avoid circular imports
                from ..schema import FlowStepState, StepStatus
                step_state = FlowStepState(
                    step_name=step_name,
                    status=StepStatus(status) if status in ["pending", "running", "completed", "failed", "skipped"] else status,
                    started_at=started_at or datetime.now(timezone.utc)
                )
                flow_state.steps.append(step_state)
            
            # Update step state
            if status in ["pending", "running", "completed", "failed", "skipped"]:
                from ..schema import StepStatus
                step_state.status = StepStatus(status)
            else:
                step_state.status = status
            if inputs is not None:
                step_state.inputs = inputs
            if outputs is not None:
                step_state.outputs = outputs
            if error is not None:
                step_state.error = error
            if started_at is not None:
                step_state.started_at = started_at
            if completed_at is not None:
                step_state.completed_at = completed_at
            if metadata is not None:
                step_state.metadata = metadata
            
            # Update flow state timestamps
            flow_state.updated_at = datetime.now(timezone.utc).isoformat()
            
            # Save updated flow state
            await self.state_store.save_flow_state(flow_state)
            
            logger.debug(f"💾 Saved step state: {flow_id}/{step_name} -> {status}")
            
        except Exception as e:
            logger.warning(f"Failed to save step state for {flow_id}/{step_name}: {e}")
            # Don't raise exception - state persistence should not break flow execution
    
    async def _update_flow_progress(self, flow_id: str, current_step: str, status: str):
        """
        Update overall flow progress in Redis.
        
        Args:
            flow_id: Unique flow execution ID
            current_step: Name of the currently executing step
            status: Overall flow status (running, completed, failed)
        """
        if not self.redis_enabled or not self.state_store:
            return  # Skip if Redis not available
        
        try:
            # Get existing flow state
            flow_state = await self.state_store.get_flow_state(flow_id)
            if flow_state:
                # Update flow progress
                flow_state.current_step = current_step
                flow_state.status = status
                flow_state.updated_at = datetime.now(timezone.utc)
                
                # Calculate progress statistics
                total_steps = len(flow_state.steps)
                completed_steps = len([s for s in flow_state.steps if s.status == "completed"])
                failed_steps = len([s for s in flow_state.steps if s.status == "failed"])
                
                # Update metadata with progress info
                if not flow_state.metadata:
                    flow_state.metadata = {}
                
                flow_state.metadata.update({
                    "total_steps": total_steps,
                    "completed_steps": completed_steps,
                    "failed_steps": failed_steps,
                    "progress_percentage": (completed_steps / total_steps * 100) if total_steps > 0 else 0
                })
                
                # Save updated flow state
                await self.state_store.save_flow_state(flow_state)
                
                logger.debug(f"📊 Updated flow progress: {flow_id} -> {current_step} ({status})")
            
        except Exception as e:
            logger.warning(f"Failed to update flow progress for {flow_id}: {e}")
            # Don't raise exception - progress tracking should not break flow execution
    
    async def _initialize_flow_state(self, flow_id: str, flow_name: str, inputs: Dict[str, Any], status: str = "running"):
        """
        Initialize flow state in Redis at the start of execution.
        
        Args:
            flow_id: Unique flow execution ID
            flow_name: Name of the flow being executed
            inputs: Flow input parameters
            status: Initial status (default: "running", can be "queued" for async API)
        """
        if not self.redis_enabled or not self.state_store:
            logger.warning(f"Failed to initialize flow state for {flow_id}: Redis not available")
            return  # Skip if Redis not available
        
        try:
            # Create a sanitized copy of inputs without binary data
            sanitized_inputs = {}
            for key, value in inputs.items():
                if key == 'file' and isinstance(value, dict):
                    # Store file metadata instead of binary content
                    sanitized_inputs[key] = {
                        'filename': value.get('filename', 'unknown'),
                        'content_type': value.get('content_type', 'application/octet-stream'),
                        'size': value.get('size', 0)
                    }
                else:
                    sanitized_inputs[key] = value
            
            logger.info(f"Initializing flow state for {flow_id} with status '{status}'")
            
            # Create initial flow state with sanitized inputs
            flow_state = FlowState(
                flow_id=flow_id,
                flow_name=flow_name,
                status=status,
                inputs=sanitized_inputs,  # Sanitized inputs without binary data
                started_at=datetime.now(timezone.utc),  # Required field
                steps=[],
                created_at=datetime.now(timezone.utc).isoformat(),  # Legacy field as string
                updated_at=datetime.now(timezone.utc).isoformat(),  # Legacy field as string
                metadata={
                    "total_steps": 0,
                    "completed_steps": 0,
                    "failed_steps": 0,
                    "progress_percentage": 0
                }
            )
            
            # Save initial flow state
            await self.state_store.save_flow_state(flow_state)
            
            logger.info(f"🚀 Initialized flow state: {flow_id} ({flow_name}) - Status: {status}")
            
        except Exception as e:
            logger.error(f"Failed to initialize flow state for {flow_id}: {e}")
            # Don't raise exception - state initialization should not break flow execution
    
    async def _finalize_flow_state(self, flow_id: str, status: str, 
                                  result: Optional[Dict[str, Any]] = None,
                                  error: Optional[str] = None):
        """
        Finalize flow state in Redis at the end of execution.
        
        Args:
            flow_id: Unique flow execution ID
            status: Final flow status (completed, failed)
            result: Flow execution result (if successful)
            error: Error message (if failed)
        """
        if not self.redis_enabled or not self.state_store:
            return  # Skip if Redis not available
        
        try:
            # Get existing flow state
            flow_state = await self.state_store.get_flow_state(flow_id)
            if flow_state:
                # Update final flow state
                flow_state.status = status
                flow_state.completed_at = datetime.now(timezone.utc)
                flow_state.updated_at = datetime.now(timezone.utc).isoformat()  # Legacy field as string
                
                if result is not None:
                    flow_state.outputs = result
                if error is not None:
                    # Store error in metadata since FlowState doesn't have error field
                    if not flow_state.metadata:
                        flow_state.metadata = {}
                    flow_state.metadata["error"] = error
                
                # Calculate final execution time
                if flow_state.started_at:
                    execution_time = (flow_state.completed_at - flow_state.started_at).total_seconds()
                    flow_state.total_execution_time = execution_time
                    if not flow_state.metadata:
                        flow_state.metadata = {}
                    flow_state.metadata["total_execution_time"] = execution_time
                
                # Save final flow state
                await self.state_store.save_flow_state(flow_state)
                
                status_emoji = "✅" if status == "completed" else "❌"
                logger.info(f"{status_emoji} Finalized flow state: {flow_id} -> {status}")
            
        except Exception as e:
            logger.warning(f"Failed to finalize flow state for {flow_id}: {e}")
            # Don't raise exception - state finalization should not break flow execution
    
    # ========================================
    # PHASE 3: Flow Resumption & Recovery
    # ========================================
    
    async def resume_flow(self, flow_id: str) -> Dict[str, Any]:
        """
        Resume a paused or failed flow from last successful step.
        
        Args:
            flow_id: Unique flow execution ID to resume
            
        Returns:
            Flow execution result
            
        Raises:
            ValueError: If flow not found or cannot be resumed
        """
        if not self.redis_enabled or not self.state_store:
            raise ValueError("Redis not available - cannot resume flows without persistent state")
        
        try:
            # Get existing flow state
            flow_state = await self.state_store.get_flow_state(flow_id)
            if not flow_state:
                raise ValueError(f"Flow {flow_id} not found in Redis")
            
            # Check if flow can be resumed
            if flow_state.status == "completed":
                logger.info(f"Flow {flow_id} already completed")
                return flow_state.outputs or {"message": "Flow already completed"}
            
            if flow_state.status not in ["failed", "running"]:
                raise ValueError(f"Flow {flow_id} cannot be resumed (status: {flow_state.status})")
            
            # Get the flow definition
            flow = self.get_flow(flow_state.flow_name)
            if not flow:
                raise ValueError(f"Flow definition '{flow_state.flow_name}' not found")
            
            logger.info(f"🔄 Resuming flow: {flow_id} ({flow_state.flow_name})")
            
            # Find last completed step
            completed_steps = [s.step_name for s in flow_state.steps if s.status == "completed"]
            failed_steps = [s.step_name for s in flow_state.steps if s.status == "failed"]
            
            logger.info(f"Flow resume status - Completed: {len(completed_steps)}, Failed: {len(failed_steps)}")
            
            # Resume from appropriate point
            result = await self._execute_from_step(flow_state, flow, completed_steps)
            
            # Update flow completion state
            await self._update_flow_progress(flow_id, "completed", "completed")
            await self._finalize_flow_state(flow_id, "completed", result)
            
            logger.info(f"✅ Flow {flow_id} resumed and completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Flow resumption failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Update flow failure state
            await self._update_flow_progress(flow_id, "failed", "failed")
            await self._finalize_flow_state(flow_id, "failed", None, error_msg)
            
            raise
    
    async def retry_failed_step(self, flow_id: str, step_name: str) -> Dict[str, Any]:
        """
        Retry a specific failed step in a flow.
        
        Args:
            flow_id: Unique flow execution ID
            step_name: Name of the step to retry
            
        Returns:
            Step execution result
            
        Raises:
            ValueError: If flow/step not found or cannot be retried
        """
        if not self.redis_enabled or not self.state_store:
            raise ValueError("Redis not available - cannot retry steps without persistent state")
        
        try:
            # Get existing flow state
            flow_state = await self.state_store.get_flow_state(flow_id)
            if not flow_state:
                raise ValueError(f"Flow {flow_id} not found in Redis")
            
            # Find the failed step
            failed_step = flow_state.get_step_state(step_name)
            if not failed_step:
                raise ValueError(f"Step '{step_name}' not found in flow {flow_id}")
            
            if failed_step.status != "failed":
                raise ValueError(f"Step '{step_name}' is not in failed state (current: {failed_step.status})")
            
            # Get the flow definition
            flow = self.get_flow(flow_state.flow_name)
            if not flow:
                raise ValueError(f"Flow definition '{flow_state.flow_name}' not found")
            
            # Get the step definition
            step_def = flow.get_step(step_name)
            if not step_def:
                raise ValueError(f"Step definition '{step_name}' not found in flow")
            
            logger.info(f"🔄 Retrying failed step: {flow_id}/{step_name}")
            
            # Reset step state for retry
            await self._reset_step_state(flow_id, step_name)
            
            # Execute the single step
            result = await self._execute_single_step(flow_state, flow, step_def)
            
            logger.info(f"✅ Step {step_name} retried successfully")
            return result
            
        except Exception as e:
            error_msg = f"Step retry failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Update step failure state
            await self._save_step_state(flow_id, step_name, "failed", error=error_msg)
            
            raise
    
    async def cancel_flow(self, flow_id: str) -> Dict[str, Any]:
        """
        Cancel a running flow.
        
        Args:
            flow_id: Unique flow execution ID to cancel
            
        Returns:
            Cancellation result
        """
        if not self.redis_enabled or not self.state_store:
            raise ValueError("Redis not available - cannot cancel flows without persistent state")
        
        try:
            # Get existing flow state
            flow_state = await self.state_store.get_flow_state(flow_id)
            if not flow_state:
                raise ValueError(f"Flow {flow_id} not found in Redis")
            
            if flow_state.status in ["completed", "failed", "cancelled"]:
                return {"message": f"Flow {flow_id} already {flow_state.status}"}
            
            logger.info(f"🛑 Cancelling flow: {flow_id}")
            
            # Update flow state to cancelled
            await self._update_flow_progress(flow_id, "cancelled", "cancelled")
            await self._finalize_flow_state(flow_id, "cancelled", None, "Flow cancelled by user")
            
            logger.info(f"✅ Flow {flow_id} cancelled successfully")
            return {"message": f"Flow {flow_id} cancelled", "flow_id": flow_id}
            
        except Exception as e:
            error_msg = f"Flow cancellation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
    
    async def _execute_from_step(self, flow_state: 'FlowState', flow: 'FlowDefinition', completed_steps: List[str]) -> Dict[str, Any]:
        """
        Execute flow from a specific step, skipping already completed steps.
        
        Args:
            flow_state: Current flow state from Redis
            flow: Flow definition
            completed_steps: List of already completed step names
            
        Returns:
            Flow execution result
        """
        # Reconstruct execution context from flow state
        execution = self._reconstruct_execution_context(flow_state, flow)
        
        # Get execution order (topological sort)
        execution_order = flow.get_execution_order()
        
        # Execute remaining steps
        for step_batch in execution_order:
            for step_name in step_batch:
                # Skip already completed steps
                if step_name in completed_steps:
                    logger.info(f"⏭️  Skipping completed step: {step_name}")
                    continue
                
                # Execute the step
                step = flow.get_step(step_name)
                await self._execute_step(step, execution.context, flow)
        
        # Build final output
        result = await self._build_output(flow, execution)
        return result
    
    async def _execute_single_step(self, flow_state: 'FlowState', flow: 'FlowDefinition', step_def: 'FlowStep') -> Dict[str, Any]:
        """
        Execute a single step with proper context reconstruction.
        
        Args:
            flow_state: Current flow state from Redis
            flow: Flow definition
            step_def: Step definition to execute
            
        Returns:
            Step execution result
        """
        # Reconstruct execution context from flow state
        execution = self._reconstruct_execution_context(flow_state, flow)
        
        # Execute the step
        await self._execute_step(step_def, execution.context, flow)
        
        # Return step result
        step_result = execution.context.step_results.get(step_def.name, {})
        return {"step_name": step_def.name, "result": step_result}
    
    async def _reset_step_state(self, flow_id: str, step_name: str):
        """
        Reset a step state for retry.
        
        Args:
            flow_id: Unique flow execution ID
            step_name: Name of the step to reset
        """
        if not self.redis_enabled or not self.state_store:
            return
        
        try:
            flow_state = await self.state_store.get_flow_state(flow_id)
            if flow_state:
                step_state = flow_state.get_step_state(step_name)
                if step_state:
                    # Reset step state for retry
                    step_state.status = "pending"
                    step_state.error = None
                    step_state.outputs = None
                    step_state.started_at = None
                    step_state.completed_at = None
                    step_state.retry_count = getattr(step_state, 'retry_count', 0) + 1
                    
                    # Save updated flow state
                    await self.state_store.save_flow_state(flow_state)
                    
                    logger.info(f"🔄 Reset step state for retry: {flow_id}/{step_name} (attempt #{step_state.retry_count})")
        
        except Exception as e:
            logger.warning(f"Failed to reset step state for {flow_id}/{step_name}: {e}")
    
    def _reconstruct_execution_context(self, flow_state: 'FlowState', flow: 'FlowDefinition') -> 'FlowExecution':
        """
        Reconstruct execution context from saved flow state.
        
        Args:
            flow_state: Saved flow state from Redis
            flow: Flow definition
            
        Returns:
            Reconstructed FlowExecution context
        """
        # Create new execution context
        execution = self.context_manager.create_context(flow_state.flow_name, flow_state.inputs)
        
        # Restore step results from completed steps
        for step_state in flow_state.steps:
            if step_state.status == "completed" and step_state.outputs:
                # Create ExecutionResult from saved outputs
                from ..executors.base_executor import ExecutionResult
                execution_result = ExecutionResult(
                    success=True,
                    outputs=step_state.outputs,
                    error=None,
                    execution_time=getattr(step_state, 'execution_time', 0.0)
                )
                execution.context.add_step_result(step_state.step_name, execution_result)
        
        # Set flow ID to match the resumed flow
        execution.context.flow_id = flow_state.flow_id
        
        logger.debug(f"🔧 Reconstructed execution context for {flow_state.flow_id} with {len(flow_state.steps)} steps")
        
        return execution
    
    async def _save_response_formatter_result(self, flow_id: str, formatter_outputs: Dict[str, Any]):
        """
        Save ResponseFormatter result to flow state for callback usage.
        
        Args:
            flow_id: Flow execution ID
            formatter_outputs: ResponseFormatter step outputs
        """
        if not self.redis_enabled or not self.state_store:
            return  # Skip if Redis not available
        
        try:
            # Get existing flow state
            flow_state = await self.state_store.get_flow_state(flow_id)
            if flow_state:
                # Store ResponseFormatter result in metadata for callback usage
                if not flow_state.metadata:
                    flow_state.metadata = {}
                
                flow_state.metadata["response_formatter_result"] = formatter_outputs
                
                # Save updated flow state
                await self.state_store.save_flow_state(flow_state)
            else:
                logger.warning(f"Flow state not found for {flow_id}, cannot save ResponseFormatter result")
                
        except Exception as e:
            logger.error(f"Failed to save ResponseFormatter result for {flow_id}: {str(e)}")
            # Don't raise exception - this should not break flow execution


# Global flow runner instance
flow_runner = FlowRunner()
