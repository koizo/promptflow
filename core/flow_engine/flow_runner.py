"""
Flow Runner

Main orchestration engine for executing YAML-defined flows.
Coordinates executors, manages dependencies, handles templating,
and provides the primary interface for flow execution.
"""

import asyncio
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
import logging
import importlib
import inspect

from .yaml_loader import YAMLFlowLoader, FlowDefinition, FlowStep
from .template_engine import TemplateEngine
from .context_manager import ContextManager, FlowExecution
from ..executors.base_executor import BaseExecutor, ExecutionResult, FlowContext

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
    """
    
    def __init__(self, flows_dir: Path = None):
        self.flows_dir = flows_dir or Path("flows")
        self.yaml_loader = YAMLFlowLoader()
        self.template_engine = TemplateEngine()
        self.context_manager = ContextManager()
        self.executor_registry = ExecutorRegistry()
        
        # Load flows
        self.flows: Dict[str, FlowDefinition] = {}
        self.load_flows()
        
        # Auto-discover executors
        self.executor_registry.auto_discover_executors()
    
    async def start(self):
        """Start the flow runner."""
        await self.context_manager.start()
        logger.info("Flow runner started")
    
    async def stop(self):
        """Stop the flow runner."""
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
        Execute a flow with given inputs.
        
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
        
        try:
            # Validate inputs
            self._validate_inputs(flow, inputs)
            
            # Execute flow steps
            await self._execute_flow(flow, execution)
            
            # Build final output
            result = await self._build_output(flow, execution)
            
            # Mark as completed
            self.context_manager.update_execution(execution.execution_id, "completed")
            
            return result
            
        except Exception as e:
            error_msg = f"Flow execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Mark as failed
            self.context_manager.update_execution(execution.execution_id, "failed", error_msg)
            
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
        """Execute a single step."""
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
        
        # Execute step
        context.current_step = step.name
        result = await executor._safe_execute(context, rendered_config)
        
        # Add result to context
        context.add_step_result(step.name, result)
        
        if not result.success:
            raise RuntimeError(f"Step {step.name} failed: {result.error}")
    
    def _validate_inputs(self, flow: FlowDefinition, inputs: Dict[str, Any]):
        """Validate flow inputs."""
        for input_def in flow.inputs:
            if input_def.required and input_def.name not in inputs:
                if input_def.default is None:
                    raise ValueError(f"Required input missing: {input_def.name}")
                else:
                    inputs[input_def.name] = input_def.default
            
            # Type validation could be added here
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


# Global flow runner instance
flow_runner = FlowRunner()
