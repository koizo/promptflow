"""
Base Executor Framework

Provides the foundation for all executors in the flow engine.
Executors are reusable components that perform specific tasks and can be
orchestrated through YAML flow definitions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """
    Standardized result format for executor execution.
    
    This ensures consistent output format across all executors
    and provides structured error handling.
    """
    success: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "success": self.success,
            "outputs": self.outputs,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time": self.execution_time
        }


class FlowContext:
    """
    Context object passed between executors during flow execution.
    
    Manages flow state, step results, and provides access to
    configuration and previous step outputs.
    """
    
    def __init__(self, flow_name: str, inputs: Dict[str, Any]):
        self.flow_name = flow_name
        self.inputs = inputs
        self.step_results: Dict[str, ExecutionResult] = {}
        self.current_step: Optional[str] = None
        self.started_at = datetime.utcnow()
        self.completed_steps: List[str] = []
        self.failed_steps: List[str] = []
        
    def add_step_result(self, step_name: str, result: ExecutionResult) -> None:
        """Add result from a completed step."""
        self.step_results[step_name] = result
        
        if result.success:
            self.completed_steps.append(step_name)
            logger.info(f"Step '{step_name}' completed successfully")
        else:
            self.failed_steps.append(step_name)
            logger.error(f"Step '{step_name}' failed: {result.error}")
    
    def get_step_output(self, step_name: str, output_key: str = None) -> Any:
        """Get output from a previous step."""
        if step_name not in self.step_results:
            raise ValueError(f"Step '{step_name}' not found in results")
        
        result = self.step_results[step_name]
        if not result.success:
            raise ValueError(f"Step '{step_name}' failed, cannot access outputs")
        
        if output_key is None:
            return result.outputs
        
        if output_key not in result.outputs:
            raise ValueError(f"Output '{output_key}' not found in step '{step_name}'")
        
        return result.outputs[output_key]
    
    def get_input(self, key: str, default: Any = None) -> Any:
        """Get input value with optional default."""
        return self.inputs.get(key, default)
    
    def has_step_completed(self, step_name: str) -> bool:
        """Check if a step has completed successfully."""
        return step_name in self.completed_steps
    
    def has_step_failed(self, step_name: str) -> bool:
        """Check if a step has failed."""
        return step_name in self.failed_steps
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of flow execution."""
        return {
            "flow_name": self.flow_name,
            "started_at": self.started_at.isoformat(),
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "total_steps": len(self.step_results),
            "success_rate": len(self.completed_steps) / len(self.step_results) if self.step_results else 0
        }


class BaseExecutor(ABC):
    """
    Abstract base class for all executors.
    
    Executors are reusable components that perform specific tasks
    within a flow. They receive a FlowContext and return an ExecutionResult.
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"executor.{self.name}")
    
    @abstractmethod
    async def execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Execute the task with given context and configuration.
        
        Args:
            context: Flow execution context with inputs and previous results
            config: Step-specific configuration from YAML
            
        Returns:
            ExecutionResult with success status, outputs, and metadata
        """
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate executor configuration.
        
        Override this method to add executor-specific validation.
        Raise ValueError for invalid configuration.
        """
        pass
    
    def get_required_config_keys(self) -> List[str]:
        """
        Return list of required configuration keys.
        
        Override this method to specify required configuration.
        """
        return []
    
    def get_optional_config_keys(self) -> List[str]:
        """
        Return list of optional configuration keys with defaults.
        
        Override this method to specify optional configuration.
        """
        return []
    
    async def _safe_execute(self, context: FlowContext, config: Dict[str, Any]) -> ExecutionResult:
        """
        Safely execute the task with error handling and timing.
        
        This method wraps the actual execute() method with:
        - Configuration validation
        - Error handling
        - Execution timing
        - Logging
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate configuration
            self.validate_config(config)
            
            # Check required config keys
            required_keys = self.get_required_config_keys()
            missing_keys = [key for key in required_keys if key not in config]
            if missing_keys:
                raise ValueError(f"Missing required configuration keys: {missing_keys}")
            
            self.logger.info(f"Executing {self.name} with config: {config}")
            
            # Execute the task
            result = await self.execute(context, config)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # Add executor metadata
            result.metadata.update({
                "executor": self.name,
                "executed_at": start_time.isoformat(),
                "execution_time": execution_time
            })
            
            self.logger.info(f"{self.name} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = f"Executor {self.name} failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_time=execution_time,
                metadata={
                    "executor": self.name,
                    "executed_at": start_time.isoformat(),
                    "execution_time": execution_time,
                    "error_type": type(e).__name__
                }
            )
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this executor.
        
        Returns metadata about the executor including its capabilities,
        required/optional configuration, and description.
        """
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "module": self.__class__.__module__,
            "required_config": self.get_required_config_keys(),
            "optional_config": self.get_optional_config_keys(),
            "description": self.__doc__ or "No description available"
        }
