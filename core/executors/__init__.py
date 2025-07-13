"""
Core Executors Package

This package contains reusable execution units that can be orchestrated
by the flow engine to create declarative, YAML-based flows.

All executors inherit from BaseExecutor and follow a standardized interface
for consistent execution and result handling.
"""

from .base_executor import BaseExecutor, ExecutionResult, FlowContext

__all__ = [
    "BaseExecutor",
    "ExecutionResult", 
    "FlowContext"
]
