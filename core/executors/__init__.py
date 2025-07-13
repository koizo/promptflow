"""
Core Executors Package

This package contains reusable execution units that can be orchestrated
by the flow engine to create declarative, YAML-based flows.

All executors inherit from BaseExecutor and follow a standardized interface
for consistent execution and result handling.
"""

from .base_executor import BaseExecutor, ExecutionResult, FlowContext
from .document_extractor import DocumentExtractor
from .llm_analyzer import LLMAnalyzer
from .file_handler import FileHandler
from .ocr_processor import OCRProcessor
from .image_handler import ImageHandler
from .data_combiner import DataCombiner
from .response_formatter import ResponseFormatter

__all__ = [
    "BaseExecutor",
    "ExecutionResult", 
    "FlowContext",
    "DocumentExtractor",
    "LLMAnalyzer",
    "FileHandler",
    "OCRProcessor",
    "ImageHandler",
    "DataCombiner",
    "ResponseFormatter"
]
