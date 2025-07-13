"""
Document Analysis Flow

A complete flow that combines document text extraction with LLM analysis.
This flow follows the established architecture pattern by separating concerns:
- Core document extraction providers remain in core/document_extraction/
- Flow-specific analysis logic and API endpoints are here in flows/document_analysis/
"""

from .router import router

__all__ = ["router"]
