"""Tools package - imports all tools to trigger registration."""
from app.tools.registry import get_registry, Tool, ToolResult, ToolRegistry
from app.tools import weather, websearch

__all__ = ["get_registry", "Tool", "ToolResult", "ToolRegistry"]
