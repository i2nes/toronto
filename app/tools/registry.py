"""Tool Registry for MCP-style tool calling.

Provides a dataclass-based tool system where LLMs can call tools
via JSON-formatted requests.
"""
import asyncio
import json
from dataclasses import dataclass
from typing import Callable, Awaitable, Dict, Any, Optional
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()


@dataclass
class Tool:
    """Tool definition with input/output schemas and handler."""
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[BaseModel]]


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ToolRegistry:
    """Registry for available tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._timeout = 30.0  # 30 second timeout for tool execution

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        logger.info("tool_registered", tool_name=tool.name)

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self.tools.values())

    def get_tools_description(self) -> str:
        """Get a formatted description of all tools for the LLM system prompt."""
        if not self.tools:
            return "No tools available."

        descriptions = []
        for tool in self.tools.values():
            # Get input schema from Pydantic model
            input_schema = tool.input_model.model_json_schema()

            descriptions.append(f"""
Tool: {tool.name}
Description: {tool.description}
Input schema: {json.dumps(input_schema, indent=2)}
""")

        return "\n".join(descriptions)

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool

        Returns:
            ToolResult with success status and data or error
        """
        tool = self.get_tool(tool_name)

        if not tool:
            logger.error("tool_not_found", tool_name=tool_name)
            return ToolResult(success=False, error=f"Tool '{tool_name}' not found")

        try:
            # Validate input with Pydantic
            validated_input = tool.input_model(**args)

            # Execute with timeout
            async with asyncio.timeout(self._timeout):
                result = await tool.handler(validated_input)

            # Convert result to dict
            result_dict = result.model_dump() if hasattr(result, 'model_dump') else {}

            logger.info(
                "tool_executed",
                tool_name=tool_name,
                success=True,
                result_preview=str(result_dict)[:100]
            )

            return ToolResult(success=True, data=result_dict)

        except asyncio.TimeoutError:
            logger.error("tool_timeout", tool_name=tool_name, timeout=self._timeout)
            return ToolResult(
                success=False,
                error=f"Tool execution timeout after {self._timeout}s"
            )

        except Exception as e:
            logger.exception("tool_execution_failed", tool_name=tool_name, error=str(e))
            return ToolResult(success=False, error=f"Tool execution failed: {str(e)}")


# Global registry instance
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry


def register_tool(tool: Tool) -> None:
    """Register a tool in the global registry."""
    _registry.register(tool)
