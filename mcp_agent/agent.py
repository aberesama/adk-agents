from google.genai import types

from google.adk.agents import LlmAgent, Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.runners import InMemoryRunner

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.tools.function_tool import FunctionTool

retry_config = types.HttpRetryOptions(
    attempts = 5,
    exp_base = 7,
    initial_delay = 1,
    http_status_codes = [429, 500, 503, 504]
)

mcp_image_server = McpToolset(
    connection_params = StdioConnectionParams(
        server_params = StdioServerParameters(
            command = 'npx',
            args = [
                '-y',
                '@modelcontextprotocol/server-everything',
            ],
        ),
        timeout = 30,
    ),
    tool_filter = ['getTinyImage'],
)

image_agent = LlmAgent(
    name = 'image_agent',
    model = Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config,
    ),
    instruction = "Use the MCP Tool to generate images for user queries",
    tools = [mcp_image_server],
)

runner = InMemoryRunner(agent = image_agent)