from typing import Any, Dict

from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

retry_config = types.HttpRetryOptions(
    exp_base = 7,
    http_status_codes = [429, 500, 503, 504],
    initial_delay = 1,
    attempts = 5,
)

APP_NAME = 'default'
USER_ID = 'default'
SESSION = 'default'

MODEL_NAME = 'gemini-2.5-flash-lite'

root_agent = Agent(
    name = 'text_chat_bot',
    model = Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config
    ),
    instruction = 'A chatbot for recipes'
)

session_service = InMemorySessionService()

runner = Runner(agent = root_agent, app_name = APP_NAME, session_service = session_service)