from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory
from google.genai import types

retry_config = types.HttpRetryOptions(
    attempts = 5,
    exp_base = 7,
    initial_delay = 1,
    http_status_codes = [429, 500, 503, 504],
)

memory_service = (InMemoryMemoryService())

session_service = InMemorySessionService()

APP_NAME = 'MemoryDemoApp'
USER_ID = 'demo_user'

async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )   

root_agent = LlmAgent(
    model=Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config,
    ),
    name='PersistentMechanic',
    description='A helpful assistant for user questions.',
    instruction='Answer user questions on vehicle repair to the best of your knowledge. Make sure to give instructions that are legible across all knowledge types for vehicle terminology. always begin the conversation by mentioning what you can do to the user.',
    tools = [preload_memory],
    after_agent_callback = auto_save_to_memory,
)

runner = Runner(
    agent = root_agent,
    app_name = APP_NAME,
    session_service = session_service,
    memory_service = memory_service,
)