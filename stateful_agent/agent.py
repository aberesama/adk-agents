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

USER_NAME_SCOPE_LEVELS = ('temp', 'user', 'app')

def save_userinfo(
        tool_context: ToolContext, user_name: str, country: str
) -> Dict[str, Any]:
    """Tool to record and save user name and country in session state.

    Args:
        user_name: The username to store in session state
        country: The name of the user's country"""
    tool_context.state['user:name'] = user_name
    tool_context.state['user:country'] = country

    return {'status': 'success'}

def retrieve_user_info(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to retrieve user name and country from session state.
    """

    user_name = tool_context.state.get('user:name', 'Usernam not found')
    country = tool_context.state.get('user:country', 'Country not found')

    return {'status': 'success', 'user_name': user_name, 'country': country}


root_agent = LlmAgent(
    name = 'text_chat_bot',
    model = Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config
    ),
    description = """A chatbot wih persistent memory for recipes. * To record username and country when provided use `save_userinfo` tool. 
    * To fetch username and country when required use `retrieve_userinfo` tool.""",
    tools = [save_userinfo, retrieve_user_info],
)

research_app_compacting = App(
    name = 'research_app_compacting',
    root_agent = root_agent,
    events_compaction_config = EventsCompactionConfig(
        compaction_interval = 3,                    overlap_size = 1,),
)

#session_service = InMemorySessionService()

db_url = 'sqlite:///my_agent_data.db'
session_service = DatabaseSessionService(db_url = db_url)

#runner = Runner(agent = root_agent, app_name = APP_NAME, session_service = session_service)

research_runner_compacting = Runner(
    app =research_app_compacting, session_service = session_service
)