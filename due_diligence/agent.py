from google.adk.agents.llm_agent import Agent
from google.genai import types
from google.adk.models.google_llm import Gemini

retry_config = types.HttpRetryOptions(
    attempts = 5,
    exp_base = 7,
    initial_delay = 1,
    http_status_codes = [429, 500, 503, 504]
)

root_agent = Agent(
    model = Gemini(retry_options = retry_config, model='gemini-2.5-flash-lite'),
    name = 'root_agent',
    description = 'A helpful assistant for user questions.',
    instruction = 'Answer user questions to the best of your knowledge',
)

regulation_confirmation_agent = Agent(
    model = Gemini(retry_options = retry_config, model='gemini-2.5-flash-lite'),
    name = 'regulation_confirmation_agent',
    description = 'An agent that confirms if the stated investment scheme is regulated in Kenya.',
    instruction = """ """,
    tools = []
)

net_interest_agent = Agent(
    model = Gemini(retry_options = retry_config, model = 'gemini-2.5-flash-lite'),
    name = 'net_interest_agent',
    description = 'An agent that calculates the net interest rate after accounting for all fees and charges.',
    instruction = """ """,
    tools = []
)

business_nature_agent = Agent(
    model = Gemini(retry_options = retry_config, model='gemini-2.5-flash-lite'),
    name = 'business_nature_agent',
    description = 'An agent that evaluates the nature of the business offering the investment scheme.',
    instruction = """ """,
    tools = []
)