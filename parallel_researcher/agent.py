from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.plugins.logging_plugin import (LoggingPlugin,)
from google.adk.tools import AgentTool, FunctionTool, google_search

retry_config = types.HttpRetryOptions(
    attempts = 5,
    exp_base = 7,
    initial_delay = 1,
    http_status_codes = [429, 500, 503, 504]
)

tech_researcher = Agent(
    name = 'TechResearcher',
    model = Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config
    ),
    instruction = """ Research the latest AI/ML trends. Include 3 key developments, the main companies involved, and the potential impact. Keep the report very concise (100 words).""",
    tools = [google_search],
    output_key = 'tech_research',
)

health_researcher = Agent(
    name = 'HealthResearcher',
    model = Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config
    ),
    instruction = """ Research recent medical breakthroughs. Include 3 significant advances, their practical applications, and estimated timelines. Keep the report concise (100 words).""",
    tools = [google_search],
    output_key = 'health_research',
)

finance_researcher = Agent(
    name = 'FinanceResearcher',
    model = Gemini(
        model = 'gemini-2.5-flash',
        retry_options = retry_config
    ),
    instruction = """ """,
    tools = [google_search],
    output_key = 'finance_research',
)

aggregator_agent = Agent(
    name = 'AggregatorAgent',
    model = Gemini(
        model = 'gemini-2.5-flash',
        retry_options = retry_config
    ),
    instruction = """ Combine these three research findings into a single executive summary:

    **Technology Trends:**
    {tech_research}
    
    **Health Breakthroughs:**
    {health_research}
    
    **Finance Innovations:**
    {finance_research}
    
    Your summary should highlight common themes, surprising connections, and the most important key takeaways from all three reports. The final summary should be around 200 words. """,
    output_key = 'executive_summary',
)

parallel_researcher = ParallelAgent(
    name = 'ParallelResearcher',
    sub_agents = [tech_researcher, health_researcher, finance_researcher],
)

root_agent = SequentialAgent(
    name = 'ResearchSystem',
    sub_agents = [parallel_researcher, aggregator_agent],
)

runner = InMemoryRunner(
    agent = root_agent,
    plugins = [LoggingPlugin()]
)