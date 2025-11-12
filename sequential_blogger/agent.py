from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search

retry_config = types.HttpRetryOptions(
    attempts = 5,
    exp_base = 7,
    initial_delay = 1,
    http_status_codes = [429, 500, 503, 504]
)

outline_agent = Agent(
    name = 'OutlineAgent',
    model = Gemini(
        model='gemini-2.5-flash-lite',
        retry_options = retry_config
    ),
    instruction = """
    Create a blog outline for the given topic with:
    1. A catchy headline
    2. An introduction hook
    3. 3-5 main sections with 2-3 bullet points for each
    4. A concluding thought
    """,
    output_key = 'blog_outline',
)

writer_agent = Agent(
    name = 'WriterAgent',
    model = Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config
    ),
    instruction = """
    Following this outline strictly: {blog_outline} 
    Write a brief, 200 to 300-word blog post with an engaging and informative tone.
    """,
    output_key = 'blog_draft',
)

editor_agent = Agent(
    name = 'EditorAgent',
    model = Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config
    ),
    instruction = """
    Edit this draft: {blog_draft}
    Your task is to polish the text by fixing any grammatical errors, improving the flow and sentence structure, and enhancing overall clarity.
    """,
    output_key = 'final_blog',
)

root_agent = Agent(
    name='BlogPipeline',
    sub_agents = [outline_agent, writer_agent, editor_agent]
)
