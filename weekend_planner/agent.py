from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini
from google.genai import types
from google.adk.tools import AgentTool, FunctionTool

retry_config = types.HttpRetryOptions(
    exp_base = 7,
    http_status_codes=[429, 500, 503, 504],
    attempts = 5,
    initial_delay = 1,
)

event_sourcing_agent = Agent(
    model = Gemini(
        retry_options = retry_config,
        model = 'gemini-2.5-flash'
    ),
    name = 'EventSourscingAgent',
    instruction = """"
    Your sole purpose is to gather comprehensive event data based on the provided location and date range, maximizing recall. Process the output from scraping tools to convert raw, unstructured text into a predictable JSON schema (Activity Name, Address, Price, Time, URL). If a tool fails (e.g., API times out), use the Real-Time Search Tool to find a high-level list of alternatives, ensuring you fail gracefully and maintain Robustness
    """,
    description = 'A specialist agent dedicated to querying and processing real-time, unstructured data from external web sources (scrapers like Luma/Eventbrite) and third-party APIs. Its goal is to return a clean, structured list of potential activities.' ,
    tools = [scrape_listings, fact_checker],
    output_key = 'filtered_activities',
)

itinerary_planning_agent = Agent(
    name = 'ItineraryPlanningAgent',
    description = 'Responsible for taking the raw activity list, applying user constraints (budget, memory), and performing complex calculations (e.g., travel time) using the Geospatial tool to synthesize a personalized, logical, and conflict-free 2-day schedule',
    model = Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config
    ),
    instruction = """You are the expert logistician. Your goal is to construct a realistic, personalized itinerary. Use the user preferences provided by the User Memory Agent to rank activities. For the top candidates, use the Geospatial Tool to ensure the itinerary is feasible (e.g., travel time between activities is reasonable). If the user's budget is ambiguous (e.g., 'low budget') or two highly-ranked activities conflict, invoke the Human-in-the-Loop Tool to ask the WPA to seek clarification""",
    tools = [calculate_route, filter_by_budget, ],
    output_key = 'itinerary_content',
)

user_memory_agent = Agent(
    name = 'UserMemoryAgent',
    model = Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config
    ),
    instruction = """
    You are the personalized expert on the user. When queried, you must retrieve the most semantically relevant memories to augment the Planner Agent's context. Your memory store is strictly isolated per user to ensure Safety and Alignment. When performing retrieval, prioritize blending relevance (semantic similarity) with recency (time-based score) to find the best fit context for the current query.
    """ ,
    description = 'Manages the long-term persistence of user preferences, historical interactions, and declarative memories (e.g., favorite foods, past successful trips, specific budget constraints). The primary WPA will call this agent using the Memory-as-a-Tool pattern',
    tools = [retrieve_user_preferences, save_user_preferences],
    output_key = 'user_preferences',

)

root_agent = Agent(
    model = Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config
    ),
    name='WeekendPlannerCoordinator',
    description = 'The top-level Coordinator that manages the entire itinerary creation workflow.',
    instruction = """
    You are the Project Manager for the Weekend Planner System. Your primary function is orchestration.
    Your mission is to decompose the user's request (Location, Budget, Interests) into a precise, multi-step trajectory
    and delegate each step to the appropriate specialized Agent Tools.

    You must execute the planning in the following sequential order:
    1. CONTEXT RETRIEVAL: You MUST call the UserMemoryAgent first to gather personal context.
    2. DATA SOURCING: You MUST call the EventSourcingAgent next, using the retrieved user data (especially budget and location) as input for searching.
    3. LOGISTICS PLANNING: You MUST call the ItineraryPlanningAgent last, using the resulting user context and activity list to synthesize the final 2-day itinerary.

    Ensure you track the output of each sub-agent call (Observation) and use it as input for the next agent call (Action) to maintain coherence. Do not synthesize the final answer yourself; rely on the final output from the ItineraryPlanningAgent.
    """,
    tools = [AgentTool(agent = event_sourcing_agent ), AgentTool( agent = user_memory_agent ), AgentTool( agent = itinerary_planning_agent )]
)
