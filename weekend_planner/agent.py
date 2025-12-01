from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool, FunctionTool, google_search, google_maps_grounding, ToolContext
from typing import List, Dict, Any
import json
import os

# HTTP Retry Configuration
retry_config = types.HttpRetryOptions(
    exp_base=7,
    http_status_codes=[429, 500, 503, 504],
    attempts=5,
    initial_delay=1,
)

# Budget thresholds for filtering
BUDGET_THRESHOLDS = {
    'low budget': {'max_price': 1000},
    'mid-range': {'min_price': 1001, 'max_price': 5000},
    'high-end': {'min_price': 5001}
}

# Simple in-memory storage for user preferences (replace with actual database in production)
USER_PREFERENCES_STORE = {}

# ==================== TOOL FUNCTIONS ====================

def filter_by_budget(ctx: ToolContext, activities: List[Dict], budget_preference: str) -> List[Dict]:
    """
    Filters a list of activities based on the user's budget preference 
    ('low budget', 'mid-range', 'high-end').
    
    Args:
        ctx: Tool context
        activities: List of activity dictionaries
        budget_preference: Budget category as string
        
    Returns:
        Filtered list of activities matching the budget
    """
    preference = budget_preference.lower()
    thresholds = BUDGET_THRESHOLDS.get(preference)
    
    if not thresholds:
        # Fail gracefully: If preference is unknown, no filtering is applied
        return activities 

    filtered_activities = []
    
    for activity in activities:
        price = activity.get('Price', 0) 
        
        try:
            # Safely parse the price string (e.g., "$35" -> 35.0)
            price_str = str(price).replace('$', '').replace(',', '').split()[0]
            price_value = float(price_str)
        except (ValueError, TypeError, IndexError):
            # Treat non-numeric (e.g., 'Free', 'TBD') as passing low/mid-range filters
            if str(price).lower() in ['free', 'tbd', 'n/a', 'contact for price']:
                filtered_activities.append(activity)
            continue 

        min_p = thresholds.get('min_price', 0)
        max_p = thresholds.get('max_price', float('inf'))

        if min_p <= price_value <= max_p:
            filtered_activities.append(activity)
            
    return filtered_activities


def retrieve_user_preferences(ctx: ToolContext, user_id: str) -> Dict[str, Any]:
    """
    Retrieves stored user preferences from the memory store.
    
    Args:
        ctx: Tool context
        user_id: Unique identifier for the user
        
    Returns:
        Dictionary containing user preferences or empty dict if not found
    """
    preferences = USER_PREFERENCES_STORE.get(user_id, {})
    
    if not preferences:
        # Return default preferences if none exist
        return {
            'user_id': user_id,
            'budget_preference': 'mid-range',
            'interests': [],
            'dietary_restrictions': [],
            'past_activities': [],
            'favorite_locations': []
        }
    
    return preferences


def save_user_preferences(ctx: ToolContext, user_id: str, preferences: Dict[str, Any]) -> Dict[str, str]:
    """
    Saves user preferences to the memory store.
    
    Args:
        ctx: Tool context
        user_id: Unique identifier for the user
        preferences: Dictionary of user preferences to save
        
    Returns:
        Status message indicating success or failure
    """
    try:
        # Merge with existing preferences
        existing = USER_PREFERENCES_STORE.get(user_id, {})
        existing.update(preferences)
        existing['user_id'] = user_id
        
        USER_PREFERENCES_STORE[user_id] = existing
        
        return {
            'status': 'success',
            'message': f'Preferences saved for user {user_id}',
            'saved_preferences': existing
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to save preferences: {str(e)}'
        }


# ==================== SPECIALIZED AGENTS ====================

# Fact Checker Specialist Agent
fact_checker_specialist = Agent(
    name="FactCheckerAgent",
    model=Gemini(model="gemini-2.0-flash-exp"),
    instruction="""
    Your sole task is to take raw, unstructured event data (text) and convert it into a clean, 
    predictable JSON list. The required schema is a list of objects, each containing: 
    - 'Activity Name' (string): The name of the event or activity
    - 'Address' (string): Full street address
    - 'Price' (string or number): Cost in local currency or 'Free', 'TBD'
    - 'Time' (string): Date and time of the event
    - 'URL' (string): Web link to more information
    
    If data is missing (e.g., Price), try to infer it from context or mark it as 'TBD' or 
    'Contact for Price'. Always prioritize accuracy over completeness.
    """,
    tools=[google_search],
    description="Specialist agent for fact-checking and structuring event data"
)

# Geospatial Agent for route optimization
geospatial_agent = Agent(
    name="GeospatialAgent",
    model=Gemini(model='gemini-2.0-flash-exp'),
    instruction="""
    Your sole task is to calculate the most efficient travel route, distance, and estimated 
    time (in minutes) between a given list of sequential stops (addresses). 
    
    Use the 'google_maps_grounding' tool to get accurate travel information.
    
    Return the result as a structured JSON list of segments, each including:
    - 'start_address': Starting location
    - 'end_address': Destination
    - 'travel_time_minutes': Estimated travel time
    - 'distance_km': Distance in kilometers
    
    If addresses are unclear, try to geocode them first. Optimize for minimal total travel time.
    """,
    tools=[google_maps_grounding],
    description="Specialist for calculating optimal routes and travel times"
)

# ==================== MAIN WORKFLOW AGENTS ====================

# Event Sourcing Agent
event_sourcing_agent = Agent(
    model=Gemini(
        retry_options=retry_config,
        model='gemini-2.0-flash-exp'
    ),
    name='EventSourcingAgent',
    instruction="""
    Your sole purpose is to gather comprehensive event data based on the provided location 
    and date range, maximizing recall and accuracy.
    
    Process:
    1. Use Google Search to find events, activities, and attractions in the specified location
    2. Search for events on popular platforms like Eventbrite, Luma, Meetup, local event calendars
    3. Pass the raw results to the FactCheckerAgent to structure the data
    4. Return a clean JSON list of activities
    
    If a tool fails (e.g., API timeout), use alternative search strategies and fail gracefully.
    Always aim to return at least 10-15 diverse activity options.
    
    Search for various categories: cultural events, outdoor activities, food/dining, 
    entertainment, workshops, sports, and local attractions.
    """,
    description='Specialist agent for gathering and structuring real-time event data from web sources',
    tools=[google_search, AgentTool(agent=fact_checker_specialist)],
)

# Itinerary Planning Agent
itinerary_planning_agent = Agent(
    name='ItineraryPlanningAgent',
    description='Creates personalized, logical itineraries by applying user preferences and calculating travel logistics',
    model=Gemini(
        model='gemini-2.0-flash-exp',
        retry_options=retry_config
    ),
    instruction="""
    You are the expert logistician responsible for creating the perfect 2-day weekend itinerary.
    
    Your process:
    1. Review the user preferences provided (budget, interests, past activities)
    2. Apply budget filtering using the filter_by_budget tool
    3. Rank activities based on user interests and preferences
    4. For top candidates, use the GeospatialAgent to calculate travel times
    5. Create a realistic schedule that:
       - Balances activity types (mix of relaxation and excitement)
       - Minimizes travel time between activities
       - Allows adequate time for each activity
       - Stays within budget constraints
       - Avoids scheduling conflicts
    
    Output format should be a structured 2-day itinerary with:
    - Day 1 and Day 2 sections
    - Chronological ordering with times
    - Activity details (name, location, price, duration)
    - Travel time between activities
    - Total estimated cost
    
    If there are ambiguities or conflicts, note them clearly in your output.
    """,
    tools=[
        AgentTool(agent=geospatial_agent), 
        FunctionTool(filter_by_budget)
    ],
)

# User Memory Agent
user_memory_agent = Agent(
    name='UserMemoryAgent',
    model=Gemini(
        model='gemini-2.0-flash-exp',
        retry_options=retry_config
    ),
    instruction="""
    You are the personalized memory expert for the user. Your responsibilities:
    
    RETRIEVAL MODE:
    - When queried, retrieve the most relevant user preferences
    - Prioritize semantic relevance and recency
    - Return preferences that will help personalize the itinerary
    
    STORAGE MODE:
    - When provided with new preference information, save it appropriately
    - Update existing preferences intelligently (don't overwrite everything)
    - Maintain user privacy and data isolation
    
    Key preference categories to track:
    - Budget preference (low budget, mid-range, high-end)
    - Interests and hobbies
    - Dietary restrictions
    - Past activities and ratings
    - Favorite locations and venues
    - Activity pace preference (relaxed vs. packed schedule)
    
    Always ensure data is properly structured and up-to-date.
    """,
    description='Manages user preferences, historical interactions, and personalized context',
    tools=[
        FunctionTool(retrieve_user_preferences), 
        FunctionTool(save_user_preferences)
    ],
)

# Root Coordinator Agent
root_agent = Agent(
    model=Gemini(
        model='gemini-2.0-flash-exp',
        retry_options=retry_config
    ),
    name='WeekendPlannerCoordinator',
    description='Top-level orchestrator that manages the entire itinerary creation workflow',
    instruction="""
    You are the Project Manager for the Weekend Planner System. Your mission is orchestration.
    
    You must execute the planning workflow in this EXACT sequential order:
    
    STEP 1 - CONTEXT RETRIEVAL:
    - Call the UserMemoryAgent first to gather user preferences
    - Extract user_id from the query (or use 'default_user' if not provided)
    - Use retrieve_user_preferences to get budget, interests, and history
    
    STEP 2 - DATA SOURCING:
    - Call the EventSourcingAgent with location and date range
    - Include user budget and interests in the search context
    - Ensure you get a comprehensive list of activities (aim for 15+ options)
    
    STEP 3 - ITINERARY SYNTHESIS:
    - Call the ItineraryPlanningAgent with:
      * The activity list from Step 2
      * The user preferences from Step 1
    - Let it create the optimized 2-day schedule
    
    STEP 4 - MEMORY UPDATE (Optional):
    - If the user provides new preferences during the interaction, 
      call UserMemoryAgent to save them
    
    Track the output of each step and use it as input for the next step.
    Do not synthesize the final answer yourself - rely on the ItineraryPlanningAgent's output.
    
    Present the final itinerary in a clear, user-friendly format with all relevant details.
    """,
    tools=[
        AgentTool(agent=user_memory_agent),
        AgentTool(agent=event_sourcing_agent),
        AgentTool(agent=itinerary_planning_agent)
    ]
)

# ==================== RUNNER INITIALIZATION ====================

# Initialize session service (required for Runner)
session_service = InMemorySessionService()

# Initialize runner with agent and session service
runner = Runner(
    agent = root_agent,
    app_name = 'weekend_planner',
    session_service = session_service
)