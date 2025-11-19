import uuid
from google.genai import types

from google.adk.agents import LlmAgent, Agent
from google.adk.models.google_llm import Gemini

from google.adk.tools.tool_context import ToolContext
from google.adk.tools.function_tool import FunctionTool

from google.adk.apps.app import App, ResumabilityConfig

from google.adk.sessions import InMemorySessionService

from google.adk.runners import Runner

retry_config = types.HttpRetryOptions(
    attempts = 5,
    exp_base = 7,
    initial_delay = 1,
    http_status_codes = [429, 500, 503, 504]
)

LARGER_ORDER_THRESHOLD = 5

def place_shipping_order( 
        num_containers: int, destination: str, tool_context: ToolContext
) -> dict:
    """Places a shipping order. Requires approval if ordering more than 5 containers (LARGE_ORDER_THRESHOLD).

    Args:
        num_containers: Number of containers to ship
        destination: Shipping destination

    Returns:
        Dictionary with order status"""
    
    if num_containers <= LARGER_ORDER_THRESHOLD:
        return {
            "status": "approved",
            "order_id": f"ORD-{num_containers}-AUTO",
            "num_containers": num_containers,
            destination: destination,
            "message": f"Order uto-approved: {num_containers} containers to {destination}",
        }
    
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint = f"Large Order: {num_containers} containers to {destination}. Do you want to approve?",
            payload = {"num_containers": num_containers, "destination": destination},
        )
        return {
            "status": "pending",
            "message": f"Order for {num_containers} containers requires approval",
        }
    
    if tool_context.tool_confirmation.confirmed:
        return{
            "status": "approved",
            "order_id": f"ORD-{num_containers}-HUMAN",
            "num_containers": num_containers,
            "destination": destination,
            "message":f"Order approved: {num_containers} containers to {destination}",
        }
    else:
        return{
            "status": "rejected",
            "message": f"Order rejected: {num_containers} containers to {destination}",
        }

shipping_agent = LlmAgent(
    name = 'shipping_agent',
    model = Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config),
    instruction = """You are a shipping coordinator assistant.
  
  When users request to ship containers:
   1. Use the place_shipping_order tool with the number of containers and destination
   2. If the order status is 'pending', inform the user that approval is required
   3. After receiving the final result, provide a clear summary including:
      - Order status (approved/rejected)
      - Order ID (if available)
      - Number of containers and destination
   4. Keep responses concise but informative""",
    tools = [FunctionTool(func=place_shipping_order)],
)

shipping_app = App(
    name = 'shipping_coordinator',
    root_agent = shipping_agent,
    resumability_config = ResumabilityConfig(is_resumable=True),
)

session_service = InMemorySessionService()

shipping_runner = Runner(
    app = shipping_app,
    session_service = session_service,
)

def check_for_approval(events):
    """Check if events contain an approval request.

    Returns:
        dict with approval details or None
    """
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if(
                    part.function_call and part.function_call.name == "adk_request_confirmation"
                ):
                    return{
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None
def print_agent_response(events):
    """Print agent's text responses from events."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")

def create_approval_response(approval_info, approved):
    """Create approval response message."""
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(
        role="user", parts=[types.Part(function_response=confirmation_response)]
    )

async def run_shipping_workflow(query: str, auto_approve: bool = True):
    """Runs a shipping workflow with approval handling.

    Args:
        query: User's shipping request
        auto_approve: Whether to auto-approve large orders (simulates human decision)
    """

    print(f"\n{'='*60}")
    print(f"User > {query}\n")

    session_id = f"order_{uuid.uuid4().hex[:8]}"

    await session_service.create_session(
        app_name="shipping_coordinator", user_id="test_user", session_id=session_id
    )

    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    events = []

    async for event in shipping_runner.run_async(
        user_id="test_user", session_id=session_id, new_message=query_content
    ):
        events.append(event)

    approval_info = check_for_approval(events)

    if approval_info:
        print(f"⏸️  Pausing for approval...")
        print(f"🤔 Human Decision: {'APPROVE ✅' if auto_approve else 'REJECT ❌'}\n") 
        
        async for event in shipping_runner.run_async(
            user_id="test_user",
            session_id=session_id,
            new_message=create_approval_response(
                approval_info, auto_approve
            ), 
            invocation_id=approval_info[
                "invocation_id"
            ], 
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent > {part.text}")

    else:
        print_agent_response(events)

    print(f"{'='*60}\n")
    
