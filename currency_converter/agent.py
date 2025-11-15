from google.genai import types

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search, AgentTool, ToolContext
from google.adk.code_executors import BuiltInCodeExecutor

retry_config = types.HttpRetryOptions(
    attempts = 5,
    exp_base = 7,
    initial_delay = 1,
    http_status_codes = [429, 500, 503, 504],
)

def get_fee_for_payment_method(method: str) -> dict:
    """Looks up the transaction fee percentage for a given payment method.

    This tool simulates looking up a company's internal fee structure based on
    the name of the payment method provided by the user.

    Args:
        method: The name of the payment method. It should be descriptive,
                e.g., "platinum credit card" or "bank transfer".

    Returns:
        Dictionary with status and fee information.
        Success: {"status": "success", "fee_percentage": 0.02}
        Error: {"status": "error", "error_message": "Payment method not found"}"""
    fee_database = {
        "platinum credit card": 0.02,
        "gold debit card": 0.035, 
        "bank transfer": 0.01,
    }

    fee = fee_database.get(method.lower())
    if fee is not None:
        return {'status': 'success', 'fee_percentage': fee}
    else:
        return {
            "status": "error",
            "error_message": f"Payment method '{method}' not found",
        }

def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """Looks up and returns the exchange rate between two currencies.

    Args:
        base_currency: The ISO 4217 currency code of the currency you
                       are converting from (e.g., "USD").
        target_currency: The ISO 4217 currency code of the currency you
                         are converting to (e.g., "EUR").

    Returns:
        Dictionary with status and rate information.
        Success: {"status": "success", "rate": 0.93}
        Error: {"status": "error", "error_message": "Unsupported currency pair"}"""
    
    rate_database = {
        "usd": {
            "eur": 0.93,
            "jpy": 157.50,
            "inr": 83.58,
        }
    }

    base = base_currency.lower()
    target = target_currency.lower()

    rate = rate_database.get(base, {}).get(target)
    if rate is not None:
        return {"status": "success", "rate": rate}
    else: 
        return {
            "status": "error",
            "error_message": f"Unsupported currency pair: {base_currency}/{target_currency}",
        }
    
calculation_agent = LlmAgent(
    name = 'CalculationAgent',
    model = Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config
    ),
    instruction = """You are a specialized calculator that ONLY responds with Python code. You are forbidden from providing any text, explanations, or conversational responses.
 
     Your task is to take a request for a calculation and translate it into a single block of Python code that calculates the answer.
     
     **RULES:**
    1.  Your output MUST be ONLY a Python code block.
    2.  Do NOT write any text before or after the code block.
    3.  The Python code MUST calculate the result.
    4.  The Python code MUST print the final result to stdout.
    5.  You are PROHIBITED from performing the calculation yourself. Your only job is to generate the code that will perform the calculation.
   
    Failure to follow these rules will result in an error.""",
    code_executor = BuiltInCodeExecutor(),
)

root_agent = LlmAgent(
    name = 'currency_agent',
    model = Gemini(
        model = 'gemini-2.5-flash-lite',
        retry_options = retry_config
    ),
    instruction = """You are a smart currency conversion assistant. You must strictly follow these steps and use the available tools.

  For any currency conversion request:

   1. Get Transaction Fee: Use the get_fee_for_payment_method() tool to determine the transaction fee.
   2. Get Exchange Rate: Use the get_exchange_rate() tool to get the currency conversion rate.
   3. Error Check: After each tool call, you must check the "status" field in the response. If the status is "error", you must stop and clearly explain the issue to the user.
   4. Calculate Final Amount (CRITICAL): You are strictly prohibited from performing any arithmetic calculations yourself. You must use the calculation_agent tool to generate Python code that calculates the final converted amount. This 
      code will use the fee information from step 1 and the exchange rate from step 2.
   5. Provide Detailed Breakdown: In your summary, you must:
       * State the final converted amount.
       * Explain how the result was calculated, including:
           * The fee percentage and the fee amount in the original currency.
           * The amount remaining after deducting the fee.
           * The exchange rate applied.""",
    tools = [get_exchange_rate, get_fee_for_payment_method, AgentTool(agent = calculation_agent)],
)