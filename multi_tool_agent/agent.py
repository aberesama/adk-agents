from google.adk.agents import Agent
from zoneinfo import ZoneInfo
import datetime


def get_category(vehicle: str) -> dict:
    """
    Determines the required driving license class based on the vehicle type. 
    Use this tool when the user asks which driving class they should take for a specific vehicle.
    """
    if vehicle.lower() == "motorbike":
        return {
            "Status" : "Success",
            "Recommendation" : (f"Since you want to drive a {vehicle}, you should take the class A training"),
        }
    elif vehicle.lower() == "lorry":
        return {
            "Status" : "Success",
            "Recommendation" : f"Since you want to drive a {vehicle}, you should take class C.",
        }
    elif vehicle.lower() in ("car","pickup","wagon"):
        return {
            "Status" : "Success",
            "Recommendation" : f"Since you want to drive a {vehicle}, you should take class B.",
        }
    else:
        return {
            "Status" : "Error",
            "Error_message" : f"I cannot recommend ny class for driving a {vehicle}",
        }

def get_location(city: str) -> dict:
    """
    Determines which driving school branch is near your location and recommends it. Use this tool when the user gives you their location.
    """
    if city.lower() in ("kisii", "mwihoko", "kangemi"):
        return {
            "status" : "success",
            "recommendation" : f" Since you are in {city}, you should go to Zionlink Driving School branch",
        }
    elif city.lower() in ("nairobi", "narok", "busia", "kakamega", "bungoma", "eldoret"):
        return {
            "status" : "success",
            "recommendation" : f"since you are in {city}, you shoud go to Rocky Driving School branch",
        }
    else:
        return {
            "status" : "error",
            "recommendation" : f"there are no driving schools in {city}",
        }



root_agent = Agent(
    model='gemini-2.5-flash',
    name='driving_class_agent',
    description='A helpful assistant to answer questions about driving school classes',
    instruction='You are a helpful agent who can answer questions to help people decide which driving school class to take. Take their vehicle type and location then use tools given to output a one sentence string recommending their class and school from options given.',
    tools = [get_category, get_location]
)
