# src/tools/order_tools.py
import json
import logging
from livekit.agents import function_tool, RunContext

logger = logging.getLogger(__name__)

@function_tool
async def save_order_to_json(context: RunContext, order: dict):
    """
    Save the final order to a JSON file.
    The LLM should call this tool only when all fields are filled.
    
    Args:
        order: A dictionary containing order details.
    """
    try:
        with open("orders.json", "a") as f:
            f.write(json.dumps(order) + "\n")
        return "Order saved successfully."
    except Exception as e:
        logger.error(f"Error saving order: {e}")
        return "Failed to save the order."
