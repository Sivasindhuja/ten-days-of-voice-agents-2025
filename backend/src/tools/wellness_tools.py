# src/tools/wellness_tools.py

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from livekit.agents import function_tool, RunContext

logger = logging.getLogger(__name__)

LOG_FILE = "wellness_log.json"


def _load_all_entries() -> List[Dict[str, Any]]:
    """Internal helper: load all entries from wellness_log.json."""
    if not os.path.exists(LOG_FILE):
        return []

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                logger.warning("Unexpected JSON structure in wellness_log.json, resetting to []")
                return []
    except Exception as e:
        logger.error(f"Failed to read {LOG_FILE}: {e}")
        return []


def _save_all_entries(entries: List[Dict[str, Any]]) -> None:
    """Internal helper: save all entries to wellness_log.json."""
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to write {LOG_FILE}: {e}")


@function_tool
async def get_last_wellness_entry(context: RunContext) -> str:
    """
    Return the last stored wellness check-in entry as a JSON string.
    If no data exists, return an empty string.
    The LLM should parse or summarize this text and reference it gently.
    """
    entries = _load_all_entries()
    if not entries:
        return ""

    last = entries[-1]
    # Return a compact JSON string the LLM can read.
    return json.dumps(last, ensure_ascii=False)


@function_tool
async def log_wellness_entry(
    context: RunContext,
    mood: str,
    energy: str,
    stressors: str,
    objectives: List[str],
    summary: str,
) -> str:
    """
    Store a new wellness check-in entry in wellness_log.json.

    Args:
        mood: Short text describing current mood (e.g. "anxious but hopeful").
        energy: Short text for energy ("low", "medium", "high", etc.).
        stressors: Short description of main stressors, or "none".
        objectives: List of 1–3 short, concrete goals or intentions.
        summary: One short sentence summarizing the check-in.

    Returns:
        A short confirmation message.
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mood": mood,
        "energy": energy,
        "stressors": stressors,
        "objectives": objectives,
        "summary": summary,
    }

    entries = _load_all_entries()
    entries.append(entry)
    _save_all_entries(entries)

    logger.info(f"Saved wellness entry: {entry}")
    return "Wellness entry saved."
