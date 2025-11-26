"""
tutor_tools.py
This file contains helper functions for loading and retrieving tutor content
from the JSON file used in the Teach-the-Tutor voice agent (Day 4).
"""

import json
import os

# Load content from JSON file
def load_tutor_content():
    try:
        file_path = os.path.join("shared-data", "day4_tutor_content.json")
        with open(file_path, "r") as f:
            content = json.load(f)
        return content
    except Exception as e:
        print(f"Error loading content: {e}")
        return []


# Get concept by title (e.g., "variables", "loops")
def get_concept_by_title(content, title):
    for item in content:
        if item["title"].lower() == title.lower():
            return item
    return None


# Get a concept summary for LEARN mode
def get_summary(content, title):
    concept = get_concept_by_title(content, title)
    if concept:
        return concept["summary"]
    return "Sorry, I couldn't find that concept."


# Get sample question for QUIZ or TEACH_BACK mode
def get_sample_question(content, title):
    concept = get_concept_by_title(content, title)
    if concept:
        return concept["sample_question"]
    return "Sorry, no question found for that concept."
