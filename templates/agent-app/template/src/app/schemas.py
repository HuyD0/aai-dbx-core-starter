"""Structured-output contracts for the agent's final answers."""

FINAL_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The final answer for the user.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Self-assessed confidence in the answer.",
        },
        "tools_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of the tools consulted for this answer.",
        },
    },
    "required": ["answer", "confidence"],
}
