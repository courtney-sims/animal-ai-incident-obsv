from typing import TypedDict


class Judgment(TypedDict):
    entry_id: str
    keep: bool
    reasoning: str
    confidence: str  # one of VALID_CONFIDENCES

VALID_CONFIDENCES = ("High", "Medium", "Low")
