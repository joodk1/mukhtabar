"""Grading: multiple-choice answers are right or wrong (exact match)."""


def grade(question, chosen: int) -> float:
    """1.0 if the chosen option (0-3) is the correct one, else 0.0."""
    return 1.0 if chosen == int(question["answer"]) else 0.0
