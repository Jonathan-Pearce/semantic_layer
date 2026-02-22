"""Evaluator module.

Loads the test question set, dispatches questions to both the
baseline and semantic agents, and collects results for comparison.
"""

import json
import os

QUESTIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.json")


def load_questions(path: str | None = None) -> list[dict]:
    """Load the evaluation question set from a JSON file.

    Parameters
    ----------
    path : str | None
        Path to the questions JSON file.  Defaults to
        ``evaluator/questions.json``.

    Returns
    -------
    list[dict]
        The list of question objects.
    """
    path = path or QUESTIONS_PATH
    with open(path, "r") as f:
        data = json.load(f)
    return data["question_set"]["questions"]


def save_results(results: list[dict], path: str) -> None:
    """Save evaluation results to a JSON file.

    Parameters
    ----------
    results : list[dict]
        List of result dicts from agent runs.
    path : str
        Destination file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
