"""Baseline agent module.

This agent operates WITHOUT access to the semantic layer.
It must be completely isolated from any semantic layer files,
chat history, or cached tokens from the semantic agent.
"""

import json
import os

import duckdb
import pandas as pd
import requests

GITHUB_MODELS_URL = "https://models.inference.ai.azure.com"

SYSTEM_PROMPT = (
    "You are a data analysis assistant. You have access to a dataset in "
    "parquet format. You have NO access to any semantic layer, data dictionary, "
    "or metadata definitions. Answer questions using only the raw data. "
    "When asked, provide code in Python, SQL (DuckDB dialect), or R as requested."
)


def load_data(parquet_path: str) -> pd.DataFrame:
    """Load a parquet file into a pandas DataFrame."""
    return pd.read_parquet(parquet_path)


def query_duckdb(parquet_path: str, sql: str) -> pd.DataFrame:
    """Execute a SQL query against a parquet file using DuckDB."""
    conn = duckdb.connect()
    conn.execute(
        f"CREATE VIEW dataset AS SELECT * FROM read_parquet('{parquet_path}')"
    )
    result = conn.execute(sql).fetchdf()
    conn.close()
    return result


def call_llm(
    question: str,
    model: str = "openai/gpt-4o",
    token: str | None = None,
) -> str:
    """Send a question to the GitHub Models API and return the response.

    Parameters
    ----------
    question : str
        The user question to send.
    model : str
        The model identifier to use.
    token : str | None
        GitHub token for authentication. Falls back to the
        ``GITHUB_TOKEN`` environment variable.

    Returns
    -------
    str
        The assistant's response text.
    """
    token = token or os.environ.get("GITHUB_TOKEN", "")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    }
    response = requests.post(
        f"{GITHUB_MODELS_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def run_question(question: str, **kwargs) -> dict:
    """Run a single question through the baseline agent.

    Returns a dict with the question and the agent's answer.
    """
    answer = call_llm(question, **kwargs)
    return {"agent": "baseline", "question": question, "answer": answer}


def run_questions(questions: list[dict], **kwargs) -> list[dict]:
    """Run a list of question dicts through the baseline agent.

    Each item in *questions* must have a ``"question"`` key.
    """
    results = []
    for q in questions:
        result = run_question(q["question"], **kwargs)
        result["id"] = q.get("id")
        result["category"] = q.get("category")
        results.append(result)
    return results
