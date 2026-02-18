"""Semantic agent module.

This agent operates WITH access to the semantic layer.
It loads semantic layer definitions (JSON or YAML) and includes
them in its system prompt so the LLM can use business context
when answering questions.
"""

import json
import os

import duckdb
import pandas as pd
import requests
import yaml

GITHUB_MODELS_URL = "https://models.inference.ai.azure.com"

SEMANTIC_LAYER_DIR = os.path.dirname(os.path.abspath(__file__))


def load_semantic_layer(fmt: str = "json") -> dict:
    """Load the semantic layer definition.

    Parameters
    ----------
    fmt : str
        Format to load — ``"json"`` or ``"yaml"``.
    """
    if fmt == "yaml":
        path = os.path.join(SEMANTIC_LAYER_DIR, "semantic_layer.yaml")
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        path = os.path.join(SEMANTIC_LAYER_DIR, "semantic_layer.json")
        with open(path, "r") as f:
            return json.load(f)


def build_system_prompt(semantic_layer: dict) -> str:
    """Build the system prompt including semantic layer context."""
    layer_text = json.dumps(semantic_layer, indent=2)
    return (
        "You are a data analysis assistant. You have access to a dataset in "
        "parquet format and the following semantic layer that describes the "
        "data's business context, metrics, dimensions, and glossary:\n\n"
        f"{layer_text}\n\n"
        "Use this semantic layer to inform your answers. "
        "When asked, provide code in Python, SQL (DuckDB dialect), or R as requested."
    )


def load_data(parquet_path: str) -> pd.DataFrame:
    """Load a parquet file into a pandas DataFrame."""
    return pd.read_parquet(parquet_path)


def query_duckdb(parquet_path: str, sql: str) -> pd.DataFrame:
    """Execute a SQL query against a parquet file using DuckDB."""
    parquet_path = os.path.realpath(parquet_path)
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    conn = duckdb.connect()
    conn.execute("CREATE VIEW dataset AS SELECT * FROM read_parquet(?)", [parquet_path])
    result = conn.execute(sql).fetchdf()
    conn.close()
    return result


def call_llm(
    question: str,
    semantic_layer: dict | None = None,
    model: str = "openai/gpt-4o",
    token: str | None = None,
) -> str:
    """Send a question to the GitHub Models API with semantic layer context.

    Parameters
    ----------
    question : str
        The user question to send.
    semantic_layer : dict | None
        Semantic layer definition to include in the system prompt.
        If ``None``, the layer is loaded from the default JSON file.
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
    if semantic_layer is None:
        semantic_layer = load_semantic_layer()

    system_prompt = build_system_prompt(semantic_layer)

    token = token or os.environ.get("GITHUB_TOKEN", "")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
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
    """Run a single question through the semantic agent.

    Returns a dict with the question and the agent's answer.
    """
    answer = call_llm(question, **kwargs)
    return {"agent": "semantic", "question": question, "answer": answer}


def run_questions(questions: list[dict], **kwargs) -> list[dict]:
    """Run a list of question dicts through the semantic agent.

    Each item in *questions* must have a ``"question"`` key.
    """
    results = []
    for q in questions:
        result = run_question(q["question"], **kwargs)
        result["id"] = q.get("id")
        result["category"] = q.get("category")
        results.append(result)
    return results
