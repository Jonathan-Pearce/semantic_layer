# Semantic Layer Evaluation Framework

An A/B testing framework to evaluate how a semantic layer improves LLM-based data analysis and metrics reporting.

## Project Structure

```
├── baseline_agent/        # LLM agent WITHOUT semantic layer access
│   ├── __init__.py
│   └── agent.py
├── semantic_agent/        # LLM agent WITH semantic layer access
│   ├── __init__.py
│   ├── agent.py
│   ├── semantic_layer.json
│   └── semantic_layer.yaml
├── evaluator/             # Evaluation harness and question set
│   ├── __init__.py
│   ├── evaluator.py
│   └── questions.json
├── data/                  # Dataset directory (parquet files)
├── run_evaluation.py      # Main entry point
└── requirements.txt
```

## Overview

The framework runs the same set of questions against two isolated LLM agents:

- **Baseline Agent** — has access only to the raw dataset. No semantic layer, glossary, or business context is provided.
- **Semantic Agent** — receives the semantic layer (metric definitions, dimensions, glossary) in its system prompt alongside the raw dataset.

The evaluation question set (`evaluator/questions.json`) begins with **isolation checks** that verify the baseline agent cannot access the semantic layer, chat history, or cached tokens from the semantic agent.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Set the `GITHUB_TOKEN` environment variable, then run:

```bash
export GITHUB_TOKEN="your_github_token"
python run_evaluation.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--questions` | `evaluator/questions.json` | Path to the question set |
| `--model` | `openai/gpt-4o` | GitHub Models API model identifier |
| `--output-dir` | `results` | Directory for output files |

Results are written to the output directory as:

- `baseline_results.json` — raw answers from the baseline agent
- `semantic_results.json` — raw answers from the semantic agent
- `comparison.json` — side-by-side comparison

## Semantic Layer

Template semantic layer files are provided in both JSON and YAML formats under `semantic_agent/`. Replace the template content with actual dataset definitions:

- **Entities** — tables or logical objects in the dataset
- **Metrics** — aggregate expressions and KPIs
- **Dimensions** — categorical or temporal columns for grouping
- **Glossary** — business term definitions

## Data

Place parquet files in the `data/` directory. The agents use [DuckDB](https://duckdb.org/) for SQL queries and [pandas](https://pandas.pydata.org/) / [PyArrow](https://arrow.apache.org/docs/python/) for dataframe operations.

## Question Set

Questions are defined in `evaluator/questions.json` and support requesting answers in multiple output formats: Python, SQL (DuckDB), and R.

Questions are grouped into categories:
- `isolation_check` — verify agent isolation
- `data_analysis` — test data exploration and understanding
- `metrics_reporting` — test metric computation and reporting
