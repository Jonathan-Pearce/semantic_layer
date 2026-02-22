"""Run the evaluation suite.

Loads the question set, sends each question to both the baseline and
semantic agents, and writes results to ``results/``.

Usage
-----
    python run_evaluation.py [--questions PATH] [--model MODEL]

Environment variables
---------------------
GITHUB_TOKEN
    Required. A GitHub personal access token used to authenticate
    with the GitHub Models API.
"""

import argparse
import json
import os
import sys

from baseline_agent.agent import run_questions as baseline_run
from semantic_agent.agent import run_questions as semantic_run
from evaluator.evaluator import load_questions, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic layer evaluation")
    parser.add_argument(
        "--questions",
        default=None,
        help="Path to the questions JSON file (default: evaluator/questions.json)",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o",
        help="Model identifier for the GitHub Models API (default: openai/gpt-4o)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for result files (default: results)",
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable is required.", file=sys.stderr)
        sys.exit(1)

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions")

    # Run baseline agent (no semantic layer)
    print("Running baseline agent...")
    baseline_results = baseline_run(questions, model=args.model, token=token)

    # Run semantic agent (with semantic layer)
    print("Running semantic agent...")
    semantic_results = semantic_run(questions, model=args.model, token=token)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results(baseline_results, os.path.join(args.output_dir, "baseline_results.json"))
    save_results(semantic_results, os.path.join(args.output_dir, "semantic_results.json"))

    # Combined results for easy comparison
    if len(baseline_results) != len(semantic_results):
        print(
            f"Warning: result count mismatch — baseline={len(baseline_results)}, "
            f"semantic={len(semantic_results)}",
            file=sys.stderr,
        )
    combined = []
    for b, s in zip(baseline_results, semantic_results):
        combined.append({
            "id": b.get("id"),
            "category": b.get("category"),
            "question": b["question"],
            "baseline_answer": b["answer"],
            "semantic_answer": s["answer"],
        })
    save_results(combined, os.path.join(args.output_dir, "comparison.json"))

    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
