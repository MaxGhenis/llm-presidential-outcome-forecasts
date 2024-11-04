from openai import OpenAI
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime
from .config import Config
from .utils import format_history
from .validation import extract_numbers_vectorized


def get_gpt4_responses(
    client: OpenAI, prompt: str, model: str, n: int = 50
) -> List[str]:
    """Get multiple responses from GPT-4 in one call"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=1.0,
        )
        return [choice.message.content for choice in response.choices]
    except Exception as e:
        print(f"Error getting GPT-4 responses: {e}")
        return []


def print_validation_summary(results_df: pd.DataFrame, metric: Dict) -> None:
    """Print detailed validation summary with min/max violations"""
    total = len(results_df)

    # Check range validation
    in_range_mask = (
        results_df["value"].notna()
        & (results_df["value"] >= metric["validation"]["min_value"])
        & (results_df["value"] <= metric["validation"]["max_value"])
    )
    valid_count = in_range_mask.sum()

    print(f"\n  Results ({valid_count}/{total} valid responses):")

    if total - valid_count > 0:
        invalid_df = results_df[~in_range_mask].copy()

        # Categorize validation failures
        def get_failure_reason(row):
            if pd.isna(row["value"]):
                return f"No number found: '{row['last_line']}'"
            if row["value"] < metric["validation"]["min_value"]:
                return f"Below minimum {metric['validation']['min_value']}: {row['value']}"
            if row["value"] > metric["validation"]["max_value"]:
                return f"Above maximum {metric['validation']['max_value']}: {row['value']}"
            return f"Other issue: {row['last_line']}"

        print("\n  Invalid responses:")
        for _, row in invalid_df.iterrows():
            print(f"    {get_failure_reason(row)}")

    # Show distribution of valid responses
    if valid_count > 0:
        valid_df = results_df[in_range_mask]
        desc = valid_df["value"].describe()
        print("\n  Distribution of valid responses:")
        print(f"    Mean ± Std: {desc['mean']:.2f} ± {desc['std']:.2f}")
        print(f"    Range: [{desc['min']:.2f}, {desc['max']:.2f}]")
        print(
            f"    Quartiles: {desc['25%']:.2f} | {desc['50%']:.2f} | {desc['75%']:.2f}"
        )


def run_prediction_batch(
    client: OpenAI, metric: Dict, candidate: str, n_runs: int = 50
) -> Tuple[List[Dict], pd.DataFrame]:
    """Run predictions in batch and return both valid results and validation info"""
    history_str = format_history(metric.get("history", {}))
    prompt = metric["prompt_template"].format(
        candidate=candidate, history=history_str
    )

    print(f"\nRunning {n_runs} predictions for {candidate}...")

    # Get all responses at once
    responses = get_gpt4_responses(client, prompt, metric["model"], n=n_runs)
    if not responses:
        return [], pd.DataFrame()

    # Extract numbers and validation info
    results_df = extract_numbers_vectorized(responses)

    # Add metadata
    results_df["timestamp"] = datetime.now()
    results_df["outcome"] = metric["name"]
    results_df["candidate"] = candidate
    results_df["run"] = range(len(responses))

    # Add range validation
    results_df["in_valid_range"] = (
        results_df["value"].notna()
        & (results_df["value"] >= metric["validation"]["min_value"])
        & (results_df["value"] <= metric["validation"]["max_value"])
    )

    # Print validation summary
    print_validation_summary(results_df, metric)

    # Create valid results list
    valid_results = results_df[results_df["in_valid_range"]].to_dict("records")

    return valid_results, results_df
