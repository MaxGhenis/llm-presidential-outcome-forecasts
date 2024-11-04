from openai import OpenAI
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime
import requests
from .config import Config
from .utils import format_history
from .validation import extract_numbers_vectorized


def get_model_response(
    client: OpenAI, prompt: str, model: str, n: int = 50
) -> List[str]:
    """Get multiple responses from model API"""
    try:
        if model.startswith("gpt"):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                n=n,
                temperature=1.0,
            )
            return [choice.message.content for choice in response.choices]
        elif model == "grok-beta":
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {Config.XAI_API_KEY}",
                },
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "model": "grok-beta",
                    "temperature": 1.0,
                    "stream": False,
                    "n": n,
                },
            )
            return [
                choice["message"]["content"]
                for choice in response.json()["choices"]
            ]
    except Exception as e:
        print(f"Error getting model responses: {e}")
        return []


def print_validation_summary(results_df: pd.DataFrame, metric: Dict) -> None:
    """Print detailed validation summary with min/max violations"""
    total = len(results_df)

    # Consider valid statuses
    valid_status_mask = results_df["validation_status"].isin(
        ["valid_with_unit", "valid_decimal"]
    )
    value_in_range_mask = (
        results_df["value"].notna()
        & (results_df["value"] >= metric["validation"]["min_value"])
        & (results_df["value"] <= metric["validation"]["max_value"])
    )
    valid_mask = valid_status_mask & value_in_range_mask
    valid_count = valid_mask.sum()

    print(f"\n  Results ({valid_count}/{total} valid responses):")

    if valid_count < total:
        # Create a copy for manipulation
        invalid_df = results_df.loc[~valid_mask].copy()

        # Count by status for non-numeric and out-of-range separately
        status_counts = invalid_df["validation_status"].value_counts()

        # Identify out-of-range values that had valid status
        out_of_range_mask = ~value_in_range_mask & valid_status_mask
        n_out_of_range = out_of_range_mask.sum()

        if len(status_counts) > 0 or n_out_of_range > 0:
            print("\n  Invalid responses by reason:")
            for status, count in status_counts.items():
                print(f"    {status}: {count}")
            if n_out_of_range > 0:
                print(f"    out_of_range: {n_out_of_range}")

            # Show examples
            print("\n  Example invalid responses:")

            # Show non-numeric examples
            for status in status_counts.index:
                examples = invalid_df[
                    invalid_df["validation_status"] == status
                ]
                if len(examples) > 0:
                    print(f"\n    {status}:")
                    for _, row in examples.head(2).iterrows():
                        print(f"      {row['last_line']}")

            # Show out-of-range examples
            if n_out_of_range > 0:
                print("\n    out_of_range:")
                out_of_range = results_df.loc[out_of_range_mask]
                for _, row in out_of_range.head(2).iterrows():
                    print(f"      {row['value']:.1f} (in: {row['last_line']})")

    # Show distribution of valid responses
    if valid_count > 0:
        valid_df = results_df.loc[valid_mask]
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
    responses = get_model_response(client, prompt, metric["model"], n=n_runs)
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
