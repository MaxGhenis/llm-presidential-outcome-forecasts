import pandas as pd
from datetime import datetime
from .config import Config
from .analysis import Analysis
import scipy.stats as stats
import numpy as np
import yaml
import time
from openai import OpenAI
from .llm import run_prediction_batch
import argparse


def format_p_value(p: float) -> str:
    """Format p-value for display"""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def format_effect(effect: float, se: float, p: float) -> str:
    """Format effect size with significance stars"""
    stars = (
        "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    return f"{effect:6.2f} (SE: {se:.2f}){stars}"


def main(
    results_file: str = None,
    force_generate: bool = False,
    additional_runs: bool = False,
):
    """
    Run analysis, generating results if they don't exist or combining if additional_runs=True.
    Args:
        results_file: Optional specific results file to use
        force_generate: If True, generate new results even if file exists
        additional_runs: If True, append to existing results
    """
    Config.setup()

    existing_df = None

    # Find most recent results file if not specified
    if results_file is None:
        results_files = list(
            Config.OUTPUTS_DIR.glob("model_comparison_results_*.csv")
        )
        if results_files:
            results_file = max(results_files, key=lambda x: x.stat().st_mtime)
            if not force_generate:
                if additional_runs:
                    # Load but don't return - we'll add more results
                    existing_df = pd.read_csv(results_file)
                    print(
                        f"\nLoaded {len(existing_df)} existing results from {results_file}"
                    )
                else:
                    # Check if we have all models and return if complete
                    df = pd.read_csv(results_file)
                    if set(df["model"].unique()) == set(Config.MODELS):
                        print(
                            f"\nLoaded {len(df)} results from {results_file}"
                        )
                        run_analysis(df)
                        return

    # Store original run count if doing additional runs
    original_runs = Config.RUNS_PER_CONDITION
    if additional_runs:
        Config.RUNS_PER_CONDITION = 100  # Set to get additional 100

    # Generate new results
    print(
        "\nGenerating "
        + ("additional" if additional_runs else "new")
        + " results..."
    )

    # Load outcomes
    with open("outcomes.yaml", "r") as f:
        outcomes = yaml.safe_load(f)

    all_results = []
    all_validations = []

    # Initialize clients
    openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

    # Run predictions for all models
    for model_name in Config.MODELS:
        print(f"\nProcessing with model: {model_name}")

        for metric in outcomes["metrics"]:
            print(f"\nProcessing metric: {metric['name']}")

            for candidate in Config.CANDIDATES:
                print(f"Running predictions for {candidate}...")

                metric_copy = metric.copy()
                metric_copy["model"] = model_name

                results, validation_df = run_prediction_batch(
                    openai_client,
                    metric_copy,
                    candidate,
                    Config.RUNS_PER_CONDITION,
                )

                for result in results:
                    result["model"] = model_name
                validation_df["model"] = model_name

                all_results.extend(results)
                all_validations.append(validation_df)

                time.sleep(Config.RATE_LIMIT_DELAY)

    # Restore original run count if needed
    if additional_runs:
        Config.RUNS_PER_CONDITION = original_runs

    # Convert to DataFrame
    new_df = pd.DataFrame(all_results)
    validation_df = pd.concat(all_validations, ignore_index=True)

    # Combine with existing results if available
    if existing_df is not None:
        df = pd.concat([existing_df, new_df], ignore_index=True)
        print(
            f"\nCombined {len(existing_df)} existing and {len(new_df)} new results"
        )
    else:
        df = new_df

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = (
        Config.OUTPUTS_DIR / f"model_comparison_results_{timestamp}.csv"
    )
    validation_file = (
        Config.OUTPUTS_DIR / f"model_comparison_validation_{timestamp}.csv"
    )

    df.to_csv(results_file, index=False)
    validation_df.to_csv(validation_file, index=False)

    print(f"\nSaved {len(df)} total results to {results_file}")

    # Run analysis on combined results
    run_analysis(df)


def run_analysis(df: pd.DataFrame):
    """Run analysis on existing results DataFrame"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize analysis
    analysis = Analysis(df)

    # Model comparison effects
    effects_df = analysis.model_comparison_effects()
    effects_df.to_csv(Config.OUTPUTS_DIR / f"model_effects_{timestamp}.csv")

    print("\nSUMMARY OF MODEL COMPARISON ANALYSIS")
    print("=" * 80)

    print("\n1. HARRIS VS TRUMP EFFECTS")
    print("-" * 80)
    for _, row in effects_df.iterrows():
        print(f"\n{row['outcome']}:")
        print(
            "  GPT-4o:      "
            + format_effect(row["base_effect"], row["base_se"], row["base_p"])
        )
        print(
            "  GPT-4o-mini: "
            + format_effect(
                row["mini_effect"],
                row["mini_se"],
                stats.norm.sf(abs(row["mini_effect"] / row["mini_se"])) * 2,
            )
        )
        print(
            f"    Diff from GPT-4o: {row['mini_diff']:6.2f} {format_p_value(row['mini_diff_p'])}"
        )
        print(
            "  Grok:        "
            + format_effect(
                row["grok_effect"],
                row["grok_se"],
                stats.norm.sf(abs(row["grok_effect"] / row["grok_se"])) * 2,
            )
        )
        print(
            f"    Diff from GPT-4o: {row['grok_diff']:6.2f} {format_p_value(row['grok_diff_p'])}"
        )

    # Model consistency analysis
    consistency_df = analysis.model_consistency_analysis()
    consistency_df.to_csv(
        Config.OUTPUTS_DIR / f"model_consistency_{timestamp}.csv"
    )

    print("\n2. MODEL CONSISTENCY ANALYSIS")
    print("-" * 80)
    for outcome in consistency_df["outcome"].unique():
        outcome_data = consistency_df[consistency_df["outcome"] == outcome]
        print(f"\n{outcome}:")

        # Model properties for all models
        print("  Model properties:")
        for model in Config.MODELS:
            model_data = outcome_data[outcome_data["model"] == model]
            if not model_data.empty:
                print(f"    {model:12s}")
                print(
                    f"      Effect:     {model_data['effect'].iloc[0]:6.2f} "
                    f"(SE: {model_data['effect_se'].iloc[0]:.2f})"
                )
                print(
                    f"      Var ratio:  {model_data['variance_ratio'].iloc[0]:6.2f}"
                )

    print("\n* p<0.05, ** p<0.01, *** p<0.001")

    # Generate LaTeX table
    latex_table = analysis.generate_latex_table()
    with open(
        Config.OUTPUTS_DIR / f"model_comparison_table_{timestamp}.tex", "w"
    ) as f:
        f.write(latex_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model comparison analysis"
    )
    parser.add_argument(
        "--results_file", type=str, help="Path to results file"
    )
    parser.add_argument(
        "--force_generate",
        action="store_true",
        help="Force generation of new results",
    )
    parser.add_argument(
        "--additional_runs",
        action="store_true",
        help="Generate additional runs and combine with existing",
    )

    args = parser.parse_args()

    main(
        results_file=args.results_file,
        force_generate=args.force_generate,
        additional_runs=args.additional_runs,
    )
