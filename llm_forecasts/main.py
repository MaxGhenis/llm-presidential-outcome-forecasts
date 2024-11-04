import pandas as pd
from datetime import datetime
from .config import Config
from .analysis import Analysis
import scipy.stats as stats
import numpy as np
import yaml
import time
from openai import OpenAI
from .gpt import run_prediction_batch


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


def main(results_file: str = None, force_generate: bool = False):
    """
    Run analysis, generating results if they don't exist.
    Args:
        results_file: Optional specific results file to use
        force_generate: If True, generate new results even if file exists
    """
    Config.setup()

    # Try to find existing results unless force_generate is True
    if not force_generate and results_file is None:
        results_files = list(
            Config.OUTPUTS_DIR.glob("model_comparison_results_*.csv")
        )
        if results_files:
            results_file = max(results_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(results_file)
            # Check if we have all models
            if set(df["model"].unique()) == set(Config.MODELS):
                print(f"\nLoaded {len(df)} results from {results_file}")
                run_analysis(df)
                return

    # If we get here, we need to generate results
    print("\nGenerating new results...")

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

                # Pass model in metric dictionary
                metric_copy = metric.copy()
                metric_copy["model"] = model_name

                # Get results
                results, validation_df = run_prediction_batch(
                    openai_client,
                    metric_copy,
                    candidate,
                    Config.RUNS_PER_CONDITION,
                )

                # Add model information
                for result in results:
                    result["model"] = model_name
                validation_df["model"] = model_name

                all_results.extend(results)
                all_validations.append(validation_df)

                # Add small delay between runs
                time.sleep(Config.RATE_LIMIT_DELAY)

    # Convert to DataFrame and save
    df = pd.DataFrame(all_results)
    validation_df = pd.concat(all_validations, ignore_index=True)

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = (
        Config.OUTPUTS_DIR / f"model_comparison_results_{timestamp}.csv"
    )
    validation_file = (
        Config.OUTPUTS_DIR / f"model_comparison_validation_{timestamp}.csv"
    )

    df.to_csv(results_file, index=False)
    validation_df.to_csv(validation_file, index=False)

    print(f"\nSaved results to {results_file}")

    # Run analysis on new results
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
    main()
