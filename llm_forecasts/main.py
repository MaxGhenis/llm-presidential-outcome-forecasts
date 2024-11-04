import pandas as pd
from datetime import datetime
from .config import Config
from .analysis import Analysis
import scipy.stats as stats
import numpy as np


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


def main(results_file: str = None):
    """Run analysis on existing results."""
    Config.setup()

    # Find most recent results file if not specified
    if results_file is None:
        results_files = list(
            Config.OUTPUTS_DIR.glob("model_comparison_results_*.csv")
        )
        if not results_files:
            raise FileNotFoundError(
                "No results files found in outputs directory"
            )
        results_file = max(results_files, key=lambda x: x.stat().st_mtime)

    # Load results
    df = pd.read_csv(results_file)
    print(f"\nLoaded {len(df)} results from {results_file}")

    # Run analysis
    analysis = Analysis(df)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
            + format_effect(
                row["gpt4o_effect"],
                row["gpt4o_se"],
                stats.norm.sf(abs(row["gpt4o_effect"] / row["gpt4o_se"])) * 2,
            )
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
            f"  Difference:   {row['interaction']:6.2f} {format_p_value(row['interaction_p'])}"
        )

    # Model consistency
    consistency_df = analysis.model_consistency_analysis()
    consistency_df.to_csv(
        Config.OUTPUTS_DIR / f"model_consistency_{timestamp}.csv"
    )

    print("\n2. MODEL CONSISTENCY ANALYSIS")
    print("-" * 80)
    for outcome in consistency_df["outcome"].unique():
        outcome_data = consistency_df[consistency_df["outcome"] == outcome]
        corr = outcome_data["correlation_between_models"].iloc[0]
        n = (
            len(df[df["outcome"] == outcome]) // 4
        )  # Divide by 4 for 2 models * 2 candidates
        corr_p = 2 * stats.norm.sf(
            abs(corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2))
        )

        print(f"\n{outcome}:")
        print(f"  Correlation:  {corr:6.3f} {format_p_value(corr_p)}")
        print("  Model properties:")
        for _, row in outcome_data.iterrows():
            print(f"    {row['model']:12s}")
            print(
                f"      Effect:     {row['effect']:6.2f} (SE: {row['effect_se']:.2f})"
            )
            print(f"      Var ratio:  {row['variance_ratio']:6.2f}")

    print("\n* p<0.05, ** p<0.01, *** p<0.001")

    # Generate LaTeX table
    latex_table = analysis.generate_latex_table()
    with open(
        Config.OUTPUTS_DIR / f"model_comparison_table_{timestamp}.tex", "w"
    ) as f:
        f.write(latex_table)


if __name__ == "__main__":
    main()
