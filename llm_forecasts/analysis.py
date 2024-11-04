import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import statsmodels.api as sm
from pathlib import Path


class Analysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def basic_statistics(self) -> pd.DataFrame:
        """Calculate basic statistics for each outcome and candidate"""
        stats_df = (
            self.df.groupby(["outcome", "candidate"])["value"]
            .agg(
                [
                    "count",
                    "mean",
                    "std",
                    lambda x: x.quantile(0.025),
                    lambda x: x.quantile(0.975),
                ]
            )
            .round(3)
        )
        stats_df.columns = ["n", "mean", "std", "ci_lower", "ci_upper"]
        return stats_df

    def harris_vs_trump_effects(self) -> pd.DataFrame:
        """Calculate effects of Harris vs Trump for each outcome"""
        effects = []

        for outcome in self.df["outcome"].unique():
            outcome_df = self.df[self.df["outcome"] == outcome].copy()

            # Create dummy variable (1 for Harris, 0 for Trump)
            outcome_df["harris"] = (
                outcome_df["candidate"] == "Kamala Harris"
            ).astype(int)

            # Run simple regression
            X = outcome_df["harris"]
            y = outcome_df["value"]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            effects.append(
                {
                    "outcome": outcome,
                    "harris_effect": model.params["harris"],
                    "std_error": model.bse["harris"],
                    "p_value": model.pvalues["harris"],
                    "t_stat": model.tvalues["harris"],
                    "r_squared": model.rsquared,
                }
            )

        return pd.DataFrame(effects)

    def plot_distributions(self, output_dir: Path):
        """Plot distribution of predictions for each outcome"""
        for outcome in self.df["outcome"].unique():
            outcome_df = self.df[self.df["outcome"] == outcome]

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=outcome_df, x="candidate", y="value")
            plt.title(f"Distribution of Predictions: {outcome}")
            plt.savefig(output_dir / f"{outcome}_boxplot.png")
            plt.close()

            # Also create violin plots for better distribution visualization
            plt.figure(figsize=(10, 6))
            sns.violinplot(data=outcome_df, x="candidate", y="value")
            plt.title(f"Distribution of Predictions: {outcome}")
            plt.savefig(output_dir / f"{outcome}_violin.png")
            plt.close()

    def generate_latex_table(self) -> str:
        """Generate LaTeX table with results"""
        stats_df = self.basic_statistics()
        effects_df = self.harris_vs_trump_effects()

        latex_str = r"""\begin{table}[htbp]
\centering
\caption{GPT-4 Predictions by Candidate and Harris vs Trump Effects}
\begin{tabular}{lccccc}
\hline
Outcome & Candidate & Mean & 95\% CI & Effect & p-value \\
\hline
"""

        for outcome in stats_df.index.get_level_values(0).unique():
            effect_row = effects_df[effects_df["outcome"] == outcome].iloc[0]

            # Add Trump row
            trump_row = stats_df.loc[(outcome, "Donald Trump")]
            latex_str += f"{outcome} & Trump & {trump_row['mean']:.2f} & [{trump_row['ci_lower']:.2f}, {trump_row['ci_upper']:.2f}] & -- & -- \\\\\n"

            # Add Harris row
            harris_row = stats_df.loc[(outcome, "Kamala Harris")]
            latex_str += f"& Harris & {harris_row['mean']:.2f} & [{harris_row['ci_lower']:.2f}, {harris_row['ci_upper']:.2f}] & {effect_row['harris_effect']:.2f} & {effect_row['p_value']:.3f} \\\\\n"

        latex_str += r"""\hline
\end{tabular}
\end{table}"""

        return latex_str
