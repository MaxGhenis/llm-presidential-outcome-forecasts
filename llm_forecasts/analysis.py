import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import statsmodels.api as sm
from pathlib import Path
import numpy as np
import scipy.stats as stats


class Analysis:
    def __init__(self, df):
        """Initialize with results dataframe."""
        self.df = df

    def basic_statistics(self) -> pd.DataFrame:
        """Calculate basic statistics for each outcome, model, and candidate"""
        stats_df = (
            self.df.groupby(["outcome", "model", "candidate"])["value"]
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

    # In Analysis class
    def model_comparison_effects(self) -> pd.DataFrame:
        """Calculate effects with base model comparisons"""
        effects = []

        for outcome in self.df["outcome"].unique():
            outcome_df = self.df[self.df["outcome"] == outcome].copy()

            # Get actual models present in data
            present_models = outcome_df["model"].unique()

            # Create dummy variables
            outcome_df["harris"] = (
                outcome_df["candidate"] == "Kamala Harris"
            ).astype(int)

            # Only create dummies for models that exist
            outcome_df["mini"] = (
                (outcome_df["model"] == "gpt-4o-mini").astype(int)
                if "gpt-4o-mini" in present_models
                else 0
            )
            outcome_df["grok"] = (
                (outcome_df["model"] == "grok-beta").astype(int)
                if "grok-beta" in present_models
                else 0
            )

            # Create interaction terms for present models
            outcome_df["harris_mini"] = (
                outcome_df["harris"] * outcome_df["mini"]
            )
            outcome_df["harris_grok"] = (
                outcome_df["harris"] * outcome_df["grok"]
            )

            # Run regression
            X = sm.add_constant(
                outcome_df[
                    ["harris", "mini", "grok", "harris_mini", "harris_grok"]
                ]
            )
            y = outcome_df["value"]
            model = sm.OLS(y, X).fit()

            # Calculate effects for each present model
            result = {
                "outcome": outcome,
                "base_effect": model.params["harris"],
                "base_se": model.bse["harris"],
                "base_p": model.pvalues["harris"],
            }

            # Add mini effects if present
            if "gpt-4o-mini" in present_models:
                mini_effect = (
                    model.params["harris"] + model.params["harris_mini"]
                )
                mini_se = np.sqrt(
                    model.bse["harris"] ** 2
                    + model.bse["harris_mini"] ** 2
                    + 2 * model.cov_params().loc["harris", "harris_mini"]
                )
                result.update(
                    {
                        "mini_effect": mini_effect,
                        "mini_se": mini_se,
                        "mini_diff": model.params["harris_mini"],
                        "mini_diff_p": model.pvalues["harris_mini"],
                    }
                )

            # Add grok effects if present
            if "grok-beta" in present_models:
                grok_effect = (
                    model.params["harris"] + model.params["harris_grok"]
                )
                grok_se = np.sqrt(
                    model.bse["harris"] ** 2
                    + model.bse["harris_grok"] ** 2
                    + 2 * model.cov_params().loc["harris", "harris_grok"]
                )
                result.update(
                    {
                        "grok_effect": grok_effect,
                        "grok_se": grok_se,
                        "grok_diff": model.params["harris_grok"],
                        "grok_diff_p": model.pvalues["harris_grok"],
                    }
                )

            result.update(
                {"r_squared": model.rsquared, "n_obs": len(outcome_df)}
            )

            effects.append(result)

        return pd.DataFrame(effects)

    def generate_latex_table(self) -> str:
        """Generate LaTeX table with model comparison results"""
        effects_df = self.model_comparison_effects()

        latex_str = r"""\begin{table}[htbp]
    \centering
    \caption{Harris vs Trump Effects by Model}
    \begin{tabular}{lccccc}
    \hline
    & \multicolumn{1}{c}{GPT-4o} & \multicolumn{2}{c}{GPT-4o-mini} & \multicolumn{2}{c}{Grok} \\
    Outcome & Effect & Effect & Diff & Effect & Diff \\
    \hline
    """

        for _, row in effects_df.iterrows():
            latex_str += (
                f"{row['outcome']} & "
                f"{row['base_effect']:.2f} ({row['base_se']:.2f}) & "
                f"{row['mini_effect']:.2f} ({row['mini_se']:.2f}) & "
                f"{row['mini_diff']:.2f}{'***' if row['mini_diff_p'] < 0.001 else '**' if row['mini_diff_p'] < 0.01 else '*' if row['mini_diff_p'] < 0.05 else ''} & "
                f"{row['grok_effect']:.2f} ({row['grok_se']:.2f}) & "
                f"{row['grok_diff']:.2f}{'***' if row['grok_diff_p'] < 0.001 else '**' if row['grok_diff_p'] < 0.01 else '*' if row['grok_diff_p'] < 0.05 else ''} \\\\\n"
            )

        latex_str += r"""\hline
    \multicolumn{6}{p{0.95\textwidth}}{\small Note: Effects show difference in predictions between Harris and Trump. 
    Standard errors in parentheses. Diff columns show difference from GPT-4o effect. * p<0.05, ** p<0.01, *** p<0.001} \\
    \end{tabular}
    \label{tab:model_comparison}
    \end{table}"""

        return latex_str

    def conduct_robustness_checks(self) -> pd.DataFrame:
        """Conduct additional robustness checks"""
        checks = []

        for outcome in self.df["outcome"].unique():
            outcome_df = self.df[self.df["outcome"] == outcome]

            # Test for normality and variance homogeneity by model and candidate
            for model in outcome_df["model"].unique():
                for candidate in outcome_df["candidate"].unique():
                    mask = (outcome_df["model"] == model) & (
                        outcome_df["candidate"] == candidate
                    )
                    values = outcome_df[mask]["value"]

                    _, shapiro_p = stats.shapiro(values)
                    _, levene_p = stats.levene(
                        values, outcome_df[~mask]["value"]
                    )

                    checks.append(
                        {
                            "outcome": outcome,
                            "model": model,
                            "candidate": candidate,
                            "shapiro_p": shapiro_p,
                            "levene_p": levene_p,
                            "mean": values.mean(),
                            "std": values.std(),
                            "n": len(values),
                        }
                    )

        return pd.DataFrame(checks)

    def model_consistency_analysis(self) -> pd.DataFrame:
        """Analyze consistency between models"""
        consistency = []

        for outcome in self.df["outcome"].unique():
            outcome_df = self.df[self.df["outcome"] == outcome]

            # Calculate effects for each model
            models = outcome_df["model"].unique()
            model_effects = {}

            for model in models:
                model_data = outcome_df[outcome_df["model"] == model]
                harris_data = model_data[
                    model_data["candidate"] == "Kamala Harris"
                ]["value"]
                trump_data = model_data[
                    model_data["candidate"] == "Donald Trump"
                ]["value"]

                effect = harris_data.mean() - trump_data.mean()
                effect_se = np.sqrt(
                    harris_data.var() / len(harris_data)
                    + trump_data.var() / len(trump_data)
                )
                model_effects[model] = {
                    "effect": effect,
                    "effect_se": effect_se,
                }

            # Calculate correlation by reshaping data
            pivot_df = pd.pivot_table(
                outcome_df,
                values="value",
                index=["run", "candidate"],
                columns=["model"],
            ).reset_index()

            correlation = pivot_df[models[0]].corr(pivot_df[models[1]])

            # Add results
            for model in models:
                consistency.append(
                    {
                        "outcome": outcome,
                        "model": model,
                        "effect": model_effects[model]["effect"],
                        "effect_se": model_effects[model]["effect_se"],
                        "correlation_between_models": correlation,
                        "variance_ratio": (
                            outcome_df[outcome_df["model"] == model][
                                "value"
                            ].var()
                            / outcome_df[outcome_df["model"] != model][
                                "value"
                            ].var()
                        ),
                    }
                )

        return pd.DataFrame(consistency)

    def plot_distributions(self, output_dir: Path):
        """Plot distribution of predictions by model and candidate"""
        for outcome in self.df["outcome"].unique():
            outcome_df = self.df[self.df["outcome"] == outcome]

            plt.figure(figsize=(12, 6))
            sns.violinplot(
                data=outcome_df,
                x="candidate",
                y="value",
                hue="model",
                split=True,
            )
            plt.title(f"Distribution of Predictions: {outcome}")
            plt.savefig(output_dir / f"{outcome}_violin_by_model.png")
            plt.close()

            # Create boxplot with individual points
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=outcome_df,
                x="candidate",
                y="value",
                hue="model",
                showfliers=False,
            )
            sns.stripplot(
                data=outcome_df,
                x="candidate",
                y="value",
                hue="model",
                dodge=True,
                alpha=0.3,
                size=4,
            )
            plt.title(f"Distribution of Predictions: {outcome}")
            plt.savefig(output_dir / f"{outcome}_boxplot_with_points.png")
            plt.close()
