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

    def model_comparison_effects(self) -> pd.DataFrame:
        """Calculate effects with model interactions"""
        effects = []

        for outcome in self.df["outcome"].unique():
            outcome_df = self.df[self.df["outcome"] == outcome].copy()

            # Create dummy variables
            outcome_df["harris"] = (
                outcome_df["candidate"] == "Kamala Harris"
            ).astype(int)
            outcome_df["gpt4o"] = (outcome_df["model"] == "gpt-4o").astype(int)
            outcome_df["interaction"] = (
                outcome_df["harris"] * outcome_df["gpt4o"]
            )

            # Run regression with interaction
            X = sm.add_constant(outcome_df[["harris", "gpt4o", "interaction"]])
            y = outcome_df["value"]
            model = sm.OLS(y, X).fit()

            # Calculate effects for each model
            mini_effect = model.params["harris"]
            gpt4o_effect = model.params["harris"] + model.params["interaction"]

            # Calculate standard errors
            mini_se = model.bse["harris"]
            gpt4o_se = np.sqrt(
                model.bse["harris"] ** 2
                + model.bse["interaction"] ** 2
                + 2 * model.cov_params().loc["harris", "interaction"]
            )

            # Add to results
            effects.append(
                {
                    "outcome": outcome,
                    "mini_effect": mini_effect,
                    "mini_se": mini_se,
                    "mini_p": model.pvalues["harris"],
                    "gpt4o_effect": gpt4o_effect,
                    "gpt4o_se": gpt4o_se,
                    "interaction": model.params["interaction"],
                    "interaction_p": model.pvalues["interaction"],
                    "r_squared": model.rsquared,
                    "n_obs": len(outcome_df),
                }
            )

        return pd.DataFrame(effects)

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

    def generate_latex_table(self) -> str:
        """Generate LaTeX table with model comparison results"""
        effects_df = self.model_comparison_effects()

        latex_str = r"""\begin{table}[htbp]
\centering
\caption{Model Comparison of Harris vs Trump Effects}
\begin{tabular}{lcccccc}
\hline
& \multicolumn{2}{c}{GPT-4o} & \multicolumn{2}{c}{GPT-4o-mini} & \multicolumn{2}{c}{Interaction} \\
Outcome & Effect & SE & Effect & SE & Diff & p-value \\
\hline
"""

        for _, row in effects_df.iterrows():
            latex_str += (
                f"{row['outcome']} & "
                f"{row['gpt4o_effect']:.2f} & ({row['gpt4o_se']:.2f}) & "
                f"{row['mini_effect']:.2f} & ({row['mini_se']:.2f}) & "
                f"{row['interaction']:.2f} & "
                f"{'<0.001' if row['interaction_p'] < 0.001 else f'{row['interaction_p']:.3f}'} \\\\\n"
            )

        latex_str += r"""\hline
\multicolumn{7}{p{0.8\textwidth}}{\small Note: Effects show difference in predictions between Harris and Trump. 
Interaction shows difference between GPT-4o and GPT-4o-mini effects. Standard errors in parentheses.} \\
\end{tabular}
\label{tab:model_comparison}
\end{table}"""

        return latex_str
