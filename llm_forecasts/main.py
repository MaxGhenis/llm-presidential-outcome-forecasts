from openai import OpenAI
import yaml
import pandas as pd
from datetime import datetime
from .config import Config
from .analysis import Analysis
from .gpt import run_prediction_batch


def main():
    # Setup
    Config.setup()

    # Initialize OpenAI client
    client = OpenAI(api_key=Config.API_KEY)

    # Load outcomes
    with open("outcomes.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Run predictions
    all_results = []
    all_validations = []

    for metric in config["metrics"]:
        print(f"\nProcessing metric: {metric['name']}")
        for candidate in Config.CANDIDATES:
            print(f"  Running predictions for {candidate}...")
            results, validation_df = run_prediction_batch(
                client,
                metric,
                candidate,
                config["analysis_parameters"]["runs_per_condition"],
            )
            all_results.extend(results)
            all_validations.append(validation_df)

            # Print validation summary
            print(f"  Validation summary:")
            print(f"    Valid predictions: {len(results)}")
            print("    Invalid predictions by reason:")
            print(
                validation_df[~validation_df["in_valid_range"]][
                    "validation_status"
                ]
                .value_counts()
                .to_string()
            )

    # Save all validation info
    validation_df = pd.concat(all_validations, ignore_index=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_df.to_csv(
        Config.OUTPUTS_DIR / f"validation_results_{timestamp}.csv", index=False
    )

    # Analysis
    df = pd.DataFrame(all_results)
    df.to_csv(
        Config.OUTPUTS_DIR / f"prediction_results_{timestamp}.csv", index=False
    )

    # Run analysis
    analysis = Analysis(df)
    stats_df = analysis.basic_statistics()
    effects_df = analysis.harris_vs_trump_effects()

    # Output results
    print("\nBasic Statistics:")
    print(stats_df)
    print("\nHarris vs Trump Effects:")
    print(effects_df)

    # Generate plots and LaTeX
    analysis.plot_distributions(Config.OUTPUTS_DIR)
    latex_table = analysis.generate_latex_table()
    with open(Config.OUTPUTS_DIR / "results_table.tex", "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    main()
