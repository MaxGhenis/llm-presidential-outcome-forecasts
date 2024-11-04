"""Main validation module for extracting numbers from responses."""

import pandas as pd
from typing import List
from .extractors import extract_number_with_validation


def extract_numbers_vectorized(responses: List[str]) -> pd.DataFrame:
    """Extract numbers from responses and return full validation info"""
    # Convert responses to DataFrame for vectorized operations
    df = pd.DataFrame({"raw_response": responses})

    # Extract last line of each response
    df["last_line"] = (
        df["raw_response"].str.strip().str.split("\n").str[-1].str.strip()
    )

    validation_df = df.apply(extract_number_with_validation, axis=1)
    return pd.concat([df, validation_df], axis=1)
