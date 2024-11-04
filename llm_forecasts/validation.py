import pandas as pd
import re
from typing import List


def extract_numbers_vectorized(responses: List[str]) -> pd.DataFrame:
    """Extract numbers from responses and return full validation info"""
    # Convert responses to DataFrame for vectorized operations
    df = pd.DataFrame({"raw_response": responses})

    # Extract last line of each response
    df["last_line"] = (
        df["raw_response"].str.strip().str.split("\n").str[-1].str.strip()
    )

    def extract_number_with_validation(row):
        last_line = row["last_line"]

        # Skip if line is empty
        if not last_line:
            return pd.Series(
                {
                    "value": None,
                    "validation_status": "empty_line",
                    "extracted_text": "",
                }
            )

        # First check if it's just a year
        year_match = re.match(r"^\s*20\d{2}\s*$", last_line)
        if year_match:
            return pd.Series(
                {
                    "value": float(year_match.group(0)),
                    "validation_status": "appears_to_be_year",
                    "extracted_text": last_line,
                }
            )

        # Try to extract PM2.5 specific format
        if any(term in last_line for term in ["PM2.5", "μg/m³", "ug/m3"]):
            matches = re.findall(r"(\d*\.?\d+)\s*(?:μg/m³|ug/m3)?", last_line)
            if not matches:
                return pd.Series(
                    {
                        "value": None,
                        "validation_status": "no_number_found",
                        "extracted_text": last_line,
                    }
                )
            try:
                value = float(matches[0])
                return pd.Series(
                    {
                        "value": value,
                        "validation_status": "valid_pm25",
                        "extracted_text": last_line,
                    }
                )
            except ValueError:
                return pd.Series(
                    {
                        "value": None,
                        "validation_status": "invalid_number_format",
                        "extracted_text": last_line,
                    }
                )

        # For GDP and poverty rate, try to extract any number
        matches = re.findall(r"[-+]?\d*\.?\d+", last_line)
        if not matches:
            return pd.Series(
                {
                    "value": None,
                    "validation_status": "no_number_found",
                    "extracted_text": last_line,
                }
            )

        try:
            # Look for the first number that's not a year
            for match in matches:
                value = float(match)
                if not (1900 <= value <= 2100):
                    return pd.Series(
                        {
                            "value": value,
                            "validation_status": "valid",
                            "extracted_text": last_line,
                        }
                    )
            # If we only found years, return the first number but mark it
            return pd.Series(
                {
                    "value": float(matches[0]),
                    "validation_status": "appears_to_be_year",
                    "extracted_text": last_line,
                }
            )
        except ValueError:
            return pd.Series(
                {
                    "value": None,
                    "validation_status": "invalid_number_format",
                    "extracted_text": last_line,
                }
            )

    validation_df = df.apply(extract_number_with_validation, axis=1)
    return pd.concat([df, validation_df], axis=1)
