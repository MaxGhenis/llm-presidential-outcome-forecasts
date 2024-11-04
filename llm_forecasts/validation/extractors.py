"""Number extraction functions for different types of responses."""

import re
import pandas as pd
from typing import Optional, Tuple
from .patterns import (
    UNIT_PATTERN,
    POST_YEAR_PATTERN,
    CONTEXT_PATTERN,
    GDP_PATTERN,
    NUMBER_PATTERN,
)


def extract_with_unit(clean_line: str) -> Optional[Tuple[float, str]]:
    """Extract numbers with units (μg/m³ or %)."""
    unit_match = re.search(UNIT_PATTERN, clean_line)
    if unit_match:
        try:
            value = float(unit_match.group(1))
            return value, "valid_with_unit"
        except ValueError:
            pass
    return None


def extract_post_year(clean_line: str) -> Optional[Tuple[float, str]]:
    """Extract numbers that appear after year mentions."""
    post_year_match = re.search(POST_YEAR_PATTERN, clean_line)
    if post_year_match:
        try:
            value = float(post_year_match.group(1))
            return value, "valid_decimal"
        except ValueError:
            pass
    return None


def extract_from_context(clean_line: str) -> Optional[Tuple[float, str]]:
    """Extract numbers from context phrases."""
    context_match = re.search(CONTEXT_PATTERN, clean_line, re.IGNORECASE)
    if context_match:
        try:
            value = float(context_match.group(1))
            return value, "valid_decimal"
        except ValueError:
            pass
    return None


def extract_gdp(clean_line: str) -> Optional[Tuple[float, str]]:
    """Extract GDP values with commas."""
    # First remove any pure years so they don't interfere
    clean_line = re.sub(YEAR_PATTERN, "", clean_line)

    gdp_match = re.search(GDP_PATTERN, clean_line)
    if gdp_match:
        try:
            value = float(gdp_match.group(0).replace(",", ""))
            # Validate it's in a reasonable GDP range
            if value >= 50000:  # Only accept GDP-sized numbers
                return value, "valid_decimal"
        except ValueError:
            pass
    return None


def extract_any_number(clean_line: str) -> Optional[Tuple[float, str]]:
    """Extract any reasonable number from the text."""
    clean_line = re.sub(r"(\d),(\d)", r"\1\2", clean_line)
    number_matches = re.finditer(NUMBER_PATTERN, clean_line)

    values = []
    for match in number_matches:
        try:
            value = float(match.group(1))
            if not (1900 <= value <= 2100):  # Not a year
                values.append(value)
        except ValueError:
            continue

    if values:
        if any(x > 1000 for x in values):  # GDP values
            value = max(values)  # Take largest for GDP
        else:
            value = min(
                values,
                key=lambda x: abs(x - 10) if x < 100 else float("inf"),
            )
        return value, "valid_decimal"
    return None


def extract_number_with_validation(row) -> pd.Series:
    """Extract and validate numbers from a response row."""
    last_line = row["last_line"]

    # Skip if line is empty or just contains markdown
    if not last_line or last_line.strip("`* -") == "":
        return pd.Series(
            {
                "value": None,
                "validation_status": "empty_line",
                "extracted_text": "",
            }
        )

    # Remove markdown and other formatting but keep numbers
    clean_line = re.sub(r'\*\*|_|\.\.\.|"|`', "", last_line)

    # Try extractors in order of specificity
    for extractor in [
        extract_with_unit,
        extract_post_year,
        extract_from_context,
        extract_gdp,
        extract_any_number,
    ]:
        result = extractor(clean_line)
        if result:
            value, status = result
            return pd.Series(
                {
                    "value": value,
                    "validation_status": status,
                    "extracted_text": last_line,
                }
            )

    return pd.Series(
        {
            "value": None,
            "validation_status": "no_number_found",
            "extracted_text": last_line,
        }
    )
