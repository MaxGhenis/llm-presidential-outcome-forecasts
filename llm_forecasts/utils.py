from typing import Dict


def format_history(history: Dict) -> str:
    """Format historical data into a string for the prompt"""
    if not history:
        return ""
    return "\n".join(f"{year}: {value}" for year, value in history.items())
