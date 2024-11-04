"""Regular expression patterns for number extraction."""

# Pattern to identify just years
YEAR_PATTERN = r"\b20\d{2}\b"

# Numbers with units (μg/m³ or %)
UNIT_PATTERN = r"(\d+\.?\d*)\s*(?:μg/m³|ug/m3|%)"

# GDP values with commas (must come before post_year pattern)
GDP_PATTERN = r"(?:[\$]?\d{2,3}(?:,\d{3})+(?:\.\d+)?|\b\d{5,6}(?:\.\d+)?)\b"

# Numbers after year mentions (e.g., "2026: 8.9")
# Modified to not match GDP-sized numbers
POST_YEAR_PATTERN = r"20\d{2}[:\s]+(?!(?:\d{2,3}(?:,\d{3})+))(\d+\.?\d*)"

# Numbers in context phrases
CONTEXT_PATTERN = r"(?:concentration|rate|level)[:\s]+(\d+\.?\d*)"

# Any reasonable number pattern
NUMBER_PATTERN = r"(?:^|[^\d])(\d+\.?\d*|\d*\.\d+)(?:[^\d]|$)"
