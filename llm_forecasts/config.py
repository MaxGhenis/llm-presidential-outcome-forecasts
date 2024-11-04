import os
from pathlib import Path


class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    OUTPUTS_DIR = BASE_DIR / "outputs"

    # API Settings
    API_KEY = os.getenv(
        "OPENAI_API_KEY"
    )  # Make sure to set this environment variable
    MODELS = ["gpt-4o-mini", "gpt-4o"]
    RATE_LIMIT_DELAY = 1  # seconds

    # Analysis Settings
    CANDIDATES = ["Donald Trump", "Kamala Harris"]
    RANDOM_SEED = 42

    @classmethod
    def setup(cls):
        """Create necessary directories"""
        cls.OUTPUTS_DIR.mkdir(exist_ok=True)
