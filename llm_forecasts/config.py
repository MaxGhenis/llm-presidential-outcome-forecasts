import os
from pathlib import Path


class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    OUTPUTS_DIR = BASE_DIR / "outputs"

    # API Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    XAI_API_KEY = os.getenv("XAI_API_KEY")
    BASE_MODEL = "gpt-4o"  # Base model for comparisons
    MODELS = ["grok-beta", "gpt-4o", "gpt-4o-mini"]
    RATE_LIMIT_DELAY = 1  # seconds

    # Analysis Settings
    CANDIDATES = ["Donald Trump", "Kamala Harris"]
    RANDOM_SEED = 42
    RUNS_PER_CONDITION = 100

    @classmethod
    def setup(cls):
        """Create necessary directories"""
        cls.OUTPUTS_DIR.mkdir(exist_ok=True)
