import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)


class Settings:
    def __init__(self):
        self.API_TOKEN: str = os.getenv("API_TOKEN", "")
        self.MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", "models"))
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.MODEL_VERSION: str = os.getenv("MODEL_VERSION", "0.1.0")

        if not self.API_TOKEN:
            raise ValueError("API_TOKEN must be set in environment or .env file")

        if not self.MODEL_DIR.is_absolute():
            self.MODEL_DIR = Path(__file__).resolve().parent / self.MODEL_DIR


settings = Settings()
