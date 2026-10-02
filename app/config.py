import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    APP_NAME: str = "Indian Cattle & Buffalo Recognition"
    UPLOAD_DIR: str = "uploads"
    MODEL_PATH: str = "saved_models/cattle_classifier.pth"
    
    # AI API Settings
    AI_API_KEY: str = os.getenv("AI_API_KEY", "")
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "gemini")  # gemini, openai, etc.
    AI_MODEL_NAME: str = os.getenv("AI_MODEL_NAME", "gemini-1.5-flash")
    
    class Config:
        env_file = ".env"

settings = Settings()
