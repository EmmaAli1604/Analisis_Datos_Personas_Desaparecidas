# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Mi App"
    APP_ICON: str = "🚀"
    VERSION: str  = "1.0.0"
    LAYOUT: str   = "wide"
    DB_URL: str   = ""
    API_KEY: str  = ""

    class Config:
        env_file = ".env"

settings = Settings()