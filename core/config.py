from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "sqlite:///data/traffic.db"
    upload_dir: str = "./data/uploads"
    default_tz: str = "Asia/Kolkata"
    debug: bool = False
    refresh_interval: int = 30

    model_config = {"env_prefix": "TRAFFIC_", "env_file": ".env"}

settings = Settings()
