from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///data/traffic.db"
    upload_dir: str = "./data/uploads"
    archive_dir: str = "./data/archive"
    default_tz: str = "Asia/Kolkata"
    log_level: str = "INFO"
    debug: bool = False

    model_config = {"env_prefix": "TRAFFIC_", "env_file": ".env"}


settings = Settings()
