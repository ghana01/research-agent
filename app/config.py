
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_model: str = "gpt-4o"
    openai_timeout: float = Field(default=10, gt=0)
    openai_max_retries: int = Field(default=2, ge=0)
    embedding_model: str = "text-embedding-3-small"
    vector_db_path: str = "./data/chroma"


settings = Settings()
