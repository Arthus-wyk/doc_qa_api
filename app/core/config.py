from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "qwen2.5:7b"
    chroma_persist_dir: str = "data/chroma"
    chroma_collection_name: str = "documents"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()