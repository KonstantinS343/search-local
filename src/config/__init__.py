from pydantic_settings import BaseSettings

from .fastapi_config import FastApiConfig, fastapi_settings
from .redis_config import RedisConfig, redis_settings

class Settings(BaseSettings):
    fastapi_settings: FastApiConfig = fastapi_settings
    redis_settings: RedisConfig = redis_settings


settings = Settings()