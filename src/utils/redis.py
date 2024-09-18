from redis import from_url  # type: ignore

from typing import Any

from config import settings


def get_cache(key: str):
    redis = from_url(
        f'redis://{settings.redis_settings.host}/{settings.redis_settings.index_db}'
    )

    result = redis.get(key)
    redis.close()
    return result.decode("utf-8") if result else None


def set_cache(key: str, value: Any):
    redis = from_url(
        f'redis://{settings.redis_settings.host}/{settings.redis_settings.index_db}'
    )

    redis.set(key, str(value))
    redis.close()


def clear_cache(key: str):
    redis = from_url(
        f'redis://{settings.redis_settings.host}/{settings.redis_settings.index_db}'
    )

    redis.delete(key)