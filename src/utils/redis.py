from redis import from_url  # type: ignore

from typing import Any



def get_cache(key: str):
    redis = from_url(
        f'redis://redis/'
    )

    result = redis.get(key)
    redis.close()
    return result.decode("utf-8") if result else None


def set_cache(key: str, value: Any):
    redis = from_url(
        f'redis://redis/'
    )

    redis.set(key, str(value))
    redis.close()


def clear_cache(key: str):
    redis = from_url(
        f'redis://redis/'
    )

    redis.delete(key)