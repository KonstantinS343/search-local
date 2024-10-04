from redis.asyncio import from_url  # type: ignore

from typing import Any



async def get_by_key(key: str, db: int = 1):
    redis = await from_url(
        f'redis://localhost/{db}/'
    )

    result = (await redis.get(key)).decode()
    await redis.close()
    return result


async def get_values(path: str, db: int = 1):
    redis = await from_url(
        f'redis://localhost/{db}/'
    )
    
    my_keys = []
    
    for key in await redis.keys():
        if (await redis.get(key)).decode() == path:
            my_keys.append(key.decode())

    await redis.close()
    return my_keys


async def set_key(key: str, value: Any, db: int = 1):
    redis = await from_url(
        f'redis://localhost/{db}/'
    )

    await redis.set(key, str(value))
    await redis.close()


async def delete_key(key: str, db: int = 1):
    redis = await from_url(
        f'redis://localhost/{db}/'
    )

    await redis.delete(key)
    await redis.close()