from fastapi import APIRouter, Response
import json

from collections.abc import Sequence


echo_router = APIRouter(prefix='/echo', tags=["Echo"])
config_router = APIRouter(prefix='/config', tags=["Config"])


@echo_router.post('/')
async def post_echo(message: str) -> Response:
    """
    It'll just return what you send. Just to make sure it's working.
    """
    return Response(content=message)
    
@config_router.post('/set/')
async def post_config(config: Sequence[str]) -> Response:
    """
    This endpoint is used to send a list of files to be tracked.
    """
    with open('config', 'a') as file:
        for item in config:
            file.writelines([item, '\n'])
    
    return Response(content='OK', status_code=201, ia_type='text/plain')

@config_router.get('/')
async def get_config() -> Response:
    """
    You can use this endpoint to get a list of files to be tracked. Required on machines on the local network
    """
    with open('config', 'r') as file:
        files = file.readlines()
    
    return Response(content=json.dumps({'paths': files}), status_code=200, media_type="application/json")
