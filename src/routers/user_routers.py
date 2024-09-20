from fastapi import APIRouter, Response
from i_s.i_s import i_s
import json

from utils.redis import get_by_key, set_key, delete_key, get_values

  
user_router = APIRouter(prefix='/query', tags=["User"])
        
@user_router.post('/')
async def user_query(query: str):
    """
    The primary endpoint for user experience, accepts the request and gives the documents.
    """
    
    res = await i_s.handle_user_query(query)
    
    results = []
    
    for i in res:
        path = get_by_key(i)
        snippet = get_by_key(i, 2)
        
        with open(path, 'r') as file:
            start, end = snippet.split(' ')
            content = file.read()[int(start):int(end)]
        
        results.append({'doc': 'http://localhost/static/{path}', 'text': content})
    
    return Response(content=json.dumps(results), status_code=200, media_type='application/json')