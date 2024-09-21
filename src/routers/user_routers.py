from fastapi import APIRouter, Response
from i_s.i_s import i_s
import json

from utils.redis import get_by_key
from model.query import Query

  
user_router = APIRouter(prefix='/query', tags=["User"])
        
@user_router.post('/')
async def user_query(query: Query):
    """
    The primary endpoint for user experience, accepts the request and gives the documents.
    """
    
    res = await i_s.handle_user_query(query.query, query.limit)
    
    results = []
    print(res)
    
    for i in res:
        path = await get_by_key(i)
        snippet = await get_by_key(i, 2)
        
        with open('static/' + path, 'r') as file:
            start, end = snippet.split(' ')
            content = file.read()[int(start):int(end)]
        
        results.append({'doc': f'http://localhost/static/{path}', 'text': content})
    
    return Response(content=json.dumps(results), status_code=200, media_type='application/json')