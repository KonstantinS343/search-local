from fastapi import APIRouter, Response

import json

  
user_router = APIRouter(prefix='/query', tags=["User"])
        
@user_router.post('/')
async def user_query(query: str):
    """
    The primary endpoint for user experience, accepts the request and gives the documents.
    """
    return Response(content=json.dumps([{'doc': 'http://localhost/static/test.txt', 'text': 'Test'},
                                        {'doc': 'http://localhost/static/test2.txtt', 'text': 'Teeeest'}, 
                                        {'doc': 'http://localhost/static/test3.txt', 'text': 'Testttt'}]), status_code=200, media_type='application/json')