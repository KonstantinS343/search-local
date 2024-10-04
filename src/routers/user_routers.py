from fastapi import APIRouter, Response
from i_s.i_s import i_s
from i_s.metrics import calculate_metrics
import json

from utils.redis import get_by_key
from model.query import Query

  
user_router = APIRouter(prefix='/query', tags=["User"])
        
@user_router.post('/')
async def user_query(query: Query):
    """
    The primary endpoint for user experience, accepts the request and gives the documents.
    """
    
    with open('ip', 'r') as ip_file:
        host_ip = ip_file.read().strip()
    
    res = await i_s.handle_user_query(query.query, query.limit)
    
    results = []
    
    for i in res:
        path = await get_by_key(i)
        snippet = await get_by_key(i, 2)
        
        with open('static/' + path, 'r') as file:
            start, end = snippet.split(' ')
            content = file.read()[int(start):int(end)]
        
        results.append({'doc': f'{host_ip}/static/{path}', 'text': content.replace('\n', ' ')})
        
    recall, precision, accuracy, error, f_measure = await calculate_metrics(results, query.query)
    
    return Response(content=json.dumps({'results': results,
                                        'recall': recall, 
                                        'precision': precision, 
                                        'accuracy': accuracy, 
                                        'error': error, 
                                        'f_measure': f_measure}), status_code=200, media_type='application/json')