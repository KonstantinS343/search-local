from pydantic import BaseModel


class Query(BaseModel):
    query: str
    limit: int = 1
    
