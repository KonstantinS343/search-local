from pydantic import BaseModel


class File(BaseModel):
    filename: str
    content: str
    
