from fastapi import APIRouter
import os
import glob

from utils.redis import get_cache, set_cache, clear_cache
from model.file_sync import File

  
sync_router = APIRouter(prefix='/sync', tags=["Sync"])
        
@sync_router.post('/')
async def sync(file: File):
    """
    This endpoint is needed to synchronize files on different machines.
    It accepts the name of the file and its contents.
    """
    file_path = os.path.join('static', file.filename.replace('/', '-')[1:])
    with open(file_path, "w") as buffer:
        buffer.write(file.content)
    return file

@sync_router.delete('/delete/')
async def sync_delete(file: str):
    """
    This endpoint is needed to delete files on different machines.
    It accepts the name of the file name.
    """
    file_path = os.path.join('static', file.replace('/', '-')[1:])
    for file_path in glob.glob(file_path+'*'):
        os.remove(file_path)
    return file

# @sync_router.post('/')
# async def sync(file: File):
#     """
#     This endpoint is needed to synchronize files on different machines.
#     It accepts the name of the file and its contents.
#     """
#     file_path = 'static' + file.filename
    
#     try:
#         os.makedirs(file_path, exist_ok=True)
#     except FileExistsError:
#         pass
#     print
#     with open(file_path, "w") as buffer:
#         buffer.write(file.content)
#     return file