from fastapi import APIRouter
import os
import glob

from utils.redis import get_by_key, set_key, delete_key, get_values
from model.file_sync import File
from i_s.i_s import i_s

  
sync_router = APIRouter(prefix='/sync', tags=["Sync"])
        
@sync_router.post('/')
async def sync(file: File):
    """
    This endpoint is needed to synchronize files on different machines.
    It accepts the name of the file and its contents.
    """
    
    if file.content:
    
        old_indexes = await get_values(file.filename.replace('/', '-')[1:])
        
        if old_indexes:
            indexes, indexes_to_delete, snippets = await i_s.reindex_existing_document(file.filename.replace('/', '-')[1:], file.content, old_indexes)
            
            for i in indexes_to_delete:
                await delete_key(i)
            
            for i in indexes_to_delete:
                await delete_key(i, 2)
                
            for id, snippet in zip(indexes, snippets):
                await set_key(id, snippet.snippet_start + ' ' + snippet.snippet_end, 2)
        else:
            indexes, snippets = await i_s.index_new_document(file.filename.replace('/', '-')[1:], file.content)
        
        for i in indexes:
            await set_key(i, file.filename.replace('/', '-')[1:])
            
        for id, snippet in zip(indexes, snippets):
            await set_key(id, snippet.snippet_start + ' ' + snippet.snippet_end, 2)

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

    old_indexes = await get_values(file.replace('/', '-')[1:])
    
    for i in old_indexes:
        await delete_key(i)
    
    for i in old_indexes:
        await delete_key(i, 2)
        
    await i_s.delete_document(old_indexes)
    
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