from transformers import BertTokenizerFast, BertModel
import torch
import chromadb
from sync.query_index import QueryIndexSubsystem
import asyncio

async def get_subsystem():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")
    model = BertModel.from_pretrained("bert-large-uncased")
    model = model.to(device)

    client = await chromadb.AsyncHttpClient(
        host="chromadb", port=8000
    )
    collection = await client.get_or_create_collection(name="files", metadata={"hnsw:space": "l2"})    
        
    return collection, device, model, tokenizer


i_s = QueryIndexSubsystem(*asyncio.run(get_subsystem()), 128, 64)