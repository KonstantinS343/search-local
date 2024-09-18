
#import chromadb
from transformers import BertTokenizerFast, BertModel
import torch
import chromadb
import asyncio


class QueryIndexSubsystem:

    def __init__(
        self, chroma_db_collection, device, model, tokenizer, max_tokens_on_window, window_stride
    ):
        self._chroma_db_collection = chroma_db_collection
        self._model = model
        self._tokenizer = tokenizer
        self._max_tokens_on_window = max_tokens_on_window
        self._window_stride = window_stride
        self._device = device


    def _tokenize_text(self, document_text):
        tokenized_text = self._tokenizer(
            document_text,
            padding="max_length", truncation=True,
            max_length=self._max_tokens_on_window, stride=self._window_stride,
            return_overflowing_tokens=True,
            return_tensors='pt', 
        )

        tokenized_text.pop("overflow_to_sample_mapping")

        for key in tokenized_text.keys():
            tokenized_text[key] = tokenized_text[key].type(torch.int32).to(self._device)
        return tokenized_text


    def _count_text_embeddings(self, document_tokens, mean_dim_0 = False):
        with torch.no_grad():
            document_embeddings = self._model(
                **document_tokens
            ).last_hidden_state.mean(dim=1)
            
            if mean_dim_0:
                return [
                    document_embeddings.mean(dim=0).detach().cpu().tolist()
                ]  # Additional list is needed because mean reduce dims
            else:
                return document_embeddings.detach().cpu().tolist()


    async def index_new_document(self, document_path, document_text):
        # document_embeddings is List[List[float]]
        document_tokens = self._tokenize_text(document_text)
        document_embeddings = self._count_text_embeddings(document_tokens)
        index_ids = [
            f"{document_path}_window{window_number}"
            for window_number in range(len(document_embeddings))
        ]

        await self._chroma_db_collection.add(
            embeddings=document_embeddings, ids=index_ids
        )
        return index_ids


    async def handle_user_query(self, user_query, limit):
        with torch.no_grad():
            query_tokens = self._tokenize_text(user_query)
            if query_tokens["input_ids"].shape[0] <= 3:
                query_embeddings = self._count_text_embeddings(
                    query_tokens, mean_dim_0=True
                )
            else:
                query_embeddings = self._count_text_embeddings(
                    query_tokens, mean_dim_0=False
                )
            # TODO different query to db for this cases
            result = await self._chroma_db_collection.query(
                query_embeddings=query_embeddings,
                n_results=limit,
                include=["ids", "distance"]
            )
            print(result)  # TODO remove
            return result


    async def reindex_existing_document(
        self, document_path, new_document_text, old_index_ids
    ):
        # document_embeddings is List[List[float]]
        new_document_tokens = self._tokenize_text(new_document_text)

        if new_document_tokens["input_ids"].shape[0] < old_index_ids:
            for window_number in range(new_document_tokens["input_ids"].shape[0]):
                old_index_ids.remove(f"{document_path}_window{window_number}")

            await self._chroma_db_collection.delete(ids=old_index_ids)  # removes overflowing windows
            
        new_document_embeddings = self._count_text_embeddings(new_document_tokens)
        index_ids = [
            f"{document_path}_window{window_number}"
            for window_number in range(len(new_document_embeddings))
        ]

        await self._chroma_db_collection.upsert(
            embeddings=new_document_embeddings, ids=index_ids
        )
        return index_ids

    
    async def rename_document(self, new_document_path, old_index_ids):
        document_embeddings = await self._chroma_db_collection.get(
            ids=old_index_ids, include=["embeddings"]
        )
        print(document_embeddings)  # TODO remove
        new_index_ids = [
            f"{new_document_path}_window{window_number}"
            for window_number in range(len(old_index_ids))
        ]

        await self.delete_document(old_index_ids)

        await self._chroma_db_collection.add(
            embeddings=document_embeddings, ids=new_index_ids
        )
        return new_index_ids


    async def delete_document(self, index_ids):
        await self._chroma_db_collection.delete(ids=index_ids)


async def main():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")
    model = BertModel.from_pretrained("bert-large-uncased")
    model = model.to(device)

    client = await chromadb.AsyncHttpClient(
        host="localhost", port=1000
        #,settings=chromadb.config.Settings(allow_reset=True, anonymized_telemetry=False)
    )
    collection = await client.get_or_create_collection(name="nigga", metadata={"hnsw:space": "l2"})    
        
    i_s = QueryIndexSubsystem(collection, device, model, tokenizer, 512, 256)
    

    await i_s.index_new_document("nigga.zs", "Rad dead redemption" * 500)


    



asyncio.run(main())




