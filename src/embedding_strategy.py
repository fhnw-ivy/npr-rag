from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.embeddings import Embeddings

from src.preprocessing.base_processor import BaseProcessor
from src.vectorstore import VectorStore


class EmbeddingStrategy:
    def __init__(self, embedding_model: Embeddings, processor: BaseProcessor, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.processor = processor
        self.vector_store = vector_store

    @staticmethod
    def get_default_strategy():
        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )

        vector_store = VectorStore(embedding_function=hf,
                                   collection="cleantech-bge-small-en")

        return EmbeddingStrategy(embedding_model=hf, processor=BaseProcessor(hf), vector_store=vector_store)