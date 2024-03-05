from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.embeddings import Embeddings

from src.preprocessing.base_processor import BaseProcessor


class EmbeddingStrategy:
    def __init__(self, embedding_model: Embeddings, processor: BaseProcessor):
        self.embedding_model = embedding_model
        self.processor = processor

    @staticmethod
    def get_default_strategy():
        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )

        return EmbeddingStrategy(embedding_model=hf, processor=BaseProcessor(hf))