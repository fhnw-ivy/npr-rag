from src.preprocessing.base_processor import BaseProcessor
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


class HuggingFaceProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_splitter = SemanticChunker(self.embedding_model)

    def preprocess_documents(self, documents):
        # Implement specific preprocessing logic for HuggingFaceEmbeddings
        return super().preprocess_documents(documents)

    def chunk_documents(self, documents):
        # Implement chunking with SemanticChunker for HuggingFaceEmbeddings
        return super().chunk_documents(documents)

