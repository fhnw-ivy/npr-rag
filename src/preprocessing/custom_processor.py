from langchain_experimental.text_splitter import SemanticChunker

from src.preprocessing.base_processor import BaseProcessor


class HuggingFaceProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess_documents(self, documents):
        # Implement specific preprocessing logic for HuggingFaceEmbeddings
        return super().preprocess_documents(documents)

    def chunk_documents(self, documents):
        # Implement chunking with SemanticChunker for HuggingFaceEmbeddings
        return super().chunk_documents(documents)

