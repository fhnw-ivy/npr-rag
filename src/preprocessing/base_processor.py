from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings


class BaseProcessor:
    def __init__(self, embedding_model: Embeddings):
        self.embedding_model = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=32, length_function=len)

    def clean(self, corpus: str) -> str:
        return corpus

    def chunk(self, corpus: str, metadata) -> list[Document]:
        docs = []
        for chunk in self.text_splitter.split_text(corpus):
            docs.append(Document(page_content=chunk, metadata=metadata))

        return docs

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        self.text_splitter.split_documents(documents)
        return documents