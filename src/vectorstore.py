import chromadb
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStoreRetriever


class VectorStore:
    def __init__(self,
                 embedding_function,
                 host,
                 port,
                 collection):
        self.client = chromadb.Client(host=host, port=port)
        self.vector_store = Chroma(self.client, collection=collection)
        self.embedding_function = embedding_function

    def heartbeat(self) -> int:
        """Check the heartbeat of the vector store client."""
        return self.client.heartbeat()

    def reset(self) -> None:
        """Reset the vector store client."""
        self.client.reset()

    def get_retriever(self) -> VectorStoreRetriever:
        """Retrieve a VectorStoreRetriever from the Chroma vector store."""
        return self.vector_store.as_retriever()

    def add_documents(self, docs: list[Document]):
        """Add a list of documents to the vector store."""
        self.vector_store.add_documents(docs, embedding=self.embedding_function)

    def similarity_search(self, query: str) -> list[Document]:
        """Perform a similarity search in the vector store with a given query."""
        return self.vector_store.similarity_search(query)
