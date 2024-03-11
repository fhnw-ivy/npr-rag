import chromadb
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from tqdm import tqdm

from src import config


class VectorStore:
    def __init__(self,
                 embedding_function: Embeddings,
                 collection: str,
                 host: str = config.vector_store['host'],
                 port: int = config.vector_store['port']):
        self.client = chromadb.HttpClient(host=host, port=port)

        # TODO: Remove this later
        self.client.reset()

        self.vector_store = Chroma(client=self.client,
                                   collection_name=collection,
                                   embedding_function=embedding_function)

    def heartbeat(self) -> int:
        """Check the heartbeat of the vector store client."""
        return self.client.heartbeat()

    def reset(self) -> None:
        """Reset the vector store client."""
        self.client.reset()

    def get_retriever(self) -> BaseRetriever:
        """Retrieve a VectorStoreRetriever from the Chroma vector store."""
        return self.vector_store.as_retriever()

    def add_documents(self, docs: list[Document], batch_size=41666, verbose: bool = False):
        """Add a list of documents to the vector store."""
        batch_size = min(batch_size, 41666)
        batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)] # Check if this embeds documents twice when total size is not a multiple of batch_size

        for batch in tqdm(batches):
            self.vector_store.add_documents(documents=batch, verbose=verbose)

    def similarity_search(self, query: str) -> list[Document]:
        """Perform a similarity search in the vector store with a given query."""
        return self.vector_store.similarity_search(query)