import chromadb
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from tqdm import tqdm

load_dotenv()

CHROMADB_PERSISTENT_PATH = 'chroma'


class VectorStore:
    def __init__(self,
                 embedding_function: Embeddings,
                 collection: str,
                 persistent_path: str = CHROMADB_PERSISTENT_PATH):

        self.client = chromadb.PersistentClient(path=persistent_path)
        self.embedding_function = embedding_function
        self.collection = collection

        self.vector_store = Chroma(client=self.client,
                                   collection_name=collection,
                                   embedding_function=embedding_function)

    def heartbeat(self) -> int:
        """Check the heartbeat of the vector store client."""
        return self.client.heartbeat()

    def reset(self) -> None:
        """Reset the vector store client."""
        self.client.reset()

    def delete_collection(self) -> None:
        """Reset the collection in the vector store."""
        self.vector_store.delete_collection()

    def get_retriever(self) -> BaseRetriever:
        """Retrieve a VectorStoreRetriever from the Chroma vector store."""
        return self.vector_store.as_retriever()

    def add_documents(self, docs: list[Document], batch_size=41666, verbose: bool = False, overwrite: bool = False):
        """Add a list of documents to the vector store."""
        if not self.collection_is_empty() and not overwrite:
            print(f"Collection {self.collection} already exists in the vector store.")
            return

        batch_size = min(batch_size, 41666)
        batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]

        if verbose:
            for batch in tqdm(batches):
                self.vector_store.add_documents(documents=batch, verbose=verbose)
        else:
            for batch in batches:
                self.vector_store.add_documents(documents=batch, verbose=verbose)

    def similarity_search(self, query: str) -> list[Document]:
        """Perform a similarity search in the vector store with a given query."""
        return self.vector_store.similarity_search(query)

    def similarity_search_w_scores(self, query: str) -> list[tuple[Document, float]]:
        """Perform a similarity search in the vector store with a given query."""
        return self.vector_store.similarity_search_with_score(query)

    def collection_exists(self) -> bool:
        """Check if the collection exists in the vector store."""
        print(self.client.list_collections())
        return np.any([collection.name == self.collection for collection in self.client.list_collections()])

    def collection_is_empty(self) -> bool:
        """Check if the collection is empty in the vector store."""
        return self.client.get_collection(self.collection).count() == 0

    def __repr__(self) -> str:
        return f"VectorStore(embedding_function={self.embedding_function.__class__.__name__}, " \
               f"collection={self.collection})"
