import chromadb
from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()


if __name__ == "__main__":
    # Embedding
    text = "This is a test sentence."
    collection_name = "sample_collection"

    embeddings = default_ef([text])
    print(embeddings)

    # Add to collection
    chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    collection = chroma_client.get_or_create_collection(collection_name)

    collection.add(
        documents=[text],
        metadatas=[{"source": "my_source"}],
        ids=["test1"],
        embeddings=embeddings,
    )

    # Query
    query = "is a test"
    query_embeddings = default_ef([query])

    query_results = collection.query(
        query_embeddings=query_embeddings,
        n_results=10,
        where={"source": "my_source"},
    )

    print(query_results)
