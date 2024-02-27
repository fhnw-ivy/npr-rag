import chromadb
from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()

if __name__ == "__main__":
    text = "This is a test sentence."

    embedding = default_ef([text])

    client = chromadb.Client()
    collection = client.create_collection("sample_collection")

    print(embedding)

    print(type(embedding))
    # Add docs to the collection. Can also update and delete. Row-based API coming soon!
    collection.add(
        documents=[text],
        metadatas=[{"source": "my_source"}],
        ids=["test1"],
        embeddings=embedding,
    )
