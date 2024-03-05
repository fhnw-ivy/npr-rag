from src import config
from src.vectorstore import VectorStore

vector_store = VectorStore(
    embedding_function=None,
    host=config.vector_store['host'],
    port=config.vector_store['port'],
    collection=config.vector_store['collection']
)