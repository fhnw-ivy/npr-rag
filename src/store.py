import chromadb


def get_chroma_client():
    return chromadb.HttpClient(host='localhost', port=8000)
