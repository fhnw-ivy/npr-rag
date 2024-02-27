from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from store import get_chroma_client

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()

chroma_client = get_chroma_client()

langchain_chroma = Chroma(
    client=chroma_client,
    collection_name="my_langchain_collection",
    embedding_function=embeddings,
)

documents = [
    "This is a test sentence.",
    "This is another test sentence.",
    "This is a third test sentence.",
]

langchain_chroma.add_texts(texts=documents)
