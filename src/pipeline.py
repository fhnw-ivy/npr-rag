from generator import Generator
from dotenv import load_dotenv
import os

load_dotenv()

import config
from vectorstore import VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

vector_store = VectorStore(
    embedding_function=embeddings,
    host=config.vector_store['host'],
    port=config.vector_store['port'],
    collection=config.vector_store['collection']
)

gen = Generator(openai_api_key=os.getenv("OPENAI_API_KEY"), vectorstore=vector_store)

gen.ask("Who was in Paris?")