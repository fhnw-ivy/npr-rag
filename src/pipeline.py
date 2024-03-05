from generator import Generator
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
import os

load_dotenv()

sample_documents = [
    "The quick brown fox jumps over the lazy kangaroo.",
    "The lazy Wombat jumps over the lazy fox."
]

vectorstore = FAISS.from_texts(
    sample_documents, embedding=OpenAIEmbeddings()
)

gen = Generator(openai_api_key=os.getenv("OPENAI_API_KEY"), vectorstore=vectorstore)

gen.ask("What is the color of the fox?")