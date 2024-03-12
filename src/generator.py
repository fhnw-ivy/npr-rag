import os

import openai
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.embedding_strategy import EmbeddingStrategy
from src.langfuse import LangfuseHandler
from src.prompts import Prompt

load_dotenv()


def get_openai_model():
    # TODO: implement Azure model retrieval based on availability
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI()


class Generator:
    def __init__(self,
                 embedding_strategy: EmbeddingStrategy,
                 rag_prompt_key: str = "base_template"):
        self.rag_prompt = Prompt.get(rag_prompt_key)
        self.vectorstore = embedding_strategy.vector_store
        self.retriever = embedding_strategy.retriever

    def ask(self, question: str) -> tuple[str, list[Document]]:
        prompt_template = self.rag_prompt.template

        model = get_openai_model()

        chain = (
                {
                    "context": self.retriever,
                    "question": RunnablePassthrough()
                }
                | prompt_template  # TODO: Add token limit check
                | model
                | StrOutputParser()
        )

        handler = LangfuseHandler()
        answer = chain.invoke(question, config={"callbacks": [handler.get_callback_handler()]})

        return answer, self.retriever.get_relevant_documents(question)
