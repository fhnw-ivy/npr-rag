import os

import openai
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.embedding_strategy import EmbeddingStrategy
from src.langfuse import LangfuseHandler
from src.prompts import get_prompt

load_dotenv()


def get_openai_model():
    # TODO: implement Azure model retrieval based on availability
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI()


class Generator:
    def __init__(self,
                 embedding_strategy: EmbeddingStrategy,
                 rag_prompt_key: str = "base_template"):
        self.rag_prompt_template = None
        self.set_rag_prompt_template(rag_prompt_key)

        self.vectorstore = embedding_strategy.vector_store
        self.retriever = embedding_strategy.retriever

    def set_rag_prompt_template(self, prompt_key: str, use_langfuse: bool = True):
        self.rag_prompt_template = get_prompt(prompt_key, from_langfuse=use_langfuse)

    def ask(self, question: str) -> tuple[str, list[Document]]:
        prompt = self.rag_prompt_template

        model = get_openai_model()

        chain = (
                {
                    "context": self.retriever,
                    "question": RunnablePassthrough()
                }
                | prompt  # TODO: Add token limit check
                | model
                | StrOutputParser()
        )

        handler = LangfuseHandler()
        answer = chain.invoke(question, config={"callbacks": [handler.get_callback_handler()]})

        handler.score("test_metric", 3)

        return answer, self.retriever.get_relevant_documents(question)
