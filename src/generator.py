import openai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.documents.base import Document

from src.embedding_strategy import EmbeddingStrategy
from src.templates import base_template


def get_openai_model():
    # TODO: implement Azure model retrieval

    return ChatOpenAI()


class Generator:
    def __init__(self,
                 openai_api_key: str,
                 embedding_strategy: EmbeddingStrategy,
                 template: str = base_template):

        self.openai_api_key = openai_api_key
        self.template = template
        self.vectorstore = embedding_strategy.vector_store
        self.retriever = embedding_strategy.retriever
        openai.api_key = self.openai_api_key

    def set_template(self, template: str):
        self.template = template

    def ask(self, question: str) -> tuple[str, list[Document]]:
        prompt = ChatPromptTemplate.from_template(self.template)
        model = get_openai_model()

        chain = (
                {
                    "context": self.retriever,
                    "question": RunnablePassthrough()
                }
                | prompt
                | model
                | StrOutputParser()
        )

        answer = chain.invoke(question)
        return answer, self.retriever.get_relevant_documents(question)
