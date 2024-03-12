import os

import openai
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from src.embedding_strategy import EmbeddingStrategy
from src.langfuse import TraceManager, TraceTag
from src.prompts import Prompt

load_dotenv()


def get_openai_model():
    # TODO: implement Azure model retrieval based on availability
    # todo is this even necessary?
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI()

def get_azure_openai_model():
    return AzureChatOpenAI()


class Generator:
    def __init__(self,
                 embedding_strategy: EmbeddingStrategy,
                 rag_prompt_key: str = "base_template"):
        self.rag_prompt_key = rag_prompt_key
        self.rag_prompt = Prompt.get(rag_prompt_key)

        self.embedding_strategy = embedding_strategy
        self.vectorstore = embedding_strategy.vector_store
        self.retriever = embedding_strategy.retriever
        self.manager = None

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

        metadata = {
            "embedding_strategy": self.embedding_strategy.get_info(),
            "prompt_versioning": {
                self.rag_prompt_key: self.rag_prompt.version
            }
        }

        self.manager = TraceManager(version=self.embedding_strategy.get_version_string(),
                                    tags=[TraceTag.production],
                                    metadata=metadata)

        answer = chain.invoke(question, config={"callbacks": [self.manager.get_callback_handler()]})

        self.manager.add_query(question)
        self.manager.add_output(answer)

        return answer, self.retriever.get_relevant_documents(question)


