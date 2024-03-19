import os
from enum import Enum

import openai
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from src.embedding_strategy import EmbeddingStrategy
from src.evaluation import EvaluationAssistant
from src.langfuse import TraceManager, TraceTag
from src.prompts import Prompt

load_dotenv()

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
)


class LLMModel(Enum):
    GPT_3_AZURE = "gpt-3-azure"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


def get_llm_model(model: LLMModel = LLMModel.GPT_3_5_TURBO):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if model == LLMModel.GPT_3_5_TURBO:
        return ChatOpenAI(model_name="gpt-3.5-turbo")

    if model == LLMModel.GPT_3_AZURE:
        return AzureChatOpenAI()

    raise ValueError(f"Model {model} not supported.")


class Generator:
    def __init__(self,
                 embedding_strategy: EmbeddingStrategy,
                 rag_prompt_key: str = "base_template"):

        self.rag_prompt_key = rag_prompt_key
        self.rag_prompt = Prompt.get(rag_prompt_key)

        self.embedding_strategy = embedding_strategy
        self.vectorstore = embedding_strategy.vector_store
        self.retriever = embedding_strategy.retriever
        self.manager = None,
        self.chain = None
        self.model = None

    def set_chain(self, chain):
        self.chain = chain

    def set_model(self, model):
        self.model = model

    def ask(self, question: str) -> tuple[str, list[Document]]:
        if self.model is None:
            model = get_llm_model()
        else:
            model = self.model

        if self.chain is None:
            chain = (
                    {
                        "context": self.retriever,
                        "question": RunnablePassthrough()
                    }
                    | ChatPromptTemplate.from_template(Prompt.DEFAULT_PROMPT.get("base_template")[0])
                    | model
                    | StrOutputParser()
            )
        else:
            chain = self.chain

        metadata = {
            "embedding_strategy": self.embedding_strategy.get_info(),
        }

        self.manager = TraceManager(version=self.embedding_strategy.get_version_string(),
                                    tags=[TraceTag.production],
                                    metadata=metadata)

        answer = chain.invoke(question, config={"callbacks": [self.manager.get_callback_handler()]})

        metrics = [
            faithfulness,
            answer_relevancy
        ]

        evaluator = EvaluationAssistant(metrics=metrics)
        scores = evaluator.assess(question, answer, self.retriever.get_relevant_documents(question))

        for k, v in scores.items():
            value = v if str(v) != 'nan' else 0.0
            self.manager.add_score(k, value)

        self.manager.add_query(question)
        self.manager.add_output(answer)

        return answer, self.retriever.get_relevant_documents(question)
