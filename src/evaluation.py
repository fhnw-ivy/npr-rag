import pandas as pd
from datasets import Dataset
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from ragas import evaluate as ragas_eval
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.testset.generator import TestsetGenerator


def create_dataset(question: str, answer: str, contexts: list[str], ground_truth: str = "") -> Dataset:
    """Converts input parameters into a Dataset object."""
    return Dataset.from_dict({
        'question': [question],
        'answer': [answer],
        'contexts': [contexts],
        'ground_truth': [ground_truth] if ground_truth else []
    })


def generate_testset(documents: list[Document],
                     generator_llm: BaseChatModel,
                     critic_llm: BaseChatModel,
                     embeddings: Embeddings,
                     test_size: int = 10,
                     distributions=None,
                     verbose=False) -> Dataset:
    """Generates a testset based on given documents and distribution."""
    if distributions is None:
        distributions = {simple: 0.5, reasoning: 0.25, multi_context: 0.25}

    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)
    return generator.generate_with_langchain_docs(documents,
                                                  test_size=test_size,
                                                  distributions=distributions,
                                                  with_debugging_logs=verbose)


class EvaluationAssistant:
    def __init__(self, metrics=None):
        if metrics is None:
            metrics = [answer_relevancy, faithfulness, context_recall, context_precision]
        self.metrics = metrics

    def assess(self, question: str, answer: str, contexts: list[str], ground_truth: str = "") -> pd.DataFrame:
        """Assesses the quality of an answer with respect to a question, contexts, and an optional ground truth."""
        dataset = create_dataset(question, answer, contexts, ground_truth)
        scores = ragas_eval(dataset, metrics=self.metrics).to_pandas()
        return scores
