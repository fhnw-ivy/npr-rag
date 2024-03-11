import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

from langchain_core.documents.base import Document


def turn_to_dataset(question: str, answer: str, context: list[str], ground_truth: str) -> Dataset:
    return Dataset.from_dict({
        'question': [question],
        'answer': [answer],
        'contexts': [context],
        'ground_truth': [ground_truth]
    })


def generate_testset(documents: list[Document]) -> Dataset:
    generator = TestsetGenerator.with_openai()
    testset = generator.generate_with_langchain_docs(documents,
                                                     test_size=10,
                                                     distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

    return testset


class Assessor:
    def __init__(self,
                 metrics: list[(MetricWithLLM, MetricWithEmbeddings)] = [answer_relevancy, faithfulness, context_recall,
                                                                         context_precision]):
        self.metrics = metrics

    def assess_dataset(self, dataset: Dataset) -> pd.DataFrame:
        scores = self._evaluate_dataset(dataset)
        return scores

    def assess_example(self, question: str, answer: str, context: list[str], ground_truth: str) -> pd.DataFrame:
        dataset = turn_to_dataset(question, answer, context, ground_truth)

        scores = self._evaluate_dataset(dataset)
        return scores

    def _evaluate_dataset(self, dataset: Dataset) -> pd.DataFrame:
        scores = evaluate(dataset, metrics=self.metrics).to_pandas()
        return scores
