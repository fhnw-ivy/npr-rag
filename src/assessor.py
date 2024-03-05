from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM
from datasets import Dataset 
from ragas import evaluate

class Assessor:
    def __init__(self, metrics: list[(MetricWithLLM, MetricWithEmbeddings)]=[answer_relevancy, faithfulness, context_recall, context_precision]):
        self.metrics = metrics

    def _turn_to_dataset(self, question, answer, context: list[str]):
        return Dataset.from_dict({
            'question': question,
            'answer': answer,
            'contexts': context
        })

    def assess(self, question, answer, context: list[str]):
        dataset = self._turn_to_dataset(question, answer, context)
        scores = evaluate(dataset, metrics = self.metrics).to_pandas()
        return scores