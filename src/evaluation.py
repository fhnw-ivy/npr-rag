import pandas as pd
from datasets import Dataset
from ragas import evaluate as ragas_eval
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)


def create_dataset(question: str, answer: str, contexts: list[str], ground_truth: str = "") -> Dataset:
    """Converts input parameters into a Dataset object."""
    return Dataset.from_dict({
        'question': [question],
        'answer': [answer],
        'contexts': [contexts],
        'ground_truth': [ground_truth] if ground_truth else []
    })


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
