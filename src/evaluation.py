from langfuse import Langfuse
from ragas import evaluate as ragas_eval
from datasets import Dataset
from langchain_core.documents import Document
from ragas.metrics.base import Metric

from ragas.metrics import (
    answer_relevancy,
    faithfulness
)

langfuse = Langfuse()

langfuse.auth_check()

class Evaluator:
    def __init__(self):
        self.langfuse = langfuse

    def evaluate(self, code: str, language: str, input: str = ""):
        return self.langfuse.evaluate(code, language, input)
    
    def _convert_to_dataset(self, question: str, answer: str, context: list[Document]):
        content_list = [doc.page_content for doc in context if doc.page_content is not None]

        return Dataset.from_dict({
            'question': [question],
            'answer': [answer],
            'contexts' : [content_list]
        })

    def ragas_evaluate(self, question: str, answer: str, context: list[Document], metrics: list[Metric] = None):
        custom_metrics = metrics or [
            faithfulness,
            answer_relevancy
        ]

        result = ragas_eval(
            dataset=self._convert_to_dataset(question, answer, context),
            metrics=custom_metrics,
        )

        return result
