import os

import nest_asyncio
import pandas as pd
import seaborn as sns
from datasets import Dataset
from langchain.chains import LLMChain
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM
from matplotlib import pyplot as plt
from ragas import evaluate, RunConfig
from ragas.metrics import (
    answer_correctness,
    context_precision,
    answer_relevancy,
    answer_similarity,
    context_entity_recall,
)
from ragas.metrics.critique import (
    conciseness,
    coherence,
    correctness
)
from tqdm.auto import tqdm


def create_dataset(question: str, answer: str, contexts: list[str], ground_truth: str) -> Dataset:
    """Converts input parameters into a Dataset object."""
    return Dataset.from_dict({
        'question': [question],
        'answer': [answer],
        'contexts': [contexts],
        'ground_truth': [ground_truth] if ground_truth else []
    })


class RAGEvaluator:
    def __init__(self,
                 name: str,
                 chain: LLMChain,
                 embeddings: Embeddings,
                 llm_model: LLM,
                 dataset: Dataset = None):
        nest_asyncio.apply()

        self.metrics = [
            answer_correctness,
            answer_relevancy,
            answer_similarity,

            context_precision,
            context_entity_recall,

            conciseness,
            coherence,
            correctness
        ]

        self.name = name
        self.dataset = dataset
        self.chain = chain
        self.embeddings = embeddings
        self.llm_model = llm_model

        self.eval_results = None
        self.load_results()

    def _create_dataset_from_df(self, df) -> Dataset:
        if self.dataset is not None:
            return self.dataset

        expected_columns = ['question', 'answer', 'question_complexity']
        assert all(col in df.columns for col in expected_columns), f"Columns {expected_columns} are required."

        dataset = {"question": [],
                   "answer": [],
                   "contexts": [],
                   "ground_truth": [],
                   "actual_answer": [],
                   "question_complexity": []}

        for _, item in tqdm(df.iterrows(), total=len(df), desc="Creating dataset"):
            question = item['question']
            dataset['question'] += [question]

            chain_result = self.chain.invoke(question)
            print(type(chain_result['context']))
            answer, contexts = chain_result['answer'], chain_result['context']

            # RAG pipeline results
            dataset['answer'] += [answer]
            dataset['contexts'] += [[str(c.page_content) for c in contexts]]

            # Ground truth from labeled data
            dataset['ground_truth'] += [item['answer']]

            dataset["question_complexity"] += [item["question_complexity"]]

        self.dataset = Dataset.from_dict(dataset)
        return self.dataset

    def evaluate(self,
                 eval_df: pd.DataFrame = None,
                 raise_exceptions=True,
                 is_async=True,
                 timeout=60,
                 max_retries=10,
                 max_wait=60,
                 max_workers=16) -> pd.DataFrame:

        if self.eval_results is not None:
            return self.eval_results

        if self.dataset is None and eval_df is not None:
            self._create_dataset_from_df(eval_df)

        run_config = RunConfig(
            timeout=timeout,
            max_retries=max_retries,
            max_wait=max_wait,
            max_workers=max_workers,
        )

        ragas_eval_results = evaluate(
            self.dataset,
            metrics=self.metrics,
            embeddings=self.embeddings,
            is_async=is_async,
            raise_exceptions=raise_exceptions,
            llm=self.llm_model,
            run_config=run_config
        )

        eval_results_df = ragas_eval_results.to_pandas()

        for i, row in tqdm(eval_results_df.iterrows(), total=len(eval_results_df), desc="Reasoning"):
            question, answer, contexts = row['question'], row['answer'], row['contexts']
            result = self.reason(question, answer, contexts)
            reasoning = result['reasoning']
            eval_results_df.at[i, 'reasoning'] = reasoning

        self.eval_results = eval_results_df
        self.save_results()
        return eval_results_df

    def reason(self, question: str, answer: str, retrieved_context: str):
        evaluator = load_evaluator(EvaluatorType.LABELED_SCORE_STRING, llm=self.llm_model)
        return evaluator.evaluate_strings(prediction=answer,
                                          input=question,
                                          reference=retrieved_context)

    def summarize_metrics(self):
        metrics_data = self.eval_results.iloc[:, 5:]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=metrics_data)
        plt.title('Evaluation Metrics Summary')
        plt.xticks(rotation=45)
        plt.ylabel('Score')
        plt.tight_layout()
        plt.show()

    def _get_persistence_path(self):
        clean_name = self.name.replace(" ", "_").lower()
        return f"./tmp/evaluation_results_{clean_name}.csv"

    def save_results(self):
        if self.eval_results is None:
            raise ValueError("Evaluation results must be set before saving.")

        os.makedirs("./tmp", exist_ok=True)

        self.eval_results.to_csv(self._get_persistence_path(), index=False)
        return self._get_persistence_path()

    def load_results(self, overwrite=False):
        if self.eval_results is not None and not overwrite:
            print("Evaluation results already set. Explicitly set the property to overwrite it.")

        if not os.path.exists(self._get_persistence_path()):
            return None

        self.eval_results = pd.read_csv(self._get_persistence_path())
        return self.eval_results

    def __repr__(self):
        return f"RAGEvaluator(name={self.name}, metrics={self.metrics})"
