import nest_asyncio
import pandas as pd
from datasets import Dataset
from langchain.chains import LLMChain
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM
from ragas import evaluate, RunConfig
from ragas.metrics import (
    answer_correctness,
    context_precision,
    context_recall,
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
                 chain: LLMChain,
                 embeddings: Embeddings,
                 llm_model: LLM,
                 dataset: Dataset = None,
                 metrics=None):
        """
        Initializes the RAG evaluator with the specified chain, metrics, embeddings, and LLM model.
        :param chain: The chain to use for generating answers and contexts.
        :param metrics: A list of metric functions to evaluate the dataset. When None, the default metrics (answer correctness, context precision, and context recall) are used.
        :param embeddings: Embeddings to use for evaluation.
        :param llm_model: The LLM model to use.
        """
        nest_asyncio.apply()

        if metrics is None:
            self.metrics = [
                answer_correctness,
                context_precision,
                context_recall,
            ]
        else:
            self.metrics = metrics

        self.dataset = dataset
        self.chain = chain
        self.embeddings = embeddings
        self.llm_model = llm_model

    def create_dataset_from_df(self, df) -> Dataset:
        """
        Creates a dataset from the provided DataFrame.
        :param df: A DataFrame with questions and relevant chunks for evaluation.
        :return: A dataset for evaluation.
        """
        if self.dataset is not None:
            print("Dataset already set. Explicitly set the property to overwrite it.")
            return self.dataset

        required_cols = ["question", "relevant_chunk"]
        assert all(col in df.columns for col in required_cols), \
            f"DataFrame must contain columns {required_cols}, but got {df.columns} instead."

        dataset = {"question": [], "answer": [], "contexts": [], "ground_truth": [], "question_complexity": []}

        df_has_actual_answers = "answer" in df.columns
        if df_has_actual_answers:
            dataset["actual_answer"] = []

        for _, item in tqdm(df.iterrows(), total=len(df)):
            question = item['question']
            dataset['question'] += [question]

            chain_result = self.chain.invoke(question)
            answer, contexts = chain_result['answer'], chain_result['context']

            assert type(contexts) == list, f"Contexts must be a list, but got {type(contexts)} instead."
            assert all(type(c) == Document for c in contexts), \
                f"Contexts must be a list of Documents, but got {type(contexts[0])} instead."

            dataset['answer'] += [answer]
            dataset['contexts'] += [[str(c.page_content) for c in contexts]]
            dataset['ground_truth'] += [item['relevant_chunk']]

            if df_has_actual_answers:
                dataset['actual_answer'] += [item['answer']]
                dataset['question_complexity'] += [item['question_complexity']]
                # dataset['answer_complexity'] += [item['answer_complexity']]

        self.dataset = Dataset.from_dict(dataset)
        return self.dataset

    def evaluate(self,
                 raise_exceptions=True,
                 is_async=True,
                 timeout=60,
                 max_retries=10,
                 max_wait=60,
                 max_workers=16) -> pd.DataFrame:
        """
        Evaluates the RAG pipeline using the specified dataset and metrics.
        :return: Evaluation results.
        """
        assert self.dataset is not None, "Dataset must be set before evaluation."
        assert self.metrics is not None, "Metrics must be set before evaluation."

        run_config = RunConfig(
            timeout=timeout,
            max_retries=max_retries,
            max_wait=max_wait,
            max_workers=max_workers,
        )

        eval_results = evaluate(
            self.dataset,
            metrics=self.metrics,
            embeddings=self.embeddings,
            is_async=is_async,
            raise_exceptions=raise_exceptions,
            llm=self.llm_model,
            run_config=run_config
        )
        return eval_results.to_pandas()
