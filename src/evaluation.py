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
    context_recall,
    context_relevancy,
    faithfulness,
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
                 chain: LLMChain,
                 embeddings: Embeddings,
                 llm_model: LLM,
                 dataset: Dataset = None):
        """
        Initializes the RAG evaluator with the specified chain, metrics, embeddings, and LLM model.
        :param chain: The chain to use for generating answers and contexts.
        :param embeddings: Embeddings to use for evaluation.
        :param llm_model: The LLM model to use.
        :param dataset: The ragas dataset to use for evaluation. When None, the dataset must be set explicitly before evaluation.
        """
        nest_asyncio.apply()

        self.metrics = [
            faithfulness,
            answer_correctness,

            context_precision,
            context_recall,
            context_relevancy,
            context_entity_recall,

            answer_relevancy,
            answer_similarity,

            conciseness,
            coherence,
            correctness
        ]

        self.dataset = dataset
        self.chain = chain
        self.embeddings = embeddings
        self.llm_model = llm_model

        self.eval_results = None

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

        dataset = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        df_has_actual_answers = "answer" in df.columns
        if df_has_actual_answers:
            dataset["actual_answer"] = []
            dataset["question_complexity"] = []

        for _, item in tqdm(df.iterrows(), total=len(df), desc="Creating dataset"):
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
        """
        Evaluates the RAG pipeline using the specified dataset and metrics.
        :return: Evaluation results.
        """
        assert self.metrics is not None, "Metrics must be set before evaluation."

        if self.dataset is None and eval_df is not None:
            try:
                self.create_dataset_from_df(eval_df)
            except Exception as e:
                raise ValueError(f"Failed to create dataset from DataFrame: {e}")
        elif eval_df is None and self.dataset is None:
            raise ValueError(
                "Dataset must be set before evaluation. Set the dataset explicitly or set create_dataset to True.")

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
        return eval_results_df

    def reason(self, question: str, answer: str, retrieved_context: str):
        evaluator = load_evaluator(EvaluatorType.LABELED_SCORE_STRING, llm=self.llm_model)
        return evaluator.evaluate_strings(prediction=answer,
                                          input=question,
                                          reference=retrieved_context)

    def summarize_metrics(self):
        """
        Summarizes the metrics data for the Retriever Augmented Generation Pipeline.
        Returns:
            A summary plot of the metrics.
        """
        metrics_data = self.eval_results.iloc[:, 5:]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=metrics_data)
        plt.title('Evaluation Metrics Summary')
        plt.xticks(rotation=45)
        plt.ylabel('Score')
        plt.tight_layout()
        plt.show()