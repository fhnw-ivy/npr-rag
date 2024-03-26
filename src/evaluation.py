import nest_asyncio
from datasets import Dataset
from langchain.chains import LLMChain
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM
from ragas import evaluate
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
                 metrics=None,
                 is_async=True,
                 raise_exceptions=True):
        """
        Initializes the RAG evaluator with the specified chain, metrics, embeddings, and LLM model.
        :param chain: The chain to use for generating answers and contexts.
        :param metrics: A list of metric functions to evaluate the dataset.
        :param embeddings: Embeddings to use for evaluation.
        :param llm_model: The LLM model to use.
        :param is_async: Whether to evaluate asynchronously.
        :param raise_exceptions: Whether to raise exceptions during evaluation.
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
        self.is_async = is_async
        self.raise_exceptions = raise_exceptions

    def create_dataset_from_df(self, df) -> Dataset:
        """
        Creates a dataset from the provided DataFrame.
        :param df: A DataFrame with questions and relevant chunks for evaluation.
        :return: A dataset for evaluation.
        """
        required_cols = ["question", "relevant_chunk"]
        assert all(col in df.columns for col in required_cols), \
            f"DataFrame must contain columns {required_cols}, but got {df.columns} instead."

        dataset = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        df_has_actual_answers = "answer" in df.columns
        if df_has_actual_answers:
            dataset["actual_answer"] = []

        for _, item in tqdm(df.iterrows(), total=len(df)):
            question = item['question']
            dataset['question'] += [question]

            chain_result = self.chain.invoke(question)
            answer, contexts = chain_result['answer'], chain_result['context']

            assert type(contexts) == list, f"Contexts must be a list, but got {type(contexts)} instead."

            dataset['answer'] += [answer]
            dataset['contexts'] += contexts
            dataset['ground_truth'] += [item['relevant_chunk']]

            if df_has_actual_answers:
                dataset['actual_answer'] += [item['answer']]

        self.dataset = Dataset.from_dict(dataset)
        return self.dataset

    def create_dataset_from_df_async(self, df, max_concurrency=2) -> Dataset:
        assert all(col in df.columns for col in ["question", "relevant_chunk"]), \
            "DataFrame must contain columns ['question', 'relevant_chunk']."

        # If the chain can be invoked in batches for efficiency:
        questions = df['question'].tolist()
        # Assuming batch_invoke is implemented to handle batch processing
        results = self.chain.batch(questions, config={"max_concurrency": max_concurrency})

        # Directly use results to create the dataset dictionary
        dataset = {
            "question": questions,
            "answer": [result['answer'] for result in results],
            "contexts": [result['context'] for result in results],
            "ground_truth": df['relevant_chunk'].tolist()
        }

        if "answer" in df.columns:
            dataset["actual_answer"] = df['answer'].tolist()

        return Dataset.from_dict(dataset)

    def evaluate(self):
        """
        Evaluates the RAG pipeline using the specified dataset and metrics.
        :return: Evaluation results.
        """
        assert self.dataset is not None, "Dataset must be set before evaluation."
        assert self.metrics is not None, "Metrics must be set before evaluation."

        eval_results = evaluate(
            self.dataset,
            metrics=self.metrics,
            embeddings=self.embeddings,
            is_async=self.is_async,
            raise_exceptions=self.raise_exceptions,
            llm=self.llm_model
        )
        return eval_results
