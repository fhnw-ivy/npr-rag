from pathlib import Path

import nest_asyncio
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset
from langchain.evaluation import load_evaluator, EvaluatorType
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LLM
from matplotlib import pyplot as plt
from ragas import evaluate, RunConfig
from ragas.metrics import (answer_correctness, context_precision, answer_relevancy, answer_similarity,
                           context_entity_recall)
from tqdm.auto import tqdm

nest_asyncio.apply()


class DatasetCreator:
    def __init__(self, chain):
        self.chain = chain

    def validate_dataframe(self, df):
        """
        Validate if all expected columns exist in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Raises:
            ValueError: If any expected columns are missing in the DataFrame.
        """
        expected_columns = ['question', 'answer', 'question_complexity']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"Expected columns {expected_columns} not found in DataFrame.")

    def populate_dataset(self, df):
        """
        Populate a dataset dictionary from the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame from which to create the dataset.

        Returns:
            dict: A dictionary representing the dataset.
        """
        # Meaning of columns taken from: https://cobusgreyling.medium.com/rag-evaluation-9813a931b3d4
        dataset = {"question": [],  # The question posed
                   "answer": [],  # The generated answer from the RAG pipeline
                   "contexts": [],  # The retrieved contexts from the RAG pipeline
                   "ground_truth": [],  # The ground truth answer from the labeled data
                   "actual_answer": [],  # The actual answer from the labeled data (not used in eval but for reference)
                   "question_complexity": [],  # The complexity of the question (from RAGAS dataset generation)
                   "contexts_origin_doc_ids": [],  # The origin document IDs of the retrieved contexts
                   "best_match_id": []  # The best match ID from the default dataset
                   }

        dataset['question'] = df['question'].tolist()
        dataset['ground_truth'] = df['answer'].tolist()
        dataset['actual_answer'] = df['answer'].tolist()
        dataset['question_complexity'] = df['question_complexity'].tolist()
        dataset['best_match_id'] = df['best_match_id'].tolist()
        return dataset

    def apply_rag_chain(self, question):
        """
        Apply the RAG chain to a question to generate answers and contexts.

        Args:
            question (str): The question to process.

        Returns:
            tuple: A tuple containing the answer and contexts.
        """
        result = self.chain.invoke(question)
        return result['answer'], result['context']

    def create_dataset_from_df(self, df, verbose=True):
        """
        Create a dataset from a DataFrame after validating and populating necessary fields.

        Args:
            df (pd.DataFrame): The DataFrame from which to create the dataset.
            verbose (bool): Whether to print verbose logs. Default is False.

        Returns:
            Dataset: A Dataset object populated with the processed data.
        """
        self.validate_dataframe(df)

        dataset = self.populate_dataset(df)
        for question in tqdm(df['question'].tolist(),
                             desc="Running RAG chain",
                             disable=not verbose):
            answer, contexts = self.apply_rag_chain(question)

            dataset['answer'].append(answer)
            dataset['contexts'].append([str(c.page_content) for c in contexts])
            dataset['contexts_origin_doc_ids'].append(
                [c.metadata["origin_doc_id"] for c in contexts])

        return Dataset.from_dict(dataset)


class Evaluator:
    def __init__(self, name, rag_chain, llm_model, embeddings, cache_results=True, cache_dir="cache"):
        """
        Initialize the Evaluator.

        Args:
            name (str): The name of the evaluator.
            rag_chain (LangchainChain): The RAG chain to use for evaluations.
            llm_model (LLM): The language model to use for evaluations.
            embeddings (Embeddings): The embeddings to use for evaluations.
            cache_results (bool): Whether to cache the evaluation results. Default is True.
            cache_dir (str): The directory to use for caching results. Default is "cache".
        """
        self.name = name
        self.dataset = None
        self.dataset_creator = DatasetCreator(rag_chain)
        self.llm_model = llm_model
        self.embeddings = embeddings
        self.eval_results = None
        self.ragas_metrics = [answer_correctness,
                              answer_relevancy,
                              answer_similarity,
                              context_precision,
                              context_entity_recall]

        self.cache_results = cache_results
        self.cache_dir = Path(cache_dir)
        if self.cache_results:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_default_ragas_run_config(self):
        return RunConfig(timeout=60, max_retries=10, max_wait=60, max_workers=16)

    def _cache_file_path(self, df):
        """
        Generate a file path for caching results based on DataFrame content.

        Args:
            df (pd.DataFrame): DataFrame to be evaluated.

        Returns:
            Path: File path for the cached result.
        """
        df_hash = pd.util.hash_pandas_object(df).sum()
        file_name = f"{self.name}_{df_hash}.pkl"
        return self.cache_dir / file_name

    def evaluate(self,
                 df_eval,
                 ragas_run_config=None,
                 raise_exceptions=False,
                 is_async=True,
                 verbose=True):
        """
        Evaluate a given DataFrame using the specified metrics and configurations.

        Args:
            df_eval (pd.DataFrame): The DataFrame to evaluate.
            ragas_run_config (RunConfig): The RunConfig object to use for the RAGAS evaluation.
            raise_exceptions (bool): Whether to raise exceptions during evaluation. Default is False.
            is_async (bool): Whether to run the evaluation asynchronously. Default is True.
            verbose (bool): Whether to print verbose logs. Default is True.

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation results.
        """
        if self.cache_results:
            cache_file = self._cache_file_path(df_eval)
            if cache_file.exists():
                self.eval_results = pd.read_pickle(cache_file)
                if verbose:
                    print(f"Loaded cached results from {cache_file}")
                return self.eval_results

        if self.dataset is None:
            self.dataset = self.dataset_creator.create_dataset_from_df(df_eval, verbose=verbose)

        print(len(df_eval), len(self.dataset))

        if ragas_run_config is None:
            ragas_run_config = self._get_default_ragas_run_config()

        eval_results = evaluate(self.dataset,
                                metrics=self.ragas_metrics,
                                embeddings=self.embeddings,
                                is_async=is_async,
                                raise_exceptions=raise_exceptions,
                                llm=self.llm_model,
                                run_config=ragas_run_config)

        eval_results = eval_results.to_pandas()
        eval_results = self.add_reasoning(eval_results, verbose=verbose)
        eval_results = self.add_reciprocal_rank(eval_results, verbose=verbose)

        self.eval_results = eval_results

        if self.cache_results:
            cache_file = self._cache_file_path(df_eval)
            self.eval_results.to_pickle(cache_file)
            if verbose:
                print(f"Cached results at {cache_file}")

        return self.eval_results

    def add_reasoning(self, eval_df, verbose=True):
        """
        Process the raw evaluation results to augment them with reasoning.

        Args:
            eval_df (pd.DataFrame): The DataFrame containing raw evaluation results.
            verbose (bool): Whether to print verbose logs. Default is True.

        Returns:
            pd.DataFrame: The processed evaluation DataFrame.
        """
        evaluator = load_evaluator(EvaluatorType.LABELED_SCORE_STRING, llm=self.llm_model)
        for i, row in tqdm(eval_df.iterrows(), desc="Adding reasoning", disable=not verbose, total=len(eval_df)):
            result = evaluator.evaluate_strings(prediction=row['answer'],
                                                input=row['question'],
                                                reference=row['contexts'])
            eval_df.at[i, 'reasoning'] = result['reasoning']
        return eval_df

    def add_reciprocal_rank(self, eval_df, verbose=True):
        """
        Compute the Reciprocal Rank (RR) for the evaluation DataFrame.

        Args:
            eval_df (pd.DataFrame): The DataFrame containing the evaluation results.
            verbose (bool): Whether to print verbose logs. Default is True.

        Returns:
            pd.DataFrame: The evaluation DataFrame augmented with the reciprocal ranks.
        """

        for i, row in tqdm(eval_df.iterrows(), desc="Computing MRR", disable=not verbose, total=len(eval_df)):
            retrieved_docs = row['contexts_origin_doc_ids']
            ground_truth_origin_doc_ids = row['best_match_id']

            indices = np.where(retrieved_docs == ground_truth_origin_doc_ids)[0]
            if indices.size > 0:
                rr = 1 / (indices[0] + 1)
            else:
                rr = 0

            eval_df.at[i, 'rr'] = rr

        return eval_df

    def summarize(self):
        """
        Summarize the evaluation results in plots and print the overall Mean Reciprocal Rank (MRR).

        Returns:
            None
        """
        if self.eval_results is None:
            raise ValueError("No evaluation results found. Run the evaluation first.")

        sns.set(style="whitegrid")

        ragas_metrics_data = (self.eval_results
                              .select_dtypes(include=[np.float64])
                              .drop('rr', axis=1))

        normalized_ragas_metrics_data = (
                (ragas_metrics_data - ragas_metrics_data.min()) /
                (ragas_metrics_data.max() - ragas_metrics_data.min()))

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=normalized_ragas_metrics_data, palette="Set2")
        plt.title('Comparison of All Normalized RAGAS Evaluation Metrics')
        plt.ylabel('Scores (Normalized)')
        plt.xlabel('Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        means = ragas_metrics_data.mean()
        # stds = ragas_metrics_data.std() # TODO: Add standard deviation to the plot

        plt.figure(figsize=(12, 6))
        sns.barplot(x=means.index, y=means, palette="Set2")
        plt.title('Mean of RAGAS Evaluation Metrics')
        plt.ylabel('Mean Scores')
        plt.xlabel('Metrics')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        mrr_overall = self.eval_results['rr'].mean()
        print(f"Overall Mean Reciprocal Rank (MRR): {mrr_overall:.3f}")
