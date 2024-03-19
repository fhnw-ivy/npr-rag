#!/usr/bin/env python
"""
Description:
This script generates a test set from documents. When run from the CLI, it processes a dataset from a specified path,
including embedding and chunking.

Example Usage:
CLI: python scripts/generate_testset.py --dataset_path "<path_to_csv>" --testset_path "<path_to_save_testset.csv>" --testset_size 1000 --verbose
"""

import argparse
import logging

import pandas as pd
from datasets import Dataset
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.testset.generator import TestsetGenerator
from tqdm import tqdm

from src.embedding_strategy import EmbeddingStrategy
from src.generation import get_llm_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_dataset(dataset_path, strategy, verbose):
    """Processes a dataset from a CSV file, including embedding and chunking given a strategy."""
    logging.info(f"Processing dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    documents = []

    for index, row in tqdm(df.iterrows(), disable=not verbose):
        content = row['content']
        row = row.fillna('')
        metadata = {
            "url": row['url'],
            "domain": row['domain'],
            "title": row['title'],
            "author": row['author'],
            "date": row['date']
        }
        row_docs = strategy.processor.chunk(content, metadata)
        documents += row_docs

    logging.info(f"Processed {len(documents)} documents.")
    return documents


def generate_testset(documents: list[Document],
                     generator_llm: BaseChatModel,
                     critic_llm: BaseChatModel,
                     embeddings: Embeddings,
                     test_size: int = 10,
                     distributions=None,
                     verbose=False) -> Dataset:
    """Generates a testset based on given documents and distribution."""
    if distributions is None:
        distributions = {simple: 0.6, reasoning: 0.3, multi_context: 0.1}
    else:
        assert sum(distributions.values()) == 1
        assert all([d in [simple, reasoning, multi_context] for d in distributions.keys()])

    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)
    return generator.generate_with_langchain_docs(documents,
                                                  test_size=test_size,
                                                  distributions=distributions,
                                                  with_debugging_logs=verbose)


def generate_testset_and_save(documents, strategy, testset_path, testset_size, verbose):
    generator_llm, critic_llm = get_llm_model(), get_llm_model()  # Get default LLMs
    # get total characters for documents
    total_chars = sum([len(doc.page_content) for doc in documents]) + sum([len(doc.metadata) for doc in documents])

    logging.info(f"Total characters in documents: {total_chars}")

    for i in range(0, len(documents), 100):
        batch = documents[i:i+100]
        testset = generate_testset(
            documents=batch,
            generator_llm=generator_llm,
            critic_llm=critic_llm,
            embeddings=strategy.embedding_model,
            test_size=testset_size,
            verbose=verbose
        )

        testset_df = testset.to_pandas()
        testset_df.to_csv(testset_path, index=False, mode='a', header=False)


def main_cli(dataset_path, testset_path, testset_size, verbose):
    strategy = EmbeddingStrategy.get_default_strategy()
    documents = process_dataset(dataset_path, strategy, verbose)
    generate_testset_and_save(documents, strategy, testset_path, testset_size, verbose)
    logging.info(f"Generated testset. Saved to {testset_path}")


if __name__ == "__main__":
    print("This process may be very costly in terms of money. Are you sure you want to proceed?")
    user_input = input("Type 'yes' to proceed: ")
    if user_input.lower() != "yes":
        print("Aborted.")
        exit(0)

    parser = argparse.ArgumentParser(description="Generate a test set from a dataset or pre-processed documents.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset CSV file")
    parser.add_argument("--testset_path", required=True, help="Path to save the generated testset CSV")
    parser.add_argument("--testset_size", type=int, default=50, help="Size of the testset to generate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if args.dataset_path:
        main_cli(args.dataset_path, args.testset_path, args.testset_size, args.verbose)
    else:
        logging.error("Dataset path is required when running from CLI.")
