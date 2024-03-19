import logging

from datasets import Dataset
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.testset.generator import TestsetGenerator

from src.generation import get_llm_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_testset(documents: list[Document],
                     generator_llm: BaseChatModel,
                     critic_llm: BaseChatModel,
                     embeddings: Embeddings,
                     test_size: int = 10,
                     distributions=None,
                     verbose=False) -> Dataset:
    """Generates a testset based on given documents and distribution."""
    if distributions is None:
        distributions = {simple: 0.5, reasoning: 0.25, multi_context: 0.25}
    else:
        assert sum(distributions.values()) == 1
        assert all([d in [simple, reasoning, multi_context] for d in distributions.keys()])

    generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)
    return generator.generate_with_langchain_docs(documents,
                                                  test_size=test_size,
                                                  distributions=distributions,
                                                  with_debugging_logs=verbose)


def generate_testset_and_save(documents, embeddings, testset_path, testset_size, verbose):
    generator_llm, critic_llm = get_llm_model(), get_llm_model()  # Get default LLMs

    testset = generate_testset(
        documents=documents[:5],
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embeddings,
        test_size=testset_size,
        verbose=verbose
    )

    testset_df = testset.to_pandas()
    testset_df.to_csv(testset_path, index=False)
