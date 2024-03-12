from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAI

from src.preprocessing.base_processor import BaseProcessor
from src.vectorstore import VectorStore


class EmbeddingStrategy:
    def __init__(self,
                 name: str,
                 version: int,
                 embedding_model: Embeddings,
                 processor: BaseProcessor,
                 vector_store: VectorStore,
                 retriever: BaseRetriever = None):
        self.name = name
        self.version = version

        self.embedding_model = embedding_model
        self.processor = processor
        self.vector_store = vector_store
        self.retriever = vector_store.get_retriever() if retriever is None else retriever

    def get_version_string(self) -> str:
        assert self.version is not None
        assert self.name is not None
        return f"{self.name}_v{self.version}"

    def get_info(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "embedding_model": self.embedding_model.__class__.__name__,
            "processor": self.processor.__class__.__name__,
            "vector_store": self.vector_store.__class__.__name__,
            "retriever": self.retriever.__class__.__name__
        }

    @staticmethod
    def get_default_strategy():
        fake_embeddings = FastEmbedEmbeddings()

        vector_store = VectorStore(embedding_function=fake_embeddings,
                                   collection="cleantech-bge-small-en-fake")

        return EmbeddingStrategy(
            name="DefaultStrategy",
            version=1,
            embedding_model=fake_embeddings,
            processor=BaseProcessor(fake_embeddings),
            vector_store=vector_store)

    @staticmethod
    def get_bge_strategy():
        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}

        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )

        # Metadata schema based on the values on the CSV
        vector_store = VectorStore(embedding_function=hf,
                                   collection="cleantech-bge-small-en")

        return EmbeddingStrategy(
            name="BGEStrategy",
            version=1,
            embedding_model=hf,
            processor=BaseProcessor(hf),
            vector_store=vector_store
        )

    @staticmethod
    def get_custom_strategy():
        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}

        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )

        # Metadata schema based on the values on the CSV
        metadata_field_info = [
            AttributeInfo(
                name="url",
                description="Url of the document",
                type="string",
            ),
            AttributeInfo(
                name="title",
                description="Title of the document",
                type="string",
            ),
            AttributeInfo(
                name="date",
                description="Date of the document",
                type="string",
            ),
            AttributeInfo(
                name="author",
                description="Author of the document",
                type="string",
            ),
            AttributeInfo(
                name="domain",
                description="Domain of the document, closely related to the source of the document",
                type="string",
            ),
        ]
        document_content_description = "Part of text content of article, blog, or other"

        vector_store = VectorStore(embedding_function=hf,
                                   collection="cleantech-bge-small-en")

        llm = OpenAI(temperature=0)
        retriever = SelfQueryRetriever.from_llm(
            llm, vector_store.vector_store, document_content_description, metadata_field_info, verbose=True
        )

        return EmbeddingStrategy(
            name="CustomStrategy",
            version=1,
            embedding_model=hf,
            processor=BaseProcessor(hf),
            vector_store=vector_store,
            retriever=retriever)
