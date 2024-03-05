from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from src.preprocessing.base_processor import BaseProcessor
from src.vectorstore import VectorStore

from langchain_openai import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

class EmbeddingStrategy:

    def __init__(self, embedding_model: Embeddings, processor: BaseProcessor, vector_store: VectorStore, retriever: BaseRetriever=None):
        self.embedding_model = embedding_model
        self.processor = processor
        self.vector_store = vector_store
        self.retriever = vector_store.get_retriever() if retriever is None else retriever

    @staticmethod
    def get_default_strategy():
        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )

        vector_store = VectorStore(embedding_function=hf,
                                   collection="cleantech-bge-small-en")

        return EmbeddingStrategy(embedding_model=hf, processor=BaseProcessor(hf), vector_store=vector_store)

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

        # Configure retriver
        llm = OpenAI(temperature=0)
        retriever = SelfQueryRetriever.from_llm(
            llm, vector_store.vector_store, document_content_description, metadata_field_info, verbose=True
        )


        return EmbeddingStrategy(embedding_model=hf, processor=BaseProcessor(hf), vector_store=vector_store, retriever=retriever)
