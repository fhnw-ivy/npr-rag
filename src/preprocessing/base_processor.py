import langchain_community
import pandas as pd
from ast import literal_eval

from langchain_core.embeddings import Embeddings
from pandas import DataFrame
from pydantic.v1 import BaseModel


class BaseProcessor:

    def __init__(self, embedding_model: (BaseModel, Embeddings), batch_size: int = 41666, *args, **kwargs):
        self.batch_size = batch_size
        self.embedding_model = embedding_model

    @staticmethod
    def clean(dataframe: DataFrame):
        dataframe['content'] = dataframe['content'].apply(literal_eval)
        return dataframe.explode('content')

    def preprocess_documents(self, documents):
        # Placeholder for common preprocessing steps
        return documents

    def chunk_documents(self, documents):
        # Placeholder for a generic chunking strategy
        return documents

    def clean_text(self, text):
        # Placeholder for generic text cleaning

        return text
