import re
import pandas as pd
import ast
import numpy as np
from langdetect import detect, LangDetectException
import hashlib


def hash_string(s):
    s = str(s).lower().strip()
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8


class Preprocessor:
    def __init__(self, dataframe: pd.DataFrame, explode=True, verbose=False, concatenate_contents=False) -> None:
        self.df = dataframe.copy()
        self.explode = explode
        self.verbose = verbose
        self.concatenate_contents = concatenate_contents

        for col in ['content', 'title', 'date', 'author', 'domain', 'url']:
            assert col in self.df.columns, f'Column {col} not found in dataframe'

    def _add_id(self):
        self.df['id'] = self.df['content'].apply(lambda x: hash_string(x))

        if self.verbose:
            print(f'Added unique id to each chunk. Duplicate ids: {self.df["id"].duplicated().sum()}')

    def _remove_duplicate_chunks(self):
        n_rows = self.df.shape[0]
        self.df = self.df.drop_duplicates(subset=['content'], keep='first')

        if self.verbose:
            print(f'Dropped {n_rows - len(self.df)} duplicate chunks')

    def _remove_language(self):
        n_rows = self.df.shape[0]
        non_en_rows = self.df['language'] != 'en'

        self.df.loc[non_en_rows, 'language'] = self.df.loc[non_en_rows, 'content'].apply(self._safe_detect)
        self.df = self.df[self.df['language'] == 'en']

        if self.verbose:
            print(f'Dropped {n_rows - len(self.df)} non-english chunks')

    @staticmethod
    def _safe_detect(text):
        text = str(text)
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return np.nan

    def _remove_html(self):
        self.df['content'] = self.df['content'].apply(self._clean_html)

    def _remove_special_chars(self):
        self.df['content'] = self.df['content'].apply(self.clean_special_chars)

    @staticmethod
    def _clean_html(text):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', text)
        return cleantext

    @staticmethod
    def clean_special_chars(text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    def _concatenate_contents(self):
        self.df['content'] = self.df['content'].apply(lambda x: ' '.join(x))

    def preprocess(self) -> pd.DataFrame:
        self.df['language'] = self.df['content'].apply(self._safe_detect)
        self.df['content'] = self.df['content'].apply(ast.literal_eval)

        self._add_id()

        if self.explode:
            self.df = self.df.explode('content')

        self._remove_language()
        self._remove_html()
        self._remove_special_chars()
        self._remove_duplicate_chunks()

        if not self.explode:
            self.df = self.df.groupby('Unnamed: 0').agg({'content': list,
                                                         'id': 'first',
                                                         'language': 'first',
                                                         'title': 'first',
                                                         'date': 'first',
                                                         'author': 'first',
                                                         'domain': 'first',
                                                         'url': 'first'}).reset_index()

        if self.concatenate_contents:
            self._concatenate_contents()

        return self.df
