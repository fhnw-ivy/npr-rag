import ast
import hashlib
import re

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from joblib import Parallel, delayed
from langdetect import detect, LangDetectException


def hash_string(s):
    s = str(s).lower().strip()
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8


class Preprocessor:
    def __init__(self, dataframe: pd.DataFrame, verbose=False) -> None:
        self.df = dataframe.copy()
        self.verbose = verbose

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
        if type(text) == list:
            text = ' '.join(text)
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

        self._remove_language()
        self._remove_html()
        self._remove_special_chars()
        self._remove_duplicate_chunks()

        self.df = self.df.groupby('Unnamed: 0').agg({'content': list,
                                                     'language': 'first',
                                                     'title': 'first',
                                                     'date': 'first',
                                                     'author': 'first',
                                                     'domain': 'first',
                                                     'url': 'first'}).reset_index()
        self._concatenate_contents()
        self._add_id()
        return self.df


class EvaluationPreprocessor:
    def __init__(self, default_df: pd.DataFrame, df_eval: pd.DataFrame, verbose: bool = False):
        self.default_df = default_df
        self.df_eval = df_eval
        self.verbose = verbose

    @staticmethod
    def compute_similarity(a, b):
        """Compute similarity score between two text strings using SequenceMatcher."""
        return fuzz.token_set_ratio(a, b)

    def _find_best_match(self, eval_chunk):
        """Find the best match in the default dataset for a given evaluation chunk."""
        best_match_score = 0
        best_match_index = None
        for i, doc in self.default_df.iterrows():
            score = self.compute_similarity(eval_chunk, doc['content'])
            if self.verbose:
                print(f"Comparing with {doc['id']}: Score = {score}")
            if score > best_match_score:
                best_match_score = score
                best_match_index = doc['id']
        return best_match_score, best_match_index

    def preprocess(self):
        """Process the evaluation DataFrame to find the best text matches in parallel."""
        df_eval = self.df_eval.drop_duplicates().copy()

        len_before = len(df_eval)
        try:
            df_eval['relevant_chunk'] = df_eval['relevant_chunk'].apply(eval).explode()
        except:
            print("Could not explode the DataFrame. Make sure the 'relevant_chunk' column is a list.")

        assert len_before == len(df_eval), "Expected the same number of rows after exploding the DataFrame."

        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(self._find_best_match)(chunk) for chunk in df_eval['relevant_chunk']
        )

        scores, match_ids = zip(*results)
        df_eval.loc[:, 'best_match_score'] = scores
        df_eval.loc[:, 'best_match_id'] = match_ids

        if 'answer' in df_eval.columns:
            print("Renaming 'answer' to 'ground_truth'.")
            df_eval.rename(columns={'answer': 'ground_truth'}, inplace=True)
        elif 'relevant_chunk' in df_eval.columns:
            print("Renaming 'relevant_chunk' to 'ground_truth'.")
            df_eval.rename(columns={'relevant_chunk': 'ground_truth'}, inplace=True)

        return df_eval
