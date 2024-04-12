import pandas as pd
from joblib import Parallel, delayed
from difflib import SequenceMatcher


class EvaluationSetPreprocessor:
    def __init__(self, default_df: pd.DataFrame, df_eval: pd.DataFrame, verbose: bool = False):
        self.default_df = default_df
        self.df_eval = df_eval
        self.verbose = verbose

    @staticmethod
    def compute_similarity(a, b):
        """Compute similarity score between two text strings using SequenceMatcher."""
        return SequenceMatcher(None, a, b).ratio()

    def _find_best_match(self, eval_chunk):
        """Find the best match in the default dataset for a given evaluation chunk."""
        best_match_score = 0
        best_match_index = None
        for i, row in self.default_df.iterrows():
            score = self.compute_similarity(eval_chunk, row['content'])
            if score > best_match_score:
                best_match_score = score
                best_match_index = row['id']
        return best_match_score, best_match_index

    def preprocess(self):
        """Process the evaluation DataFrame to find the best text matches in parallel."""

        len_before = len(self.df_eval)
        try:
            self.df_eval['relevant_chunk'] = self.df_eval['relevant_chunk'].apply(eval).explode()
        except:
            pass

        assert len_before == len(self.df_eval), "Expected the same number of rows after exploding the DataFrame."

        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(self._find_best_match)(chunk) for chunk in self.df_eval['relevant_chunk']
        )

        self.df_eval['best_match_score'], self.df_eval['best_match_id'] = zip(*results)
        return self.df_eval
