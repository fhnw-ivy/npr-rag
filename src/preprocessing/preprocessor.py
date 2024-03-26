import pandas as pd
import ast
import numpy as np
from langdetect import detect, LangDetectException

class Preprocessor:
    def __init__(self, dataframe: pd.DataFrame, verbose=False) -> None:
        self.df = dataframe
        self.verbose = verbose
    
    def _remove_duplicate_chunks(self):
        n_rows = self.df.shape[0]
        self.df = self.df.drop_duplicates(subset=['content'], keep='first')

        if self.verbose:
            print(f'Dropped {n_rows-len(self.df)} duplicate chunks')
    
    def _remove_language(self):
        n_rows = self.df.shape[0]
        non_en_rows = self.df['language'] != 'en'
    
        self.df.loc[non_en_rows, 'language'] = self.df.loc[non_en_rows, 'content'].apply(self._safe_detect)
        self.df = self.df[self.df['language'] == 'en']
        
        if self.verbose:
            print(f'Dropped {n_rows - len(self.df)} non-english chunks')
        
    
    def _safe_detect(self, text):
        text = str(text)
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return np.nan
    
    def preprocess(self) -> pd.DataFrame:
        self.df['language'] = self.df['content'].apply(self._safe_detect)
        self.df['content'] = self.df['content'].apply(ast.literal_eval)
        self.df = self.df.explode('content')
        
        self._remove_language()
        self._remove_duplicate_chunks()
        
        return self.df