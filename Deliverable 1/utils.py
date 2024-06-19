import re
import os
import nltk
import itertools
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from functools import lru_cache
from nltk.corpus import stopwords
from scipy.sparse.csr import csr_matrix
from nltk import PorterStemmer, word_tokenize
from typing import Dict, Callable, Iterable, Tuple, List, Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

if not os.path.isdir(f"{os.path.expanduser('~')}/nltk_data/corpora/stopwords"):
    nltk.download('stopwords')

if not os.path.isdir(f"{os.path.expanduser('~')}/nltk_data/corpora/punkt"):
    nltk.download('punkt')

__stemmer = PorterStemmer()

# auxilliary function

# remove samples that contain a NaN in at least one question
def remove_nan_questions(x_train: pd.DataFrame, y_train: pd.DataFrame) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    dropped_x_train = x_train.dropna(how="any")
    idx = set(x_train.index).intersection(dropped_x_train.index)
    dropped_y_train = y_train.loc[list(idx)]
    return dropped_x_train, dropped_y_train

# aggregating feature vectors 

# horizontally stack the feature matrices
def _horizontal_stacking(x_q1: csr_matrix, x_q2: csr_matrix) -> csr_matrix:
    return hstack((x_q1, x_q2))

def _cosine_similarity(x_q1: csr_matrix, x_q2: csr_matrix) -> csr_matrix:
    _cos_sim = x_q1.tocsr() * x_q2.tocsr().T
    return _cos_sim

def _abs_difference(x_q1: csr_matrix, x_q2: csr_matrix) -> csr_matrix:
    return np.abs(x_q1 - x_q2)

# preprocessing questions

def _remove_punctuation(text: str):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        text = np.char.replace(text, symbols[i], ' ')
        text = np.char.replace(text, "  ", " ")
    text = np.char.replace(text, ',', '')
    return text

def _remove_stop_words(text: str, stop_words: Iterable[str]) -> str:
    return ' '.join([word for word in text.split() if word not in stop_words])

def _stemming(text: str) -> str:
    return " ".join([__stemmer.stem(w) for w in word_tokenize(text)])

def _to_british(text: str) -> str:
    text = re.sub(r"(...)our\b", r"\1or", text)
    text = re.sub(r"([bt])re\b", r"\1er", text)
    text = re.sub(r"([iy])s(e\b|ing|ation)", r"\1z\2", text)
    text = re.sub(r"ogue\b", "og", text)
    return text

# generating extra features 

# return the question's length ratio (first with respect to the second)
def _length_ratio(q1_w: List[str], q2_w: List[str]) -> float:
    if any([len(_q) for _q in (q1_w, q2_w)]):
        return 0.
    return len(q1_w) / len(q2_w)

# count the ratio of coincident words with respect the total number 
def _get_coincident_words_ratio(q1_w: List[str], q2_w: List[str]) -> float:
    unique_q1, unique_q2 = set(q1_w), set(q2_w)
    return 2 * len(unique_q1 & unique_q2) / (len(unique_q1) + len(unique_q2))

# identify whether the 2 questions have coincident keywords and return a proportional normalized value
def _coincident_keyword(q1_w: List[str], q2_w: List[str]) -> float:
    
    keywords = {"What", "what", "Who", "who", "Which", "which", "Where",
                "where", "Why", "why", "When", "when", "How", "how", "Whose",
                "whose", "Can", "can"}
    
    unique_q1, unique_q2 = set(q1_w), set(q2_w)
    keywords_q1 = unique_q1 & keywords 
    keywords_q2 = unique_q2 & keywords

    if len(keywords_q1) == 0 and len(keywords_q2) == 0:
        denominator = 1
    else:
        denominator = (len(keywords_q1) + len(keywords_q2))

    return 2*len(keywords_q1 & keywords_q2)/denominator


def _jaccard_distance(q1_w: List[str], q2_w: List[str]) -> float:
    unique_q1, unique_q2 = set(q1_w), set(q2_w)
    return 1 - len(unique_q1 & unique_q2) / len(unique_q1.union(unique_q2))


def _levenshtein_sim_w(q1_w: List[str], q2_w: List[str]) -> float:
    @lru_cache(None)  
    def min_dist(s1, s2):

        if s1 == len(q1_w) or s2 == len(q2_w):
            return len(q1_w) - s1 + len(q2_w) - s2

        # no change required
        if q1_w[s1] == q2_w[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),      # insert character
            min_dist(s1 + 1, s2),      # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)

# extractors and aggregators 

_SUPPORTED_EXTRACTORS: dict = {
    'cv': CountVectorizer(
        ngram_range=(1, 1), lowercase=False),
    'cv_2w': CountVectorizer(
        ngram_range=(1, 2), lowercase=False),
    'tf_idf': TfidfVectorizer(ngram_range=(1, 1), lowercase=False),
    'tf_idf_2w': TfidfVectorizer(ngram_range=(1, 2), lowercase=False),
}

_SUPPORTED_AGGREGATORS: Dict[str, Callable] = {
    'stack': _horizontal_stacking,
    'absolute': _abs_difference,
}

_SUPPORTED_EXTRA_FEATURES: Dict[str, Callable] = {
    'coincident_ratio': _get_coincident_words_ratio,
    'coincident_keyword': _coincident_keyword,
    'jaccard': _jaccard_distance,
    'levenshtein_w': _levenshtein_sim_w,
    'length_ratio': _length_ratio,
}

# pipeline classes

class TextPreprocessor:
    def __init__(self,
                 remove_stop_words: bool = False,
                 remove_punctuation: bool = False,
                 to_lower: bool = False,
                 apply_stemming: bool = False,
                 british: bool = False) \
            -> None:
        self.remove_stop_words = remove_stop_words
        self.remove_punctuation = remove_punctuation
        self.to_lower = to_lower
        self.apply_stemming = apply_stemming
        self.british = british

        self.custom_stop_words: set = {
            "What", "what", "Who", "who", "Which",
            "which", "Where", "where", "Why", "why",
            "When", "when", "How", "how", "Whose", "whose", "Can"}
        self.stop_words = set(stopwords.words(
            'english')) - self.custom_stop_words

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):

        df_prep = df.copy()

        # preprocessing
        if self.to_lower:
            df_prep['question1'] = df_prep['question1'].str.lower()
            df_prep['question2'] = df_prep['question2'].str.lower()

        # apply functions to the 'text' column of the DataFrame
        if self.remove_stop_words:
            df_prep.loc[:, 'question1'] = df_prep['question1'].apply(
                lambda _t: _remove_stop_words(_t, self.stop_words))
            df_prep.loc[:, 'question2'] = df_prep['question2'].apply(
                lambda _t: _remove_stop_words(_t, self.stop_words))
        if self.remove_punctuation:
            df_prep.loc[:, 'question1'] = df_prep['question1'].apply(
                lambda _t: _remove_punctuation(_t))
            df_prep.loc[:, 'question2'] = df_prep['question2'].apply(
                lambda _t: _remove_punctuation(_t))
        if self.british:
            df_prep.loc[:, 'question1'] = df_prep['question1'].apply(
                lambda _t: _to_british(_t))
            df_prep.loc[:, 'question2'] = df_prep['question2'].apply(
                lambda _t: _to_british(_t))
        if self.apply_stemming:
            df_prep.loc[:, 'question1'] = df_prep['question1'].apply(
                lambda _t: _stemming(_t))
            df_prep.loc[:, 'question2'] = df_prep['question2'].apply(
                lambda _t: _stemming(_t))

        return df_prep

class FeatureGenerator:
    def __init__(self,
                 exts: Tuple = ('cv', ),
                 aggs: Tuple = ('stack', ),
                 extra_features: Tuple[str] = 'all') -> None:
        assert len(exts) == len(aggs), \
            "Extractor and aggregator lists must be of the same length"
        self.extractors = [_get_extractor(ext) for ext in exts]
        self.extractor_names: Tuple[str] = exts
        self.aggregators = [_SUPPORTED_AGGREGATORS[agg] for agg in aggs]
        self.extra_features_creator = ExtraFeaturesCreator(extra_features)

    def set_params(self,
                   exts: Tuple = ('cv', ),
                   aggs: Tuple = ('stack', ),
                   extra_features: Tuple[str] = 'all') -> None:
        self.__class__(exts, aggs, extra_features)

    def fit(self, questions_df: pd.DataFrame, y: pd.DataFrame = None):
        self.extractors = [ext if name.startswith('spacy') else ext.fit(
            questions_df.values.flatten()) for name, ext in zip(
            self.extractor_names, self.extractors)]
        return self

    def transform(self, questions_df: pd.DataFrame, y: pd.DataFrame = None):
        agg_features = []
        for name, ext, agg in zip(self.extractor_names, self.extractors,
                                  self.aggregators):
            # apply the extractor to each question
            if name.startswith('spacy'):
                print("Using spacy word embedding: please WAIT, "
                      "this may take some time")
                # use a spacy embedding
                x_q1 = questions_df.iloc[:, 0].apply(
                    lambda x: ext(x).vector)
                x_q2 = questions_df.iloc[:, 1].apply(
                    lambda x: ext(x).vector)

            else:
                x_q1 = ext.transform(questions_df.iloc[:, 0])
                x_q2 = ext.transform(questions_df.iloc[:, 1])

            # aggregate them
            x_agg = agg(x_q1, x_q2)
            agg_features.append(x_agg)

        if len(self.extra_features_creator.features_functions) != 0:
            # compute the extra features
            x_extra: np.ndarray = self.extra_features_creator.transform(
                questions_df)
            # merge them
            return hstack((hstack(agg_features), x_extra))
        return hstack(agg_features)

def _get_extractor(ext: str):
    if ext in ['spacy_small', 'spacy_medium']:
        _spacy_version: str = _SUPPORTED_EXTRACTORS[ext]
        try:
            import spacy # type: ignore
            spacy.load(_spacy_version)
        except OSError:
            os.system(f'python -m spacy download {_spacy_version}')
        finally:
            import spacy # type: ignore
            return spacy.load(_spacy_version)
    else:
        return _SUPPORTED_EXTRACTORS[ext]

class ExtraFeaturesCreator:
    def __init__(self, features_to_add: Union[Tuple, str]) -> None:
        if isinstance(features_to_add, str):
            assert features_to_add == 'all', "Unrecognized extra features list"
            self.features_functions: Dict[str, callable] = \
                _SUPPORTED_EXTRA_FEATURES
        else:
            self.features_functions = {
                _n: _SUPPORTED_EXTRA_FEATURES[_n] for _n in features_to_add}

    def transform(self, questions_df: pd.DataFrame) -> np.ndarray:
        if len(self.features_functions) == 0:
            raise ValueError("There is no extra features to be aggregated")

        for _c in ('question1', 'question2'):
            questions_df[_c] = questions_df[_c].str.split()
        extra_features = pd.DataFrame()

        for _f_name, _f_function in self.features_functions.items():
            extra_features[_f_name] = questions_df.apply(
                lambda x: _f_function(x.question1, x.question2), axis=1)

        return extra_features.values

# grid search functions

def get_param_grid(name: str, seed: int):
    
    preprocessor_grid = {
        "preprocessor__remove_stop_words": [True, False],
        "preprocessor__remove_punctuation": [True, False],
        "preprocessor__to_lower": [True, False],
        "preprocessor__apply_stemming": [True, False],
        "preprocessor__british": [True, False],
    }

    exts = list(_SUPPORTED_AGGREGATORS.keys())
    aggs = list(_SUPPORTED_AGGREGATORS.keys())
    combinations_exts = [list(comb) for i in range(
        1, len(exts) + 1) for comb in itertools.combinations(exts, i)]
    combinations_aggs = [list(comb) for i in range(
        1, len(aggs) + 1) for comb in itertools.combinations(aggs, i)]

    generator_grid = {
        "generator__ext": combinations_exts,
        "generator__agg": combinations_aggs,
        "generator__extra_features": list(_SUPPORTED_EXTRA_FEATURES.keys())
    }

    classifier_grid = {
        "LogisticRegression": {
            "classifier__random_state": [seed],
            "classifier__penalty": ["l2"],
            "classifier__C": [0.01, 0.1, 1],  # , 10, 100]
        },
        "RandomForestClassifier": {
            "classifier__random_state": [seed],
            "classifier__n_estimators": [10, 50, 100],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__bootstrap": [True, False],
        },
        "SVC": {
            "classifier__random_state": [seed],
            "classifier__kernel": ["linear", "poly", "rbf", "sigmoid"],
            "classifier__C": [0.01, 0.1, 1, 2],
            "classifier__gamma": ["scale", "auto"],
        },
        "KNeighborsClassifier": {
            "classifier__n_neighbors": [3, 5, 7],
            "classifier__weights": ["uniform", "distance"],
        },
        "GradientBoostingClassifier": {
            "classifier__random_state": [seed],
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [3, 5, None],
            "classifier__learning_rate": [0.01, 0.1, 1],
            "classifier__subsample": [0.5, 0.8, 1.0],
        },
        "AdaBoostClassifier": {
            "classifier__random_state": [seed],
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.1, 1],
            "classifier__algorithm": ["SAMME", "SAMME.R"],
        },
        "GaussianNB": {},
        "BernoulliNB": {},
        "LinearDiscriminantAnalysis": {
            "classifier__solver": ["svd", "lsqr", "eigen"],
            "classifier__shrinkage": [None, "auto", 0.1, 0.5, 0.9],
            "classifier__n_components": [None, 1],
            "classifier__store_covariance": [True, False],
            "classifier__tol": [1e-4, 1e-3, 1e-2],
        },
        "QuadraticDiscriminantAnalysis": {
            "classifier__reg_param": [0.0, 0.1, 0.5, 1.0],
            "classifier__store_covariance": [True, False],
            "classifier__tol": [1e-4, 1e-3, 1e-2],
        },
    }
    return preprocessor_grid | generator_grid | classifier_grid[name]