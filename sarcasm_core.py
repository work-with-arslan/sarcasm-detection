"""
sarcasm_core.py
================
Assignment functionality module – all ML logic, zero Streamlit dependencies.
Contains every task from the assignment as clearly labelled functions/classes.

Tasks implemented:
  - Setup         : derive_seed_config()
  - Task 1        : basic_clean(), apply_negation(), remove_stopwords(),
                    generate_ngrams(), preprocess(), preprocess_trace(),
                    build_vocabulary()
  - Task 2        : compute_document_frequencies(), build_count_matrix(),
                    compute_tfidf()
  - Task 3        : MultinomialNB, BernoulliNB
  - Task 4        : extract_numeric_features(), inject_missing_values(),
                    mean_impute(), class_conditional_impute()
  - Task 5        : detect_outliers_iqr(), clamp_outliers()
  - Task 6        : MinMaxScaler, StandardScaler, GaussianNB
  - Task 7        : random_oversample(), tune_threshold()
  - Task 8        : get_misclassified()
  - Pipeline      : run_pipeline()  – runs all tasks end-to-end

Usage:
    from sarcasm_core import run_pipeline
    results = run_pipeline("Sarcasm_Headlines_Dataset.json", student_id=277211)
"""

import json
import re
import math
import random
import warnings
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# SETUP – Seed & Config Derivation
# ══════════════════════════════════════════════════════════════════════════════

def derive_seed_config(student_id: int) -> dict:
    """
    Derive all experiment parameters from the student ID.

    seed          = student_id % 100000
    smoothing set = determined by seed % 3
    k (ablation)  = seed % 4
    t (rare-word) = (seed % 6) + 2

    Returns
    -------
    dict with keys: seed, alphas, k, t
    """
    seed = student_id % 100000
    np.random.seed(seed)
    random.seed(seed)

    sm = seed % 3
    if sm == 0:
        alphas = [0, 0.1, 1]
    elif sm == 1:
        alphas = [0, 0.5, 2]
    else:
        alphas = [0, 1, 5]

    k = seed % 4
    t = (seed % 6) + 2

    return dict(seed=seed, alphas=alphas, k=k, t=t)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 – Tokenisation & Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

# --- Constants ---

NEGATION_WORDS = {'not', 'no', 'never', "don't", "didn't", "isn't", "wasn't"}

STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'that', 'this', 'these', 'those', 'it', 'its', 'than', 'then', 'so',
    'up', 'out', 'about', 'into', 'after', 'before', 'between', 'through',
    'he', 'she', 'they', 'we', 'you', 'i', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'our', 'their', 'what', 'which', 'who', 'how',
    'all', 'also', 'just', 'more', 'most', 'other', 'over', 'same',
    'own', 'under', 'again', 'further', 'once', 'here', 'there', 'when',
    'where', 'why', 'both', 'each', 'few', 'very', 'too', 'while',
}


def count_vowels(word: str) -> int:
    """Return the number of vowels in a word (used for negation window size)."""
    return sum(1 for ch in word.lower() if ch in 'aeiou')


# Task 1 – Step 1: Basic Cleaning
def basic_clean(text: str) -> str:
    """
    Step 1 – Basic Cleaning:
    - Lowercase
    - Strip HTML tags
    - Substitute tonal markers:  !! → [EXC_SEQ], ?? → [QUE_SEQ],
                                  ?!/!? → [INT_MARK], ... → [ELLIP]
    - Remove remaining punctuation (apostrophes inside words preserved)
    """
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Tonal marker substitution (order matters)
    text = re.sub(r'[?!]*[!][?!]*[!][?!]*', '[EXC_SEQ]', text)   # !! or more
    text = re.sub(r'[?][?]+', '[QUE_SEQ]', text)                   # ?? or more
    text = re.sub(r'([?][!]+|[!][?]+)', '[INT_MARK]', text)        # ?! or !?
    text = re.sub(r'\.{3,}', '[ELLIP]', text)                      # ...
    # Keep apostrophes inside words, remove everything else non-word
    text = re.sub(r"(?<!\w)'(?!\w)|(?<=\w)'(?!\w)|(?<!\w)'(?=\w)", ' ', text)
    text = re.sub(r"[^\w\s'\[\]]", ' ', text)
    return text


# Task 1 – Step 2: Negation Handling
def apply_negation(tokens: list) -> list:
    """
    Step 2 – Negation Handling:
    For every negation word, count its vowels (N).
    Prefix the next N tokens with NOT_.
    e.g. "not" (1 vowel) → prefix next 1 token
         "never" (2 vowels) → prefix next 2 tokens
    NOT_-prefixed tokens are immune to stopword removal.
    """
    result, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in NEGATION_WORDS:
            result.append(tok)
            n_vowels = count_vowels(tok)
            i += 1
            for _ in range(n_vowels):
                if i < len(tokens):
                    result.append('NOT_' + tokens[i])
                    i += 1
        else:
            result.append(tok)
            i += 1
    return result


# Task 1 – Step 3: Stopword Removal
def remove_stopwords(tokens: list) -> list:
    """
    Step 3 – Stopword Removal:
    Remove tokens in STOPWORDS, BUT preserve any token starting with NOT_.
    Stopword removal is applied AFTER negation handling.
    """
    return [t for t in tokens if t.startswith('NOT_') or t not in STOPWORDS]


# Task 1 – Step 4: N-gram Generation
def generate_ngrams(tokens: list) -> list:
    """
    Step 4 – N-gram Generation:
    Append bigrams (unigram_unigram) to the token list.
    Returns unigrams + bigrams combined.
    """
    bigrams = ['_'.join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
    return tokens + bigrams


# Task 1 – Full Pipeline
def preprocess(text: str, vocab: set = None) -> list:
    """
    Full preprocessing pipeline (Steps 1–5):
    clean → tokenise → negation → stopwords → n-grams → OOV mapping.
    If vocab is provided, tokens not in vocab are replaced with <UNK>.
    """
    text = basic_clean(text)
    tokens = re.findall(r"[\w'\[\]]+", text)  # Step 1 tokenise
    tokens = apply_negation(tokens)            # Step 2
    tokens = remove_stopwords(tokens)          # Step 3
    tokens = generate_ngrams(tokens)           # Step 4
    # Step 5 – OOV handling
    if vocab is not None:
        tokens = [t if t in vocab else '<UNK>' for t in tokens]
    return tokens


def preprocess_trace(text: str, vocab: set = None) -> dict:
    """
    Same as preprocess() but returns intermediate token lists at each step.
    Useful for the Streamlit preprocessing explorer.
    Returns dict with keys: cleaned, raw, neg, sw, ng, final
    """
    cleaned = basic_clean(text)
    raw     = re.findall(r"[\w'\[\]]+", cleaned)
    neg     = apply_negation(raw)
    sw      = remove_stopwords(neg)
    ng      = generate_ngrams(sw)
    final   = [t if (vocab is None or t in vocab) else '<UNK>' for t in ng]
    return dict(cleaned=cleaned, raw=raw, neg=neg, sw=sw, ng=ng, final=final)


def build_vocabulary(token_lists: list) -> set:
    """
    Task 1 – Step 5 (OOV):
    Build vocabulary from training token lists only.
    Always includes the <UNK> sentinel.
    """
    vocab = set(tok for toks in token_lists for tok in toks)
    vocab.add('<UNK>')
    return vocab


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 – Manual Feature Representation
# ══════════════════════════════════════════════════════════════════════════════

def _is_prime(n: int) -> bool:
    """Return True if n is a prime number (used for prime-index feature scheme)."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def compute_document_frequencies(train_token_lists: list) -> dict:
    """
    Task 2 – Compute Document Frequency (DF) for each token.
    DF(token) = number of training documents in which token appears ≥ once.
    Returns a dict: {token: df_count}
    """
    df_counts = defaultdict(int)
    for toks in train_token_lists:
        for tok in set(toks):
            df_counts[tok] += 1
    return dict(df_counts)


def build_count_matrix(
    token_lists: list,
    df_counts: dict,
    t: int,
    _state: dict = None,
) -> tuple:
    """
    Task 2A – Count Vectoriser with Prime-Index Feature Scheme.

    Steps:
    1. Remove rare tokens (DF < t)
    2. Sort remaining tokens alphabetically
    3. Tokens at prime positions (2,3,5,7,...) → individual columns
    4. Tokens at non-prime positions → aggregated into column 0
    5. Build document-term matrix

    Parameters
    ----------
    token_lists : list of token lists (train / val / test)
    df_counts   : dict from compute_document_frequencies()
    t           : minimum document frequency threshold
    _state      : if provided, reuse the prime/non-prime mappings from a
                  previous call (so val/test use the same column layout as train)

    Returns
    -------
    mat    : np.ndarray shape (n_docs, n_features)
    state  : dict with keys tok2col, non_prime_set, n_feat, prime_toks,
             non_prime_set_len – pass as _state for val/test matrices
    """
    if _state is None:
        # Build column mappings from df_counts (training set only)
        filtered = sorted([tok for tok, cnt in df_counts.items() if cnt >= t])
        prime_toks    = [filtered[i - 1] for i in range(1, len(filtered) + 1) if _is_prime(i)]
        non_prime_set = set(filtered[i - 1] for i in range(1, len(filtered) + 1) if not _is_prime(i))
        tok2col = {tok: idx + 1 for idx, tok in enumerate(prime_toks)}
        n_feat  = len(prime_toks) + 1   # col 0 = non-prime aggregate
        state = dict(
            tok2col=tok2col, non_prime_set=non_prime_set,
            n_feat=n_feat, prime_toks=prime_toks,
            non_prime_set_len=len(non_prime_set),
            filtered_size=len(filtered),
        )
    else:
        state = _state

    tok2col       = state['tok2col']
    non_prime_set = state['non_prime_set']
    n_feat        = state['n_feat']

    mat = np.zeros((len(token_lists), n_feat), dtype=np.float32)
    for row, toks in enumerate(token_lists):
        for tok in toks:
            if tok in tok2col:
                mat[row, tok2col[tok]] += 1
            elif tok in non_prime_set:
                mat[row, 0] += 1

    return mat, state


def compute_tfidf(
    count_mat: np.ndarray,
    df_counts: dict,
    state: dict,
    n_train_docs: int,
) -> np.ndarray:
    """
    Task 2B – TF-IDF.

    TF  = raw count / row sum  (normalised term frequency)
    IDF = log((N+1)/(df+1)) + 1  (smoothed inverse document frequency)

    Parameters
    ----------
    count_mat    : matrix from build_count_matrix()
    df_counts    : dict from compute_document_frequencies()
    state        : state dict from build_count_matrix()
    n_train_docs : number of training documents (N)

    Returns
    -------
    tfidf_mat : np.ndarray same shape as count_mat
    """
    tok2col       = state['tok2col']
    non_prime_set = state['non_prime_set']
    n_feat        = state['n_feat']

    idf = np.zeros(n_feat, dtype=np.float32)
    np_df_sum = sum(df_counts.get(tok, 0) for tok in non_prime_set)
    idf[0] = math.log((n_train_docs + 1) / (np_df_sum + 1)) + 1
    for tok, col in tok2col.items():
        idf[col] = math.log((n_train_docs + 1) / (df_counts.get(tok, 0) + 1)) + 1

    row_sums = count_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    tf = count_mat / row_sums
    return tf * idf


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 – Naïve Bayes Variants (log-space)
# ══════════════════════════════════════════════════════════════════════════════

class MultinomialNB:
    """
    Task 3A – Multinomial Naïve Bayes.

    Models word count frequencies. Uses log-space computations to prevent
    numerical underflow. Supports Laplace smoothing (alpha > 0).

    log P(c|x) ∝ log P(c) + Σ x_i · log P(w_i | c)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        n = X.shape[0]
        self.log_priors_      = {}
        self.log_likelihoods_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            # Log prior: log P(c) = log(n_c / n)
            self.log_priors_[c] = math.log(Xc.shape[0] / n)
            # Smoothed likelihood: P(w|c) = (count(w,c) + α) / (Σ count + α·V)
            tc = Xc.sum(axis=0) + self.alpha
            with np.errstate(divide='ignore'):
                ll = np.where(tc > 0, np.log(tc / tc.sum()), -np.inf)
            self.log_likelihoods_[c] = ll
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for idx, c in enumerate(self.classes_):
            ll = np.where(np.isfinite(self.log_likelihoods_[c]),
                          self.log_likelihoods_[c], 0.0)
            scores[:, idx] = self.log_priors_[c] + X.dot(ll)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


class BernoulliNB:
    """
    Task 3B – Bernoulli Naïve Bayes.

    Models binary word presence (not counts). Explicitly penalises absent
    features, making it well-suited for short documents.

    log P(c|x) ∝ log P(c) + Σ x_i·log P(w|c) + (1-x_i)·log(1-P(w|c))
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        n  = X.shape[0]
        Xb = (X > 0).astype(np.float32)   # binarise
        self.log_priors_ = {}
        self.log_p_      = {}   # log P(feature=1 | class)
        self.log_1p_     = {}   # log P(feature=0 | class)
        for c in self.classes_:
            Xc = Xb[y == c]
            nc = Xc.shape[0]
            self.log_priors_[c] = math.log(nc / n)
            p = (Xc.sum(axis=0) + self.alpha) / (nc + 2 * self.alpha)
            self.log_p_[c]  = np.log(np.clip(p,     1e-10, 1 - 1e-10))
            self.log_1p_[c] = np.log(np.clip(1 - p, 1e-10, 1 - 1e-10))
        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = (X > 0).astype(np.float32)
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for idx, c in enumerate(self.classes_):
            scores[:, idx] = (self.log_priors_[c]
                              + Xb.dot(self.log_p_[c])
                              + (1 - Xb).dot(self.log_1p_[c]))
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 – Numeric Meta-Features & Imputation
# ══════════════════════════════════════════════════════════════════════════════

def extract_numeric_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Task 4 – Extract 9 numeric features from raw headline text.

    Features:
        word_count      : total words in headline
        char_length     : total character length
        uppercase_words : number of fully uppercase words
        excl_count      : number of '!' characters
        ques_count      : number of '?' characters
        negation_count  : number of negation words
        digit_count     : number of digit characters
        uppercase_ratio : uppercase_words / word_count
        sarcasm_ratio   : uppercase_words / word_count
                          (0.5 "Ambiguity Constant" if word_count == 0)

    Note: domain name (theonion / huffpost) is PROHIBITED as a feature.
    """
    headlines = df_in['headline'].fillna('')

    word_count      = headlines.apply(lambda x: len(x.split()))
    char_length     = headlines.apply(len)
    uppercase_words = headlines.apply(
        lambda x: sum(1 for w in x.split() if w.isupper()))
    excl_count      = headlines.apply(lambda x: x.count('!'))
    ques_count      = headlines.apply(lambda x: x.count('?'))
    negation_count  = headlines.apply(
        lambda x: sum(1 for w in x.lower().split() if w in NEGATION_WORDS))
    digit_count     = headlines.apply(lambda x: sum(c.isdigit() for c in x))
    uppercase_ratio = uppercase_words / word_count.replace(0, np.nan)
    uppercase_ratio = uppercase_ratio.fillna(0.0)

    # Feature 7 – Sarcasm Ratio with Ambiguity Constant (0.5 when word_count=0)
    sarcasm_ratio = np.where(word_count == 0, 0.5,
                             uppercase_words / word_count)

    return pd.DataFrame({
        'word_count':      word_count,
        'char_length':     char_length,
        'uppercase_words': uppercase_words,
        'excl_count':      excl_count,
        'ques_count':      ques_count,
        'negation_count':  negation_count,
        'digit_count':     digit_count,
        'uppercase_ratio': uppercase_ratio,
        'sarcasm_ratio':   sarcasm_ratio,
    })


def inject_missing_values(
    feat_df: pd.DataFrame,
    seed: int,
    frac: float = 0.05,
) -> pd.DataFrame:
    """
    Task 4A – Inject 5% random missing values using the seed for reproducibility.
    Guarantees at least one NaN in the 'sarcasm_ratio' column.
    """
    rng = np.random.RandomState(seed)
    dm  = feat_df.copy().astype(float)
    nr, nc   = dm.shape
    n_missing = int(frac * nr * nc)
    rows = rng.randint(0, nr, n_missing)
    cols = rng.randint(0, nc, n_missing)
    for r, c in zip(rows, cols):
        dm.iat[r, c] = np.nan
    # Ensure at least one NaN in sarcasm_ratio
    if dm['sarcasm_ratio'].isna().sum() == 0:
        dm.iat[0, dm.columns.get_loc('sarcasm_ratio')] = np.nan
    return dm


def mean_impute(
    df_missing: pd.DataFrame,
    col_means: pd.Series = None,
) -> tuple:
    """
    Task 4B – Global Mean Imputation.
    Replace NaN with the column mean computed on training data only.

    Returns (imputed_df, col_means) — pass col_means when imputing val/test.
    """
    if col_means is None:
        col_means = df_missing.mean()
    return df_missing.fillna(col_means), col_means


def class_conditional_impute(
    df_missing: pd.DataFrame,
    y_labels: np.ndarray,
    class_means: dict = None,
    col_means: pd.Series = None,
) -> tuple:
    """
    Task 4B – Class-Conditional Mean Imputation.
    Fill NaN in class-c rows with the mean computed from class-c training rows.
    Falls back to global mean for any remaining NaN.

    Returns (imputed_df, class_means, col_means).
    """
    if class_means is None:
        class_means = {c: df_missing[y_labels == c].mean() for c in [0, 1]}
    if col_means is None:
        col_means = df_missing.mean()

    df_out = df_missing.copy()
    for c, cm in class_means.items():
        mask = y_labels == c
        df_out.loc[mask] = df_out.loc[mask].fillna(cm)
    df_out = df_out.fillna(col_means)   # fallback for any remaining NaN
    return df_out, class_means, col_means


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 – Outlier Detection & Treatment
# ══════════════════════════════════════════════════════════════════════════════

def detect_outliers_iqr(
    df_train: pd.DataFrame,
) -> tuple:
    """
    Task 5A – Detect outliers using the IQR method.
    IQR = Q3 − Q1
    Lower fence = Q1 − 1.5 · IQR
    Upper fence = Q3 + 1.5 · IQR

    Fences are computed on training data only.

    Returns (outlier_mask, lower_bounds, upper_bounds)
    """
    Q1     = df_train.quantile(0.25)
    Q3     = df_train.quantile(0.75)
    IQR    = Q3 - Q1
    lower  = Q1 - 1.5 * IQR
    upper  = Q3 + 1.5 * IQR
    mask   = ((df_train < lower) | (df_train > upper)).any(axis=1)
    return mask, lower, upper


def clamp_outliers(
    df: pd.DataFrame,
    lower: pd.Series,
    upper: pd.Series,
) -> pd.DataFrame:
    """
    Task 5B – Treat outliers by clamping (Winsorisation).
    Values below lower fence → lower fence.
    Values above upper fence → upper fence.
    No rows are removed, preserving class distribution.
    """
    return df.clip(lower=lower, upper=upper, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# TASK 6 – Scaling & Gaussian NB
# ══════════════════════════════════════════════════════════════════════════════

class MinMaxScaler:
    """
    Task 6 – MinMax Scaling.
    Maps each feature to [0, 1]:  x_scaled = (x − min) / (max − min)
    Fit on training data only to prevent data leakage.
    """

    def fit(self, X: np.ndarray):
        self.min_  = X.min(axis=0)
        self.max_  = X.max(axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ == 0] = 1   # avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class StandardScaler:
    """
    Task 6 – Standardisation.
    Maps each feature to zero mean, unit variance:  x_scaled = (x − μ) / σ
    Fit on training data only to prevent data leakage.
    """

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1   # avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class GaussianNB:
    """
    Task 6 – Gaussian Naïve Bayes (for numeric features).
    Assumes each feature follows a Gaussian distribution per class.
    Used after scaling to compare MinMax vs Standardisation.
    """

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        n = X.shape[0]
        self.priors_, self.mean_, self.var_ = {}, {}, {}
        for c in self.classes_:
            Xc = X[y == c]
            self.priors_[c] = Xc.shape[0] / n
            self.mean_[c]   = Xc.mean(axis=0)
            self.var_[c]    = Xc.var(axis=0) + 1e-9   # variance smoothing
        return self

    def _log_likelihood(self, X: np.ndarray, c) -> np.ndarray:
        m, v = self.mean_[c], self.var_[c]
        return -0.5 * np.sum(np.log(2 * np.pi * v) + (X - m) ** 2 / v, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.column_stack([
            math.log(self.priors_[c]) + self._log_likelihood(X, c)
            for c in self.classes_
        ])
        return self.classes_[np.argmax(scores, axis=1)]


# ══════════════════════════════════════════════════════════════════════════════
# TASK 7 – Class Imbalance & Performance Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def random_oversample(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple:
    """
    Task 7B – Random Oversampling.
    Duplicate minority-class rows (with replacement) until both classes
    have equal counts. Shuffles the result using the seed.
    Returns (X_balanced, y_balanced).
    """
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    X_parts, y_parts = [], []
    for c, cnt in zip(classes, counts):
        mask = y == c
        Xc, yc = X[mask], y[mask]
        if cnt < max_count:
            n_extra = max_count - cnt
            idx = rng.choice(cnt, n_extra, replace=True)
            Xc = np.vstack([Xc, Xc[idx]])
            yc = np.concatenate([yc, yc[idx]])
        X_parts.append(Xc)
        y_parts.append(yc)
    X_bal = np.vstack(X_parts)
    y_bal = np.concatenate(y_parts)
    perm  = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def tune_threshold(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    thresholds: np.ndarray = None,
) -> tuple:
    """
    Task 7C – Decision Threshold Tuning.
    Sweep thresholds on the VALIDATION set (never test set!) and
    return the threshold that maximises F1-score for the sarcastic class.

    Returns (best_threshold, best_f1, threshold_curve)
    where threshold_curve is a list of (threshold, f1) pairs.
    """
    if thresholds is None:
        thresholds = np.arange(0.30, 0.81, 0.05)

    val_proba = log_softmax(model.predict_log_proba(X_val))
    best_thresh, best_f1 = 0.5, 0.0
    curve = []
    for th in thresholds:
        preds = (val_proba[:, 1] >= th).astype(int)
        f1    = f1_score(y_val, preds, zero_division=0)
        curve.append((round(float(th), 2), float(f1)))
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(th)

    return best_thresh, best_f1, curve


# ══════════════════════════════════════════════════════════════════════════════
# TASK 8 – Error Analysis
# ══════════════════════════════════════════════════════════════════════════════

def get_misclassified(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    headlines: list,
    threshold: float,
    n: int = 8,
) -> list:
    """
    Task 8A – Return up to n misclassified test examples.
    Each entry is a dict with keys:
        headline, actual, pred, prob, type ('FP' or 'FN')
    """
    test_proba = log_softmax(model.predict_log_proba(X_test))
    y_pred     = (test_proba[:, 1] >= threshold).astype(int)
    mis_idx    = np.where(y_pred != y_test)[0][:n]
    return [
        {
            'headline': headlines[i],
            'actual':   int(y_test[i]),
            'pred':     int(y_pred[i]),
            'prob':     float(test_proba[i, 1]),
            'type':     'FP' if y_pred[i] == 1 and y_test[i] == 0 else 'FN',
        }
        for i in mis_idx
    ]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def log_softmax(scores: np.ndarray) -> np.ndarray:
    """Convert raw log-probability scores to normalised probabilities."""
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard classification metrics and return as a dict."""
    return dict(
        accuracy  = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0),
        recall    = recall_score(y_true, y_pred, average='macro', zero_division=0),
        f1_macro  = f1_score(y_true, y_pred, average='macro', zero_division=0),
        f1_sarc   = f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        cm        = confusion_matrix(y_true, y_pred),
    )


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the JSON-lines sarcasm dataset into a DataFrame."""
    records = []
    with open(dataset_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(records)


def split_dataset(df: pd.DataFrame, seed: int) -> tuple:
    """
    Deterministic 70/15/15 train/val/test split.
    Returns (train_df, val_df, test_df).
    """
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n   = len(df)
    ntr = int(0.70 * n)
    nv  = int(0.15 * n)
    return (
        df.iloc[:ntr].reset_index(drop=True),
        df.iloc[ntr:ntr + nv].reset_index(drop=True),
        df.iloc[ntr + nv:].reset_index(drop=True),
    )


# ══════════════════════════════════════════════════════════════════════════════
# END-TO-END PIPELINE  (runs all tasks, returns a single results dict)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(dataset_path: str, student_id: int) -> dict:
    """
    Run the complete sarcasm detection pipeline (Tasks 1–8).

    Parameters
    ----------
    dataset_path : path to Sarcasm_Headlines_Dataset.json
    student_id   : used to derive seed and all experiment parameters

    Returns
    -------
    Large results dict consumed by the Streamlit UI (app.py).
    Keys are documented inline below.
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    cfg   = derive_seed_config(student_id)
    seed  = cfg['seed']
    alphas= cfg['alphas']
    k     = cfg['k']
    t     = cfg['t']

    # ── Load & split dataset ──────────────────────────────────────────────────
    df = load_dataset(dataset_path)
    tr, vl, te = split_dataset(df, seed)
    y_train = tr['is_sarcastic'].values
    y_val   = vl['is_sarcastic'].values
    y_test  = te['is_sarcastic'].values

    # ── Task 1 – Preprocessing ────────────────────────────────────────────────
    train_tokens = [preprocess(h) for h in tr['headline']]
    vocab        = build_vocabulary(train_tokens)
    val_tokens   = [preprocess(h, vocab) for h in vl['headline']]
    test_tokens  = [preprocess(h, vocab) for h in te['headline']]

    # ── Task 2 – Feature representation ──────────────────────────────────────
    df_counts = compute_document_frequencies(train_tokens)
    top10     = sorted(df_counts.items(), key=lambda x: -x[1])[:10]

    Xtr, feat_state = build_count_matrix(train_tokens, df_counts, t)
    Xv,  _          = build_count_matrix(val_tokens,   df_counts, t, feat_state)
    Xte, _          = build_count_matrix(test_tokens,  df_counts, t, feat_state)

    n_train = len(train_tokens)
    Xtr_tf = compute_tfidf(Xtr, df_counts, feat_state, n_train)
    Xv_tf  = compute_tfidf(Xv,  df_counts, feat_state, n_train)
    Xte_tf = compute_tfidf(Xte, df_counts, feat_state, n_train)

    # ── Task 3 – NB variants ──────────────────────────────────────────────────
    nb_results = {}
    for a in alphas:
        mnb    = MultinomialNB(a).fit(Xtr, y_train)
        bnb    = BernoulliNB(a).fit(Xtr, y_train)
        mnb_tf = MultinomialNB(a).fit(Xtr_tf, y_train)
        nb_results[a] = dict(
            mnb_count = metrics_dict(y_val, mnb.predict(Xv)),
            bnb_count = metrics_dict(y_val, bnb.predict(Xv)),
            mnb_tfidf = metrics_dict(y_val, mnb_tf.predict(Xv_tf)),
        )

    # ── Task 4 – Numeric features & imputation ───────────────────────────────
    num_tr = extract_numeric_features(tr)
    num_vl = extract_numeric_features(vl)
    num_te = extract_numeric_features(te)

    num_tr_m = inject_missing_values(num_tr, seed)
    num_vl_m = inject_missing_values(num_vl, seed)
    num_te_m = inject_missing_values(num_te, seed)

    # Mean imputation
    num_tr_mi, col_means = mean_impute(num_tr_m)
    num_vl_mi, _         = mean_impute(num_vl_m, col_means)
    num_te_mi, _         = mean_impute(num_te_m, col_means)

    # Class-conditional imputation
    num_tr_cc, class_means, col_means = class_conditional_impute(
        num_tr_m, y_train)
    num_vl_cc, _, _ = class_conditional_impute(
        num_vl_m, y_val, class_means, col_means)
    num_te_cc, _, _ = class_conditional_impute(
        num_te_m, y_test, class_means, col_means)

    # ── Task 5 – Outlier detection & clamping ────────────────────────────────
    outlier_mask, lower, upper = detect_outliers_iqr(num_tr_cc)
    num_tr_cl = clamp_outliers(num_tr_cc, lower, upper)
    num_vl_cl = clamp_outliers(num_vl_cc, lower, upper)
    num_te_cl = clamp_outliers(num_te_cc, lower, upper)

    # ── Task 6 – Scaling + Gaussian NB ───────────────────────────────────────
    Xn_tr = num_tr_cl.values.astype(np.float64)
    Xn_vl = num_vl_cl.values.astype(np.float64)

    mm_scaler  = MinMaxScaler().fit(Xn_tr)
    std_scaler = StandardScaler().fit(Xn_tr)

    gnb_mm  = GaussianNB().fit(mm_scaler.transform(Xn_tr),  y_train)
    gnb_std = GaussianNB().fit(std_scaler.transform(Xn_tr), y_train)

    scale_results = dict(
        minmax = metrics_dict(y_val, gnb_mm.predict(mm_scaler.transform(Xn_vl))),
        stdize = metrics_dict(y_val, gnb_std.predict(std_scaler.transform(Xn_vl))),
    )

    # ── Task 7 – Imbalance + oversampling + threshold tuning ─────────────────
    class_dist    = dict(Counter(y_train))
    Xtr_os, ytr_os = random_oversample(Xtr, y_train, seed)

    best_mnb = MultinomialNB(alphas[-1]).fit(Xtr_os, ytr_os)
    best_thresh, best_f1, thresh_curve = tune_threshold(best_mnb, Xv, y_val)

    val_proba = log_softmax(best_mnb.predict_log_proba(Xv))
    yv_pred   = (val_proba[:, 1] >= best_thresh).astype(int)
    val_met   = metrics_dict(y_val, yv_pred)

    # ── Task 8 – Error analysis ───────────────────────────────────────────────
    test_proba    = log_softmax(best_mnb.predict_log_proba(Xte))
    yt_pred       = (test_proba[:, 1] >= best_thresh).astype(int)
    test_met      = metrics_dict(y_test, yt_pred)
    misclassified = get_misclassified(
        best_mnb, Xte, y_test,
        te['headline'].tolist(), best_thresh,
    )

    # ── Return everything the UI needs ────────────────────────────────────────
    return dict(
        # Config
        seed=seed, alphas=alphas, k=k, t=t,
        # Task 1
        vocab=vocab, vocab_size=len(vocab),
        # Task 2
        tok2col        = feat_state['tok2col'],
        non_prime_set  = feat_state['non_prime_set'],
        n_feat         = feat_state['n_feat'],
        prime_toks     = feat_state['prime_toks'],
        non_prime_set_len = feat_state['non_prime_set_len'],
        filtered_size  = feat_state['filtered_size'],
        top10          = top10,
        df_counts      = df_counts,
        # Dataset split sizes
        n_train=len(tr), n_val=len(vl), n_test=len(te),
        # Task 3
        nb_results=nb_results,
        # Task 4
        num_tr_sample    = num_tr.head(5),
        num_tr_miss_na   = num_tr_m.isna().sum().to_dict(),
        # Task 5
        outlier_count    = int(outlier_mask.sum()),
        # Task 6
        scale_results    = scale_results,
        # Task 7
        class_dist       = class_dist,
        thresh_curve     = thresh_curve,
        best_thresh      = best_thresh,
        best_f1          = best_f1,
        val_met          = val_met,
        # Task 8
        test_met         = test_met,
        misclassified    = misclassified,
        # Model + data for live prediction
        best_mnb         = best_mnb,
        _Xtr_os          = Xtr_os,
        _ytr_os          = ytr_os,
    )
