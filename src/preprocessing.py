import numpy as np
import pandas as pd
import requests

import torch
import nltk
from sentence_transformers import models, SentenceTransformer
import os
import requests
from pathlib import Path
import nltk
from nltk import sent_tokenize, word_tokenize, regexp_tokenize
from nltk.corpus import stopwords
import pymorphy2

from tqdm import tqdm


def get_text(url, encoding="utf-8", to_lower=True):
    url = str(url)
    if url.startswith("http"):
        r = requests.get(url)
        if not r.ok:
            r.raise_for_status()
        return r.text.lower() if to_lower else r.text
    elif os.path.exists(url):
        with open(url, encoding=encoding) as f:
            return f.read().lower() if to_lower else f.read()
    else:
        raise Exception("parameter [url] can be either URL or a filename")


def normalize_tokens(tokens):
    return [morph.parse(tok)[0].normal_form for tok in tokens]


def remove_stopwords(tokens, stopwords=None, min_length=4):
    if not stopwords:
        return tokens
    stopwords = set(stopwords)
    tokens = [tok for tok in tokens if tok not in stopwords and len(tok) >= min_length]
    return tokens


def normalize_rem_stops(words, stopwords=None, normalize=True):
    if normalize:
        words = normalize_tokens(words)
    if stopwords:
        words = remove_stopwords(words, stopwords)
    return words


def prep_text(text):
    words = nltk.tokenize.word_tokenize(text, language="russian")
    words = normalize_rem_stops(words, stopwords=stopwords_ru)
    return words

url_stopwords_ru = (
    "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"
)
stopwords_ru = get_text(url_stopwords_ru).splitlines()
morph = pymorphy2.MorphAnalyzer()
nltk.download("punkt")
WINE_DATA_PATH = "../data/wine/wines_winestyle.csv"
MAIN_COLS = ["id", "name", "aroma", "taste", "food_pairing", "description"]
DESC_COLS = ["aroma", "taste", "food_pairing"]


MODEL_INFO = "DeepPavlov/rubert-base-cased"

wine_df = pd.read_csv(WINE_DATA_PATH, usecols=MAIN_COLS)
wine_df = wine_df.head(10000)
wine_df[DESC_COLS] = wine_df[DESC_COLS].fillna("")

tokens_col = []
wine_df["full_desc"] = wine_df[DESC_COLS].agg(" ".join, axis=1)
wine_df["tokens"] = [prep_text(text) for text in tqdm(wine_df["full_desc"])]

wine_tokens_df = wine_df[["id", "name", "tokens"]]
wine_tokens_df = wine_tokens_df[wine_tokens_df.tokens.map(len) > 0]

transformer = models.Transformer(MODEL_INFO)
pooling_module = models.Pooling(transformer.get_word_embedding_dimension())
model = SentenceTransformer(modules=[transformer, pooling_module])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

wine_vectors = [
    model.encode(token).mean(axis=0).tolist() for token in tqdm(wine_tokens_df["tokens"])
]
wine_tokens_df["vector"] = wine_vectors

wine_vectors_df = wine_tokens_df[["id", "name", "vector"]]
wine_vectors_df.to_csv("../data/wine/wines_vectors_local.csv", index=False)
wine_vectors_df.to_parquet("../data/wine/wines_vectors.parquet", index=False)
