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
import itertools
import argparse

from ast import literal_eval

from sklearn.metrics.pairwise import cosine_similarity
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
MODEL_INFO = "DeepPavlov/rubert-base-cased"

transformer = models.Transformer(MODEL_INFO)
pooling_module = models.Pooling(transformer.get_word_embedding_dimension())
model = SentenceTransformer(modules=[transformer, pooling_module])

WINE_DATA_PATH = "../data/wine/wines_winestyle_10k.csv"
MAIN_COLS = ["id", "name", "aroma", "taste", "food_pairing", "description"]

WINE_VEC_DATA_PATH = "../data/wine/wines_vectors.parquet"

wine_df = pd.read_csv(WINE_DATA_PATH, usecols=MAIN_COLS)
wine_vectors_df = pd.read_parquet(WINE_VEC_DATA_PATH)


class Recommendator:
    def __init__(self, casual_df, vectors_df):
        self.casual_df = casual_df
        self.vectors_df = vectors_df
        self.vectors = np.concatenate(self.vectors_df["vector"]).reshape(
            (wine_vectors_df.shape[0], -1)
        )
        self.cosim_matrix = None  # Для расчёта нужно большое кол-во ресурсов

    def _calculate_cosim_matrix(self):
        self.cosim_matrix = cosine_similarity(vectors, vectors)

    def get_jaccard_sim(self, set1, set2):
        c = set1.intersection(set2)
        return c, float(len(c)) / (len(set1) + len(set2) - len(c))

    def recommend_from_description(self, desc, n=10):
        wine_vector = model.encode(prep_text(desc)).mean(axis=0).tolist()
        cosim_vec = cosine_similarity(np.array(wine_vector).reshape(1, -1), self.vectors)[0]
        best_idxs = np.argsort(cosim_vec)[::-1][: n + 1]
        best_ids = self.vectors_df.iloc[best_idxs]["id"].tolist()

        recommendations = self.casual_df.iloc[pd.Index(self.casual_df["id"]).get_indexer(best_ids)]
        recommendations.reset_index(drop=True, inplace=True)
        return recommendations

    def recommend_from_wine(self, wine_identifier, n=10):
        if isinstance(wine_identifier, int):
            idx = self.vectors_df[self.vectors_df["id"] == wine_identifier].index[0]
        elif isinstance(wine_identifier, str):
            idx = self.vectors_df[self.vectors_df["name"] == wine_identifier].index[0]

        cosim_vec = cosine_similarity(self.vectors[idx].reshape(1, -1), self.vectors)[0]
        best_idxs = np.argsort(cosim_vec)[::-1][: n + 1]
        best_ids = self.vectors_df.iloc[best_idxs]["id"].tolist()

        recommendations = self.casual_df.iloc[pd.Index(self.casual_df["id"]).get_indexer(best_ids)]
        recommendations.reset_index(drop=True, inplace=True)
        recommend_tokens = set(tuple(itertools.chain(*recommendations["tokens"])))

        query_tokens = set(recommendations["tokens"][0])

        random_wines = self.casual_df.iloc[
            pd.Index(self.casual_df["id"]).get_indexer(
                np.random.randint(1, self.casual_df.shape[0], n)
            )
        ]
        random_wines.reset_index(drop=True, inplace=True)
        random_tokens = set(tuple(itertools.chain(*random_wines["tokens"])))
        matched_recommended, rec_jacc = self.get_jaccard_sim(query_tokens, recommend_tokens)
        matched_random, rand_jacc = self.get_jaccard_sim(query_tokens, random_tokens)
        print("similarity in recommended: ", "{0:0.2f}".format(rec_jacc))
        print("similarity in random: ", "{0:0.2f}".format(rand_jacc))
        print("matched tokens in recommended: ", matched_recommended)
        print("matched tokens in random: ", matched_random)
        return recommendations


sommelier = Recommendator(wine_df, wine_vectors_df)


def get_full_desc(wine_df, idx):
    wine = wine_df.iloc[idx]
    full_desc = f"\nName: {wine['name']}\n\n"
    full_desc += f"Aroma: {wine['aroma']}\n\n"
    full_desc += f"Taste: {wine['taste']}\n\n"
    full_desc += f"Food pairing: {wine['food_pairing']}\n\n"
    full_desc += f"{'='*127}"
    return full_desc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Videos to images')
    parser.add_argument('indir', type=str, help='Input dir for videos')
    parser.add_argument('outdir', type=str, help='Output dir for image')
    args = parser.parse_args()
wine_identificator = 119  # "Rafael Cambra, Soplo, Valencia DO, 2014"
wine_identificator = "Rafael Cambra, Soplo, Valencia DO, 2014"

# wine_identificator = 25467 # "Trimbach, Sylvaner, Alsace AOC"
rec = sommelier.recommend_from_wine(wine_identificator)

print(get_full_desc(rec, 0))
print(get_full_desc(rec, 1))

rec = sommelier.recommend_from_description("путин")

print(get_full_desc(rec, 0))
print(get_full_desc(rec, 1))
