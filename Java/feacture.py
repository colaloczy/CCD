import os
import time
import javalang
import pandas as pd
import multiprocessing as mp
from functools import lru_cache

import Levenshtein
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

@lru_cache(maxsize=None)
def get_java_tokens(file_path):
    tokens = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()

        for tok in javalang.tokenizer.tokenize(code):
            if isinstance(tok, javalang.tokenizer.Identifier):
                tokens.append("id")
            else:
                tokens.append(tok.value)

    except Exception:
        pass

    return tokens

def jaccard_sim(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / (len(sa | sb) + 1e-6)


def jaro_sim(a, b):
    return Levenshtein.jaro(" ".join(a), " ".join(b))


def levenshtein_ratio(a, b):
    return Levenshtein.ratio(" ".join(a), " ".join(b))

def build_global_lda(all_files, n_topics=5):
    docs = []
    index = {}

    for i, f in enumerate(all_files):
        tokens = get_java_tokens(f)
        docs.append(" ".join(tokens))
        index[f] = i

    vectorizer = CountVectorizer(min_df=2)
    X = vectorizer.fit_transform(docs)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )
    topic_matrix = lda.fit_transform(X)

    return topic_matrix, index

def compute_pair(args):
    f1, f2, label, topic_matrix, index = args

    t1 = get_java_tokens(f1)
    t2 = get_java_tokens(f2)

    jac = jaccard_sim(t1, t2)
    # jaro = jaro_sim(t1, t2)
    lev = levenshtein_ratio(t1, t2)

    v1 = topic_matrix[index[f1]]
    v2 = topic_matrix[index[f2]]
    lda_sim = cosine_similarity([v1], [v2])[0][0]

    len_ratio = len(t1) / (len(t2) + 1e-6)
    # rename_ratio = 1 - jac

    return [
        jac,lev, lda_sim,
        len_ratio,
        label
    ]

def main():
    inputcsv = "dataset/nonclone.csv"
    # inputcsv = "dataset/clone.csv"
    source_root = "dataset/id2sourcecode/"

    pairs = pd.read_csv(inputcsv, header=None)
    pairs = pairs.drop(labels=0)
    pairs.columns = ['FunID1', 'FunID2']

    file_pairs = []
    all_files = set()

    for _, row in pairs.iterrows():
        f1 = os.path.join(source_root, str(row.FunID1) + ".java")
        f2 = os.path.join(source_root, str(row.FunID2) + ".java")

        file_pairs.append((f1, f2))
        all_files.add(f1)
        all_files.add(f2)

    print("Training global LDA (Java)...")
    topic_matrix, index = build_global_lda(list(all_files))

    tasks = [(f1, f2, 1, topic_matrix, index) for f1, f2 in file_pairs]

    with mp.Pool(mp.cpu_count()) as pool:
        rows = pool.map(compute_pair, tasks)

    df = pd.DataFrame(rows, columns=[
        "Jaccard", "Levenshtein",
        "LDA", "LenRatio", "Label"
    ])

    df.to_csv(
        "java_nonclone_features_global_lda.csv",
        index=False
    )

if __name__ == "__main__":
    start = time.time()
    main()
    print("Total time:", time.time() - start)
