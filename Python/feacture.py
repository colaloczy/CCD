import os
import re
import time
import random
import multiprocessing as mp
from functools import lru_cache

import numpy as np
import pandas as pd
import Levenshtein

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. Tokenization (cached)
# =========================
@lru_cache(maxsize=None)
def get_code_tokens(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = "\n".join([l for l in code.splitlines() if l.strip()])

        tokens = re.findall(
            r'[a-zA-Z_]\w*|\d+|[+\-*/=<>!]+|[(){}\[\],.;]',
            code
        )
        return tokens
    except:
        return []


# =========================
# 2. Similarity metrics
# =========================
def jaccard_sim(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / (len(sa | sb) + 1e-6)


def jaro_sim(a, b):
    return Levenshtein.jaro(" ".join(a), " ".join(b))


def levenshtein_sim(a, b):
    return Levenshtein.ratio(" ".join(a), " ".join(b))


# =========================
# 3. Pair generation
# =========================
def generate_pairs(root_dir, samples_per_folder=50):
    clone_pairs = []
    nonclone_pairs = []

    folders = os.listdir(root_dir)
    folder_files = {}

    # ===== Clone pairs =====
    for folder in folders:
        python_dir = os.path.join(root_dir, folder, "Python")
        if not os.path.isdir(python_dir):
            continue

        files = [os.path.join(python_dir, f)
                 for f in os.listdir(python_dir)
                 if f.endswith(".py")]

        if len(files) > samples_per_folder:
            files = random.sample(files, samples_per_folder)

        folder_files[folder] = files

        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                clone_pairs.append((files[i], files[j]))

    # ===== Non-clone pairs =====
    folder_list = list(folder_files.keys())

    for i in range(len(folder_list)):
        for j in range(i + 1, len(folder_list)):
            f1_list = folder_files[folder_list[i]]
            f2_list = folder_files[folder_list[j]]

            pair_num = min(len(f1_list), len(f2_list))
            f1_sample = random.sample(f1_list, pair_num)
            f2_sample = random.sample(f2_list, pair_num)

            for f1, f2 in zip(f1_sample, f2_sample):
                nonclone_pairs.append((f1, f2))

    # ===== Balance =====
    n = min(len(clone_pairs), len(nonclone_pairs))
    clone_pairs = random.sample(clone_pairs, n)
    nonclone_pairs = random.sample(nonclone_pairs, n)

    print(f"Clone pairs: {len(clone_pairs)} | Non-clone pairs: {len(nonclone_pairs)}")
    return clone_pairs, nonclone_pairs


# =========================
# 4. Global LDA (ONCE)
# =========================
def build_global_lda(all_files, n_topics=5):
    docs = []
    index = {}

    for i, f in enumerate(all_files):
        tokens = get_code_tokens(f)
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


# =========================
# 5. Pair feature extraction
# =========================
def compute_pair(args):
    f1, f2, label, topic_matrix, index = args

    t1 = get_code_tokens(f1)
    t2 = get_code_tokens(f2)

    jac = jaccard_sim(t1, t2)
    # jaro = jaro_sim(t1, t2)
    lev = levenshtein_sim(t1, t2)

    v1 = topic_matrix[index[f1]]
    v2 = topic_matrix[index[f2]]
    lda_sim = cosine_similarity([v1], [v2])[0][0]

    len_ratio = len(t1) / (len(t2) + 1e-6)
    # rename_ratio = 1 - jac

    return [
        jac, lev, lda_sim,
        len_ratio, 
        label
    ]


# =========================
# 6. Main
# =========================
def main():
    root = "PythonCodeNet/data"  
    start = time.time()

    clone_pairs, nonclone_pairs = generate_pairs(root)

    all_pairs = [(a, b, 1) for a, b in clone_pairs] + \
                [(a, b, 0) for a, b in nonclone_pairs]

    all_files = list(set([f for p in all_pairs for f in p[:2]]))

    print("Training global LDA...")
    topic_matrix, index = build_global_lda(all_files)

    tasks = [(a, b, y, topic_matrix, index) for a, b, y in all_pairs]

    with mp.Pool(mp.cpu_count()) as pool:
        rows = pool.map(compute_pair, tasks)

    df = pd.DataFrame(rows, columns=[
        "Jaccard", "Levenshtein",
        "LDA", "LenRatio", "Label"
    ])

    df.to_csv("features.csv", index=False)
    print("Saved features.csv")
    print("Total time:", time.time() - start)


if __name__ == "__main__":
    main()
