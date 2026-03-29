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
# Paper-level Tokenizer
# =========================

@lru_cache(maxsize=None)
def get_code_tokens(file_path):
    """
    Lightweight, language-agnostic tokenizer for
    Java / C++ / Python / C# code clone detection.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()

        # -------------------------------------------------
        # 1. Remove comments (multi-language)
        # -------------------------------------------------
        # Python comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        # C / C++ / Java / C# single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        # C / C++ / Java / C# multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # -------------------------------------------------
        # 2. Normalize literals
        # -------------------------------------------------
        # String literals
        code = re.sub(r'"([^"\\]|\\.)*"', ' STR_LITERAL ', code)
        # Char literals
        code = re.sub(r"'([^'\\]|\\.)*'", ' CHAR_LITERAL ', code)
        # Numeric literals (int / float)
        code = re.sub(r'\b\d+(\.\d+)?\b', ' NUM_LITERAL ', code)

        # -------------------------------------------------
        # 3. Remove empty lines & normalize spaces
        # -------------------------------------------------
        code = "\n".join(l for l in code.splitlines() if l.strip())

        # -------------------------------------------------
        # 4. Tokenization
        # -------------------------------------------------
        token_pattern = re.compile(
            r'''
            [A-Za-z_]\w*            |  # identifiers / keywords
            STR_LITERAL             |  # normalized string
            CHAR_LITERAL            |  # normalized char
            NUM_LITERAL             |  # normalized number
            ==|!=|<=|>=|\+\+|--     |  # multi-char operators
            &&|\|\|                 |  # logical operators
            [+\-*/=<>!&|]           |  # single-char operators
            [(){}\[\],.;]              # delimiters
            ''',
            re.VERBOSE
        )

        tokens = token_pattern.findall(code)
        return tokens

    except Exception:
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
    

    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):

            all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
            if len(all_files) > samples_per_folder:
                files = random.sample(all_files, samples_per_folder)
            else:
                files = all_files
                

            for i in range(len(files)):
                for j in range(i+1, len(files)):
                    clone_pairs.append((
                        os.path.join(folder_path, files[i]),
                        os.path.join(folder_path, files[j])
                    ))
    

    folder_files = {}
    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

            if len(all_files) > samples_per_folder:
                folder_files[folder] = [
                    os.path.join(folder_path, f) 
                    for f in random.sample(all_files, samples_per_folder)
                ]
            else:
                folder_files[folder] = [
                    os.path.join(folder_path, f) 
                    for f in all_files
                ]
    
    folder_list = list(folder_files.keys())

    for i in range(len(folder_list)):
        for j in range(i+1, len(folder_list)):
            folder1 = folder_list[i]
            folder2 = folder_list[j]
            files1 = folder_files[folder1]
            files2 = folder_files[folder2]
            
            pairs_needed = min(len(files1), len(files2))
            selected_files1 = random.sample(files1, pairs_needed)
            selected_files2 = random.sample(files2, pairs_needed)
            
            for f1, f2 in zip(selected_files1, selected_files2):
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
    root = "poj-104/ProgramData"  
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
