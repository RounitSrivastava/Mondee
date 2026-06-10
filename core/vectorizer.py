"""
core/vectorizer.py

Hybrid Embedding Engine

Vector = [RoBERTa Embedding + LDA Topic Vector]

Benefits:
- Semantic similarity (RoBERTa)
- Topic similarity (LDA)
- Better retrieval quality
"""

import os
import pickle
import torch
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

torch.set_num_threads(os.cpu_count())
# =====================================================
# CONFIG
# =====================================================

N_TOPICS = 30

INDEX_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "index"
)

LDA_FILE = os.path.join(
    INDEX_DIR,
    "lda_model.pkl"
)

VECTORIZER_FILE = os.path.join(
    INDEX_DIR,
    "count_vectorizer.pkl"
)

# ROBERTA


_model = None


def get_model():

    global _model

    if _model is None:

        print(
            "[vectorizer] Loading RoBERTa model..."
        )

        _model = SentenceTransformer(
    "intfloat/e5-large-v2"
)

    return _model



# LDA


_vectorizer = None
_lda = None


def fit_models(
    texts: list[str]
):

    global _vectorizer
    global _lda

    print(
        "[vectorizer] Training CountVectorizer..."
    )

    _vectorizer = CountVectorizer(
        stop_words="english",
        max_features=3000,
        min_df=2,
        max_df=0.95
    )

    X = _vectorizer.fit_transform(texts)

    print(
        f"[vectorizer] Training LDA ({N_TOPICS} topics)..."
    )

    _lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=42,
    learning_method="batch",
    n_jobs=1
)

    _lda.fit(X)

    os.makedirs(INDEX_DIR, exist_ok=True)

    with open(
        VECTORIZER_FILE,
        "wb"
    ) as f:
        pickle.dump(
            _vectorizer,
            f
        )

    with open(
        LDA_FILE,
        "wb"
    ) as f:
        pickle.dump(
            _lda,
            f
        )

    print(
        "[vectorizer] Saved LDA artifacts"
    )


def load_models():

    global _vectorizer
    global _lda

    if _vectorizer is None:

        with open(
            VECTORIZER_FILE,
            "rb"
        ) as f:
            _vectorizer = pickle.load(f)

    if _lda is None:

        with open(
            LDA_FILE,
            "rb"
        ) as f:
            _lda = pickle.load(f)


# ROBERTA EMBEDDINGS


def embed_roberta(
    texts: list[str]
) -> np.ndarray:

    model = get_model()

    e5_texts = [

        f"passage: {text}"

        for text in texts

    ]

    vectors = model.encode(

        e5_texts,

        batch_size=128,

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False

    )

    return vectors.astype(
        "float32"
    )
def embed_query(
    query: str
) -> np.ndarray:

    model = get_model()

    vector = model.encode(

        [f"query: {query}"],

        convert_to_numpy=True,

        normalize_embeddings=True

    )

    return vector.astype(
        "float32"
    )

# LDA EMBEDDINGS


def lda_transform(
    texts: list[str]
) -> np.ndarray:

    load_models()

    X = _vectorizer.transform(
        texts
    )

    topics = _lda.transform(
        X
    )

    return topics.astype(
        "float32"
    )


# HYBRID EMBEDDINGS


def embed(
    texts: list[str]
) -> np.ndarray:

    roberta_vecs = embed_roberta(
        texts
    )

    lda_vecs = lda_transform(
        texts
    )

    vectors = np.hstack(
        [
            roberta_vecs,
            lda_vecs
        ]
    )

    vectors = vectors.astype(
        "float32"
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    norms = np.where(
        norms == 0,
        1,
        norms
    )

    vectors = vectors / norms

    return vectors.astype(
        "float32"
    )


def embed_one(
    text: str
) -> np.ndarray:

    return embed(
        [text]
    )


# COMPATIBILITY


def fit(
    texts: list[str]
):

    fit_models(
        texts
    )



# DEBUG


if __name__ == "__main__":

    sample = [

        "software engineer internship",

        "marketing manager role",

        "data scientist position",

        "frontend react developer"
    ]

    fit(sample)

    emb = embed(sample)

    print(
        "\nEmbedding Shape:",
        emb.shape
    )

    print(
        "RoBERTa Dim:",
        emb.shape[1] - N_TOPICS
    )

    print(
        "LDA Topics:",
        N_TOPICS
    )