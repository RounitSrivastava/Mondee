"""
core/indexer.py

Hybrid Retrieval Index

Vector =
    0.8 * RoBERTa Embedding
    +
    0.2 * LDA Topic Distribution

FAISS:
    IVF + Inner Product

Designed for:
- Semantic retrieval
- Topic retrieval
- Large catalogs
"""

import os
import json
import faiss
import numpy as np

from tqdm import tqdm

from core.vectorizer import (
    fit_models,
    embed_roberta,
    lda_transform
)

#paths
INDEX_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "index"
)

INDEX_FILE = os.path.join(
    INDEX_DIR,
    "experiences.index"
)

META_FILE = os.path.join(
    INDEX_DIR,
    "experiences_meta.json"
)

EMBEDDING_CKPT = os.path.join(
    INDEX_DIR,
    "embedding_checkpoint.npy"
)

#config

EMBEDDING_WEIGHT = 0.7
LDA_WEIGHT = 0.3

BATCH_SIZE = 128

USE_IVF = True

# HELPERS

def _normalize(x):

    x = x.astype("float32")

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True
    )

    norms = np.where(
        norms == 0,
        1,
        norms
    )

    return x / norms


def _save_embedding_checkpoint(
    vectors: np.ndarray,
    processed_count: int
):

    os.makedirs(
        INDEX_DIR,
        exist_ok=True
    )

    np.save(
        EMBEDDING_CKPT,
        {
            "vectors": vectors,
            "processed_count": processed_count
        }
    )


def _load_embedding_checkpoint():

    if not os.path.exists(
        EMBEDDING_CKPT
    ):

        return None

    print(
        f"\n[indexer] Resuming from embedding checkpoint..."
    )

    data = np.load(
        EMBEDDING_CKPT,
        allow_pickle=True
    ).item()

    print(
        f"[indexer] Restored "
        f"{data['processed_count']} embeddings"
    )

    return data


def _remove_embedding_checkpoint():

    if os.path.exists(
        EMBEDDING_CKPT
    ):

        os.remove(
            EMBEDDING_CKPT
        )

        print(
            "[indexer] Embedding checkpoint removed"
        )


# BUILD INDEX

def build_index(
    experiences: list[dict]
):

    if not experiences:

        raise ValueError(
            "No experiences supplied."
        )

    os.makedirs(
        INDEX_DIR,
        exist_ok=True
    )

    print(
        f"\n[indexer] Building index for "
        f"{len(experiences)} experiences"
    )

    texts = [

        f"{e.get('name','')} "
        f"{e.get('details','')}"

        for e in experiences
    ]

     # TRAIN LDA

    print(
        "[indexer] Training LDA..."
    )

    fit_models(texts)

    #models - E5 + LDA

    print(
        "[indexer] Creating E5 vectors..."
    )

    ckpt = _load_embedding_checkpoint()

    if ckpt is not None:

        embedding_vectors = ckpt["vectors"]

        processed_count = ckpt["processed_count"]

        start_batch = processed_count // BATCH_SIZE

        if start_batch * BATCH_SIZE >= len(
            texts
        ):

            final_embeddings = _normalize(
                embedding_vectors[:len(texts)]
            )

        else:

            remaining_texts = texts[
                processed_count:
            ]

            remaining_parts = []

            for i in tqdm(
                range(
                    0,
                    len(remaining_texts),
                    BATCH_SIZE
                ),
                desc="E5-Large-v2 (resume)"
            ):

                batch = remaining_texts[
                    i:i + BATCH_SIZE
                ]

                remaining_parts.append(
                    embed_roberta(batch)
                )

            if not remaining_parts:

                final_embeddings = _normalize(
                    embedding_vectors[:len(texts)]
                )

            else:

                remaining_vectors = np.vstack(
                    remaining_parts
                )

                full_vectors = np.vstack(
                    [
                        embedding_vectors,
                        remaining_vectors
                    ]
                )[:len(texts)]

                final_embeddings = _normalize(
                    full_vectors
                )

    else:

        embedding_parts = []

        for i in tqdm(
            range(
                0,
                len(texts),
                BATCH_SIZE
            ),
            desc="E5-Large-v2"
        ):

            batch = texts[
                i:i + BATCH_SIZE
            ]

            batch_embeddings = embed_roberta(
                batch
            )

            embedding_parts.append(
                batch_embeddings
            )

            processed = min(
                i + BATCH_SIZE,
                len(texts)
            )

            _save_embedding_checkpoint(
                np.vstack(
                    embedding_parts
                ),
                processed
            )

        final_embeddings = _normalize(
            np.vstack(
                embedding_parts
            )
        )

    embedding_vectors = final_embeddings

    #LDA

    print(
        "[indexer] Creating topic vectors..."
    )

    lda_vectors = lda_transform(
        texts
    )

    # NORMALIZE INDIVIDUALLY

    embedding_vectors = _normalize(
        embedding_vectors
    )

    lda_vectors = _normalize(
        lda_vectors
    )

# WEIGHTED HYBRID

    vectors = np.hstack(

    [
        embedding_vectors * EMBEDDING_WEIGHT,
        lda_vectors * LDA_WEIGHT
    ]

)

    vectors = _normalize(
        vectors
    )

    dim = vectors.shape[1]

    print(
        f"[indexer] Vector shape = "
        f"{vectors.shape}"
    )

    # BUILD FAISS

    if USE_IVF and len(vectors) > 5000:

        print(
            "[indexer] Building IVF index..."
        )

        nlist = min(
            256,
            max(
                32,
                int(
                    np.sqrt(
                        len(vectors)
                    )
                )
            )
        )

        quantizer = faiss.IndexFlatIP(
            dim
        )

        index = faiss.IndexIVFFlat(
            quantizer,
            dim,
            nlist,
            faiss.METRIC_INNER_PRODUCT
        )

        index.train(
            vectors
        )

        index.add(
            vectors
        )

        index.nprobe = min(
            40,
            nlist
        )

    else:

        print(
            "[indexer] Building FlatIP index..."
        )

        index = faiss.IndexFlatIP(
            dim
        )

        index.add(
            vectors
        )

    # SAVE INDEX

    faiss.write_index(
        index,
        INDEX_FILE
    )

    print(
        f"[indexer] Saved "
        f"{index.ntotal} vectors"
    )
  # METADATA

    metadata = []

    for exp in experiences:

        metadata.append(

            {
                "exp_id":
                    exp["exp_id"],

                "name":
                    exp["name"],

                "details":
                    exp["details"]
            }

        )

    with open(
        META_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "[indexer] Metadata saved"
    )


# LOAD INDEX

def load_index():

    if not os.path.exists(
        INDEX_FILE
    ):

        raise FileNotFoundError(

            "Index not found.\n"
            "Run build_index.py first."
        )

    index = faiss.read_index(
        INDEX_FILE
    )

    with open(
        META_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(
            f
        )

    print(
        f"[indexer] Loaded "
        f"{index.ntotal} vectors"
    )

    return (
        index,
        metadata
    )