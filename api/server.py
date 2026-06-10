"""
api/server.py

REST endpoint for persona-based recommendations.

POST /recommend
  Body: {
    "user_id": "u_001",
    "persona": "truck_driver",
    "signals": [...],
    "query_string": "...",
    "destination": "..."
    [optional] "top_k": 30
  }
  Returns: same schema as core.recommender.recommend()

GET /health → {"status": "ok", "index_loaded": <bool>}
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

from flask import (
    Flask,
    request,
    jsonify
)

from core.recommender import (
    recommend,
    _load_once
)
from core.indexer import (
    load_index
)

app = Flask(__name__)

_index_loaded = False


@app.route("/health", methods=["GET"])

def health():

    global _index_loaded

    try:

        idx, _ = load_index()

        _index_loaded = True

        return jsonify({

            "status": "ok",
            "index_loaded": True,
            "vectors": idx.ntotal,
            "metric": "inner_product"
        })

    except Exception as exc:

        return jsonify({

            "status": "error",
            "index_loaded": False,
            "detail": str(exc)
        }), 503


@app.route("/recommend", methods=["POST"])

def recommend_endpoint():

    _load_once()

    body = request.get_json(
        silent=True
    )

    if not body:

        return jsonify({
            "error": "JSON body required"
        }), 400

    top_k = body.get(
        "top_k",
        30
    )

    try:

        result = recommend(
            body,
            top_k=top_k
        )

        return jsonify(result)

    except ValueError as exc:

        return jsonify({
            "error": str(exc)
        }), 400

    except FileNotFoundError as exc:

        return jsonify({
            "error": str(exc)
        }), 503


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    debug = os.environ.get(
        "DEBUG",
        "0"
    ) == "1"

    _load_once()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug
    )
