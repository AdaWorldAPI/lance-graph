# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Functional tests for the ``POST /query/vector`` route.

The raw-vector path runs end to end: a real Lance-backed store in a temp
directory, the real Cypher engine, the real vector rerank. Only the OpenAI
embedding call on the ``query_text`` path is replaced, because it needs a
network and an API key.
"""

import pyarrow as pa
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from knowledge_graph import service as service_module
from knowledge_graph.component import KnowledgeGraphComponent
from knowledge_graph.config import KnowledgeGraphConfig
from knowledge_graph.service import LanceKnowledgeGraph
from knowledge_graph.store import LanceGraphStore
from lance_graph import GraphConfig

QUERY = "MATCH (d:Document) RETURN d.name, d.embedding"


@pytest.fixture
def client(tmp_path):
    kg_config = KnowledgeGraphConfig(
        storage_path=tmp_path / "storage",
        schema_path=tmp_path / "graph.yaml",
    )
    store = LanceGraphStore(kg_config)
    store.ensure_layout()
    # Doc1 and Doc2 point the same way; Doc3 is orthogonal to them; Doc4 is
    # anti-parallel to Doc1, so cosine and dot order it LAST while L2 still
    # ranks it by raw distance — the metrics disagree, which is what lets a
    # test see which one the route actually applied.
    store.write_tables(
        {
            "Document": pa.table(
                {
                    "id": [1, 2, 3, 4],
                    "name": ["Doc1", "Doc2", "Doc3", "Doc4"],
                    "embedding": pa.array(
                        [
                            [1.0, 0.0, 0.0],
                            [4.0, 0.2, 0.0],
                            [0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0],
                        ],
                        type=pa.list_(pa.float32()),
                    ),
                }
            )
        }
    )
    graph_config = GraphConfig.builder().with_node_label("Document", "id").build()

    component = KnowledgeGraphComponent(kg_config)
    component._service = LanceKnowledgeGraph(graph_config, storage=store)
    app = FastAPI()
    app.include_router(component.router)
    test_client = TestClient(app)
    test_client.kg_service = component._service
    return test_client


def _service_of(client):
    return client.kg_service


def _names(response):
    return [row["d.name"] for row in response.json()["rows"]]


def test_raw_vector_returns_top_k_nearest(client):
    response = client.post(
        "/query/vector",
        json={
            "query": QUERY,
            "column": "d.embedding",
            "vector": [1.0, 0.0, 0.0],
            "metric": "cosine",
            "top_k": 2,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["row_count"] == 2
    assert _names(response) == ["Doc1", "Doc2"]
    assert body["column"] == "d.embedding"
    assert body["metric"] == "cosine"
    assert body["top_k"] == 2
    assert "_distance" in body["rows"][0]


def test_metric_selects_the_distance_function(client):
    """L2 and cosine rank Doc2 differently: it points the same way as the
    query (cosine-nearest) but is far away in raw distance (L2)."""

    def top(metric):
        response = client.post(
            "/query/vector",
            json={
                "query": QUERY,
                "column": "d.embedding",
                "vector": [1.0, 0.0, 0.0],
                "metric": metric,
                "top_k": 4,
            },
        )
        assert response.status_code == 200, response.text
        return _names(response)

    cosine = top("cosine")
    l2 = top("l2")
    assert cosine[:2] == ["Doc1", "Doc2"]
    # Doc2 sits at distance ~3.0 from the query, farther than Doc3 (~1.41).
    assert l2.index("Doc3") < l2.index("Doc2")
    assert cosine != l2
    # Dot product favours the long, aligned Doc2 over the unit-length Doc1.
    assert top("dot")[0] == "Doc2"


def test_include_distance_false_drops_the_column(client):
    response = client.post(
        "/query/vector",
        json={
            "query": QUERY,
            "column": "d.embedding",
            "vector": [1.0, 0.0, 0.0],
            "top_k": 1,
            "include_distance": False,
        },
    )

    assert response.status_code == 200, response.text
    assert "_distance" not in response.json()["rows"][0]


def test_neither_vector_nor_text_is_rejected(client):
    response = client.post(
        "/query/vector", json={"query": QUERY, "column": "d.embedding"}
    )
    assert response.status_code == 400
    assert "Either 'vector' or 'query_text'" in response.json()["detail"]


def test_both_vector_and_text_is_rejected(client):
    response = client.post(
        "/query/vector",
        json={
            "query": QUERY,
            "column": "d.embedding",
            "vector": [1.0, 0.0, 0.0],
            "query_text": "anything",
        },
    )
    assert response.status_code == 400
    assert "only one of" in response.json()["detail"]


@pytest.mark.parametrize(
    "field, value",
    [("metric", "manhattan"), ("top_k", 0), ("top_k", 10001)],
)
def test_out_of_contract_fields_fail_validation(client, field, value):
    payload = {
        "query": QUERY,
        "column": "d.embedding",
        "vector": [1.0, 0.0, 0.0],
        field: value,
    }
    assert client.post("/query/vector", json=payload).status_code == 422


def test_query_text_is_embedded_then_reranked(client, monkeypatch):
    """The text path must embed with the requested model and rank by the
    resulting vector — the stub maps the text onto Doc3's direction, so a
    route that ignored the embedding would not put Doc3 first."""
    seen = {}

    class FakeEmbeddingGenerator:
        def __init__(self, model):
            seen["model"] = model

        def embed_one(self, text):
            seen["text"] = text
            return [0.0, 1.0, 0.0]

    import knowledge_graph.embeddings as embeddings

    monkeypatch.setattr(embeddings, "EmbeddingGenerator", FakeEmbeddingGenerator)

    response = client.post(
        "/query/vector",
        json={
            "query": QUERY,
            "column": "d.embedding",
            "query_text": "science",
            "top_k": 1,
            "embedding_model": "text-embedding-3-large",
        },
    )

    assert response.status_code == 200, response.text
    assert _names(response) == ["Doc3"]
    assert seen == {"model": "text-embedding-3-large", "text": "science"}


def test_failed_embedding_is_a_generic_server_error(client, monkeypatch):
    """A failure inside the service is a 500 whose body names no internals:
    the raw message quotes the caller's text and the embedding client."""

    class NoEmbedding:
        def __init__(self, model):
            pass

        def embed_one(self, text):
            return None

    import knowledge_graph.embeddings as embeddings

    monkeypatch.setattr(embeddings, "EmbeddingGenerator", NoEmbedding)

    response = client.post(
        "/query/vector",
        json={"query": QUERY, "column": "d.embedding", "query_text": "secret text"},
    )
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Vector query execution failed."
    assert "secret text" not in detail


def test_empty_query_text_is_a_client_error(client):
    response = client.post(
        "/query/vector",
        json={"query": QUERY, "column": "d.embedding", "query_text": ""},
    )
    assert response.status_code == 422


def test_query_by_text_rejects_an_unknown_metric(client, monkeypatch):
    """The service must refuse a metric it cannot honour rather than rank by
    cosine without saying so."""

    class ReachedEmbedding(Exception):
        pass

    class Unused:
        def __init__(self, model):
            raise ReachedEmbedding

    import knowledge_graph.embeddings as embeddings

    monkeypatch.setattr(embeddings, "EmbeddingGenerator", Unused)
    kg = _service_of(client)
    with pytest.raises(ValueError, match="Unsupported metric 'euclidean'"):
        kg.query_by_text(QUERY, "text", "d.embedding", metric="euclidean")
    # The three documented names are accepted case-insensitively: "L2" gets
    # past the metric check and on to the embedding step.
    with pytest.raises(ReachedEmbedding):
        kg.query_by_text(QUERY, "text", "d.embedding", metric="L2")


def test_service_module_exposes_the_rerank_entry_points():
    # Guards the import the route relies on: the component calls these two
    # methods by name, so a rename would otherwise surface only as a 500.
    assert hasattr(service_module.LanceKnowledgeGraph, "run_with_vector_rerank")
    assert hasattr(service_module.LanceKnowledgeGraph, "query_by_text")
