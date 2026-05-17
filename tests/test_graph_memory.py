import numpy as np
import pytest

from bitcache import GraphMemory


def _vec(seed, dim=64):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _build_sample_graph():
    gm = GraphMemory(dim=64)
    gm.add_entity("salesforce", _vec(1), name="Salesforce", entity_type="company")
    gm.add_entity("mulesoft", _vec(2), name="MuleSoft", entity_type="company")
    gm.add_entity("oracle", _vec(3), name="Oracle", entity_type="company")
    gm.add_entity("boomi", _vec(4), name="Boomi", entity_type="company")
    gm.add_entity("marc", _vec(5), name="Marc Benioff", entity_type="person")

    gm.add_relation("salesforce", "owns", "mulesoft")
    gm.add_relation("mulesoft", "integrates_with", "oracle")
    gm.add_relation("mulesoft", "competes_with", "boomi")
    gm.add_relation("marc", "founded", "salesforce")
    return gm


def test_add_entity():
    gm = GraphMemory(dim=64)
    gm.add_entity("e1", _vec(1), name="Entity 1", entity_type="concept")
    assert gm.num_entities == 1


def test_add_relation():
    gm = _build_sample_graph()
    assert gm.num_entities == 5
    assert gm.num_relations == 4


def test_get_relations():
    gm = _build_sample_graph()
    rels = gm.get_relations("salesforce")
    assert len(rels) == 1
    assert rels[0]["relation"] == "owns"
    assert rels[0]["target"] == "mulesoft"


def test_get_incoming_relations():
    gm = _build_sample_graph()
    incoming = gm.get_incoming_relations("mulesoft")
    assert len(incoming) == 1
    assert incoming[0]["source"] == "salesforce"


def test_search_finds_entity():
    gm = _build_sample_graph()
    results = gm.search(_vec(1), k=1, expand=False)
    assert len(results) == 1
    assert results[0]["id"] == "salesforce"


def test_search_with_expansion():
    gm = _build_sample_graph()
    results = gm.search(_vec(1), k=1, expand=True, max_hops=2)
    assert len(results) == 1
    assert results[0]["id"] == "salesforce"
    # Should expand to mulesoft (hop 1) and oracle/boomi (hop 2)
    expanded = results[0]["expanded"]
    expanded_ids = [e["id"] for e in expanded]
    assert "mulesoft" in expanded_ids


def test_search_by_relation():
    gm = _build_sample_graph()
    results = gm.search_by_relation("mulesoft", relation="integrates_with")
    assert len(results) == 1
    assert results[0]["id"] == "oracle"


def test_search_by_relation_all():
    gm = _build_sample_graph()
    results = gm.search_by_relation("mulesoft")
    assert len(results) == 2  # integrates_with oracle + competes_with boomi


def test_get_path():
    gm = _build_sample_graph()
    path = gm.get_path("marc", "oracle")
    assert path is not None
    # marc → founded → salesforce → owns → mulesoft → integrates_with → oracle
    assert len(path) <= 3


def test_get_path_no_connection():
    gm = _build_sample_graph()
    path = gm.get_path("oracle", "marc", max_depth=2)
    # No path from oracle to marc in 2 hops
    assert path is None


def test_remove_entity():
    gm = _build_sample_graph()
    assert gm.remove_entity("mulesoft") is True
    assert gm.num_entities == 4

    # Relations involving mulesoft should be gone
    rels = gm.get_relations("salesforce")
    assert len(rels) == 0  # "owns mulesoft" removed


def test_remove_nonexistent():
    gm = GraphMemory(dim=64)
    assert gm.remove_entity("ghost") is False


def test_duplicate_relation_ignored():
    gm = GraphMemory(dim=64)
    gm.add_entity("a", _vec(1), name="A")
    gm.add_entity("b", _vec(2), name="B")
    gm.add_relation("a", "knows", "b")
    gm.add_relation("a", "knows", "b")  # duplicate
    assert gm.num_relations == 1
