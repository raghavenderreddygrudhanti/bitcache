import numpy as np
import pytest

from bitcache import BinaryIndex, quantize


def test_quantize_basic():
    vec = np.array([[1.0, -1.0, 0.5, -0.5, 0.1, -0.1, 1.0, -1.0]], dtype=np.float32)
    packed = quantize(vec)
    assert packed.shape == (1, 1)  # 8 dims → 1 byte
    assert packed[0, 0] == 0b10101010  # bits 0,2,4,6 set (positive values)


def test_quantize_1536_dim():
    vec = np.random.randn(1, 1536).astype(np.float32)
    packed = quantize(vec)
    assert packed.shape == (1, 192)  # 1536/8 = 192 bytes


def test_quantize_batch():
    vecs = np.random.randn(100, 768).astype(np.float32)
    packed = quantize(vecs)
    assert packed.shape == (100, 96)  # 768/8 = 96 bytes per vector


def test_index_add_and_search():
    dim = 128
    n = 1000
    rng = np.random.default_rng(42)
    database = rng.standard_normal((n, dim)).astype(np.float32)

    index = BinaryIndex(dim=dim)
    index.add(database)
    assert len(index) == n

    query = database[0]
    dists, indices = index.search(query, k=5)
    assert len(dists) == 5
    assert indices[0] == 0  # self-match should be first


def test_index_search_batch():
    dim = 64
    n = 500
    nq = 10
    rng = np.random.default_rng(42)
    database = rng.standard_normal((n, dim)).astype(np.float32)
    queries = database[:nq]

    index = BinaryIndex(dim=dim)
    index.add(database)

    dists, indices = index.search_batch(queries, k=5)
    assert dists.shape == (nq, 5)
    assert indices.shape == (nq, 5)
    # Each query should find itself as nearest
    for i in range(nq):
        assert indices[i, 0] == i


def test_index_empty_search():
    index = BinaryIndex(dim=128)
    query = np.random.randn(128).astype(np.float32)
    dists, indices = index.search(query, k=5)
    assert len(dists) == 0
    assert len(indices) == 0


def test_index_incremental_add():
    dim = 64
    rng = np.random.default_rng(42)
    index = BinaryIndex(dim=dim)

    batch1 = rng.standard_normal((100, dim)).astype(np.float32)
    batch2 = rng.standard_normal((100, dim)).astype(np.float32)

    index.add(batch1)
    assert len(index) == 100

    index.add(batch2)
    assert len(index) == 200


def test_index_save_load(tmp_path):
    dim = 128
    rng = np.random.default_rng(42)
    database = rng.standard_normal((500, dim)).astype(np.float32)

    index = BinaryIndex(dim=dim)
    index.add(database)
    index.save(str(tmp_path / "test_index"))

    loaded = BinaryIndex.load(str(tmp_path / "test_index"))
    assert len(loaded) == 500
    assert loaded.dim == dim

    query = database[0]
    d1, i1 = index.search(query, k=5)
    d2, i2 = loaded.search(query, k=5)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_array_equal(d1, d2)


def test_compression_ratio():
    dim = 1536
    n = 10000
    rng = np.random.default_rng(42)
    database = rng.standard_normal((n, dim)).astype(np.float32)

    index = BinaryIndex(dim=dim)
    index.add(database)

    # float32: 10000 * 1536 * 4 = 61,440,000 bytes
    # binary:  10000 * 192 = 1,920,000 bytes
    # ratio: 32x
    assert index.compression_ratio == pytest.approx(32.0, rel=0.01)


def test_dim_mismatch_raises():
    index = BinaryIndex(dim=128)
    with pytest.raises(ValueError):
        index.add(np.random.randn(10, 64).astype(np.float32))


def test_invalid_dim_raises():
    with pytest.raises(ValueError):
        BinaryIndex(dim=0)
    with pytest.raises(ValueError):
        BinaryIndex(dim=-1)
