def test_base():
    from vesto import Vesto

    db = Vesto()

    collection = db.add_collection("docs", "embedding", "cosine", 2)

    collection.add_index("flat_idx", "flat")

    ids = collection.insert(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    # len(db) counts collections
    assert len(db) == 1
    assert ids == [0, 1]  # insert returns the minted EntityIds as u64s

    # search takes the index name, returns (score, vector) pairs.
    results = collection.search("flat_idx", [1.0, 0.0], 1)

    score, vector = results[0]
    assert vector == [1.0, 0.0]  # nearest to [1,0] is itself


def test_hnsw():
    from vesto import Vesto

    db = Vesto()

    collection = db.add_collection("docs", "embedding", "cosine", 2)

    collection.add_index("hnsw_idx", "hnsw")

    ids = collection.insert(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    # len(db) counts collections
    assert len(db) == 1
    assert ids == [0, 1]  # insert returns the minted EntityIds as u64s

    # search takes the index name, returns (score, vector) pairs.
    results = collection.search("hnsw_idx", [1.0, 0.0], 1)

    score, vector = results[0]
    assert vector == [1.0, 0.0]  # nearest to [1,0] is itself


def test_hnsw_flat_recall():
    import random

    from vesto import Vesto

    random.seed(0)
    dim = 10
    n = 100

    db = Vesto()
    collection = db.add_collection("docs", "embedding", "l2", dim)
    collection.add_index("flat_idx", "flat")
    collection.add_index("hnsw_idx", "hnsw")

    vectors = [[random.random() for _ in range(dim)] for _ in range(n)]
    collection.insert(vectors)

    # query with several stored vectors, compare top-10
    k = 10
    hits = 0
    total = 0
    for query in vectors[:30]:
        truth = {tuple(v) for _, v in collection.search("flat_idx", query, k)}
        got = {tuple(v) for _, v in collection.search("hnsw_idx", query, k)}
        hits += len(truth & got)
        total += len(truth)

    recall = hits / total
    print(f"recall@{k} = {recall:.3f}")
    assert recall > 0.8, f"recall too low: {recall}"
