def test_base():
    from vesto import Vesto

    db = Vesto()

    collection = db.add_collection("docs")

    vector_field = "my_vector_field"
    collection.add_index(vector_field, "flat_idx", "flat", "cosine", 2)

    ids = collection.insert(
        vector_field,
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        [{"metadata": "doc1"}, {"metadata": "doc2"}]
    )

    # len(db) counts collections
    assert len(db) == 1
    assert ids == [0, 1]  # insert returns the minted EntityIds as u64s

    # search takes the index name, returns (score, vector) pairs.
    results = collection.search(vector_field, "flat_idx", [1.0, 0.0], 1, False)

    score, vector = results[0]
    assert vector == [1.0, 0.0]  # nearest to [1,0] is itself


def test_hnsw():
    from vesto import Vesto

    db = Vesto()

    collection = db.add_collection("docs")

    vector_field = "my_vector_field"
    collection.add_index(vector_field, "hnsw_idx", "hnsw", "l2", 2)

    ids = collection.insert(
        vector_field,
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        [{"metadata": "doc1"}, {"metadata": "doc2"}]
    )

    # len(db) counts collections
    assert len(db) == 1
    assert ids == [0, 1]  # insert returns the minted EntityIds as u64s

    # search takes the index name, returns (score, vector) pairs.
    results = collection.search(vector_field, "hnsw_idx", [1.0, 0.0], 1, False)

    score, vector = results[0]
    assert vector == [1.0, 0.0]  # nearest to [1,0] is itself


def test_hnsw_flat_recall():
    import random

    from vesto import Vesto

    random.seed(0)
    dim = 10
    n = 100

    db = Vesto()
    collection = db.add_collection("docs")
    vector_field = "my_vector_field"
    collection.add_index(vector_field, "flat_idx", "flat", "l2", dim)
    collection.add_index(vector_field, "hnsw_idx", "hnsw", "l2", dim)

    vectors = [[random.random() for _ in range(dim)] for _ in range(n)]
    collection.insert(vector_field, vectors, [])

    # query with several stored vectors, compare top-10
    k = 10
    hits = 0
    total = 0
    for query in vectors[:30]:
        truth = {tuple(v) for _, v in collection.search(vector_field, "flat_idx", query, k, False)}
        got = {tuple(v) for _, v in collection.search(vector_field, "hnsw_idx", query, k, False)}
        hits += len(truth & got)
        total += len(truth)

    recall = hits / total
    print(f"recall@{k} = {recall:.3f}")
    assert recall > 0.8, f"recall too low: {recall}"
