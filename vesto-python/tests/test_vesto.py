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
