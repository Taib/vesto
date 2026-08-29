def test_base():
    from vesto import Vesto

    db = Vesto()

    db.insert([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    assert len(db) == 2

    results = db.search([1.0, 0.0], 1)

    assert results[0][1] == 0