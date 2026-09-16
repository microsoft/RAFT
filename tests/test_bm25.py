import pytest

from raft.embedding import BM25Index


def records(texts):
    return [
        {
            "id": str(i),
            "case_id": "case-a",
            "item_index": i,
            "text": text,
            "embedding": [1.0, 2.0],
            "metadata": {"product": "sync"},
        }
        for i, text in enumerate(texts)
    ]


def test_search_persistence_and_alignment(tmp_path):
    rows = records(["E42 login failure", "Password reset", "E42 network timeout"])
    index = BM25Index.from_records(iter(rows))
    assert [doc["id"] for doc in index.documents] == [row["id"] for row in rows]
    assert [doc["text"] for doc in index.documents] == [row["text"] for row in rows]
    assert all("embedding" not in doc for doc in index.documents)
    hits = index.search("E42 LOGIN", k=20)
    assert [hit["id"] for hit in hits] == ["0", "2"]
    assert hits[0]["score"] > hits[1]["score"] > 0
    assert hits[0]["case_id"] == "case-a" and hits[0]["item_index"] == 0
    assert "metadata" not in hits[0]
    assert len(index.search("E42", k=1)) == 1
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert loaded.documents == index.documents
    assert loaded.search("E42 LOGIN", k=20) == hits


@pytest.mark.parametrize("texts", [[], ["!!!"], ["!!!", "???"]])
def test_empty_corpus_or_vocabulary_and_replacing_previous_snapshot(tmp_path, texts):
    BM25Index.from_records(records(["old content"])).save(tmp_path)
    index = BM25Index.from_records(records(texts))
    assert index.search("old content") == []
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert len(loaded.documents) == len(texts)
    assert loaded.search("old content") == []


def test_query_edges_and_single_character_terms():
    index = BM25Index.from_records(records(["!!!", "X is not Y", "success"]))
    for query in ["", "   ", "???", "unseenword"]:
        assert index.search(query) == []
    assert index.search("x")[0]["id"] == "1"
    assert index.search("not")[0]["id"] == "1"
    with pytest.raises(ValueError, match="k must"):
        index.search("x", k=0)
    for query in ["", "?", "unknownword"]:
        assert index.scores(query) == [0.0, 0.0, 0.0]
    scores = index.scores("x")
    assert scores[0] == scores[2] == 0.0 and scores[1] > 0


def test_invalid_records():
    row = records(["text"])[0]
    with pytest.raises(ValueError, match="unique"):
        BM25Index.from_records([row, row])
    with pytest.raises(ValueError, match="nonempty text"):
        BM25Index.from_records([{**row, "text": " "}])


def test_failed_save_is_not_loadable(monkeypatch, tmp_path):
    index = BM25Index.from_records(records(["text"]))
    index.save(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(index._index, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        index.save(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        BM25Index.load(tmp_path)
