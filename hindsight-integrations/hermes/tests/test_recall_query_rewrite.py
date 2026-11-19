"""``recall_query_rewrite``: the prefetch path may rewrite the raw user message into a
standalone retrieval question; the rewrite is fail-open and never touches the
``hindsight_recall`` tool path (that query is already model-composed)."""

import hindsight_hermes as plugin


def test_prefetch_uses_the_rewritten_query(provider, monkeypatch):
    instance, fake = provider({"bank_id": "team", "recall_query_rewrite": True})
    monkeypatch.setattr(
        plugin, "_rewrite_recall_query", lambda q: "What did the user previously decide about the renewal policy?"
    )
    text, count = instance._do_recall("上次说的那个政策后来定了吗")
    assert fake.recalls, "recall must have been issued"
    sent = fake.recalls[0]["query"]
    assert sent == "What did the user previously decide about the renewal policy?"
    instance.shutdown()


def test_rewrite_failure_falls_back_to_the_raw_query(provider, monkeypatch):
    instance, fake = provider({"bank_id": "team", "recall_query_rewrite": True})
    monkeypatch.setattr(plugin, "_rewrite_recall_query", lambda q: "")
    instance._do_recall("what is my name?")
    assert fake.recalls[0]["query"] == "what is my name?"
    instance.shutdown()


def test_rewrite_off_by_default_keeps_the_raw_query(provider, monkeypatch):
    instance, fake = provider({"bank_id": "team"})
    monkeypatch.setattr(
        plugin, "_rewrite_recall_query", lambda q: "should never be called", raising=False
    )
    instance._do_recall("what is my name?")
    assert fake.recalls[0]["query"] == "what is my name?"
    instance.shutdown()
