import json
import logging
from pathlib import Path

import types
import tradutor.llm_backend as lb


def _fake_post(expected_payload_container):
    def _post(url, json=None, timeout=None):
        expected_payload_container["payload"] = json
        return types.SimpleNamespace(
            json=lambda: {"response": "ok"},
            raise_for_status=lambda: None,
        )

    return _post


def test_ollama_includes_num_ctx_and_keep_alive(monkeypatch):
    captured = {}
    monkeypatch.setattr(lb.requests, "post", _fake_post(captured))
    backend = lb.LLMBackend(
        backend="ollama",
        model="m",
        temperature=0.1,
        logger=logging.getLogger("ollama-test"),
        repeat_penalty=1.0,
        num_predict=10,
        num_ctx=2048,
        keep_alive="1h",
    )
    backend.generate("hi")
    payload = captured["payload"]
    assert payload["keep_alive"] == "1h"
    assert payload["options"]["num_ctx"] == 2048


def test_ollama_omits_num_ctx_when_none(monkeypatch):
    captured = {}
    monkeypatch.setattr(lb.requests, "post", _fake_post(captured))
    backend = lb.LLMBackend(
        backend="ollama",
        model="m",
        temperature=0.1,
        logger=logging.getLogger("ollama-test"),
        repeat_penalty=1.0,
        num_predict=10,
        num_ctx=None,
    )
    backend.generate("hi")
    payload = captured["payload"]
    assert "num_ctx" not in payload["options"]
