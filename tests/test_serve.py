"""tests/test_serve.py -- the local chat UI server (serve.py).

Pure-logic tests plus one real HTTP round trip on an ephemeral port with a tiny model. As with
test_generate.py, nothing here asserts anything about the CONTENT of generated text.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

import generate as gen
import serve
from nanogpt.config import DEFAULT_SEED, INT_DTYPE, ModelConfig, make_rng
from nanogpt.kv_cache import KVCache
from nanogpt.model import GPT
from nanogpt.tokenizer import load_corpus_and_tokenizer

_SEED = DEFAULT_SEED + 1101
_V = 55
_CFG = ModelConfig(V=_V, T=16, d=8, L=2, H=2)


@pytest.fixture(scope="module")
def tok():
    return load_corpus_and_tokenizer()[1]


def _model():
    m = GPT(_CFG)
    p = m.init_params(make_rng(_SEED))
    r = make_rng(_SEED + 1)
    for k in m.param_keys():
        if not (k.endswith(".gamma") or k.endswith(".beta")):
            p[k] += 0.4 * r.standard_normal(p[k].shape)
    return m, p


# ---------------------------------------------------------------------------
# sanitize / parse
# ---------------------------------------------------------------------------
def test_sanitize_drops_unknown_characters_and_reports_them(tok):
    clean, dropped = serve.sanitize_prompt("The rain z9\u2603!", tok.stoi)
    assert clean == "The rain !" and dropped == sorted({"z", "9", "\u2603"})


def test_parse_defaults_and_truncation_keep_the_tail(tok):
    req = serve.parse_request({"prompt": "Remember the rain softly"}, _CFG, tok.stoi)
    assert req["max_new"] == serve.default_max_new(_CFG) == 15  # T=16: default 40 is capped to T-1
    assert req["temperature"] == 0.8 and req["greedy"] is False
    assert serve.default_max_new(ModelConfig(V=_V)) == 40  # the default model keeps 40
    assert len(req["prompt"]) + req["max_new"] <= _CFG.T
    assert req["prompt"] == "Remember the rain softly"[-(_CFG.T - req["max_new"]):] or len(req["prompt"]) <= 1


def test_parse_truncates_long_prompt_to_fit_context_with_the_last_characters(tok):
    long = "a" * 40 + "Remember"
    req = serve.parse_request({"prompt": long, "max_new": 10}, _CFG, tok.stoi)
    assert req["prompt"] == long[-(_CFG.T - 10):] and len(req["prompt"]) + 10 == _CFG.T
    assert req["truncated"] == len(long) - (_CFG.T - 10)


def test_parse_reports_dropped_chars_and_rejects_all_unknown(tok):
    assert serve.parse_request({"prompt": "rain9"}, _CFG, tok.stoi)["dropped"] == ["9"]
    with pytest.raises(serve.RequestError):
        serve.parse_request({"prompt": "999 \u2603"[:3]}, _CFG, tok.stoi)
    with pytest.raises(serve.RequestError):
        serve.parse_request({"prompt": ""}, _CFG, tok.stoi)


@pytest.mark.parametrize("bad", [
    {"prompt": 5}, {"prompt": "a", "max_new": 0}, {"prompt": "a", "max_new": 16},
    {"prompt": "a", "max_new": 2.5}, {"prompt": "a", "max_new": True},
    {"prompt": "a", "temperature": 0}, {"prompt": "a", "temperature": 99},
    {"prompt": "a", "top_k": 0}, {"prompt": "a", "top_k": _V + 1}, {"prompt": "a", "greedy": "yes"},
    {"prompt": "a", "seed": -1}, {"prompt": "a" * (serve.MAX_PROMPT_CHARS_IN + 1)}, [], "x",
])
def test_parse_rejects_bad_requests(tok, bad):
    with pytest.raises(serve.RequestError):
        serve.parse_request(bad, _CFG, tok.stoi)


def test_parse_accepts_boundary_values(tok):
    r = serve.parse_request({"prompt": "a", "max_new": _CFG.T - 1, "top_k": _V, "seed": 0,
                             "temperature": 0.05}, _CFG, tok.stoi)
    assert r["max_new"] == _CFG.T - 1 and len(r["prompt"]) == 1 and r["top_k"] == _V and r["seed"] == 0


# ---------------------------------------------------------------------------
# streaming == generate()
# ---------------------------------------------------------------------------
def test_stream_completion_equals_generate_for_the_same_rng(tok):
    m, p = _model()
    prompt = "Remember"
    ids = tok.encode(prompt)
    for kw in ({}, {"top_k": 5}, {"temperature": 0.3}, {"greedy": True}):
        streamed = "".join(serve.stream_completion(m, p, tok, prompt, 7, make_rng(123), **kw))
        k = {"greedy_decode": kw.get("greedy", False), "temperature": kw.get("temperature", 0.8),
             "top_k": kw.get("top_k")}
        direct = gen.generate(m, p, KVCache(m.cfg, 1), ids, 7, make_rng(123), **k)
        assert streamed == tok.decode(direct), kw


def test_stream_completion_yields_one_char_per_step_and_is_seeded(tok):
    m, p = _model()
    out = list(serve.stream_completion(m, p, tok, "rain", 6, make_rng(1)))
    assert len(out) == 6 and all(len(c) == 1 for c in out)
    assert out == list(serve.stream_completion(m, p, tok, "rain", 6, make_rng(1)))
    assert out != list(serve.stream_completion(m, p, tok, "rain", 6, make_rng(2)))


def test_stream_completion_refuses_to_exceed_context(tok):
    m, p = _model()
    with pytest.raises(serve.RequestError):
        list(serve.stream_completion(m, p, tok, "a" * 10, 10, make_rng(0)))


# ---------------------------------------------------------------------------
# real HTTP round trip (ephemeral port, tiny model)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def server(tok):
    m, p = _model()
    state = serve.ChatState(m, p, tok, source="test model", trained=True)
    srv = serve.make_server(state, "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", state
    srv.shutdown()
    srv.server_close()


def _post(url, obj, raw=None):
    data = raw if raw is not None else json.dumps(obj).encode()
    req = urllib.request.Request(url + "/api/generate", data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=20)


def _lines(resp):
    return [json.loads(l) for l in resp.read().decode().splitlines() if l.strip()]


def test_http_index_page_is_served_and_is_self_contained(server):
    url, _ = server
    body = urllib.request.urlopen(url + "/", timeout=10).read().decode()
    assert "<textarea" in body and "/api/generate" in body
    assert "http://" not in body.replace("http://127", "") and "https://" not in body  # no CDN / external calls


def test_http_info_reports_limits(server):
    url, state = server
    info = json.loads(urllib.request.urlopen(url + "/api/info", timeout=10).read())
    assert info["context_length"] == _CFG.T and info["vocab_size"] == _V and info["trained"] is True
    assert info["vocab"] == "".join(state.tok.itos) and info["config"]["L"] == 2


def test_http_generate_streams_tokens_then_a_done_line(server, tok):
    url, state = server
    msgs = _lines(_post(url, {"prompt": "Remember", "max_new": 6, "seed": 7, "top_k": 5}))
    tokens = [m["token"] for m in msgs if "token" in m]
    done = msgs[-1]
    assert len(tokens) == 6 and all(len(t) == 1 for t in tokens)
    assert done["done"] is True and done["tokens"] == 6 and done["seed"] == 7
    assert done["used_prompt"] == "Remember" and done["truncated"] == 0 and done["dropped"] == []
    # the HTTP stream equals the in-process generator for the same seed
    want = "".join(serve.stream_completion(state.model, state.params, tok, "Remember", 6, make_rng(7), top_k=5))
    assert "".join(tokens) == want


def test_http_generate_reports_truncation_and_dropped_characters(server):
    url, _ = server
    done = _lines(_post(url, {"prompt": "9" + "a" * 30, "max_new": 8, "seed": 1}))[-1]
    assert done["dropped"] == ["9"] and done["truncated"] == 30 - (_CFG.T - 8) and len(done["used_prompt"]) == _CFG.T - 8


def test_http_random_seed_is_returned_so_a_run_can_be_reproduced(server):
    url, _ = server
    a = _lines(_post(url, {"prompt": "rain", "max_new": 5}))
    seed = a[-1]["seed"]
    b = _lines(_post(url, {"prompt": "rain", "max_new": 5, "seed": seed}))
    assert [m for m in a if "token" in m] == [m for m in b if "token" in m]


@pytest.mark.parametrize("payload", [{"prompt": ""}, {"prompt": "a", "max_new": 99}, {"prompt": "999"}])
def test_http_bad_requests_get_400_with_a_message(server, payload):
    url, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(url, payload)
    assert e.value.code == 400 and "error" in json.loads(e.value.read())


def test_http_malformed_and_oversized_bodies_get_400(server):
    url, _ = server
    for raw in (b"not json", b"\xff\xfe", b"[]"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(url, None, raw=raw)
        assert e.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(url, None, raw=b"x" * (serve.MAX_BODY_BYTES + 1))
    assert e.value.code == 400


def test_http_unknown_paths_404(server):
    url, _ = server
    for path in ("/nope", "/api/nope"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(url + path, timeout=10)
        assert e.value.code == 404


def test_server_binds_to_loopback_by_default():
    import inspect

    assert inspect.signature(serve.make_server).parameters["host"].default == "127.0.0.1"
    ns = serve.argparse.ArgumentParser()
    ns.add_argument("--host", default="127.0.0.1")
    assert ns.parse_args([]).host == "127.0.0.1"
    src = inspect.getsource(serve.main)
    assert 'default="127.0.0.1"' in src


def test_build_state_without_checkpoint_is_untrained_and_with_checkpoint_is_trained(tmp_path, tok):
    from nanogpt import checkpoint

    s0 = serve.build_state(str(tmp_path / "missing.ckpt"))
    assert s0.trained is False and s0.model.cfg.V == tok.vocab_size
    m, p = _model()
    path = tmp_path / "ok.ckpt"
    checkpoint.save(path, m, p)
    s1 = serve.build_state(str(path))
    assert s1.trained is True and s1.model.cfg == _CFG
    assert all(np.array_equal(s1.params[k], p[k]) for k in p)
