"""Mock backend for the BERT playground frontend.

Serves the exact JSON shapes from API_CONTRACT.md (deterministic fake data,
seeded by text hash) plus the static/ UI — so the frontend can be developed
and tested with zero ML dependencies:

    python3 tools/mock_server.py          # http://localhost:8304

No external packages; Python 3.9+ stdlib only.
"""

import hashlib
import json
import math
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
PORT = 8304

MODELS = [
    {"id": "bert-base-cased", "label": "BERT-base · cased · 12 layers", "num_layers": 12, "num_heads": 12, "dim": 768},
    {"id": "distilbert-base-uncased", "label": "DistilBERT · uncased · 6 layers", "num_layers": 6, "num_heads": 12, "dim": 768},
]
PROBES = [
    {"id": "emotion", "label": "Emotion head (demo-2 fine-tune)", "available": True,
     "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"]},
    {"id": "sst2", "label": "SST-2 sentiment", "available": True, "labels": ["negative", "positive"]},
]
MAX_TOKENS = 80


def rng_for(*parts):
    seed = int(hashlib.sha256("||".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def fake_tokens(text, model):
    """Whitespace split + [CLS]/[SEP]; wordpiece-split rare-ish long words."""
    rng = rng_for("tok", model, text)
    pieces = []
    for w in text.split():
        if len(w) > 9 and rng.random() < 0.6:
            cut = rng.randint(3, len(w) - 3)
            pieces += [w[:cut], "##" + w[cut:]]
        else:
            pieces.append(w)
    toks = ["[CLS]"] + pieces[: MAX_TOKENS - 2] + ["[SEP]"]
    special = [t.startswith("[") for t in toks]
    subword = [t.startswith("##") for t in toks]
    ids = [101] + [rng.randint(1000, 28000) for _ in toks[1:-1]] + [102]
    return toks, ids, special, subword


def norm_matrix(n, n_layers, rng):
    """Per-token norms growing across layers, [SEP] dip near the end."""
    rows = []
    for k in range(n_layers + 1):
        base = 1.0 + k * 1.1
        row = [round(base + rng.uniform(-0.4, 0.4), 2) for _ in range(n)]
        if k >= n_layers - 1 and n > 2:
            row[-1] = round(row[-1] * 0.45, 2)  # [SEP] collapses then rebounds
        rows.append(row)
    return rows


def cosine_matrices(n, n_layers, rng):
    mats, means = [], []
    for k in range(n_layers + 1):
        target = 0.06 + (0.42 - 0.06) * (k / n_layers) ** 1.6
        m = [[round(target + rng.uniform(-0.05, 0.05), 4) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            m[i][i] = 1.0
        mats.append(m)
        tri = [m[i][j] for i in range(n) for j in range(i + 1, n)]
        means.append(round(sum(tri) / len(tri), 4) if tri else 1.0)
    return mats, means


def stripe_matrices(n, n_layers, rng):
    out = []
    for k in range(n_layers + 1):
        mat = []
        for i in range(n):
            row = [round(1.6 * math.sin(0.13 * d + 0.7 * i + 0.5 * k) *
                         math.exp(-0.002 * d) + rng.uniform(-0.05, 0.05), 3)
                   for d in range(192)]
            mat.append(row)
        out.append(mat)
    return out


def attention_tensor(n, n_layers, n_heads, rng):
    weights = []
    for _l in range(n_layers):
        heads = []
        for _h in range(n_heads):
            rows = []
            for i in range(n):
                logits = [rng.random() for _ in range(n)]
                logits[0] += rng.uniform(0.5, 2.0)          # [CLS] bias
                logits[-1] += rng.uniform(0.5, 2.5)         # [SEP] bias
                if 0 < i < n - 1:
                    logits[i - 1] += rng.uniform(0.3, 1.8)  # previous-token bias
                mx = max(logits)
                exps = [math.exp(x - mx) for x in logits]
                s = sum(exps)
                rows.append([round(e / s, 4) for e in exps])
            heads.append(rows)
        weights.append(heads)
    return weights


def head_stats(n, n_layers, n_heads, weights):
    stats = []
    real = [i for i in range(n) if not (i == 0 or i == n - 1)]
    for l in range(n_layers):
        for h in range(n_heads):
            w = weights[l][h]
            if not real:
                stats.append({"layer": l, "head": h, "to_sep": 0.0, "to_cls": 0.0,
                              "to_prev": 0.0, "to_next": 0.0})
                continue
            to_sep = sum(w[i][0] + w[i][-1] for i in real) / len(real)
            to_cls = sum(w[i][0] for i in real) / len(real)
            prevs = [w[i][i - 1] for i in real if i - 1 >= 1]
            nexts = [w[i][i + 1] for i in real if i + 1 <= n - 2]
            stats.append({
                "layer": l, "head": h,
                "to_sep": round(to_sep, 4), "to_cls": round(to_cls, 4),
                "to_prev": round(sum(prevs) / len(prevs), 4) if prevs else 0.0,
                "to_next": round(sum(nexts) / len(nexts), 4) if nexts else 0.0,
            })
    return stats


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("mock %s %s\n" % (self.command, self.path))

    # ---- helpers -------------------------------------------------------
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, msg, code=400):
        self._send_json({"detail": msg}, code)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    # ---- routing -------------------------------------------------------
    def do_GET(self):
        if self.path == "/api/meta":
            return self._send_json({"models": MODELS, "default_model": MODELS[0]["id"],
                                    "probes": PROBES})
        path = self.path.split("?")[0]
        if path == "/":
            path = "/static/index.html"
        if path.startswith("/static/"):
            fpath = (STATIC / path[len("/static/"):]).resolve()
            if not str(fpath).startswith(str(STATIC.resolve())) or not fpath.is_file():
                return self._error("not found", 404)
            mime = {".html": "text/html", ".css": "text/css", ".js": "text/javascript",
                    ".svg": "image/svg+xml", ".png": "image/png"}.get(fpath.suffix, "application/octet-stream")
            body = fpath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._error("not found", 404)

    def do_POST(self):
        route = self.path.split("?")[0]
        try:
            body = self._body()
        except json.JSONDecodeError:
            return self._error("invalid JSON body")
        try:
            if route == "/api/tokenize":
                return self.api_tokenize(body)
            if route == "/api/forward":
                return self.api_forward(body)
            if route == "/api/layer":
                return self.api_layer(body)
            if route == "/api/similarity":
                return self.api_similarity(body)
            if route == "/api/probe":
                return self.api_probe(body)
        except Exception as exc:  # mock: never die on bad input
            return self._error(f"mock error: {exc}")
        self._error("not found", 404)

    # ---- endpoints -------------------------------------------------------
    def api_tokenize(self, body):
        model = body.get("model", MODELS[0]["id"])
        text = body.get("text", "")
        uncased = "uncased" in model
        normalized = text.lower() if uncased else text
        rng = rng_for("ids", model, text)
        words = normalized.split()
        pre = []
        pos = 0
        for w in words:
            start = normalized.find(w, pos)
            pre.append({"text": w, "start": max(start, 0), "end": max(start, 0) + len(w)})
            pos = max(start, 0) + len(w)
        toks, ids, special, subword = fake_tokens(text, model)
        offsets = [[0, 0]] + [[p["start"], p["end"]] for p in pre] + [[0, 0]]
        offsets = (offsets + [[0, 0]] * len(toks))[: len(toks)]
        self._send_json({
            "model": model, "text": text,
            "normalizer": "lowercase + strip accents" if uncased else "identity (cased BERT keeps case)",
            "normalized": normalized,
            "pre_tokens": pre,
            "tokens": toks, "input_ids": ids,
            "attention_mask": [1] * len(toks),
            "special_mask": special, "subword_mask": subword,
            "offsets": offsets,
            "decoded": " ".join(toks),
        })

    def api_forward(self, body):
        model = body.get("model", MODELS[0]["id"])
        text = body.get("text", "")
        cfg = next((m for m in MODELS if m["id"] == model), MODELS[0])
        n_layers, n_heads = cfg["num_layers"], cfg["num_heads"]
        toks, ids, special, subword = fake_tokens(text, model)
        if len(toks) > MAX_TOKENS:
            return self._error("text too long for the playground (max 80 tokens) — shorten it")
        rng = rng_for("fwd", model, text)
        n = len(toks)
        norms = norm_matrix(n, n_layers, rng)
        cos, means = cosine_matrices(n, n_layers, rng)
        stripes = stripe_matrices(n, n_layers, rng)
        weights = attention_tensor(n, n_layers, n_heads, rng)
        stats = head_stats(n, n_layers, n_heads, weights)
        self._send_json({
            "model": model, "text": text, "tokens": toks, "input_ids": ids,
            "special_mask": special, "subword_mask": subword,
            "num_hidden_states": n_layers + 1,
            "norms": norms, "cosine": cos, "mean_cosine": means,
            "stripes": stripes, "stripes_dim": 192,
            "attention": {"num_layers": n_layers, "num_heads": n_heads, "weights": weights},
            "head_stats": stats,
            "last_hidden_state_shape": [1, n, cfg["dim"]],
            "pooler_available": "distilbert" not in model,
            "pooler_output_shape": None if "distilbert" in model else [1, cfg["dim"]],
            "cls_preview": [round(rng.uniform(-0.6, 0.6), 4) for _ in range(8)],
            "elapsed_ms": rng.randint(40, 120),
        })

    def api_layer(self, body):
        model = body.get("model", MODELS[0]["id"])
        text = body.get("text", "")
        layer = int(body.get("layer", 0))
        cfg = next((m for m in MODELS if m["id"] == model), MODELS[0])
        if not 0 <= layer <= cfg["num_layers"]:
            return self._error(f"layer must be 0..{cfg['num_layers']}")
        rng = rng_for("layer", model, text, layer)
        toks, _, _, _ = fake_tokens(text, model)
        values = [[round(1.5 * math.sin(0.05 * d + 0.9 * i + 0.3 * layer) *
                         math.exp(-0.0015 * d) + rng.uniform(-0.05, 0.05), 3)
                   for d in range(cfg["dim"])] for i in range(len(toks))]
        self._send_json({"model": model, "text": text, "layer": layer,
                         "tokens": toks, "values": values})

    def api_similarity(self, body):
        model = body.get("model", MODELS[0]["id"])
        sentences = body.get("sentences", [])
        if not sentences:
            return self._error("sentences must be a non-empty list")
        m = len(sentences)
        rng = rng_for("sim", model, *sentences)

        def mat(base):
            out = [[round(1.0 if i == j else base + rng.uniform(-0.12, 0.12), 4)
                    for j in range(m)] for i in range(m)]
            return out

        def pca(seedname):
            r = rng_for("pca", seedname, *sentences)
            xs, ys = [], []
            for i in range(m):
                ang = 2 * math.pi * i / max(m, 1)
                rad = 1.0 + r.uniform(-0.2, 0.2)
                xs.append(round(rad * math.cos(ang), 4))
                ys.append(round(rad * math.sin(ang), 4))
            return {"x": xs, "y": ys}

        self._send_json({
            "model": model, "sentences": sentences,
            "cos_cls": mat(0.83), "cos_mean": mat(0.71),
            "pca_cls": pca("cls"), "pca_mean": pca("mean"),
            "elapsed_ms": rng.randint(120, 400),
        })

    def api_probe(self, body):
        text = body.get("text", "")
        probe_id = body.get("probe", "emotion")
        probe = next((p for p in PROBES if p["id"] == probe_id), None)
        if probe is None or not probe["available"]:
            return self._error(f"unknown or unavailable probe: {probe_id!r} "
                               f"(valid: {', '.join(p['id'] for p in PROBES)})")
        rng = rng_for("probe", probe_id, text)
        toks, _, _, _ = fake_tokens(text, "distilbert-base-uncased")
        labels = probe["labels"]
        k = len(labels)
        logits = [round(rng.uniform(-4, 5), 3) for _ in labels]
        mx = max(logits)
        exps = [math.exp(x - mx) for x in logits]
        s = sum(exps)
        probs = [round(e / s, 4) for e in exps]
        top = probs.index(max(probs))
        self._send_json({
            "probe": probe_id, "text": text, "tokens": toks,
            "labels": labels, "logits": logits, "probs": probs,
            "top_label": labels[top], "top_prob": probs[top],
            "cls_preview": [round(rng.uniform(-0.6, 0.6), 4) for _ in range(16)],
            "w_preview": [[round(rng.uniform(-0.15, 0.15), 4) for _ in range(16)]
                          for _ in labels],
            "hidden_shape": [1, len(toks), 768], "head_shape": [k, 768],
            "elapsed_ms": rng.randint(30, 90),
        })


if __name__ == "__main__":
    url = f"http://localhost:{PORT}"
    print(f"mock backend + static UI at {url}  (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
