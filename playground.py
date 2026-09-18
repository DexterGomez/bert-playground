"""BERT playground — backend logic (no web framework here).

Pure functions that load the encoders and build JSON payloads matching
API_CONTRACT.md exactly (key names, shapes, rounding). `app.py` is a thin
FastAPI wrapper around this module, so everything here can also be exercised
from a REPL or notebook during class.

What this module gives you, per endpoint:

  meta_payload()                          -> model + probe registry
  tokenize_payload(model, text)           -> tokenizer pipeline breakdown
  forward_payload(model, text)            -> all 13 hidden states, norms,
                                             cosines, stripes, attention,
                                             head stats, [CLS] preview
  layer_payload(model, text, layer)       -> one hidden state at full 768 dims
  similarity_payload(model, sentences)    -> CLS/mean-pooling cosines + PCA
  probe_payload(probe, text)              -> fine-tuned classifier head

Conventions:
  - CPU only, `eval()` mode, every forward under `torch.no_grad()`.
  - Attention must be requested with `attn_implementation="eager"` — the
    transformers 5.x default (SDPA) does not return attention weights.
  - Tensor -> JSON conversion always goes through numpy: round first
    (decimals per the contract), then `.tolist()` for native Python floats.
  - Expensive results are cached with `functools.lru_cache`; payloads are
    shared by reference between requests, so treat them as read-only.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger("bert_playground")

# --------------------------------------------------------------------------
# Registry and constants
# --------------------------------------------------------------------------

#: Playground-wide token limit (contract "Limits" section). `tokenize` is
#: exempt — it just reports what the tokenizer produced.
MAX_TOKENS = 80
TOO_LONG_MSG = "text too long for the playground (max 80 tokens) — shorten it"

DEVICE = torch.device("cpu")

#: Local fine-tune from the week-3 emotion notebook (demo 2), resolved
#: relative to this file: ../new_custom_notebooks/<name>
THIS_DIR = Path(__file__).resolve().parent
EMOTION_DIR = THIS_DIR.parent / "new_custom_notebooks" / "distilbert-base-uncased-finetuned-emotion-final"

#: SST-2 classifier from the HF cache — never downloaded at runtime.
SST2_ID = "distilbert-base-uncased-finetuned-sst-2-english"

MODELS = [
    {"id": "bert-base-cased", "label": "BERT-base · cased · 12 layers",
     "num_layers": 12, "num_heads": 12, "dim": 768},
    {"id": "distilbert-base-uncased", "label": "DistilBERT · uncased · 6 layers",
     "num_layers": 6, "num_heads": 12, "dim": 768},
]
MODEL_IDS = frozenset(m["id"] for m in MODELS)
DEFAULT_MODEL = MODELS[0]["id"]

#: Probe classifiers (tab 5). Independent of the top-bar model selector —
#: both are DistilBERT-uncased sequence classifiers.
PROBES = {
    "emotion": {
        "label": "Emotion head (demo-2 fine-tune)",
        "source": EMOTION_DIR,          # local directory
        "default_labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    },
    "sst2": {
        "label": "SST-2 sentiment",
        "source": SST2_ID,              # local HF cache only — never download
        "default_labels": ["negative", "positive"],
    },
}

# --------------------------------------------------------------------------
# Model loading (lazy + cached, race-safe under uvicorn's threadpool)
# --------------------------------------------------------------------------

_MODELS: dict[str, tuple] = {}          # model_id -> (tokenizer, model)
_PROBE_MODELS: dict[str, tuple] = {}    # probe_id -> (tokenizer, model)
_PROBE_AVAILABILITY: dict[str, bool] = {}
_LOAD_LOCK = threading.Lock()


def get_model(model_id: str):
    """Return `(tokenizer, model)` for a registry id, loading on first use.

    Raises ValueError for ids outside the registry (app.py maps that to 400).
    """
    if model_id not in MODEL_IDS:
        valid = ", ".join(m["id"] for m in MODELS)
        raise ValueError(f"unknown model id: {model_id!r} (valid: {valid})")
    if model_id not in _MODELS:
        with _LOAD_LOCK:
            if model_id not in _MODELS:  # re-check inside the lock
                t0 = time.perf_counter()
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModel.from_pretrained(
                    model_id,
                    # eager attention: SDPA never returns attention weights
                    attn_implementation="eager",
                    # keep every layer's hidden state (13 for BERT-base)
                    output_hidden_states=True,
                )
                model.to(DEVICE)
                model.eval()
                _MODELS[model_id] = (tokenizer, model)
                logger.info("loaded %s (%.1fs, eager attention, cpu)",
                            model_id, time.perf_counter() - t0)
    return _MODELS[model_id]


def probe_available(probe_id: str) -> bool:
    """Cheap availability check — never loads weights.

    emotion: directory present?  sst2: tiny local-only config load, cached
    after the first attempt. Used by /api/meta so the UI can gray out a
    probe whose checkpoint is missing.
    """
    if probe_id not in PROBES:
        return False
    if probe_id in _PROBE_AVAILABILITY:
        return _PROBE_AVAILABILITY[probe_id]
    if probe_id == "emotion":
        ok = EMOTION_DIR.is_dir()
    else:
        try:
            AutoConfig.from_pretrained(PROBES[probe_id]["source"], local_files_only=True)
            ok = True
        except Exception:
            ok = False
    _PROBE_AVAILABILITY[probe_id] = ok
    if not ok:
        logger.warning("probe %r unavailable (source not found: %s)",
                       probe_id, PROBES[probe_id]["source"])
    return ok


def _get_probe_model(probe_id: str):
    """Return `(tokenizer, classifier)` for a probe, loading lazily.

    Raises ValueError for unknown ids and for probes that fail to load —
    both surface as HTTP 400 "unknown or unavailable probe" per the contract.
    """
    if probe_id not in PROBES or (probe_id in _PROBE_AVAILABILITY
                                  and not _PROBE_AVAILABILITY[probe_id]
                                  and probe_id not in _PROBE_MODELS):
        raise ValueError(_probe_error(probe_id))
    if probe_id not in _PROBE_MODELS:
        with _LOAD_LOCK:
            if probe_id not in _PROBE_MODELS:
                try:
                    source = PROBES[probe_id]["source"]
                    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        source, local_files_only=True)
                    model.to(DEVICE)
                    model.eval()
                except Exception as exc:
                    _PROBE_AVAILABILITY[probe_id] = False
                    logger.warning("probe %r failed to load: %s", probe_id, exc)
                    raise ValueError(_probe_error(probe_id)) from exc
                _PROBE_MODELS[probe_id] = (tokenizer, model)
                logger.info("loaded probe %r from %s", probe_id, source)
    return _PROBE_MODELS[probe_id]


def _probe_error(probe_id: str) -> str:
    valid = ", ".join(PROBES)
    return f"unknown or unavailable probe: {probe_id!r} (valid: {valid})"


def _probe_labels(probe_id: str) -> list[str]:
    """Label list for /api/meta without loading any weights (config only)."""
    try:
        cfg = AutoConfig.from_pretrained(PROBES[probe_id]["source"], local_files_only=True)
        labels = [cfg.id2label[i] for i in range(len(cfg.id2label))]
    except Exception:
        return list(PROBES[probe_id]["default_labels"])
    # e.g. sst2 ships UPPERCASE labels; the contract example uses lowercase
    return [label.lower() for label in labels]


# --------------------------------------------------------------------------
# Shared tensor plumbing
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def _encode(model_id: str, text: str):
    """Tokenize once per (model, text) and cache — offsets map included."""
    tokenizer, _ = get_model(model_id)
    return tokenizer(text, return_tensors="pt", return_offsets_mapping=True)


def _checked_encode(model_id: str, text: str):
    """`_encode` plus the 80-token playground limit (forward-style routes)."""
    enc = _encode(model_id, text)
    if int(enc["input_ids"].shape[1]) > MAX_TOKENS:
        raise ValueError(TOO_LONG_MSG)
    return enc


@functools.lru_cache(maxsize=64)
def _forward(model_id: str, text: str):
    """Run the encoder once per (model, text); cache output + forward ms."""
    _, model = get_model(model_id)
    enc = _checked_encode(model_id, text)
    # the model must not receive the offset mapping
    inputs = {k: v for k, v in enc.items()
              if k in ("input_ids", "attention_mask", "token_type_ids")}
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
    return out, elapsed_ms


def _token_info(tokenizer, enc) -> tuple[list, list, list, list]:
    """(input_ids, tokens, special_mask, subword_mask) from one encoding."""
    ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    special = [t in ("[CLS]", "[SEP]") for t in tokens]
    subword = [t.startswith("##") for t in tokens]
    return ids, tokens, special, subword


def _np64(t) -> np.ndarray:
    """tensor -> float64 numpy. ALWAYS convert before rounding: rounding a
    float32 array keeps float32 noise (0.637 -> 0.6370000243...)."""
    return t.detach().double().cpu().numpy()


def _cosine_matrix(vectors) -> list:
    """Cosine matrix of L2-normalized rows: symmetric, diagonal exactly 1.0."""
    v = np.asarray(vectors, dtype=np.float64)
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    c = u @ u.T
    c = (c + c.T) / 2.0          # kill float asymmetry from BLAS
    np.fill_diagonal(c, 1.0)
    return np.round(c, 4).tolist()


def _pca_coords(vectors) -> dict:
    """2-D PCA projection as {"x": [...], "y": [...]}.

    m >= 3: two components. m == 2: one component, y padded with 0.0.
    m == 1: a single point — all zeros (PCA needs 2 points to rotate).
    """
    v = np.asarray(vectors, dtype=np.float64)
    m = v.shape[0]
    if m == 1:
        return {"x": [0.0], "y": [0.0]}
    n_components = 2 if m >= 3 else m - 1
    coords = PCA(n_components=n_components).fit_transform(v)
    x = np.round(coords[:, 0], 4).tolist()
    y = np.round(coords[:, 1], 4).tolist() if n_components == 2 else [0.0] * m
    return {"x": x, "y": y}


def _describe_normalizer(normalizer) -> str:
    """One-line human description for the tokenize tab."""
    name = type(normalizer).__name__
    if name == "BertNormalizer" and not getattr(normalizer, "lowercase", False):
        return "identity (cased BERT keeps case)"
    if name == "BertNormalizer":
        return "lowercase + strip accents (uncased BERT)"
    return name


# --------------------------------------------------------------------------
# Payload builders — one function per API route, shapes per API_CONTRACT.md
# --------------------------------------------------------------------------

def meta_payload() -> dict:
    """GET /api/meta — model + probe registry (static config, cheap)."""
    probes = [
        {
            "id": pid,
            "label": spec["label"],
            "available": probe_available(pid),
            "labels": _probe_labels(pid),
        }
        for pid, spec in PROBES.items()
    ]
    return {
        "models": [dict(m) for m in MODELS],
        "default_model": DEFAULT_MODEL,
        "probes": probes,
    }


@functools.lru_cache(maxsize=512)
def tokenize_payload(model_id: str, text: str) -> dict:
    """POST /api/tokenize — the fast tokenizer's pipeline, step by step.

    No 80-token limit here (the contract only limits the forward-style
    routes), so long texts can still be inspected at the token level.
    """
    tokenizer, _ = get_model(model_id)   # validates model_id (ValueError)
    enc = _encode(model_id, text)
    ids, tokens, special, subword = _token_info(tokenizer, enc)

    backend = tokenizer.backend_tokenizer
    normalizer = backend.normalizer
    if normalizer is None:
        normalized = text
        normalizer_desc = "none (identity)"
    else:
        normalized = normalizer.normalize_str(text)
        normalizer_desc = _describe_normalizer(normalizer)

    # pre-tokenizer runs on the NORMALIZED string; offsets are into it
    pre_tokens = [
        {"text": piece, "start": int(start), "end": int(end)}
        for piece, (start, end) in backend.pre_tokenizer.pre_tokenize_str(normalized)
    ]

    return {
        "model": model_id,
        "text": text,
        "normalizer": normalizer_desc,
        "normalized": normalized,
        "pre_tokens": pre_tokens,
        "tokens": tokens,
        "input_ids": ids,
        "attention_mask": enc["attention_mask"][0].tolist(),
        "special_mask": special,
        "subword_mask": subword,
        # offsets are relative to the RAW text; special tokens get [0, 0]
        "offsets": enc["offset_mapping"][0].tolist(),
        "decoded": tokenizer.decode(ids),
    }


@functools.lru_cache(maxsize=32)
def forward_payload(model_id: str, text: str) -> dict:
    """POST /api/forward — the full internal state for tabs 2 and 3.

    hidden_states: 0 = embedding output, 1..L = after each encoder layer
    (L+1 entries — 13 for BERT-base, 7 for DistilBERT).
    """
    tokenizer, model = get_model(model_id)
    enc = _checked_encode(model_id, text)
    out, elapsed_ms = _forward(model_id, text)

    ids, tokens, special, subword = _token_info(tokenizer, enc)
    n = len(tokens)
    real = [i for i in range(n) if not special[i]]   # non-special positions
    real_set = set(real)
    sep_positions = [j for j, t in enumerate(tokens) if t == "[SEP]"]
    cls_positions = [j for j, t in enumerate(tokens) if t == "[CLS]"]

    # one [n, 768] numpy matrix per hidden state (embeddings .. last layer)
    hidden = [_np64(h[0]) for h in out.hidden_states]

    # residual-stream growth: L2 norm of each token vector, per state
    norms = [np.round(np.linalg.norm(h, axis=1), 2).tolist() for h in hidden]

    # token-token cosines per state; mean over the upper triangle (i < j)
    cosine, mean_cosine = [], []
    for h in hidden:
        u = h / np.linalg.norm(h, axis=1, keepdims=True)
        c = (u @ u.T + (u @ u.T).T) / 2.0
        np.fill_diagonal(c, 1.0)
        mean_cosine.append(round(float(c[np.triu_indices(n, k=1)].mean()), 4))
        cosine.append(np.round(c, 4).tolist())

    # every 4th dim of the 768 -> n x 192 heatmap stripes (raw values)
    stripes = [np.round(h[:, ::4], 3).tolist() for h in hidden]

    # attention: stack the per-layer [1, H, n, n] tensors -> [L, H, n, n]
    attn = _np64(torch.stack(out.attentions).squeeze(1))
    weights = np.round(attn, 4).tolist()

    # attention habits per (layer, head), over non-special QUERY tokens
    head_stats = []
    n_layers, n_heads = attn.shape[0], attn.shape[1]
    for l in range(n_layers):
        for h_i in range(n_heads):
            w = attn[l, h_i]
            if real:
                to_sep = float(np.mean([w[i, sep_positions].sum() for i in real]))
                to_cls = float(np.mean([w[i, cls_positions].sum() for i in real]))
                prevs = [float(w[i, i - 1]) for i in real if (i - 1) in real_set]
                nexts = [float(w[i, i + 1]) for i in real if (i + 1) in real_set]
            else:  # empty text: only [CLS]/[SEP] — no real queries at all
                to_sep = to_cls = 0.0
                prevs = nexts = []
            head_stats.append({
                "layer": l, "head": h_i,
                "to_sep": round(to_sep, 4),
                "to_cls": round(to_cls, 4),
                "to_prev": round(sum(prevs) / len(prevs), 4) if prevs else 0.0,
                "to_next": round(sum(nexts) / len(nexts), 4) if nexts else 0.0,
            })

    # DistilBERT's output object has no pooler at all -> getattr, not attribute
    pooler = getattr(out, "pooler_output", None)

    return {
        "model": model_id,
        "text": text,
        "tokens": tokens,
        "input_ids": ids,
        "special_mask": special,
        "subword_mask": subword,
        "num_hidden_states": len(hidden),
        "norms": norms,
        "cosine": cosine,
        "mean_cosine": mean_cosine,
        "stripes": stripes,
        "stripes_dim": int(hidden[0].shape[1] // 4),
        "attention": {"num_layers": n_layers, "num_heads": n_heads,
                      "weights": weights},
        "head_stats": head_stats,
        "last_hidden_state_shape": [1, n, int(hidden[0].shape[1])],
        "pooler_available": pooler is not None,
        "pooler_output_shape": [int(d) for d in pooler.shape] if pooler is not None else None,
        "cls_preview": np.round(hidden[-1][0, :8], 4).tolist(),
        "elapsed_ms": elapsed_ms,
    }


@functools.lru_cache(maxsize=128)
def layer_payload(model_id: str, text: str, layer: int) -> dict:
    """POST /api/layer — one hidden state at full 768-dim resolution."""
    tokenizer, model = get_model(model_id)
    num_layers = model.config.num_hidden_layers
    if not 0 <= layer <= num_layers:
        raise ValueError(f"layer must be 0..{num_layers} for {model_id} (got {layer})")
    enc = _checked_encode(model_id, text)
    out, _ = _forward(model_id, text)
    _, tokens, _, _ = _token_info(tokenizer, enc)
    values = np.round(_np64(out.hidden_states[layer][0]), 3).tolist()
    return {"model": model_id, "text": text, "layer": layer,
            "tokens": tokens, "values": values}


def similarity_payload(model_id: str, sentences) -> dict:
    """POST /api/similarity — sentence-embedding geometry for tab 4.

    `sentences` arrives as a JSON list; validated here so a bad body is a
    clean 400 rather than an unhashable-cache TypeError.
    """
    if not isinstance(sentences, (list, tuple)) or not sentences:
        raise ValueError("sentences must be a non-empty list of strings")
    if not all(isinstance(s, str) for s in sentences):
        raise ValueError("sentences must be a non-empty list of strings")
    return _similarity_payload(model_id, tuple(sentences))


@functools.lru_cache(maxsize=32)
def _similarity_payload(model_id: str, sentences: tuple) -> dict:
    tokenizer, _ = get_model(model_id)
    t0 = time.perf_counter()

    cls_vecs, mean_vecs = [], []
    for sentence in sentences:            # each obeys the 80-token limit
        enc = _checked_encode(model_id, sentence)
        out, _ = _forward(model_id, sentence)
        last = _np64(out.last_hidden_state[0])             # [n, 768]
        _, _, special, _ = _token_info(tokenizer, enc)
        # CLS pooling: the [CLS] vector itself
        cls_vecs.append(last[0])
        # mean pooling over NON-special tokens (empty text: fall back to the
        # plain mean so we never emit NaN)
        mask = np.array([not s for s in special])
        mean_vecs.append(last[mask].mean(axis=0) if mask.any() else last.mean(axis=0))

    payload = {
        "model": model_id,
        "sentences": list(sentences),
        "cos_cls": _cosine_matrix(cls_vecs),
        "cos_mean": _cosine_matrix(mean_vecs),
        "pca_cls": _pca_coords(cls_vecs),
        "pca_mean": _pca_coords(mean_vecs),
        "elapsed_ms": int(round((time.perf_counter() - t0) * 1000)),
    }
    return payload


@functools.lru_cache(maxsize=128)
def probe_payload(probe_id: str, text: str) -> dict:
    """POST /api/probe — a fine-tuned head reads the [CLS] vector (tab 5)."""
    tokenizer, model = _get_probe_model(probe_id)
    enc = tokenizer(text, return_tensors="pt")
    if int(enc["input_ids"].shape[1]) > MAX_TOKENS:
        raise ValueError(TOO_LONG_MSG)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    elapsed_ms = int(round((time.perf_counter() - t0) * 1000))

    ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    labels = [model.config.id2label[i].lower()
              for i in range(len(model.config.id2label))]

    logits = _np64(out.logits[0])
    probs = np.exp(logits - logits.max())               # stable softmax
    probs /= probs.sum()
    top = int(probs.argmax())

    hidden_cls = _np64(out.hidden_states[-1][0, 0])     # final [CLS] vector
    w = _np64(model.classifier.weight)                  # [k, 768]

    return {
        "probe": probe_id,
        "text": text,
        "tokens": tokens,
        "labels": labels,
        "logits": np.round(logits, 3).tolist(),
        "probs": np.round(probs, 4).tolist(),
        "top_label": labels[top],
        "top_prob": round(float(probs[top]), 4),
        "cls_preview": np.round(hidden_cls[:16], 4).tolist(),
        "w_preview": np.round(w[:, :16], 4).tolist(),
        "hidden_shape": [int(d) for d in out.hidden_states[-1].shape],
        "head_shape": [int(d) for d in w.shape],
        "elapsed_ms": elapsed_ms,
    }
