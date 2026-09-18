# BERT Playground — API contract (v1)

Single source of truth shared by the backend (`app.py` + `playground.py`) and the
frontend (`static/`). **Both sides must match this exactly — key names, shapes,
rounding.** If a change is ever needed, change this file first, then both sides.

- Real backend: `.venv/bin/uvicorn app:app --port 8303` → http://localhost:8303
- Mock backend (stdlib only, no ML deps): `python3 tools/mock_server.py` → http://localhost:8304
- Both serve the UI at `GET /` → `static/index.html`, assets under `/static/...`,
  and the JSON API under `/api/...`. Same origin — no CORS needed.

All responses are `application/json`. Errors are non-2xx with body
`{"detail": "human-readable message"}` (FastAPI `HTTPException` style).
No NaN/Infinity anywhere — clamp or round first.

## Rounding rules (apply when building payloads)

| field kind | decimals |
|---|---|
| `norms` | 2 |
| `cosine`, `mean_cosine`, attention `weights`, `probs`, `pca` coords | 4 |
| `logits`, `stripes`, `/api/layer` values | 3 |
| `*_preview` | 4 |

## Limits

- If tokenizing produces **more than 80 tokens**, `/api/forward`, `/api/layer`,
  `/api/similarity`, `/api/probe` return `400` with detail
  `"text too long for the playground (max 80 tokens) — shorten it"`.
- Empty text is legal: tokenizer returns just `[CLS] [SEP]` (n=2).

---

## GET /api/meta

Model + probe registry. Static configuration, safe to cache client-side.

```json
{
  "models": [
    {"id": "bert-base-cased", "label": "BERT-base · cased · 12 layers", "num_layers": 12, "num_heads": 12, "dim": 768},
    {"id": "distilbert-base-uncased", "label": "DistilBERT · uncased · 6 layers", "num_layers": 6, "num_heads": 12, "dim": 768}
  ],
  "default_model": "bert-base-cased",
  "probes": [
    {"id": "emotion", "label": "Emotion head (demo-2 fine-tune)", "available": true, "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"]},
    {"id": "sst2", "label": "SST-2 sentiment", "available": true, "labels": ["negative", "positive"]}
  ]
}
```

`num_layers` = encoder layer count. Hidden-state indices run `0..num_layers`
inclusive where **0 = embedding output** (13 entries for BERT-base).
Probe `available: false` when the model can't be loaded (probe stays listed, UI grays it out).

---

## POST /api/tokenize  — body `{"model": "<id>", "text": "<string>"}`

Pipeline breakdown of the fast tokenizer.

```json
{
  "model": "bert-base-cased",
  "text": "I hate transformer encoders",
  "normalizer": "identity (cased BERT keeps case)",
  "normalized": "I hate transformer encoders",
  "pre_tokens": [{"text": "I", "start": 0, "end": 1}, {"text": "hate", "start": 2, "end": 6}],
  "tokens": ["[CLS]", "I", "hate", "transformer", "encoders", "[SEP]"],
  "input_ids": [101, 146, 4819, 11423, 14246, 102],
  "attention_mask": [1, 1, 1, 1, 1, 1],
  "special_mask": [true, false, false, false, false, true],
  "subword_mask": [false, false, false, false, false, false],
  "offsets": [[0, 0], [0, 1], [2, 6], [7, 19], [20, 28], [0, 0]],
  "decoded": "[CLS] I hate transformer encoders [SEP]"
}
```

- `normalizer`: one-line human description; `normalized` = the string after the
  normalizer (same as `text` for cased BERT; lowercased for uncased models).
- `pre_tokens`: `tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(normalized)`
  → `[{"text", "start", "end"}...]` with offsets into `normalized`.
- `tokens`/`input_ids`/`attention_mask`/`offsets`: from
  `tokenizer(text, return_offsets_mapping=True)` (offsets relative to `text`;
  special tokens get `[0, 0]`).
- `special_mask[i]`: true for `[CLS]`/`[SEP]`. `subword_mask[i]`: true for
  wordpiece continuations (token starts with `##`).
- `decoded`: `tokenizer.decode(input_ids)`.

---

## POST /api/forward  — body `{"model": "<id>", "text": "<string>"}`

The full internal state for tabs 2 (encoder) and 3 (attention).

```json
{
  "model": "bert-base-cased",
  "text": "I hate transformer encoders",
  "tokens": ["[CLS]", "I", "hate", "transformer", "encoders", "[SEP]"],
  "input_ids": [101, 146, 4819, 11423, 14246, 102],
  "special_mask": [true, false, false, false, false, true],
  "subword_mask": [false, false, false, false, false, false],
  "num_hidden_states": 13,
  "norms": [[6.71, 4.63, ...] /* n */ ... /* 13 arrays */],
  "cosine": [[[1.0, 0.06, ...] ... n] ... 13 matrices n×n],
  "mean_cosine": [0.0612, 0.0718, ...  /* 13, mean of upper triangle */],
  "stripes": [[[0.412, -0.108, ...] /* 192 */ ... n] ... 13 matrices n×192],
  "stripes_dim": 192,
  "attention": {
    "num_layers": 12,
    "num_heads": 12,
    "weights": [ /* [layer][head] = n×n matrix, weights[l][h][query][key] */ ]
  },
  "head_stats": [
    {"layer": 0, "head": 0, "to_sep": 0.21, "to_cls": 0.08, "to_prev": 0.11, "to_next": 0.07},
    ... /* num_layers × num_heads entries, row-major (l outer, h inner) */
  ],
  "last_hidden_state_shape": [1, 6, 768],
  "pooler_available": true,
  "pooler_output_shape": [1, 768],
  "cls_preview": [0.0891, -0.4523, ... /* first 8 dims of final [CLS] hidden state */],
  "elapsed_ms": 84
}
```

Definitions:
- `norms[k][i]` = L2 norm of hidden state `k` (0 = embeddings) for token `i`.
- `cosine[k]` = n×n matrix of cosines between token vectors at hidden state `k`
  (diagonal exactly 1.0). `mean_cosine[k]` = mean over pairs `i < j` (all tokens).
- `stripes[k]` = hidden state `k` downsampled `[:, :, ::4]` → n×192 raw values
  (client handles coloring/scaling). `stripes_dim` = 768 / 4 = 192.
- `attention.weights[l][h][i][j]` = softmax attention from query token `i` to
  key token `j` in layer `l` (0-based), head `h`. Rows sum to 1.
- `head_stats` per (layer, head), computed over **non-special query tokens**:
  - `to_sep` = mean over queries of the total mass on [SEP] positions
  - `to_cls` = mean mass on the [CLS] position
  - `to_prev` = mean of `w[i][i-1]` for real tokens that have a real predecessor
  - `to_next` = mean of `w[i][i+1]` for real tokens that have a real successor
- `pooler_available` false for DistilBERT (no pooler head); `pooler_output_shape` null then.
- `elapsed_ms` = model forward wall time (integer ms).

---

## POST /api/layer  — body `{"model": "<id>", "text": "<string>", "layer": 7}`

Full-resolution hidden state for one layer (on-demand detail view).

```json
{"model": "...", "text": "...", "layer": 7, "tokens": ["[CLS]", ...],
 "values": [[0.412, -0.108, ... /* 768 */] ... n]}
```

`layer` ∈ `0..num_layers` (0 = embeddings). Out of range → 400.

---

## POST /api/similarity  — body `{"model": "<id>", "sentences": ["...", "...", ...]}`

Embedding geometry for tab 4. Both pooling variants always returned.

```json
{
  "model": "...",
  "sentences": ["...", ...],
  "cos_cls": [[1.0, 0.83, ...] ... m×m],
  "cos_mean": [[1.0, 0.77, ...] ... m×m],
  "pca_cls": {"x": [..m], "y": [..m]},
  "pca_mean": {"x": [..m], "y": [..m]},
  "elapsed_ms": 210
}
```

Definitions:
- `cls` embedding = `last_hidden_state[:, 0, :]` (the [CLS] vector).
- `mean` embedding = mean over **non-special** token vectors of `last_hidden_state`.
- Cosines are computed on L2-normalized vectors; matrices are symmetric,
  diagonal exactly 1.0.
- PCA: `sklearn.decomposition.PCA(n_components=2)` fit separately on each
  pooling's m×768 matrix (un-normalized is fine). If `m < 3`, use
  `n_components = m - 1` and pad missing coords with `0.0` (m=1 → all zeros).
- Empty `sentences` array → 400. Sentences individually obey the 80-token limit.

---

## POST /api/probe  — body `{"text": "<string>", "probe": "emotion" | "sst2"}`

Fine-tuned classification head for tab 5. Note: probe models are **DistilBERT
uncased** regardless of the top-bar model selector.

```json
{
  "probe": "emotion",
  "text": "...",
  "tokens": ["[CLS]", ...],
  "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
  "logits": [-3.412, 4.871, ...],
  "probs": [0.0011, 0.9312, ...],
  "top_label": "joy",
  "top_prob": 0.9312,
  "cls_preview": [0.0891, ... /* first 16 dims of the [CLS] hidden state */],
  "w_preview": [[0.021, -0.113, ... /* 16 */] ... one row per label],
  "hidden_shape": [1, 12, 768],
  "head_shape": [6, 768],
  "elapsed_ms": 41
}
```

- `cls_preview` / `w_preview`: first 16 dims of the final [CLS] hidden state and
  of `classifier.weight` (row per label) — real numbers for the
  "CLS vector · W row → logit" visual.
- Unknown `probe` id or unavailable probe → 400 with detail naming valid ids.

---

## Frontend test data notes (for the mock server)

The mock generates deterministic, contract-shaped data (seeded by text hash):
plausible norms (~1 at layer 0 growing ~linearly to ~14), cosine matrices whose
mean rises 0.06 → 0.42, stripes = smooth sinusoid noise, attention rows from a
softmax with extra mass on [SEP]/[CLS]/previous token. Shape realism matters
more than numeric realism — the UI must render anything matching this contract.
