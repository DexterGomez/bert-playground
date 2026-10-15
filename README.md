# bert-playground — watch BERT think, live

Interactive lecture tool for the week-3 "inside BERT" block. Type any sentence
and walk the class through everything the model does with it: how the string
becomes tokens, what happens inside all 12 encoder layers, where every head
looks, what the output embeddings "feel" like, and how a fine-tuned head reads
them out. Built as a FastAPI backend + vanilla-JS front end (no frameworks, no
CDNs — works offline in the classroom).

## Run

```bash
cd weeks/03-encoders-transformers/demos/bert-playground
.venv/bin/python app.py          # → http://127.0.0.1:8303
```

First run loads `bert-base-cased` + `bert-base-uncased` +
`distilbert-base-uncased` + `sentence-transformers/bert-base-nli-mean-tokens`
(~10 s, all four are in the local HF cache — no network needed). The uncased
entry is the checkpoint behind the week-3 lecture numbers (the `it` → animal
example: layer 4 head 11 = sliders 3/10 in the attention tab). Forwards run on
CPU: a single
sentence is ~0.5 s cold, instant after (results are cached per model+text).

The emotion probe loads from
`../new_custom_notebooks/distilbert-base-uncased-finetuned-emotion-final`
(the demo-2 fine-tune). If that folder is missing the probe just shows as
unavailable. The SST-2 probe uses the cached HF model. Rebuild the venv with
`python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## The five tabs (and the story they tell)

1. **Tokenize** — raw text → normalizer → pre-tokens → WordPiece (`##`) →
   integer ids + attention mask. Toggle *compare cased vs uncased* to show
   what normalization does (`Los transformadores son increíbles` loses its
   accents in the uncased model). Preset chip "bank / river bank" for
   context-dependence; a long invented word shows subword splitting.
2. **Inside the encoder** — the residual stream. Layer slider 0–12 (0 =
   embeddings): each row is one token's `[768]` vector as color. *Full 768
   dims* fetches the true layer. Right: per-token norms per layer — the
   stream **grows** as each layer adds its update. Beats to narrate: the dip
   at the last step is the final LayerNorm rescaling, not information loss;
   `[SEP]` collapses at layer 11 then rebounds — special tokens are weird.
   Bottom: inter-token cosine per layer — mean cosine drifts ≈0.06 → ≈0.42:
   **anisotropy**, tokens become more similar, not less.
   **SBERT model option:** the third dropdown entry is the SAME 12-layer BERT
   fine-tuned by Sentence-BERT (NLI, mean pooling) — an amber note under the
   tab caption flags it. The class-visible delta: the mid layers go flat
   (mean cosine plateaus ≈0.24–0.34, colors barely change layer to layer),
   then L11–L12 collapse the cone *harder* than base BERT ever does (≈0.86–0.90
   at L12). Story: the SBERT loss only ever sees the mean-pooled sentence
   vector, so it reshapes exactly the pooled end — sentence-level uniformity
   is bought WITH token-level anisotropy. Warning the note carries: SBERT
   exports the mean of these token vectors, not the token vectors — don't use
   it for token-level work.
3. **Attention** — layer + head sliders; the n×n matrix (rows = the token
   looking, cols = where it looks; each row sums to 1), BertViz-style arcs
   (hover/click a token chip), and the layer×head grid colored by attention
   to [SEP] / [CLS] / previous / next — the classic "delimiter heads" and
   "positional heads". Click a grid cell to jump to that head.
4. **Embeddings** — the week-4 bridge. Two sentences → cosine of their
   embeddings, `[CLS]` vs mean pooling. Sentence bank → similarity matrix +
   PCA map. Narrate: raw BERT embeddings are anisotropic (CLS cosines all
   pile up near 1; tick *stretch to min–max* to reveal structure) — BERT was
   never trained to make cosine = meaning; week 4 swaps in embedders
   (BGE / E5) that are.
5. **Head probe** — U2T01 motivation. The demo-2 emotion fine-tune
   opened up: `last_hidden_state → [CLS] → × W[6,768] → logits → softmax`.
   Switch to SST-2 and type `I hate transformer encoders` → **NEGATIVE
   99.9%** — the model has no sense of humor; the encoder reads, the head
   labels. U2T01 swaps this head.

## Layout

```
app.py            FastAPI: routes, startup model load, serves static/ (port 8303)
playground.py     pure logic: tokenizer/forward/similarity/probe payload builders
static/           index.html + style.css + js/{api,viz,views,main}.js (no build step)
tools/mock_server.py  stdlib mock of the whole API (+ serves static/) — front-end
                  development without the ML stack: python3 tools/mock_server.py → :8304
API_CONTRACT.md   the exact JSON contract shared by backend & frontend (frozen v1)
.venv/            python 3.11, see requirements.txt (mirrors the week-3 notebook stack)
```

To change an endpoint's shape, update `API_CONTRACT.md` first, then both sides
— the mock server stays in sync for offline UI work.

## Troubleshooting

- **"backend not running?" banner** → start `app.py`; the page itself is
  served by the same process.
- **Port busy** → `lsof -ti :8303 | xargs kill` or run on another port:
  `.venv/bin/uvicorn app:app --port 8305`.
- **Slow first sentence** → normal: model forward + JSON build; subsequent
  identical sentences are cached.
- Texts longer than 80 tokens are rejected by design (payloads would blow up
  the scrubbing experience) — the error banner says so.
