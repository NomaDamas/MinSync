"""Regenerates tests/fixtures/tiny_splade (dev-only; never needed at runtime).

A tiny random ModernBERT-MLM with the exact tensor names and layout of
nlpai-lab/KURE-Sparse-nano, a tokenizer, the Sentence-Transformers module
files, and golden SPLADE vectors computed by the Hugging Face reference
(`transformers` ModernBertForMaskedLM + a literal transcription of KURE's
`ShiftedTopKSpladePooling.forward`). The Rust port must reproduce them.

    uv venv .venv && uv pip install --python .venv/bin/python \
        "transformers>=5.14" torch safetensors tokenizers
    .venv/bin/python scripts/gen_tiny_splade_fixture.py tests/fixtures/tiny_splade
"""
import json, os, sys, torch
from safetensors.torch import save_file
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import ModernBertConfig, ModernBertForMaskedLM

OUT = sys.argv[1]
os.makedirs(os.path.join(OUT, "1_ShiftedTopKSpladePooling"), exist_ok=True)
torch.manual_seed(7)

# ---- tokenizer: 70 base ids, then "[Q] " = 70 and "[D] " = 71 (non-special added tokens)
vocab = {"<unk>": 0, "<s>": 1, "<\\s>": 2, "<sep>": 3, "<mask>": 4, "<cls>": 5, "<pad>": 6}
for i in range(7, 10):
    vocab[f"<r{i}>"] = i
for i in range(60):
    vocab[f"w{i}"] = 10 + i
tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
tok.normalizer = normalizers.Lowercase()
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.add_tokens([AddedToken("[Q] ", normalized=True, special=False),
                AddedToken("[D] ", normalized=True, special=False)])
tok.post_processor = processors.TemplateProcessing(
    single="<cls> $A <sep>", special_tokens=[("<cls>", 5), ("<sep>", 3)])
tok.enable_truncation(max_length=8192)
tok.save(os.path.join(OUT, "tokenizer.json"))
Q_ID, D_ID = tok.token_to_id("[Q] "), tok.token_to_id("[D] ")
assert (Q_ID, D_ID) == (70, 71), (Q_ID, D_ID)
PREFIX = [Q_ID, D_ID]
VOCAB, HID, INTER, LAYERS, HEADS = 72, 32, 48, 5, 4

cfg = ModernBertConfig(
    vocab_size=VOCAB, hidden_size=HID, intermediate_size=INTER,
    num_hidden_layers=LAYERS, num_attention_heads=HEADS,
    max_position_embeddings=64, local_attention=8, global_attn_every_n_layers=3,
    pad_token_id=4, bos_token_id=0, eos_token_id=1, cls_token_id=5, sep_token_id=3,
    norm_eps=1e-5, hidden_activation="gelu", classifier_activation="gelu",
    decoder_bias=True, tie_word_embeddings=True,
    attention_bias=False, mlp_bias=False, norm_bias=False, classifier_bias=False,
    rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 160000.0},
                     "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
)
model = ModernBertForMaskedLM(cfg).eval()
with torch.no_grad():
    for p in model.parameters():
        p.normal_(0, 0.35)
    model.decoder.bias.normal_(0, 1.0)
    model.decoder.weight = model.model.embeddings.tok_embeddings.weight
state = {k: v.clone().contiguous() for k, v in model.state_dict().items() if k != "decoder.weight"}
save_file(state, os.path.join(OUT, "model.safetensors"))
json.dump(cfg.to_dict(), open(os.path.join(OUT, "config.json"), "w"), indent=1)

SHIFT = 2.0                      # tiny logits are small; the real model uses 10.0
canon = list(range(VOCAB))
for i in range(30, 40):          # fold a block of words onto earlier ones (WordPiece "##" folding)
    canon[i] = canon[i - 10]
seen, dense = {}, []
for c in canon:
    dense.append(seen.setdefault(c, len(seen)))
DIM = len(seen)
pool = {"pooling_strategy": "max", "activation_function": "relu", "embedding_dimension": DIM,
        "chunk_size": 4, "logit_shift": SHIFT, "top_k": None,
        "prefix_token_ids": PREFIX, "masked_vocab_ids": PREFIX, "canonical_ids": dense}
json.dump(pool, open(os.path.join(OUT, "1_ShiftedTopKSpladePooling/config.json"), "w"))
json.dump([{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.base.modules.transformer.Transformer"},
           {"idx": 1, "name": "1", "path": "1_ShiftedTopKSpladePooling", "type": "pooling.ShiftedTopKSpladePooling"}],
          open(os.path.join(OUT, "modules.json"), "w"))
json.dump({"model_type": "SparseEncoder", "prompts": {"query": "[Q] ", "document": "[D] "},
           "similarity_fn_name": "dot"}, open(os.path.join(OUT, "config_sentence_transformers.json"), "w"))


def pooled(logits, ids, mask):
    """Literal transcription of ShiftedTopKSpladePooling.forward (max, top_k=None)."""
    drop = torch.zeros_like(mask, dtype=torch.bool)
    for t in PREFIX:
        drop |= ids == t
    m = mask.masked_fill(drop, 0).unsqueeze(-1).to(logits.dtype)
    B, S, V = logits.shape
    out = torch.full((B, V), float("-inf"))
    for i in range(0, S, pool["chunk_size"]):
        x = logits[:, i:i + pool["chunk_size"], :] - SHIFT
        x = x.index_fill(-1, torch.as_tensor(PREFIX), 0.0)
        x = (x * m[:, i:i + pool["chunk_size"], :]).relu().log1p()
        out = torch.maximum(out, x.max(dim=1).values)
    folded = out.new_zeros((B, DIM))
    return folded.scatter_reduce(-1, torch.as_tensor(dense).expand_as(out), out, reduce="amax", include_self=False)


def run(texts, kind, max_length):
    prompt = {"query": "[Q] ", "document": "[D] "}[kind]
    tok.enable_truncation(max_length=max_length)
    encs = [tok.encode(prompt + t) for t in texts]
    S = max(len(e.ids) for e in encs)
    ids = torch.full((len(encs), S), 4, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, e in enumerate(encs):
        ids[i, :len(e.ids)] = torch.tensor(e.ids)
        mask[i, :len(e.ids)] = 1
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
    vec = pooled(logits, ids, mask)
    rows = []
    for i, e in enumerate(encs):
        nz = (vec[i] > 0).nonzero().flatten().tolist()
        rows.append({"ids": e.ids, "indices": nz, "values": [float(vec[i, j]) for j in nz]})
    return rows


words = lambda a, b: " ".join(f"w{i}" for i in range(a, b))
texts = [
    "w1 w2 w3",
    words(5, 45),                       # long: forces the sliding window (local_attention=8) to matter
    "w40 w12 w33 w7",                   # w33/w40 fold onto w23/w30 through canonical_ids
    "w9",
    "W4 w50 unknownword w6",            # lowercase normalizer + [UNK]
]
cases = []
for kind in ("document", "query"):
    for max_length in (64, 12):         # 12 truncates the long text
        cases.append({"kind": kind, "max_length": max_length, "texts": texts,
                      "batched": run(texts, kind, max_length),
                      "alone": [run([t], kind, max_length)[0] for t in texts]})
json.dump({"dim": DIM, "shift": SHIFT, "cases": cases}, open(os.path.join(OUT, "golden.json"), "w"))
print("ok dim", DIM, "cases", len(cases), "nnz(batched doc/64):",
      [len(r["indices"]) for r in cases[0]["batched"]])
