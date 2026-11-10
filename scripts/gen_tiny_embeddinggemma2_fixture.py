"""Regenerates tests/fixtures/tiny_embeddinggemma2 (dev-only; never needed at runtime).

A tiny random EmbeddingGemma 2 text tower with the exact tensor names
(`language_model.*`), `config.json` layout and Sentence-Transformers pipeline
(mean pooling over the attention mask, then L2 normalization) of
google/embeddinggemma-2, a tokenizer with the same post-processing
(`<bos> ... <eos>`), and golden vectors computed by the Hugging Face reference
(`transformers` EmbeddingGemma2TextModel). The Rust port must reproduce them.

    uv venv .venv && uv pip install --python .venv/bin/python \
        "transformers>=5.22" torch safetensors tokenizers
    .venv/bin/python scripts/gen_tiny_embeddinggemma2_fixture.py tests/fixtures/tiny_embeddinggemma2
"""
import json, os, sys, torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import EmbeddingGemma2TextConfig
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2TextModel

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(11)

# ---- tokenizer: <pad>=0 <eos>=1 <bos>=2 <unk>=3, then words
vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
for i in range(60):
    vocab[f"w{i}"] = 4 + i
tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
tok.normalizer = normalizers.Lowercase()
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.post_processor = processors.TemplateProcessing(
    single="<bos> $A <eos>", special_tokens=[("<bos>", 2), ("<eos>", 1)])
tok.save(os.path.join(OUT, "tokenizer.json"))

VOCAB, HID, INTER, LAYERS, HEADS, KV = 64, 32, 64, 6, 4, 2
HEAD_DIM, GLOBAL_HEAD_DIM, GLOBAL_KV = 16, 32, 1
PLE, EMB_DIM, WINDOW = 16, 24, 3

cfg = EmbeddingGemma2TextConfig(
    vocab_size=VOCAB, hidden_size=HID, intermediate_size=INTER, num_hidden_layers=LAYERS,
    num_attention_heads=HEADS, num_key_value_heads=KV, head_dim=HEAD_DIM,
    hidden_size_per_layer_input=PLE, embedding_dim=EMB_DIM, sliding_window=WINDOW,
    sliding_window_pattern=3, global_head_dim=GLOBAL_HEAD_DIM,
    num_global_key_value_heads=GLOBAL_KV, max_position_embeddings=4096)
cfg._attn_implementation = "eager"
model = EmbeddingGemma2TextModel(cfg).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.dim() == 1:
            p.copy_(1.0 + 0.3 * torch.randn_like(p))          # norm weights
        else:
            p.copy_(torch.randn_like(p) * (1.5 / p.shape[-1] ** 0.5))
    for layer in model.layers:
        layer.layer_scalar.copy_(0.6 + 0.6 * torch.rand(1))

state = {"language_model." + k: v.clone().contiguous() for k, v in model.state_dict().items()}
save_file(state, os.path.join(OUT, "model.safetensors"))

# config.json in the checkpoint's layout: text_config nested, per_layer_config keyed "NN".
text = {
    "vocab_size": VOCAB, "hidden_size": HID, "intermediate_size": INTER,
    "num_hidden_layers": LAYERS, "num_attention_heads": HEADS, "num_key_value_heads": KV,
    "head_dim": HEAD_DIM, "hidden_activation": "gelu_pytorch_tanh", "rms_norm_eps": 1e-6,
    "attention_bias": False, "sliding_window": WINDOW, "layer_types": cfg.layer_types,
    "hidden_size_per_layer_input": PLE, "embedding_dim": EMB_DIM,
    "rope_parameters": cfg.rope_parameters, "pad_token_id": 0, "bos_token_id": 2,
    "eos_token_id": 1, "max_position_embeddings": 4096,
    "per_layer_config": {f"{i:02d}": {"head_dim": GLOBAL_HEAD_DIM, "num_key_value_heads": GLOBAL_KV}
                         for i, t in enumerate(cfg.layer_types) if t == "full_attention"},
}
json.dump({"architectures": ["EmbeddingGemma2Model"], "model_type": "embedding_gemma2",
           "text_config": text}, open(os.path.join(OUT, "config.json"), "w"), indent=1)


def embed(texts, max_length):
    """Sentence-Transformers pipeline: forward, masked mean pooling, L2 normalize."""
    t = Tokenizer.from_file(os.path.join(OUT, "tokenizer.json"))
    t.enable_truncation(max_length=max_length)
    t.enable_padding(pad_id=0, pad_token="<pad>", direction="right")
    enc = t.encode_batch(texts)
    ids = torch.tensor([e.ids for e in enc])
    mask = torch.tensor([e.attention_mask for e in enc])
    with torch.no_grad():
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
    m = mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * m).sum(1) / m.sum(1)
    return torch.nn.functional.normalize(pooled, dim=-1).tolist(), [sum(e.attention_mask) for e in enc]


words = lambda a, b: " ".join(f"w{i}" for i in range(a, b))
texts = [words(0, 3), words(3, 20), words(20, 24), words(5, 11), "", words(30, 59) + " zzz"]
cases = []
for max_length in (64, 9):                    # 9 truncates the long texts
    batched, lengths = embed(texts, max_length)
    alone = [embed([x], max_length)[0][0] for x in texts]
    cases.append({"max_length": max_length, "batched": batched, "alone": alone, "lengths": lengths})
json.dump({"dim": EMB_DIM, "texts": texts, "cases": cases}, open(os.path.join(OUT, "golden.json"), "w"))
print("ok dim", EMB_DIM, "layer_types", cfg.layer_types, "lengths", cases[0]["lengths"])
