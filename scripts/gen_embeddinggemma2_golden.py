"""Regenerates tests/fixtures/embeddinggemma2/golden_real.json (dev-only).

Reference vectors of the REAL google/embeddinggemma-2 (Sentence-Transformers,
fp32, CPU, text only, truncated at MinSync's 2048-token default) for
tests/gemma2_live.rs.

    uv venv .venv && uv pip install --python .venv/bin/python \
        "transformers>=5.22" torch sentence-transformers torchvision Pillow torchaudio librosa
    .venv/bin/python scripts/gen_embeddinggemma2_golden.py
"""
import json, sys, torch, numpy as np
from sentence_transformers import SentenceTransformer
torch.manual_seed(0)
model = SentenceTransformer("google/embeddinggemma-2", model_kwargs={"dtype": torch.float32},
                            config_kwargs={"vision_config": None, "audio_config": None}, device="cpu")
model.eval()
model.max_seq_length = 2048   # MinSync's default native max_length
q = "task: search result | query: "
d = "title: none | text: "
ko = "환불 정책에 따르면 구매 증빙 인증서를 제출한 고객은 14일 이내에 전액 환불을 받을 수 있습니다. 해외 출장비는 출장 종료 후 5영업일 안에 정산해야 합니다. "
en = "The northern lights are caused by charged particles from the sun colliding with gases in the upper atmosphere. "
texts = [
  q + "환불 받으려면 어떤 인증서가 필요해?",
  q + "What causes the northern lights?",
  d + "구매 증빙 인증서를 제출하면 환불이 가능합니다.",
  d + "The northern lights are caused by charged particles from the sun.",
  d + "fn main() { println!(\"hello 🌍\"); } // ERR-48213",
  d + ko * 40 + en * 20,          # long: crosses the 512 sliding radius
  d + en * 70,                    # long, English
  "",
]
tok = model.tokenizer
enc = tok(texts, padding=False, truncation=True, max_length=2048)
ids = enc["input_ids"]
print([len(x) for x in ids], ids[0][:6], ids[-1])
emb = model.encode(texts, batch_size=3, convert_to_numpy=True, normalize_embeddings=False)
print(emb.shape, np.linalg.norm(emb, axis=1))
def trunc(e, n):
    t = e[:, :n]; return t / np.linalg.norm(t, axis=1, keepdims=True)
out = {"texts": texts, "ids": ids, "dim": 768,
       "emb": emb.round(7).tolist(), "emb256": trunc(emb, 256).round(7).tolist()}
json.dump(out, open("tests/fixtures/embeddinggemma2/golden_real.json", "w"))
