"""Bi-encoder: shared multilingual transformer, mean pooling, linear projection (hidden -> proj_dim), L2 norm.
Name and address are encoded as separate sequences with field prefixes."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class DualFieldEncoder(nn.Module):
    def __init__(self, name_or_path: str, proj_dim: int, freeze_word_embeddings: bool = True,
                 grad_checkpointing: bool = False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name_or_path)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, proj_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        if freeze_word_embeddings:
            self.backbone.get_input_embeddings().weight.requires_grad_(False)
        if grad_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        return F.normalize(self.proj(pooled).float(), dim=-1)

    def param_groups(self, lr_backbone: float, lr_proj: float, weight_decay: float):
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        return [{"params": bb, "lr": lr_backbone, "weight_decay": weight_decay},
                {"params": list(self.proj.parameters()), "lr": lr_proj, "weight_decay": 0.0}]

    def save(self, out_dir: Path, tokenizer, meta: dict):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(out_dir / "backbone")
        tokenizer.save_pretrained(out_dir / "backbone")
        torch.save(self.proj.state_dict(), out_dir / "proj.pt")
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: Path):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        m = cls(str(model_dir / "backbone"), meta["proj_dim"], freeze_word_embeddings=False)
        m.proj.load_state_dict(torch.load(model_dir / "proj.pt", map_location="cpu"))
        tok = AutoTokenizer.from_pretrained(str(model_dir / "backbone"))
        return m, tok, meta


def tokenize(tokenizer, texts: list[str], max_len: int, device):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return enc["input_ids"].to(device, non_blocking=True), enc["attention_mask"].to(device, non_blocking=True)


def encode_texts(model, tokenizer, texts: list[str], max_len: int, batch: int, device,
                 out_dtype=torch.float16) -> torch.Tensor:
    """Inference: length-sorted batches, fp16 autocast, returns a CPU tensor [len(texts), proj_dim]."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = torch.empty((len(texts), model.proj.out_features), dtype=out_dtype)
    use_amp = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            ids, am = tokenize(tokenizer, [texts[i] for i in idx], max_len, device)
            out[torch.tensor(idx)] = model(ids, am).to(out_dtype).cpu()
    return out
