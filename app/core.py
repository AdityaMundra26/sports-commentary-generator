"""Sports Commentary Generator: LoRA built from scratch on GPT-2.

One section per milestone. Each `milestone_N` name at the bottom points at the main function for that milestone.

How LoRA works here (the short version):
    A frozen weight matrix W (out x in) gets a small trainable update  dW = B @ A * (alpha / r)
    where A is (r x in) and B is (out x r), with r tiny (8). Only A and B are trained, so we train
    well under 1% of GPT-2's weights. For inference we "merge": W_new = W + dW, then throw A and B away,
    so the merged model is exactly as fast as the original GPT-2.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "dataset" / "commentary.jsonl"
ADAPTER_DIR = ROOT / "adapter"
ADAPTER_FILE = ADAPTER_DIR / "adapter.pt"
CONFIG_FILE = ADAPTER_DIR / "config.json"

BASE_MODEL = "distilgpt2"          # a smaller GPT-2; use "gpt2" for the full 124M model
TONES = ("excited", "calm", "humorous")
DEFAULT_TONE = "excited"


# =====================================================================================
# Milestone 1: dataset of simulated sports events and commentary pairs
# =====================================================================================
def load_dataset(path: str | Path = DATASET_PATH) -> list[dict]:
    """Read the JSONL dataset. Every line needs `event` and `commentary`; `sport` and `tone` are optional."""
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not row.get("event") or not row.get("commentary"):
            raise ValueError(f"line {number}: every row needs an 'event' and a 'commentary'")
        tone = row.get("tone", DEFAULT_TONE)
        if tone not in TONES:
            raise ValueError(f"line {number}: tone must be one of {TONES}, got {tone!r}")
        rows.append({"sport": row.get("sport", "general"), "tone": tone,
                     "event": row["event"].strip(), "commentary": row["commentary"].strip()})
    return rows


def build_prompt(event: str, tone: str = DEFAULT_TONE) -> str:
    """The text the model sees before it writes commentary. Training and inference use the same format."""
    if tone not in TONES:
        tone = DEFAULT_TONE
    return f"Tone: {tone}\nEvent: {event.strip()}\nCommentary:"


def format_example(row: dict) -> str:
    """One full training example: prompt + the commentary we want the model to learn to write."""
    return f"{build_prompt(row['event'], row.get('tone', DEFAULT_TONE))} {row['commentary']}\n"


# =====================================================================================
# Milestone 2: LoRA adapter injection and training (written from scratch, no PEFT library)
# =====================================================================================
def _is_conv1d(layer: nn.Module) -> bool:
    # GPT-2 uses a "Conv1D" layer that is really a Linear layer with its weight stored as (in, out).
    return type(layer).__name__ == "Conv1D"


def _in_out(layer: nn.Module) -> tuple[int, int]:
    if _is_conv1d(layer):
        return layer.weight.shape[0], layer.weight.shape[1]
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    raise TypeError(f"LoRA only wraps Linear or GPT-2 Conv1D layers, not {type(layer).__name__}")


class LoRALayer(nn.Module):
    """Wraps a frozen layer and adds a trainable low-rank update: y = base(x) + (x A^T B^T) * alpha / r."""

    def __init__(self, base: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        in_features, out_features = _in_out(base)
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))   # zeros: the model starts unchanged
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        """The update dW = B @ A * scaling, shaped (out, in)."""
        return (self.lora_B @ self.lora_A) * self.scaling


def inject_lora(model: nn.Module, target_modules=("c_attn", "c_proj"), r: int = 8, alpha: int = 16,
                dropout: float = 0.05) -> list[str]:
    """Freeze the whole model, then swap every layer whose name ends in a target for a LoRALayer.

    In GPT-2, `c_attn` makes the attention queries/keys/values and `c_proj` are the output projections.
    Returns the names of the layers that were wrapped.
    """
    for p in model.parameters():
        p.requires_grad = False
    targets = [name for name, module in model.named_modules()
               if name.split(".")[-1] in target_modules and not isinstance(module, LoRALayer)
               and (isinstance(module, nn.Linear) or _is_conv1d(module))]
    for name in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALayer(getattr(parent, child_name), r=r, alpha=alpha, dropout=dropout))
    if not targets:
        raise ValueError(f"no layers named {target_modules} found in the model")
    return targets


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the LoRA weights: this is the whole 'adapter' (about 1.5 MB instead of the 330 MB model)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def save_adapter(model: nn.Module, config: dict, directory: str | Path = ADAPTER_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), directory / "adapter.pt")
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    return directory


def load_adapter(model: nn.Module, directory: str | Path = ADAPTER_DIR) -> dict:
    """Inject LoRA layers using the saved config, then load the trained A and B matrices into them."""
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    inject_lora(model, tuple(config["target_modules"]), r=config["r"], alpha=config["alpha"], dropout=0.0)
    state = torch.load(directory / "adapter.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any("lora_" in k for k in missing):
        raise RuntimeError("the adapter does not match this model")
    return config


def encode_examples(tokenizer, texts: list[str], max_length: int = 96) -> dict[str, torch.Tensor]:
    """Tokenize training texts. Padding positions get label -100 so they do not count in the loss."""
    enc = tokenizer([t + tokenizer.eos_token for t in texts], padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


def train_lora(model: nn.Module, encodings: dict[str, torch.Tensor], epochs: int = 15, lr: float = 1e-3,
               batch_size: int = 8, seed: int = 0, log: Callable[[str], None] | None = print) -> list[float]:
    """A plain PyTorch training loop over the LoRA parameters only. Returns the average loss per epoch."""
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("nothing to train: call inject_lora first")
    optimizer = torch.optim.AdamW(params, lr=lr)
    n = encodings["input_ids"].shape[0]
    model.train()
    history = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            out = model(**{k: v[idx] for k, v in encodings.items()})
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            total += out.loss.item() * len(idx)
        history.append(total / n)
        if log:
            log(f"epoch {epoch:>2}/{epochs}  loss {history[-1]:.3f}")
    model.eval()
    return history


# =====================================================================================
# Milestone 3: generating commentary (the Streamlit page in app/main.py calls these)
# =====================================================================================
def clean_output(text: str) -> str:
    """Keep only the first line of commentary and tidy it up."""
    text = re.split(r"\n|Tone:|Event:|Commentary:", text, maxsplit=1)[0]
    text = re.sub(r"\s+", " ", text).strip()
    # Drop a trailing half-sentence if there is at least one complete sentence before it.
    last_end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if 0 < last_end < len(text) - 1:
        text = text[:last_end + 1]
    return text


@torch.no_grad()
def generate_commentary(model, tokenizer, event: str, tone: str = DEFAULT_TONE, max_new_tokens: int = 45,
                        temperature: float = 0.8, top_p: float = 0.92, seed: int | None = None) -> str:
    if not event.strip():
        raise ValueError("describe the event first")
    if seed is not None:
        torch.manual_seed(seed)
    inputs = tokenizer(build_prompt(event, tone), return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                            top_p=top_p, repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id)
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return clean_output(tokenizer.decode(new_tokens, skip_special_tokens=True))


# =====================================================================================
# Milestone 4: testing relevance and engagement of the generated commentary
# =====================================================================================
STOPWORDS = {"the", "a", "an", "by", "for", "in", "on", "of", "to", "and", "at", "with", "after", "from",
             "is", "now", "up", "one", "two", "his", "her", "their", "out", "off", "over", "into", "stays"}
ENERGY_WORDS = {"what", "incredible", "unbelievable", "stunning", "brilliant", "amazing", "wow", "huge",
                "sensational", "astonishing", "magic", "drama", "erupts", "explodes", "wins", "win", "won",
                "goal", "six", "ace", "dunk", "save", "saved", "fire", "flying", "history", "dream", "wild"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())


def event_keywords(event: str) -> set[str]:
    return {w for w in _words(event) if w not in STOPWORDS and (len(w) > 2 or w.isdigit() or "-" in w)}


def score_relevance(event: str, commentary: str) -> float:
    """0..1: how many of the event's key words (names, scores, actions) show up in the commentary.
    Mentioning three key words counts as fully relevant."""
    keywords = event_keywords(event)
    if not keywords:
        return 0.0
    hits = len(keywords & set(_words(commentary)))
    return round(min(1.0, hits / min(3, len(keywords))), 3)


def score_engagement(commentary: str, tone: str = DEFAULT_TONE) -> float:
    """0..1 from four simple signals: energy (exclamations or exciting words, matched to the tone),
    a sensible length, not repeating itself, and ending as a complete sentence."""
    words = _words(commentary)
    if not words:
        return 0.0
    score = 0.0
    energetic = "!" in commentary or any(w in ENERGY_WORDS for w in words)
    if tone == "calm":
        score += 0.3 if "!" not in commentary else 0.1     # calm commentary should not shout
    else:
        score += 0.3 if energetic else 0.0
    score += 0.25 if 6 <= len(words) <= 40 else 0.0
    score += 0.25 if len(set(words)) / len(words) >= 0.7 else 0.0
    score += 0.2 if commentary.rstrip()[-1] in ".!?" else 0.0
    return round(score, 3)


def evaluate_commentator(generate: Callable[[str, str], str], cases: list[dict]) -> dict:
    """Run `generate(event, tone)` on each case and score it. Returns averages plus every sample."""
    samples = []
    for case in cases:
        tone = case.get("tone", DEFAULT_TONE)
        text = generate(case["event"], tone)
        samples.append({"event": case["event"], "tone": tone, "commentary": text,
                        "relevance": score_relevance(case["event"], text),
                        "engagement": score_engagement(text, tone)})
    n = max(1, len(samples))
    return {"relevance": round(sum(s["relevance"] for s in samples) / n, 3),
            "engagement": round(sum(s["engagement"] for s in samples) / n, 3),
            "samples": samples}


# =====================================================================================
# Milestone 5: merging the adapter into the base weights for inference
# =====================================================================================
@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Fold every LoRA update into its base layer (W = W + B @ A * scaling) and remove the LoRA wrappers.
    After this the model is a normal GPT-2 again, with no extra cost at inference. Returns layers merged."""
    wrapped = [name for name, module in model.named_modules() if isinstance(module, LoRALayer)]
    for name in wrapped:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        lora: LoRALayer = getattr(parent, child_name)
        delta = lora.delta_weight()
        lora.base.weight += delta.T if _is_conv1d(lora.base) else delta   # Conv1D stores (in, out)
        setattr(parent, child_name, lora.base)
    return len(wrapped)


def load_commentator(adapter_dir: str | Path = ADAPTER_DIR):
    """Load base GPT-2 + the trained adapter, merge them, and return (model, tokenizer) ready to generate."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads((Path(adapter_dir) / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["base_model"])
    load_adapter(model, adapter_dir)
    merge_lora(model)
    model.eval()
    return model, tokenizer


# One name per milestone, as the starter asked.
milestone_1 = load_dataset
milestone_2 = inject_lora
milestone_3 = generate_commentary
milestone_4 = evaluate_commentator
milestone_5 = merge_lora
