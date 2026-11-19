"""Milestone 2: train the LoRA adapter.   python scripts/train.py   (about 3-8 minutes on a laptop CPU)"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app import core

parser = argparse.ArgumentParser()
parser.add_argument("--base", default=core.BASE_MODEL, help="distilgpt2 (fast) or gpt2")
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--r", type=int, default=8)
parser.add_argument("--alpha", type=int, default=16)
args = parser.parse_args()

torch.manual_seed(0)
rows = core.load_dataset()
print(f"Loaded {len(rows)} event/commentary pairs")

tokenizer = AutoTokenizer.from_pretrained(args.base)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.base)

targets = ("c_attn", "c_proj")
wrapped = core.inject_lora(model, targets, r=args.r, alpha=args.alpha)
trainable, total = core.count_parameters(model)
print(f"Injected LoRA into {len(wrapped)} layers. Training {trainable:,} of {total:,} parameters "
      f"({100 * trainable / total:.2f}%)")

encodings = core.encode_examples(tokenizer, [core.format_example(r) for r in rows])
start = time.time()
losses = core.train_lora(model, encodings, epochs=args.epochs, lr=args.lr)
print(f"Trained in {time.time() - start:.0f}s. Loss {losses[0]:.3f} -> {losses[-1]:.3f}")

core.save_adapter(model, {"base_model": args.base, "target_modules": list(targets), "r": args.r,
                          "alpha": args.alpha, "epochs": args.epochs, "lr": args.lr,
                          "examples": len(rows), "loss_first": round(losses[0], 3),
                          "loss_last": round(losses[-1], 3)})
print(f"Saved adapter to {core.ADAPTER_DIR}")

print("\nQuick check:")
for event in ["Goal by Ahmed, 90th minute, City lead 2-1", "Ace on match point, Garcia wins the final"]:
    print(f"  {event}\n  -> {core.generate_commentary(model, tokenizer, event, seed=0)}")
