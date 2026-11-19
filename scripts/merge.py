"""Milestone 5: merge the adapter into GPT-2 and save a normal model folder.   python scripts/merge.py
The app merges in memory on its own; this script is for exporting the merged model (saved to merged/)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import core

model, tokenizer = core.load_commentator()          # loads base + adapter, then merges
assert not any(isinstance(m, core.LoRALayer) for m in model.modules())
model.save_pretrained("merged")
tokenizer.save_pretrained("merged")
print("Merged model saved to merged/ (plain GPT-2 format, no LoRA layers left)")
print(core.generate_commentary(model, tokenizer, "Hat-trick for Rossi, Rovers win 3-0", seed=0))
