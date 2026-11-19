"""Milestone 4: test the model on events it never saw in training.   python scripts/evaluate.py
Writes the results to reports/evaluation.md and compares against plain GPT-2 with no adapter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from transformers import AutoModelForCausalLM

from app import core

UNSEEN = [
    {"event": "Goal by Fernandes, 75th minute, Rovers lead 1-0", "tone": "excited"},
    {"event": "Penalty missed by Walsh, still 0-0", "tone": "humorous"},
    {"event": "Six by Mehta to bring up his fifty", "tone": "excited"},
    {"event": "Wicket for Bhatt, caught at slip, 120 for 5", "tone": "calm"},
    {"event": "Slam dunk by Harris to win the game 99-97", "tone": "excited"},
    {"event": "Double fault by Ivanova on break point", "tone": "calm"},
    {"event": "Overtake by Silva for second place on lap 40", "tone": "excited"},
    {"event": "Overtime goal by Novak, Sharks win 2-1", "tone": "excited"},
    {"event": "Super raid by Deepak, three points", "tone": "excited"},
    {"event": "Goalkeeper Brandt drops an easy catch, 1-1", "tone": "humorous"},
]

model, tokenizer = core.load_commentator()
lora = core.evaluate_commentator(lambda e, t: core.generate_commentary(model, tokenizer, e, t, seed=0), UNSEEN)

base_name = json.loads(core.CONFIG_FILE.read_text())["base_model"]
plain = AutoModelForCausalLM.from_pretrained(base_name).eval()
base = core.evaluate_commentator(lambda e, t: core.generate_commentary(plain, tokenizer, e, t, seed=0), UNSEEN)

lines = ["# Evaluation on unseen events", "",
         f"Base model `{base_name}`, same prompts and seed for both.", "",
         "| Model | Relevance | Engagement |", "|---|---|---|",
         f"| Plain {base_name} (no adapter) | {base['relevance']:.2f} | {base['engagement']:.2f} |",
         f"| {base_name} + merged LoRA | {lora['relevance']:.2f} | {lora['engagement']:.2f} |", "",
         "## Samples from the LoRA model", "",
         "| Event | Tone | Commentary | Rel. | Eng. |", "|---|---|---|---|---|"]
for s in lora["samples"]:
    lines.append(f"| {s['event']} | {s['tone']} | {s['commentary'].replace('|', '/')} | "
                 f"{s['relevance']:.2f} | {s['engagement']:.2f} |")
Path("reports").mkdir(exist_ok=True)
Path("reports/evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print("\nSaved to reports/evaluation.md")
