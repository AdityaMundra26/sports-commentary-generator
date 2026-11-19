# Sports Commentary Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/gzpjhbwgdr/badge.svg)](https://ask.thili.ai/b/gzpjhbwgdr)

Type a sports event ("Goal by Rivera, 89th minute, Lions lead 2-1"), pick a tone, and a GPT-2 model fine-tuned with
**LoRA (written from scratch in PyTorch, no PEFT library)** writes a line of live commentary for it. Built for game
developers who need dynamic commentary for simulated matches.

## How it works

1. **Dataset** (`dataset/commentary.jsonl`): 84 event → commentary pairs across 7 sports (football, cricket,
   basketball, tennis, F1, hockey, kabaddi) in 3 tones (excited, calm, humorous).
2. **LoRA from scratch** (`app/core.py`, `LoRALayer`): every attention and projection layer (`c_attn`, `c_proj`) of
   GPT-2 is frozen and wrapped with two small matrices **A** (r × in) and **B** (out × r), rank r = 8.
   The layer computes `y = W·x + (B·A·x) · alpha/r`. B starts at zero, so training starts from the unchanged model.
   Only A and B train: about **1% of the parameters**, and the saved adapter is about 1.6 MB instead of 330 MB.
3. **Training** (`scripts/train.py`): plain PyTorch loop, AdamW, 15 epochs, a few minutes on a laptop CPU.
4. **Merging** (`merge_lora`): for inference each update is folded into the weight, `W_new = W + B·A·alpha/r`, and
   the LoRA wrappers are removed. The merged model gives the same outputs but runs as fast as plain GPT-2.
5. **Testing** (`scripts/evaluate.py`): 10 events never seen in training, scored for **relevance** (does it mention
   the event's names, scores and actions?) and **engagement** (energy that fits the tone, sensible length, no
   repetition, complete sentence), compared against plain GPT-2 with no adapter.
6. **Streamlit app** (`app/main.py`): event input, tone picker, creativity slider, several variations, a score for
   each line, and saved snippets you can download as JSON.

## Run it

You need Python 3.11+ and git. No API key is needed: the model runs on your laptop.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/train.py              # trains the adapter, saves it to adapter/ (first run downloads distilgpt2)
python scripts/evaluate.py           # tests on unseen events, writes reports/evaluation.md
streamlit run app/main.py            # the app
pytest                               # unit tests (tiny model, no download, runs in seconds)
```

Optional: `python scripts/merge.py` exports the merged model to `merged/` as a normal GPT-2 folder.
Use `python scripts/train.py --base gpt2` for the full 124M GPT-2 (slower, a bit better).

## Results

See [`reports/evaluation.md`](reports/evaluation.md) for scores and sample outputs, LoRA model vs plain GPT-2.

## Project layout

```
app/core.py            all the logic, one section per milestone
app/main.py            Streamlit page
dataset/               training data
scripts/               train.py, evaluate.py, merge.py
adapter/               the trained LoRA weights (created by train.py)
tests/test_core.py     one test per milestone
```

## Milestones

- [x] **1. Dataset** of simulated sports events and commentary pairs
- [x] **2. LoRA adapter injection and training** on GPT-2
- [x] **3. Streamlit interface** to input events and generate commentary
- [x] **4. Testing** relevance and engagement of the generated commentary
- [x] **5. Adapter merging** for inference

Stretch ideas done: more sports and styles, tone control (trained into the prompt), saving snippets to a file.

## Settings and secrets

This project needs no API keys. `.env` is ignored by git anyway: never commit it.
