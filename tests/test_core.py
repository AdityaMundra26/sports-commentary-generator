"""One test per milestone. They use a tiny randomly initialised GPT-2, so they run in seconds with no download."""
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from app import core


def tiny_gpt2():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=60, n_positions=32, n_embd=32, n_layer=2, n_head=2))


def fake_batch(n=8, length=12):
    torch.manual_seed(1)
    ids = torch.randint(0, 60, (n, length))
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


# Milestone 1: Create a small dataset of simulated sports events and commentary pairs.
def test_milestone_1(tmp_path):
    rows = core.load_dataset()
    assert len(rows) >= 60
    assert len({r["sport"] for r in rows}) >= 5                      # diverse sports
    assert {r["tone"] for r in rows} == set(core.TONES)               # every tone is represented
    assert len({r["event"] for r in rows}) == len(rows)               # no duplicate events
    assert all(r["event"] and r["commentary"] for r in rows)

    prompt = core.build_prompt("Goal by Rivera, 1-0", "calm")
    assert prompt == "Tone: calm\nEvent: Goal by Rivera, 1-0\nCommentary:"
    assert core.build_prompt("x", "unknown-tone").startswith("Tone: excited")
    assert core.format_example(rows[0]).startswith(core.build_prompt(rows[0]["event"], rows[0]["tone"]))

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"event": "Goal"}\n')
    with pytest.raises(ValueError):
        core.load_dataset(bad)


# Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
def test_milestone_2():
    model = tiny_gpt2().eval()
    batch = fake_batch()
    before = model(input_ids=batch["input_ids"]).logits
    base_weight = model.transformer.h[0].attn.c_attn.weight.clone()

    wrapped = core.inject_lora(model, r=4, alpha=8, dropout=0.0)
    assert len(wrapped) == 2 * 3           # c_attn, attn.c_proj, mlp.c_proj in each of the 2 layers
    trainable, total = core.count_parameters(model)
    assert 0 < trainable < 0.2 * total
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)

    # B starts at zero, so injecting LoRA must not change the model's output.
    model.eval()
    assert torch.allclose(model(input_ids=batch["input_ids"]).logits, before, atol=1e-6)

    losses = core.train_lora(model, batch, epochs=15, lr=1e-2, batch_size=4, log=None)
    assert losses[-1] < losses[0]                                      # it learns
    assert torch.equal(model.transformer.h[0].attn.c_attn.base.weight, base_weight)   # base stays frozen


# Milestone 3: Build a Streamlit interface to input event descriptions and generate commentary.
def test_milestone_3():
    assert core.clean_output(" What a goal!\nEvent: something else") == "What a goal!"
    assert core.clean_output("He scores!   The crowd goes") == "He scores!"
    assert core.clean_output("Tone: calm") == ""

    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, text, return_tensors=None):
            return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

        def decode(self, ids, skip_special_tokens=True):
            return " Rivera scores! The Lions lead.\nEvent: next"

    text = core.generate_commentary(tiny_gpt2(), FakeTokenizer(), "Goal by Rivera", seed=0, max_new_tokens=5)
    assert text == "Rivera scores! The Lions lead."
    with pytest.raises(ValueError):
        core.generate_commentary(tiny_gpt2(), FakeTokenizer(), "   ")

    page = open("app/main.py", encoding="utf-8").read()
    assert "st.text_area" in page and "st.button" in page and "generate_commentary" in page


# Milestone 4: Test the model's ability to produce relevant and engaging commentary.
def test_milestone_4():
    event = "Goal by Rivera, 89th minute, Lions lead 2-1"
    good = "Rivera scores in the 89th minute and the Lions lead 2-1! What a moment!"
    off_topic = "the the the the weather is nice"
    assert core.score_relevance(event, good) == 1.0
    assert core.score_relevance(event, off_topic) == 0.0
    assert core.score_engagement(good) > core.score_engagement(off_topic)
    assert core.score_engagement("Play is suspended for rain.", "calm") > \
        core.score_engagement("Play is suspended for rain!!!", "calm")

    report = core.evaluate_commentator(lambda e, t: good, [{"event": event}, {"event": event, "tone": "excited"}])
    assert report["relevance"] == 1.0 and len(report["samples"]) == 2


# Milestone 5: Implement adapter merging for inference.
def test_milestone_5(tmp_path):
    model = tiny_gpt2()
    core.inject_lora(model, r=4, alpha=8, dropout=0.0)
    torch.manual_seed(2)
    for name, p in model.named_parameters():          # pretend it was trained: give B real values
        if "lora_B" in name:
            p.data.normal_(0, 0.1)
    model.eval()
    ids = fake_batch()["input_ids"]
    with_adapter = model(input_ids=ids).logits

    core.save_adapter(model, {"base_model": "tiny", "target_modules": ["c_attn", "c_proj"], "r": 4, "alpha": 8},
                      tmp_path)
    assert core.merge_lora(model) == 6
    assert not any(isinstance(m, core.LoRALayer) for m in model.modules())
    assert torch.allclose(model(input_ids=ids).logits, with_adapter, atol=1e-4)    # same answers, no LoRA

    fresh = tiny_gpt2()                                 # the saved adapter loads back into a new model
    core.load_adapter(fresh, tmp_path)
    fresh.eval()
    assert torch.allclose(fresh(input_ids=ids).logits, with_adapter, atol=1e-4)
