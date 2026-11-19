"""Streamlit entry point: `streamlit run app/main.py`"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `from app import ...` works

import json

import streamlit as st

from app import core
from app.storage import Store

st.set_page_config(page_title="Sports Commentary Generator", page_icon="🎙️")
st.title("Sports Commentary Generator")
st.write("Game developers use this tool to adapt a language model with LoRA, generating dynamic and "
         "context-aware commentary snippets for simulated sports events like goal alerts or score changes.")

if not core.ADAPTER_FILE.exists():
    st.warning("No trained adapter yet. Run `python scripts/train.py` first, then reload this page.")
    st.stop()


@st.cache_resource(show_spinner="Loading GPT-2 and merging the LoRA adapter...")
def get_commentator():
    return core.load_commentator()


model, tokenizer = get_commentator()
config = json.loads(core.CONFIG_FILE.read_text())
st.caption(f"Base model: `{config['base_model']}` · LoRA rank {config['r']} · adapter merged for inference")

EXAMPLES = ["Goal by Ahmed, 90th minute, City lead 2-1",
            "Six by Mehta off the last ball to win the match",
            "Buzzer-beater three by Carter, Warriors win 110-109",
            "Ace on match point, Garcia wins the final",
            "Safety car deployed after a crash at turn 1"]

example = st.selectbox("Pick an example, or write your own below", ["(write my own)"] + EXAMPLES)
event = st.text_area("Event description", value="" if example == "(write my own)" else example,
                     placeholder="e.g. Penalty saved by Okafor, score stays 0-0")
col1, col2, col3 = st.columns(3)
tone = col1.selectbox("Tone", core.TONES)
variations = col2.number_input("Variations", min_value=1, max_value=5, value=3)
temperature = col3.slider("Creativity", 0.3, 1.3, 0.8, 0.1)
save = st.checkbox("Save generated snippets", value=True)

store = Store()

if st.button("Generate commentary", type="primary"):
    if not event.strip():
        st.error("Describe the event first.")
    else:
        with st.spinner("Commentating..."):
            results = [core.generate_commentary(model, tokenizer, event, tone, temperature=temperature)
                       for _ in range(int(variations))]
        for text in results:
            text = text or "(the model produced an empty line, try again)"
            st.markdown(f"### 🎙️ {text}")
            st.caption(f"Relevance {core.score_relevance(event, text):.2f} · "
                       f"Engagement {core.score_engagement(text, tone):.2f}")
            if save:
                store.put(datetime.now().isoformat(timespec="microseconds"),
                          {"event": event, "tone": tone, "commentary": text})

saved = store.all()
if saved:
    with st.expander(f"Saved snippets ({len(saved)})"):
        for key, item in reversed(list(saved.items())):
            st.write(f"**[{item['tone']}]** {item['event']} → {item['commentary']}")
        st.download_button("Download as JSON", json.dumps(saved, indent=2), file_name="commentary.json")
