import streamlit as st
import whisper
from transformers import pipeline
import re
import tempfile

# ---------------------------
# Streamlit App Config
# ---------------------------
st.set_page_config(page_title="AI Meeting Minutes Generator", layout="wide")
st.title("🤖 AI Meeting Minutes Generator")

# File uploader
audio_file = st.file_uploader("Upload Meeting Audio", type=["mp3", "wav", "m4a", "ogg"])

if audio_file:
    # ---------------------------
    # 1. Save uploaded file to temp
    # ---------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        temp_path = tmp.name

    # ---------------------------
    # 2. Transcription
    # ---------------------------
    st.subheader("1️⃣ Transcription")
    st.info("Transcribing audio with Whisper (CPU mode)...")
    model = whisper.load_model("small", device="cpu")
    transcription = model.transcribe(temp_path)
    text = transcription["text"]
    st.text_area("Full Transcript", text, height=200)

    # ---------------------------
    # 3. Summarization
    # ---------------------------
    st.subheader("2️⃣ Summary")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
    st.write(summary)

    # ---------------------------
    # 4. Named Entity Recognition
    # ---------------------------
    st.subheader("3️⃣ Key Entities")
    ner = pipeline("ner", grouped_entities=True)
    entities = ner(text)
    st.json(entities)

    # ---------------------------
    # 5. Extract Decisions & Action Items
    # ---------------------------
    st.subheader("4️⃣ Key Decisions & Action Items")

    decisions = []
    actions = []
    for sentence in re.split(r"(?<=[.!?]) +", text):
        if any(word in sentence.lower() for word in ["decided", "approved", "agreed", "finalized"]):
            decisions.append(sentence.strip())
        if any(word in sentence.lower() for word in ["will", "need to", "assign", "schedule", "send", "review"]):
            actions.append(sentence.strip())

    st.markdown("**Decisions:**")
    if decisions:
        for d in decisions:
            st.write(f"- {d}")
    else:
        st.write("No clear decisions detected.")

    st.markdown("**Action Items:**")
    if actions:
        for a in actions:
            st.write(f"- {a}")
    else:
        st.write("No clear action items detected.")