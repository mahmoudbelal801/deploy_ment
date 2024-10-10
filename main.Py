import streamlit as st
from transformers import pipeline

# Load pre-trained NER model
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

st.title("Named Entity Recognition (NER) & Sentence Completion")
st.subheader("Named Entity Recognition")

# Input text for NER
text = st.text_area("Enter text for NER:")


if st.button("Recognize Entities"):
    if text:
        entities = ner_model(text)
        for entity in entities:
            st.write(f"{entity['word']}: {entity['entity']} (score: {entity['score']:.4f})")