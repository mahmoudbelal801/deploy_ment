import streamlit as st  # Ensure Streamlit is imported at the top
from transformers import pipeline

st.title("Named Entity Recognition (NER) & Sentence Completion")
st.subheader("Named Entity Recognition")

# Load pre-trained NER model with error handling
try:
    ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    st.write("Model loaded successfully")
except Exception as e:
    st.write(f"Error loading model: {e}")

# Input text for NER
text = st.text_area("Enter text for NER:")

if st.button("Recognize Entities"):
    if text:
        try:
            entities = ner_model(text)
            st.write(f"Entities found: {entities}")
            for entity in entities:
                st.write(f"{entity['word']}: {entity['entity']} (score: {entity['score']:.4f})")
        except Exception as e:
            st.write(f"Error processing text: {e}")