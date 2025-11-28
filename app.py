import streamlit as st
import numpy as np
import pickle
import os

# --- Keras 3 Compatibility Fix ---
# This environment variable is still needed if Keras 3 is installed
# for other reasons, but it won't conflict with PyTorch.
os.environ['TF_USE_LEGACY_KERAS'] = '1'
# --- End Keras 3 Fix ---

from PIL import Image as PILImage

# --- START OF ROBUST NLTK FIX (v3) ---
# This block is critical to prevent all NLTK errors.
# We are now using a basic tokenizer that does NOT depend on loading the
# problematic 'english.pickle' file.
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import TreebankWordTokenizer

# Initialize a 'blank' sentence tokenizer.
sentence_tokenizer = PunktSentenceTokenizer()
word_tokenizer_instance = TreebankWordTokenizer()

def safe_word_tokenize(text):
    """
    A custom, safe word tokenizer that bypasses the broken NLTK data lookup.
    """
    if sentence_tokenizer is None:
        return ["Error:", "Tokenizer", "model", "failed", "to", "load."]
    
    sentences = sentence_tokenizer.tokenize(text)
    words = []
    for sent in sentences:
        words.extend(word_tokenizer_instance.tokenize(sent))
    return words
# --- END OF ROBUST NLTK FIX ---

# Import the correct model versions
# We are now using the PyTorch-native versions
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator

# --- CACHED FUNCTIONS TO LOAD MODELS ONLY ONCE ---

@st.cache_resource
def load_blip_model():
    """Loads the Salesforce BLIP model and processor."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # FIXED: Use the original PyTorch (PT) version
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_grammar_model():
    """Loads the grammar correction model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
    return tokenizer, model

@st.cache_resource
def get_translator():
    """Returns a Translator instance."""
    return Translator()

# --- LOAD ALL MODELS USING CACHED FUNCTIONS ---
blip_processor, blip_model = load_blip_model()
grammar_tokenizer, grammar_model = load_grammar_model()
translator = get_translator()
lang_map = {"English": "en", "Telugu": "te", "Hindi": "hi", "Tamil": "ta", "Bengali": "bn"}


# --- CORE LOGIC FUNCTIONS ---

def generate_captions_blip(image_file, processor, model, num_captions=5):
    """
    Generates 5 high-quality, diverse captions using stable beam search.
    """
    # Use the in-memory file object directly
    raw_image = PILImage.open(image_file).convert("RGB")
    
    # --- THIS IS THE FIX ---
    # Changed return_tensors from "tf" back to "pt" (PyTorch)
    # This matches the BlipForConditionalGeneration (PyTorch) model.
    inputs = processor(images=raw_image, return_tensors="pt") 
    # --- END OF FIX ---
    
    # Use stable beam search.
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_beams=10, # Use more beams for better results
        num_return_sequences=num_captions, # Ask for 5 captions
        early_stopping=True
    )
    
    # Decode and ensure captions are unique
    captions = [processor.decode(out, skip_special_tokens=True).strip() for out in outputs]
    return list(set(captions)) # Use set() to remove potential duplicates

def correct_grammar(text, tokenizer, model):
    input_text = "gec: " + text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids, max_length=64, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title("📸 Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
col1, col2 = st.columns(2)
with col1:
    language = st.selectbox("Translate to Language", list(lang_map.keys()))
with col2:
    emotion = st.selectbox("Add Emotion", ["Normal", "Romantic", "Joke", "Happy", "Sad", "Angry"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    if st.button("Generate Captions"):
        with st.spinner("Generating captions... This might take a moment."):
            
            all_captions = generate_captions_blip(uploaded_file, blip_processor, blip_model, num_captions=5)
            
            st.subheader("Generated Captions:")
            if not all_captions:
                st.error("Model failed to generate captions. Please try again.")
            
            for i, cap in enumerate(all_captions, 1):
                corrected_cap = correct_grammar(cap, grammar_tokenizer, grammar_model)
                
                if emotion != "Normal":
                    final_caption = f"[{emotion}] {corrected_cap}"
                else:
                    final_caption = corrected_cap
                
                if language != "English":
                    try:
                        final_caption = translator.translate(final_caption, dest=lang_map[language]).text
                    except Exception as e:
                        st.warning(f"Translation failed: {e}. Showing English.")
                
                st.markdown(f"**✒️ Caption {i}:** {final_caption}")

st.markdown("---")
st.markdown("### 📊 BLEU Score Evaluation (Manual Test)")

smoothie = SmoothingFunction().method4
reference_text = st.text_area("Enter reference captions (one per line)", height=100, placeholder="A dog is playing in the park.\nA brown dog runs on green grass.")
candidate_text = st.text_input("Enter your generated caption to test")

if st.button("Calculate BLEU Score") and reference_text and candidate_text:
    
    references = [safe_word_tokenize(ref.strip().lower()) for ref in reference_text.splitlines()]
    candidate = safe_word_tokenize(candidate_text.strip().lower())
    
    if references and candidate:
        bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        st.success(f"**BLEU-1 (Individual Word Match):** {bleu1:.4f}")
        st.success(f"**BLEU-4 (Phrase Match):** {bleu4:.4f}")
    else:
        st.error("Please provide valid reference and candidate captions.")

st.markdown("---")
st.markdown("**Project by Ashok**")

