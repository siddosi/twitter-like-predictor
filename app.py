# app.py

import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import difflib
import joblib
import json
import os

# --- 1. ASSET LOADING (with Caching for performance) ---

# Use Streamlit's caching to load assets only once [6]
@st.cache_resource
def load_assets():
    """Loads all necessary pre-trained assets from the 'assets' directory."""
    base_path = 'assets'
    model = joblib.load(os.path.join(base_path, 'like_predictor_model.joblib'))
    vectorizer = joblib.load(os.path.join(base_path, 'tfidf_vectorizer.joblib'))
    with open(os.path.join(base_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    return model, vectorizer, metadata

# Load assets and handle potential errors
try:
    model, vectorizer, metadata = load_assets()
except FileNotFoundError:
    st.error("Model assets not found. Please run `save_model_assets.py` first.")
    st.stop()

# Unpack metadata for easier access
company_to_avg_likes = metadata['company_to_avg_likes']
keyword_to_avg_likes = metadata['keyword_to_avg_likes']
company_keys = list(company_to_avg_likes.keys())
keyword_keys = list(keyword_to_avg_likes.keys())
feature_names = vectorizer.get_feature_names_out()
global_median_company_likes = np.median(list(company_to_avg_likes.values()))
default_keyword = max(keyword_to_avg_likes, key=keyword_to_avg_likes.get) if keyword_to_avg_likes else None

# --- 2. FEATURE ENGINEERING FUNCTIONS (Mirroring your inference logic) ---

def get_top_tfidf_word_from_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    if not text: return default_keyword
    transformed = vectorizer.transform([text])
    if np.max(transformed.toarray()) == 0: return default_keyword
    return feature_names[np.argmax(transformed.toarray())]

def get_company_avg_likes(company):
    company = company.lower()
    if company in company_to_avg_likes:
        return company_to_avg_likes[company]
    closest = difflib.get_close_matches(company, company_keys, n=1, cutoff=0.6)
    return company_to_avg_likes[closest] if closest else global_median_company_likes

def get_avg_likes_for_keyword(tweet):
    top_keyword = get_top_tfidf_word_from_text(tweet)
    if top_keyword and top_keyword in keyword_to_avg_likes:
        return keyword_to_avg_likes[top_keyword]
    closest = difflib.get_close_matches(top_keyword, keyword_keys, n=1, cutoff=0.6) if top_keyword else None
    return keyword_to_avg_likes[closest] if closest else keyword_to_avg_likes.get(default_keyword, 0)

# --- 3. STREAMLIT USER INTERFACE ---

st.set_page_config(page_title="Twitter Likes Predictor", page_icon="🐦", layout="centered")
st.title("🐦 Twitter Likes Predictor")
st.markdown("This app uses a machine learning model to estimate the number of likes a tweet might receive.")

# Use a form for efficient input handling [7]
with st.form("prediction_form"):
    tweet_input = st.text_area("Enter Tweet Text:", height=120, placeholder="What's happening?")
    company_input = st.selectbox("Select Company:", options=sorted(company_keys))
    submit_button = st.form_submit_button("Predict Likes")

# --- 4. PREDICTION LOGIC ---

if submit_button:
    if not tweet_input.strip():
        st.warning("⚠️ Please enter some text for the tweet.")
    else:
        with st.spinner("🤖 Analyzing tweet and predicting..."):
            # Build the feature vector exactly as the model was trained
            features = {
                'word_count': len(tweet_input.split()),
                'char_count': len(tweet_input),
                'has_media': 1 if "pic.twitter.com" in tweet_input else 0,
                'sentiment': 1000 * TextBlob(tweet_input).sentiment.polarity,
                'company_avg_likes': get_company_avg_likes(company_input),
                'hashtag': 1 if "#" in tweet_input else 0,
                'avg_likes_for_keyword': get_avg_likes_for_keyword(tweet_input)
            }
            
            feature_order = ['word_count', 'char_count', 'has_media', 'sentiment', 'company_avg_likes', 'hashtag', 'avg_likes_for_keyword']
            input_df = pd.DataFrame([features], columns=feature_order)

            # Make prediction on the log-transformed scale
            raw_prediction = model.predict(input_df)
            
            # Reverse the log transformation (e^x - 1)
            final_prediction = np.exp(raw_prediction) - 1
            
            # Ensure prediction is a non-negative integer
            final_prediction = max(0, int(final_prediction))

        st.subheader("📈 Prediction Result")
        st.metric(label="Predicted Likes", value=f"{final_prediction:,}")
        st.info("Note: This is an estimate. Real-world engagement can be influenced by many external factors.", icon="💡")