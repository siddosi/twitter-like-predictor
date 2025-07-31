import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
import os

print("Starting asset creation script...")

try:
   df = pd.read_csv("behaviour_simulation_train.csv", engine='python')
   print("Dataset loaded successfully.")
except FileNotFoundError:
   print("FATAL ERROR: 'behaviour_simulation_train.csv' not found.")
   print("Please place the dataset file in the same directory as this script.")
   exit()

df['likes'] = np.log(df['likes'] + 1)
df['tweets'] = df['tweets'].str.lower()
print("Data preprocessing complete.")

print("Starting feature engineering...")
df['media'] = df['media'].fillna('0')
df['has_media'] = df['media'].apply(lambda x: 1 if x!= 'no_media' else 0)
df['word_count'] = df['tweets'].apply(lambda x: len(str(x).split()))
df['char_count'] = df['tweets'].apply(lambda x: len(str(x)))
df['sentiment'] = df['tweets'].apply(lambda tweet: 1000 * TextBlob(str(tweet)).sentiment.polarity)
df['company_avg_likes'] = df.groupby('inferred company')['likes'].transform('mean')
df['hashtag'] = df['tweets'].apply(lambda x: 1 if '#' in str(x) else 0)

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['tweets'].fillna(''))
feature_names = vectorizer.get_feature_names_out()

def get_top_tfidf_word(row_index):
   scores = tfidf_matrix[row_index].toarray().flatten()
   return feature_names[np.argmax(scores)] if np.max(scores) > 0 else None

df['top_keyword'] = [get_top_tfidf_word(i) for i in range(len(df))]
keyword_avg_likes = df.groupby('top_keyword')['likes'].mean()
df['avg_likes_for_keyword'] = df['top_keyword'].map(keyword_avg_likes)
df['avg_likes_for_keyword'].fillna(df['likes'].mean(), inplace=True)
print("Feature engineering complete.")

print("Training the RandomForestRegressor model...")
features_to_train = ['word_count', 'char_count', 'has_media', 'sentiment', 'company_avg_likes', 'hashtag', 'avg_likes_for_keyword']
X = df[features_to_train]
y = df['likes']
X.fillna(0, inplace=True)
model = RandomForestRegressor(random_state=42)
model.fit(X, y)
print("Model training complete.")

print("Saving all required assets to the 'assets/' directory...")
assets_dir = 'assets'
if not os.path.exists(assets_dir):
   os.makedirs(assets_dir)

joblib.dump(model, os.path.join(assets_dir, 'like_predictor_model.joblib'))
joblib.dump(vectorizer, os.path.join(assets_dir, 'tfidf_vectorizer.joblib'))

company_to_avg_likes = df.groupby('inferred company')['likes'].mean().to_dict()
keyword_to_avg_likes = df.groupby('top_keyword')['likes'].mean().to_dict()
metadata = {
   "company_to_avg_likes": company_to_avg_likes,
   "keyword_to_avg_likes": keyword_to_avg_likes,
}
with open(os.path.join(assets_dir, 'metadata.json'), 'w') as f:
   json.dump(metadata, f, indent=4)

print("\nConfirmation: All assets have been successfully created and saved.")
