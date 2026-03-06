# app.py
import gradio as gr
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re, string

# ----------------------------
# Load trained model
# ----------------------------
model = tf.keras.models.load_model("SentimentAnalysis(me)_model.h5")  # make sure file is in the same folder

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)

# ----------------------------
# Preprocessing function
# ----------------------------
# Replace this stopwords list with your full list if you have
stop_words = set(["a", "the", "is", "and", "or", "but", "on", "in", "with", "to", "for"])

def preprocessings(text):
    text = str(text)
    text = re.sub(r"\d+", "", text)                      # Remove numbers
    text = text.lower()                                  # Lowercase
    text = ''.join(char for char in text if char not in string.punctuation)  # Remove punctuation
    text = " ".join(word for word in text.split() if word not in stop_words) # Remove stopwords
    return text

# ----------------------------
# Sentiment mapping
# ----------------------------
# Based on your snippet
sentiment_map = {-1: "negative", 0: "neutral", 1: "positive"}

# ----------------------------
# Prediction function
# ----------------------------
def predict_sentiment(new_text):
    # 1. Preprocess
    new_text = preprocessings(new_text)

    # 2. Convert to sequence
    seq = tok.texts_to_sequences([new_text])
    seq_padded = pad_sequences(seq, maxlen=108, padding='post')

    # 3. Predict
    pred_prob = model.predict(seq_padded)
    pred_class = pred_prob.argmax(axis=1)[0]

    # 4. Map to sentiment
    return sentiment_map.get(pred_class, "neutral")  # fallback to neutral

# ----------------------------
# Gradio interface
# ----------------------------
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter your review here..."),
    outputs="text",
    title="3-Class Sentiment Analysis",
    description="Type a review and see if it is negative, neutral, or positive!",
    examples=[
        ["I love this product!"],
        ["It's okay, nothing special."],
        ["I hated it, very bad experience."],
        ["great day looks like dream"]
    ]
)

# Launch the app and generate a shareable public link
iface.launch(share=True)
