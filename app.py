# app.py
import gradio as gr
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re, string

# ----------------------------
# Load trained model
# ----------------------------
model = tf.keras.models.load_model("sentiment_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)

# ----------------------------
# Preprocessing function
# ----------------------------
# Replace with your actual stopwords list
stop_words = set([
    "a", "an", "the", "is", "and", "or", "but", "on", "in", "with", "to", "for"
])

def preprocessings(text):
    text = str(text)
    text = re.sub(r"\d+", "", text)                      # Remove numbers
    text = text.lower()                                  # Lowercase
    text = ''.join(char for char in text if char not in string.punctuation)  # Remove punctuation
    text = " ".join(word for word in text.split() if word not in stop_words) # Remove stopwords
    return text

# ----------------------------
# Mapping output to sentiment
# ----------------------------
sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

# ----------------------------
# Prediction function
# ----------------------------
def predict_sentiment(text):
    text = preprocessings(text)
    seq = tok.texts_to_sequences([text])
    seq_padded = pad_sequences(seq, maxlen=81, padding='post')
    pred_prob = model.predict(seq_padded)
    pred_class = pred_prob.argmax(axis=1)[0]
    return sentiment_map[pred_class]

# ----------------------------
# Gradio interface
# ----------------------------
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter your review here..."),
    outputs="text",
    title="3-Class Sentiment Analysis",
    description="Type a review and see if it is negative, neutral, or positive!"
)

# Launch the app and generate a shareable public link
iface.launch(share=True)
