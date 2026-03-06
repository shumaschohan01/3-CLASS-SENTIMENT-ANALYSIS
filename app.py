import gradio as gr
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("sentiment_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tok = pickle.load(f)

# Sentiment preprocessing function (same as before)
import re, string
stop_words = set(["a", "the", "is", "and"])  # replace with your stop_words

def preprocessings(text):
    text = str(text)
    text = re.sub(r"\d+", "", text)
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Mapping from model output to sentiment
sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(text):
    text = preprocessings(text)
    seq = tok.texts_to_sequences([text])
    seq_padded = pad_sequences(seq, maxlen=81, padding='post')
    pred_prob = model.predict(seq_padded)
    pred_class = pred_prob.argmax(axis=1)[0]
    return sentiment_map[pred_class]

# Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter your review here..."),
    outputs="text",
    title="3-Class Sentiment Analysis",
    description="Type a review and see if it is negative, neutral, or positive!"
)

iface.launch(share=True)  # share=True gives you a public link for LinkedIn