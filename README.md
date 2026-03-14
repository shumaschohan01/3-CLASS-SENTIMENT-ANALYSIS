# Sentiment Analysis Web App (Deep Learning + Gradio)

## Project Overview

This project is a **Sentiment Analysis Web Application** built using **Deep Learning and Natural Language Processing (NLP)**.
The model predicts whether a given text is **Negative, Neutral, or Positive**.

The application is deployed using **Gradio** and hosted on **Hugging Face Spaces**, allowing users to interact with the model directly through a web interface.

---

## Live Demo

🔗 (https://huggingface.co/spaces/ShumasChohan/Sentiment-Analysis-WebApp)

---

## Features

* Text preprocessing pipeline
* Tokenization using Keras Tokenizer
* Deep learning model for sentiment classification
* Interactive web interface using Gradio
* Cloud deployment on Hugging Face Spaces

---

## Tech Stack

* Python
* TensorFlow / Keras
* Natural Language Processing (NLP)
* Gradio
* Hugging Face Spaces

---

## Project Structure

```
sentiment-analysis/
│
├── app.py                # Gradio web app
├── sentiment_model.h5    # Trained deep learning model
├── tokenizer.pkl         # Saved tokenizer
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## How the Model Works

1. User enters a text review
2. Text is preprocessed
3. Tokenizer converts text → sequence
4. Sequence is padded
5. Model predicts sentiment
6. Output label is displayed

Sentiment Classes:

* Negative
* Neutral
* Positive

---

## Installation (Run Locally)

Clone the repository:

git clone https://github.com/shumaschohan01/3-CLASS-SENTIMENT-ANALYSIS.git
cd 3-CLASS-SENTIMENT-ANALYSIS
```

Install dependencies:

pip install -r requirements.txt

---

Run the application:

python app.py

---

The app will start at:

(https://huggingface.co/spaces/ShumasChohan/Sentiment-Analysis-WebApp)

---

## Example Predictions

Input:

```
I love this product!
```

Output:

```
Positive
```

Input:

```
The experience was terrible.
```

Output:

```
Negative
---

## Future Improvements

* Improve model accuracy with larger datasets
* Add attention-based models (LSTM / Transformers)
* Deploy with GPU support
* Add visualization of prediction confidence



## Author

SHUMAS

Aspiring **AI Engineer / Data Scientist**

Connect with me on LinkedIn:
(https://www.linkedin.com/in/shumas-kashif-chohan-54b51830b/)
