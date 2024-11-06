import streamlit as st
import pickle
from re import sub
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = sub(r'[^\w\s]', '', text) # Remove punctuation
    text = sub(r'[\d+]', '', text) # Remove numbers
    
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def predict(label, text, model, vectorizer):
    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    p = model.predict(text_vec)
    label.text("Positive" if p == 1 else "Negative")

# Load the model
with open("twitter_sentiment_analysis_LR.pkl", "rb") as model_file:
    lr_model = pickle.load(model_file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as vect_file:
    vectorizer = pickle.load(vect_file)


st.markdown("""
        <h1 style="text-align:center;">Sentiment Analysis App</h1>
""", unsafe_allow_html=True)

user_input = st.text_input("Write Something :",placeholder="I'm Happy")

sentiment_result_label = st.text("")

predict_button = st.button("Predict", on_click = predict(sentiment_result_label, user_input, lr_model, vectorizer))