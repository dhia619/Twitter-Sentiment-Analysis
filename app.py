import streamlit as st
import pickle
from re import sub
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Clean and preprocess text
    text = text.lower()
    text = sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = sub(r'\d+', '', text)      # Remove numbers
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def predict_sentiment(text, model, vectorizer):
    # Preprocess and predict sentiment
    text = clean_text(text)
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return "Positive" if prediction == 1 else "Negative"

# Load the model and vectorizer
with open("twitter_sentiment_analysis_LR.pkl", "rb") as model_file:
    lr_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vect_file:
    vectorizer = pickle.load(vect_file)

# Streamlit interface
st.markdown(
    """
    <h1 style="text-align:center;">Sentiment Analysis App</h1>
    """,
    unsafe_allow_html=True
)

user_input = st.text_input("Write something:", placeholder="I'm Happy")
if st.button("Predict"):
    if user_input:
        sentiment_result = predict_sentiment(user_input, lr_model, vectorizer)
        st.write(f"Sentiment: **{sentiment_result}**")
    else:
        st.warning("Please enter some text to analyze.")
