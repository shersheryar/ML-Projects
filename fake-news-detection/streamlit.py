import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

@st.cache_resource
def load_model():
    model = joblib.load("random_forest_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")         
    return model, vectorizer

model, vectorizer = load_model()

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    processed_words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("Fake News Detection App")
st.markdown("Enter a news headline or article text to check if it's **real** or **fake**.")

user_input = st.text_area("Enter news text here:", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        cleaned_text = preprocess_text(user_input)
        transformed_text = vectorizer.transform([cleaned_text])

        prediction = model.predict(transformed_text)[0]

        # Display result
        if prediction == 1:
            st.success("This news seems **REAL**.")
        else:
            st.error("This news seems **FAKE**.")

# -----------------------------
# 📊 Optional Styling or Info
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn By **No0ne**")
