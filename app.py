import streamlit as st
import joblib
model = joblib.load('imdb-rating')
st.title('Sentiment Analyzer')
input = st.text_input('Enter your review:')
output = model.predict([input])
if st.button('Predict'):
  st.title(output[0])
