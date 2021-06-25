
import streamlit as st
import joblib
model = joblib.load('IMDB_Reviews')
st.title('Sentiment Analyser')
ip = st.text_input('Enter your review: ')
op = model.predict([ip])
ans=op[0]
if st.button('Predict'):
  st.title(op[0])
