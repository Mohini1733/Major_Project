
import streamlit as st
import joblib
import base64
st.set_page_config(layout="wide")
st.markdown("""
<style>
.big-font {
    font-size;50px  !important;
    font-family:Arial;
    color:#D5C4C4; 

}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.big-font {
    font-size;20px  !important;
    font-family:Courier;
    color:#D5C4C4; 
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">IMDB Review Sentiment Analysis !!</p>',
            unsafe_allow_html=True)

st.markdown('<p class="medium-font">Enter your review!</p>', 
            unsafe_allow_html=True)

review= st.text_input("here")
model=joblib.load('https://github.com/Mohini1733/Major_Project/blob/main/imdb-rating')

op=model.predict([review])

if st.button('Analyse'):
  st.markdown(f'<p class="medium-font">The Review is {op[0]} </p>',
              unsafe_allow_html=True)
