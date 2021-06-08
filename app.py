import streamlit as st
import joblib
import base64
st.set_page_config(layout='wide')
st.markdown("""
<style>
.big-font {
    font-size:50px !important;
    font-family:Lucida Handwriting;
    color:#D5C4C4;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.small-font {
    font-size: 18px !important;
    font-family:Courier;
    color:#DFF159;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.medium-font {
    font-size:20px !important;
    font-family:Lucida Handwriting;
    color:#D5C4C4;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">IMDB Review Sentiment Analysis !!</p>', unsafe_allow_html=True)

def fun():
  st.markdown('<p class="small-font">The website is a part of project of Internship.<br> The website deals with Sentiment analysis based on user review.<br> It generally take review of movies and predict the sentiment of user on the movie.</p>', unsafe_allow_html=True)
  st.markdown('<p class="small-font">I would like to thank my team mates, mentor for their constant support and knowledge.</p>', unsafe_allow_html=True)  
  return


st.markdown('<p class="medium-font">Enter Review !!</p>', unsafe_allow_html=True)
review = st.text_input("here")
with open('imdb-rating','rb') as f:
    model = joblib.load('imdb-rating')
op = model.predict([review])
if st.button('Analyse'):
  st.markdown(f'<p class="medium-font"> The Review is {op[0]} </p>',unsafe_allow_html=True)





if st.sidebar.button('About'):
  fun()
