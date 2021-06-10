import numpy as np
import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
nltk.download('all')

path = 'https://github.com/Mohini1733/Major_Project/blob/main/IMDB%20Dataset.csv'
df = pd.read_csv(path)
df.head()

df['sentiment'].value_counts()
sns.countplot(x = df['sentiment'])
df.drop(df.tail(10000).index,
        inplace = True)
df.sentiment.value_counts()
sns.countplot(x = df['sentiment'])

import re
stop_words = stopwords.words('english')
len(stop_words)
negative_words=['no','not',"don't","aren't","couldn't","didn't","doesn't","hadn't","hasn't","haven't","isn't","mightn't","mustn't","needn't","shouldn't","wasn't","weren't","won't","wouldn't"]
for negative_word in negative_words:
  stop_words.remove(negative_word)

len(stop_words)
    
REPLACE_BY_SPACE_RE = re.compile('[/(){}—[]|@,;‘?|।!-॥–’-]')

def clean_text(sample):
  sample = sample.lower()
  sample = sample.replace("<br /><br />", "")
  sample = REPLACE_BY_SPACE_RE.sub(' ', sample)
  sample = re.sub("[^a-z]+", " ", sample)
  sample = re.sub("[0-9]", " ",sample)
  sample = sample.split(" ")
  sample = [word for word in sample if word not in stop_words ]
  sample = " ".join(sample)
  return sample 

ps = PorterStemmer()
filter_review = []
for sentence in df['review']:
  filter_sentence = []
  sentence = sentence.replace('<br /><br />',' ')
  sentence = re.sub('[^a-zA-Z]',' ',sentence)
  sentence = sentence.lower()
  for word in nltk.word_tokenize(sentence):
    if word not in stopwords.words('english'):
      filter_sentence.append(ps.stem(word))
  filter_sentence = ' '.join(word for word in filter_sentence)
  filter_review.append(filter_sentence)
    
x = np.array(filter_review)
type(x)
tf = TfidfVectorizer()
from sklearn.svm import SVC
y = df['sentiment'].values
y = np.array(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
from sklearn.pipeline import Pipeline
text_model = Pipeline([('vect',TfidfVectorizer()),('model',SVC())])
text_model.fit(x_train,y_train)
y_pred = text_model.predict(x_test)
y_pred[:10]
y_test[:10]

from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
accuracy_score(y_pred,y_test)
classification_report(y_pred,y_test)
import joblib
joblib.dump(text_model,'imdb-rating' )

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
model=joblib.load('imdb-rating')

op=model.predict([review])

if st.button('Analyse'):
  st.markdown(f'<p class="medium-font">The Review is {op[0]} </p>',
              unsafe_allow_html=True)
