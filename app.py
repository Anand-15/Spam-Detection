import streamlit as st
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import sklearn
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input_sms=st.text_area('Enter your message: ')

if st.button('Predict'):
    # 1. Preprocess the text
    transformed_sms=transform_text(input_sms)

    #2. Vectorize the transformed text
    vector_input=tfidf.transform([transformed_sms])
    #3. Predict
    result=model.predict(vector_input)[0]

    #4.Display Result

    if result==1:
        st.header("It is a spam message")
    else :
        st.header("It is a normal message")