import streamlit as st
import pandas as pd


st.title('DOJO NLP')

phrase = input("Please insert a sentence:")

# Bouton pour générer un match idéal après soumission des informations
if st.button("Stemmer"):
    st.success(stemming(phrase))


if st.button("lemmatizing"):
    st.success(lemmatized(phrase))


if st.button("TF - IDF"):
    st.success(tf_idf(phrase))

if st.button("Word Cloud"):
    st.success(wordcloud(phrase))


