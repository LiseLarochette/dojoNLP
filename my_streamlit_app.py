import streamlit as st
import pandas as pd
import re

st.title('DOJO NLP')

phrase = input("Please insert a sentence:")

# Ajouter des boutons pour chaque fonctionnalité
if st.button("Stemmer"):
    st.write("Stemmed Sentence:")
    st.success(stemming(phrase))

if st.button("Lemmatizer"):
    st.write("Lemmatized Sentence:")
    st.success(lemmatizing(phrase))

if st.button("TF - IDF"):
    st.write("TF-IDF DataFrame:")
    st.dataframe(tf_idf(phrase))

if st.button("Word Cloud"):
    st.write("Word Cloud:")
    generate_wordcloud(phrase)
