import streamlit as st
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Charger le modèle Spacy pour le traitement en anglais
nlp = spacy.load('en_core_web_sm')

st.title('DOJO NLP')

phrase = st.text_input("Please insert a sentence:")

def stemming(phrase):
    # Tokenize the sentence, remove stopwords and apply stemming
    doc = nlp(phrase.lower())
    stemmed_words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(stemmed_words)

def lemmatizing(phrase):
    # Tokenize the sentence, remove stopwords and apply lemmatizing
    doc = nlp(phrase.lower())
    lemmed_words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(lemmed_words)

def tf_idf(phrase):
    # Créer un TfidfVectorizer
    vectorizer = TfidfVectorizer()
    # Ajuster le vecteur TF-IDF et transformer la phrase
    tfidf_matrix = vectorizer.fit_transform([phrase])
    # Convertir le résultat en DataFrame pour une meilleure visualisation
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return df_tfidf

def generate_wordcloud(phrase):
    # Suppression de la ponctuation et des stopwords
    doc = nlp(phrase.lower())
    clean_sentence = " ".join([token.text for token in doc if not token.is_stop and token.is_alpha])
    # Créer un objet WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(clean_sentence)
    # Afficher le word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

if st.button("Stemmer"):
    st.write("Stemmed Sentence:")
    st.success(stemming(phrase))

if st.button("Lemmatizing"):
    st.write("Lemmatized Sentence:")
    st.success(lemmatizing(phrase))

if st.button("TF - IDF"):
    st.write("TF-IDF DataFrame:")
    st.dataframe(tf_idf(phrase))

if st.button("Word Cloud"):
    st.write("Word Cloud:")
    generate_wordcloud(phrase)
