import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Télécharger les stopwords
nltk.download('stopwords')

# Import des règles anglaises pour Spacy
nlp = spacy.load('en_core_web_sm')

st.title('DOJO NLP')

phrase = st.text_input("Please insert a sentence:")

def stemming(phrase):
    # Tokenize the sentence and remove stopwords
    tokens_clean = [word for word in nltk.word_tokenize(phrase.lower()) if word not in stopwords.words("english")]
    # Remove punctuation
    tokens_clean = [word for word in tokens_clean if word.isalnum()]
    # Initialize the SnowballStemmer
    stem_en = SnowballStemmer("english")
    # Stem the words
    stemmed_words = [stem_en.stem(word) for word in tokens_clean]
    return " ".join(stemmed_words)

def lemmatizing(phrase):
    # Tokenize the sentence and remove stopwords
    tokens_clean = [word for word in nltk.word_tokenize(phrase.lower()) if word not in stopwords.words("english")]
    # Remove punctuation
    tokens_clean = [word for word in tokens_clean if word.isalnum()]
    # Lemmatize the words
    lemmed_words = [token.lemma_ for token in nlp(" ".join(tokens_clean))]
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
    # Suppression de la ponctuation
    clean_sentence = re.sub(r'[^\w\s]', '', phrase)
    # Suppression des stopwords
    stop_words = set(stopwords.words('french'))
    clean_sentence = ' '.join([word for word in clean_sentence.split() if word.lower() not in stop_words])
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
