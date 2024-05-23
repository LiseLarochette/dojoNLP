import streamlit as st
import pandas as pd
import nltk
nltk.download('popular')

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
  


def stemming(phrase):
    from nltk.stem import SnowballStemmer
    # Tokenize the sentence and remove stopwords
    tokens_clean = []
    for words in nltk.word_tokenize(sentence.lower()):
      if words not in nltk.corpus.stopwords.words("english"):
        tokens_clean.append(words)
    # Remove punctuation
    tokens_clean = [word for word in tokens_clean if word.isalnum()]
    # Initialize the SnowballStemmer
    stem_en = SnowballStemmer("english")
    # Stem the words
    stemmed_words = [stem_en.stem(word) for word in tokens_clean]
    # Display the results in a message box
    print("Stemming", f"Stemmed words: {stemmed_words}")

def lemmatized(phrase):
    import spacy
    # Import des règles anglaises
    nlp = spacy.load('en_core_web_sm')
    #remove stopwords
    tokens_clean = []
    for words in nltk.word_tokenize(phrase.lower()):
      if words not in nltk.corpus.stopwords.words("english"):
        tokens_clean.append(words)
    # Remove punctuation
        tokens_clean = [word for word in tokens_clean if word.isalnum()]
    # spacy découpe automatiquement en tokens avec cette syntaxe :
    lemmed_words = [nlp(word) for word in tokens_clean]
    print(" Lemmatizing:", f"Lemmatized words: {lemmed_words}")

#TF IDF
st.write("Dataframe")
def tf_idf(phrase):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Créer un TfidfVectorizer
    vectorizer = TfidfVectorizer()
    # Ajuster le vecteur TF-IDF
    # Transformer la phrase
    tfidf_matrix = vectorizer.fit_transform([phrase])
    # Convertir le résultat en df pour une meilleure visualisation
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    # Afficher le df TF-IDF
    print(df_tfidf)

def wordcloud(phrase):
  import re
  import nltk
  from nltk.corpus import stopwords
  from wordcloud import WordCloud
  import matplotlib.pyplot as plt
  # Télécharger les stopwords
  nltk.download('stopwords')
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
  plt.show()
