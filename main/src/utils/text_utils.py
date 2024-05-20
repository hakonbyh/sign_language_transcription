import nltk
import spacy
from nltk.corpus import stopwords

nlp = spacy.load("nb_core_news_sm")

nltk.download("stopwords")
nltk.download("punkt")

norwegian_stopwords = set(stopwords.words("norwegian"))
norwegian_stopwords.discard("jeg")


def lemmatize(sentence):
    doc = nlp(sentence)
    lemmatized_words = [token.lemma_ for token in doc]
    return lemmatized_words


def remove_stopwords(words):
    filtered_words = [word for word in words if word.lower() not in norwegian_stopwords]
    return " ".join(filtered_words)
