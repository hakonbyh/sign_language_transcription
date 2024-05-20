import logging
import os
import re

import spacy
from main.src.constants import SAVE_DIR
from nltk.corpus import stopwords

nlp = spacy.load("nb_core_news_sm")


logger = logging.getLogger(__name__)

norwegian_stopwords = set(stopwords.words("norwegian"))


def lemmatize(sentence):
    doc = nlp(sentence)
    return [token.lemma_ for token in doc]


def remove_stopwords(words):
    return [word for word in words if word.lower() not in norwegian_stopwords]


def get_vocab(video_files):
    txt_vocab = set()
    gls_vocab = set()

    for video_file in video_files:
        transcription_path = video_file.replace("videos", "transcripts").replace(
            ".mp4", ".txt"
        )
        try:
            with open(transcription_path, "r", encoding="utf-8") as file:
                text = file.read().lower()
                words = re.findall(r"\b\w+\b", text)

                txt_vocab.update(words)
                words_lemmatized = lemmatize(" ".join(words))

                words_no_stops_lemmatized = remove_stopwords(words_lemmatized)
                gls_vocab.update(words_no_stops_lemmatized)

        except Exception as e:
            logger.error(f"Error processing file {transcription_path}: {e}")

    return txt_vocab, gls_vocab
