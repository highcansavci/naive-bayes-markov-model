import string

import nltk
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class Tokenizer:

    def __init__(self):
        self.lemma = WordNetLemmatizer()
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, line):
        lemma_words = []
        line = line.rstrip().lower()
        if line:
            line = line.translate(str.maketrans('', '', string.punctuation))
            line = word_tokenize(line)
            for word in line:
                lemma_words.append(self.lemma.lemmatize(word))
            lemma_words = filter(self._filter_stop_words, lemma_words)
        return list(lemma_words)

    def _filter_stop_words(self, word):
        return word not in self.stop_words
