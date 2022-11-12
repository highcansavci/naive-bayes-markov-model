# Naive-Bayes Classifier Using Markov Model
# @author Can Savcı
import time

from config import EDGAR_ALAN_POE_TXT_URL, ROBERT_FROST_URL
from naive_bayes.naive_bayes_engine import NaiveBayesEngine
from naive_bayes.naive_bayes_classifier import NaiveBayesClassifier
from reader.reader import Reader
from tokenizer.tokenizer import Tokenizer

if __name__ == '__main__':
    reader = Reader(EDGAR_ALAN_POE_TXT_URL, ROBERT_FROST_URL)
    tokenizer = Tokenizer()
    naive_bayes_engine = NaiveBayesEngine(reader, tokenizer)
    naive_bayes_classifier = NaiveBayesClassifier(naive_bayes_engine)

    print("*************  Split Data  *******************")
    start_time = time.time()
    train_x, test_x, train_y, test_y = reader.train_test_split()
    start_time = time.time() - start_time
    print(f"Split data completed in {start_time}")

    print("*************  Train Model  *******************")
    start_time = time.time()
    naive_bayes_classifier.fit(train_x, train_y)
    start_time = time.time() - start_time
    print(f"Train completed in {start_time}")

    print("**********  Calculate Predictions  ************")
    start_time = time.time()
    train_pred = naive_bayes_classifier.predict(train_x)
    test_pred = naive_bayes_classifier.predict(test_x)
    print(f"Train Prediction: {train_pred}")
    print(f"Test Prediction: {test_pred}")
    start_time = time.time() - start_time
    print(f"Predictions calculated in {start_time}")

    print("**********  Calculate Scores  ************")
    start_time = time.time()
    train_score = naive_bayes_classifier.score(train_x, train_y)
    test_score = naive_bayes_classifier.score(test_x, test_y)
    print(f"Train Accuracy: {train_score}")
    print(f"Test Accuracy: {test_score}")
    start_time = time.time() - start_time
    print(f"Scores calculated in {start_time}")

    print("******************************************")
