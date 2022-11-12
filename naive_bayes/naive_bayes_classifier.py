import numpy as np

from config import NUMBER_OF_CLASSES


def predict_helper(prior, vector, matrix, input_):
    # predict
    score = 0
    for i in range(len(input_) - 1):
        if i == 0:
            score += vector[input_[i]]
        else:
            score += matrix[input_[i], input_[i + 1]]

    return score + prior


class NaiveBayesClassifier:

    def __init__(self, naive_bayes):
        self.naive_bayes = naive_bayes

    def _compute_log_likelihood(self, data, target_class):
        return predict_helper(self.naive_bayes.prior_collection[target_class],
                              self.naive_bayes.log_posterior_vector_collection[target_class],
                              self.naive_bayes.log_posterior_matrix_collection[target_class], data)

    def fit(self, train_x, train_y):
        self.naive_bayes.fit(train_x, train_y)

    def predict(self, test_x):
        predictions = np.zeros(len(test_x))
        test_list = self.naive_bayes.predict(test_x)
        for i, input_ in enumerate(test_list):
            posteriors = [self._compute_log_likelihood(input_, target) for target in range(NUMBER_OF_CLASSES)]
            predictions[i] = np.argmax(posteriors)
        return predictions

    def score(self, inputs, labels):
        predictions = self.predict(inputs)
        return np.mean(predictions == labels)
