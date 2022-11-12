import string

import numpy as np

from config import NUMBER_OF_CLASSES

UNKNOWN_WORD_SPECIAL_INDEX = 0


def decode_word(line, indexedTrainingMapping):
    encoded_words = []
    for word in line:
        if word not in indexedTrainingMapping:
            encoded_words.append(UNKNOWN_WORD_SPECIAL_INDEX)
        else:
            encoded_words.append(indexedTrainingMapping[word])
    return encoded_words


def compute_priors(train_y, number_of_classes):
    tr_y = np.array(train_y)
    shape = tr_y.shape[0]
    return [np.log(tr_y[tr_y == k].shape[0] / shape) for k in range(number_of_classes)]


def fill_matrix(map_size, input_list):
    # create add one-smoothed matrix
    matrix = np.ones((map_size, map_size))
    vector = np.ones(map_size)
    # fill out
    for tokens in input_list:
        for i in range(len(tokens) - 1):
            if i == 0:
                vector[tokens[i]] += 1
            else:
                matrix[tokens[i], tokens[i + 1]] += 1

    # normalize
    vector /= vector.sum()
    matrix /= matrix.sum(axis=1, keepdims=True)

    # find log prob
    return np.log(matrix), np.log(vector)


class NaiveBayesEngine:

    def __init__(self, reader, tokenizer):
        self.reader = reader
        self.tokenizer = tokenizer
        self.prior_collection = []
        self.log_posterior_matrix_collection = []
        self.log_posterior_vector_collection = []
        self.indexed_training_mapping = {"UNK": UNKNOWN_WORD_SPECIAL_INDEX}

    def fit(self, train_x, train_y):
        train_list = self.compute(train_x)

        for k in range(NUMBER_OF_CLASSES):
            input_list = train_list[train_y == k]
            map_size = len(self.indexed_training_mapping)
            matrix, vector = fill_matrix(map_size, input_list)
            self.log_posterior_matrix_collection.append(matrix)
            self.log_posterior_vector_collection.append(vector)

        self.prior_collection = compute_priors(train_y, NUMBER_OF_CLASSES)

    def compute(self, train_x):
        train_list = self.encode_line(train_x, self.indexed_training_mapping)
        return np.array(train_list, dtype=object)

    def predict(self, test_x):
        test_list = self.decode_line(test_x, self.indexed_training_mapping)
        return np.array(test_list, dtype=object)

    def decode_line(self, data, indexedTrainingMapping):
        encoded_lines = []
        for line in data:
            split_string = self.tokenizer.tokenize(line)
            encoded_lines.append(decode_word(split_string, indexedTrainingMapping))
        return encoded_lines

    def encode_line(self, data, indexedTrainingMapping):
        encoded_lines = []
        index = 1
        for line in data:
            split_string = self.tokenizer.tokenize(line)
            encoded_words = []
            for word in split_string:
                if word not in indexedTrainingMapping:
                    indexedTrainingMapping[word] = index
                    encoded_words.append(index)
                    index += 1
                else:
                    encoded_words.append(indexedTrainingMapping[word])
            encoded_lines.append(encoded_words)
        return encoded_lines
