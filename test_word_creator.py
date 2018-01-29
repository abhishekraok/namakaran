import model
import unittest


class TestWordCreator(unittest.TestCase):
    def test_simple_sequence_train(self):
        x_train_sequence = y_train_sequence = [1, 3, 6, 1, 2, 6, 1, 1, 6]
        word_creator_model = model.WordRNN()
        word_creator_model.train(x_train_sequence, y_train_sequence)
