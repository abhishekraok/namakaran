import model
import unittest
import training_helper
import tensorflow as tf


class TestWordCreator(unittest.TestCase):
    def test_simple_sequence_train(self):
        x_train_sequence = y_train_sequence = [[1, 3, 6, 1, 2, 6, 1, 1, 6]]
        word_creator_model = model.WordRNN(is_training=False, config=model.VerySmallConfig())
        with tf.Session as sess:
            training_helper.run_epoch(sess, word_creator_model, x_train_sequence, y_train_sequence)
