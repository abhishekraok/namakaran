import model
import unittest
import training_helper
import tensorflow as tf
import numpy as np


class TestWordCreator(unittest.TestCase):
    def test_simple_sequence_train(self):
        num_batches = 5
        word_creator_model = model.WordRNN(is_training=False, config=model.VerySmallConfig())
        x_train_batches = y_train_batches = [np.random.randint(low=0, high=10, size=(
            word_creator_model.config.batch_size, word_creator_model.config.source_sequence_length) for _ in
                                             range(num_batches)]
        with tf.Session as sess:
            training_helper.run_epoch(sess, word_creator_model, x_train_batches, y_train_batches)
