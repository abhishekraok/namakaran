import data_helper
import tensorflow as tf
import model


def main():
    x_train_sequence, y_train_sequence = data_helper.get_data()
    word_creator_model = model.WordRNN()
    with tf.Session() as sess:
        train_perplexity = run_epoch(sess, word_creator_model, word_creator_model._train_op)


if __name__ == '__main__':
    main()
