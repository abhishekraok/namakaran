import data_helper
import model


def main():
    x_train_sequence, y_train_sequence = data_helper.get_data()
    word_creator_model = model.WordRNN()
    word_creator_model.train(x_train_sequence, y_train_sequence)

if __name__ == '__main__':
    main()
