import tensorflow as tf
import inspect


def data_type():
    return tf.float32


class VerySmallConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    source_sequence_length = 20
    hidden_size = 20
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    input_vocab_size = 1000
    target_vocab_size = 26


class WordRNN:
    def __init__(self, is_training, config, input_, tgt_vocab_size):
        """

        :type config: VerySmallConfig
        """
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        def lstm_cell():
            # With the latest TensorFlow source code (as of Mar 27, 2017),
            # the BasicLSTMCell will need a reuse parameter which is unfortunately not
            # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
            # an argument check here:
            if 'reuse' in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return tf.contrib.rnn.BasicLSTMCell(
                    size, forget_bias=0.0, state_is_tuple=True,
                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(
                    size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [config.input_vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # # Build RNN cell
        # encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size)
        #
        # # Run Dynamic RNN
        # #   encoder_outputs: [max_time, batch_size, num_units]
        # #   encoder_state: [batch_size, num_units]
        # encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        #     encoder_cell, inputs,
        #     sequence_length=config.source_sequence_length, time_major=True)
        #
        # # Build RNN cell
        # decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size)
        #
        # # Helper
        # helper = tf.contrib.seq2seq.TrainingHelper(
        #     decoder_emb_inp, decoder_lengths, time_major=True)
        #
        # projection_layer = layers_core.Dense(
        #     tgt_vocab_size, use_bias=False)
        # # Decoder
        # decoder = tf.contrib.seq2seq.BasicDecoder(
        #     decoder_cell, helper, encoder_state,
        #     output_layer=projection_layer)
        # # Dynamic decoding
        # decoder_outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
        # logits = decoder_outputs.rnn_output
        # crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=decoder_outputs, logits=logits)
        # train_loss = (tf.reduce_sum(crossent * target_weights) /
        #     batch_size)
        #
        # # Calculate and clip gradients
        # params = tf.trainable_variables()
        # gradients = tf.gradients(train_loss, params)
        # clipped_gradients, _ = tf.clip_by_global_norm(
        #     gradients, config.max_grad_norm)
        #






        # Simplified version of models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(
        #     cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, config.target_vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [config.target_vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b

        # Reshape logits to be 3-D tensor for sequence loss
        logits = tf.reshape(logits, [batch_size, num_steps, config.target_vocab_size])

        # use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([batch_size, num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True
        )

        # update the cost variables
        self._cost = cost = tf.reduce_sum(loss)
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        def train(self, x, y):
            pass
