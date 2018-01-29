import model
import time
import numpy as np


def run_epoch(session, epoch_model, input_batches, target_batches, eval_op=None, verbose=False):
    """Runs the model on the given data.

    :type epoch_model: model.WordRNN
    """
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(epoch_model.initial_state)

    fetches = {
        "cost": epoch_model.cost,
        "final_state": epoch_model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    epoch_size = len(input_batches)
    for step in range(epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(epoch_model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
            feed_dict[epoch_model.input_data] = input_batches[step]
            feed_dict[epoch_model.targets] = target_batches[step]

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += epoch_model.config.batch_size

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * epoch_model.config.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)
