from DSSM.batchiterators.fileiterators import NaturalQuestionsFileIterator
from DSSM.dssm.model import *
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from DSSM.helpers.helpers import correct_guesses_of_dssm

init = tf.compat.v1.global_variables_initializer()

saver = tf.compat.v1.train.Saver()

# First just train on nq
def get_feed_dict(batch):
    q_indices_batch = batch.get_q_indices()
    p_indices_batch = batch.get_relevant_indices()
    n1_indices_batch, n2_indices_batch, n3_indices_batch, n4_indices_batch = batch.get_irrelevant_indices()

    q_values_batch = np.ones(q_indices_batch.shape[0], dtype=int)
    p_values_batch = np.ones(p_indices_batch.shape[0], dtype=int)
    n1_values_batch = np.ones(n1_indices_batch.shape[0], dtype=int)
    n2_values_batch = np.ones(n2_indices_batch.shape[0], dtype=int)
    n3_values_batch = np.ones(n3_indices_batch.shape[0], dtype=int)
    n4_values_batch = np.ones(n4_indices_batch.shape[0], dtype=int)

    feed_dict = {
        q_indices: q_indices_batch,
        q_values: q_values_batch,
        p_indices: p_indices_batch,
        p_values: p_values_batch,
        n1_indices: n1_indices_batch,
        n1_values: n1_values_batch,
        n2_indices: n2_indices_batch,
        n2_values: n2_values_batch,
        n3_indices: n3_indices_batch,
        n3_values: n3_values_batch,
        n4_indices: n4_indices_batch,
        n4_values: n4_values_batch
    }
    return feed_dict


fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()

fig.show()
fig.canvas.draw()


def update_plot(data1, data2):
    ax.clear()
    ax.plot(data1)
    ax.plot(data2)
    ax.set_ylim(bottom=0)
    fig.canvas.draw()
    # `start_event_loop` is required for console, not jupyter notebooks.
    # Don't use `plt.pause` because it steals focus and makes it hard
    # to stop the app.
    fig.canvas.start_event_loop(0.001)


with tf.compat.v1.Session() as sess:
    sess.run(init)

    train_epoch_accuracies = []
    train_losses = []
    val_epoch_accuracies = []
    val_losses = []

    trainingSet = NaturalQuestionsFileIterator("/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_backup/nq/smalltrain.csv",
                                               batch_size = BATCH_SIZE,
                                               no_of_irrelevant_samples = 4,
                                               encodingType="NGRAM")
    validationSet = NaturalQuestionsFileIterator("/Users/sahandzarrinkoub/School/year5/thesis/datasets/preprocessed_backup/nq/smallvalidation.csv",
                                                 batch_size=BATCH_SIZE,
                                                 no_of_irrelevant_samples=4,
                                                 encodingType="NGRAM")
    for epoch in range(10):
        if epoch > 0:
            trainingSet.restart()
            validationSet.restart()

        ll_train_overall = 0
        correct_train = 0
        for batch in tqdm(trainingSet):
            feed_dict = get_feed_dict(batch)

            _, ll = sess.run([optimizer, logloss], feed_dict=feed_dict)
            print(ll)
            ll_train_overall += ll
            correct_train += correct_guesses_of_dssm(sess, feed_dict, prob_p, prob_n1, prob_n2, prob_n3, prob_n4)

        train_losses.append(ll_train_overall / trainingSet.getNoOfDataPoints())
        train_epoch_accuracies.append(correct_train / trainingSet.getNoOfDataPoints())


        #evaluate on validation set
        ll_val_overall = 0
        correct_val = 0
        for batch in validationSet:
            feed_dict = get_feed_dict(batch)
            (ll_val,) = sess.run([logloss], feed_dict=feed_dict)
            correct_val += correct_guesses_of_dssm(sess, feed_dict, prob_p, prob_n1, prob_n2, prob_n3, prob_n4)
            ll_val_overall += ll_val
        val_losses.append(ll_val_overall / validationSet.getNoOfDataPoints())
        val_epoch_accuracies.append(correct_val / validationSet.getNoOfDataPoints())

        update_plot(train_losses, val_losses)

    plt.figure()
    plt.plot(train_epoch_accuracies)
    plt.plot(val_epoch_accuracies)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("accuracies")
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xlabel("batch")
    plt.title("loss")
    plt.show()
    saver.save(sess, './saved_model', global_step=epoch)
