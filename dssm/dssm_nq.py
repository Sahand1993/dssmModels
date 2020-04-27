from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from DSSM.batchiterators.batchiterators import NqSQLIterator
from DSSM.helpers.helpers import correct_guesses_of_dssm, cosine_similarity

NO_OF_TRIGRAMS = int(os.popen("wc -l /Users/sahandzarrinkoub/School/year5/thesis/DSSM/preprocessed_datasets/trigrams.txt").read().split()[0])
TRAINING_EPOCHS = 1
BATCH_SIZE = 1024

# tf Graph Input
q_indices = tf.placeholder("int64", name="q_indices", shape=[None, 2])
q_values = tf.placeholder("int64", name="q_values", shape=[None])
x_q = tf.SparseTensor(q_indices, tf.cast(q_values, tf.float32), dense_shape=[BATCH_SIZE, NO_OF_TRIGRAMS])

p_indices = tf.placeholder("int64", name="p_indices", shape=[None, 2])
p_values = tf.placeholder("int64", name="p_values", shape=[None])
x_p = tf.SparseTensor(p_indices, tf.cast(p_values, tf.float32), dense_shape=[BATCH_SIZE, NO_OF_TRIGRAMS])

n1_indices = tf.placeholder("int64", name="n1_indices", shape=[None, 2])
n1_values = tf.placeholder("int64", name="n1_values", shape=[None])
x_n1 = tf.SparseTensor(n1_indices, tf.cast(n1_values, tf.float32), dense_shape=[BATCH_SIZE, NO_OF_TRIGRAMS])

n2_indices = tf.placeholder("int64", name="n2_indices", shape=[None, 2])
n2_values = tf.placeholder("int64", name="n2_values", shape=[None])
x_n2 = tf.SparseTensor(n2_indices, tf.cast(n2_values, tf.float32), dense_shape=[BATCH_SIZE, NO_OF_TRIGRAMS])

n3_indices = tf.placeholder("int64", name="n3_indices", shape=[None, 2])
n3_values = tf.placeholder("int64", name="n3_values", shape=[None])
x_n3 = tf.SparseTensor(n3_indices, tf.cast(n3_values, tf.float32), dense_shape=[BATCH_SIZE, NO_OF_TRIGRAMS])

n4_indices = tf.placeholder("int64", name="n4_indices", shape=[None, 2])
n4_values = tf.placeholder("int64", name="n4_values", shape=[None])
x_n4 = tf.SparseTensor(n4_indices, tf.cast(n4_values, tf.float32), dense_shape=[BATCH_SIZE, NO_OF_TRIGRAMS])

W_2 = tf.Variable(tf.truncated_normal([NO_OF_TRIGRAMS, 300], mean=0, stddev=0.1, dtype=tf.float32), name="W2")
b_2 = tf.Variable(tf.truncated_normal([1, 300], mean=0, stddev=0.1, dtype=tf.float32), name="b2")

l2_q = tf.tanh(tf.sparse.sparse_dense_matmul(x_q, W_2) + b_2)
l2_p = tf.tanh(tf.sparse.sparse_dense_matmul(x_p, W_2) + b_2)
l2_n1 = tf.tanh(tf.sparse.sparse_dense_matmul(x_n1, W_2) + b_2)
l2_n2 = tf.tanh(tf.sparse.sparse_dense_matmul(x_n2, W_2) + b_2)
l2_n3 = tf.tanh(tf.sparse.sparse_dense_matmul(x_n3, W_2) + b_2)
l2_n4 = tf.tanh(tf.sparse.sparse_dense_matmul(x_n4, W_2) + b_2)

W_3 = tf.Variable(tf.truncated_normal([300, 300], mean=0, stddev=0.1, dtype=tf.float32), name="W3")
b_3 = tf.Variable(tf.truncated_normal([1, 300], mean=0, stddev=0.1, dtype=tf.float32), name="b3")

l3_q = tf.tanh(tf.matmul(l2_q, W_3) + b_3)
l3_p = tf.tanh(tf.matmul(l2_p, W_3) + b_3)
l3_n1 = tf.tanh(tf.matmul(l2_n1, W_3) + b_3)
l3_n2 = tf.tanh(tf.matmul(l2_n2, W_3) + b_3)
l3_n3 = tf.tanh(tf.matmul(l2_n3, W_3) + b_3)
l3_n4 = tf.tanh(tf.matmul(l2_n4, W_3) + b_3)

W_4 = tf.Variable(tf.truncated_normal([300, 128], mean=0, stddev=0.1, dtype=tf.float32), name="W4")
b_4 = tf.Variable(tf.truncated_normal([1, 128], mean=0, stddev=0.1, dtype=tf.float32), name="b4")

y_q = tf.tanh(tf.matmul(l3_q, W_4) + b_4)
y_p = tf.tanh(tf.matmul(l3_p, W_4) + b_4)
y_n1 = tf.tanh(tf.matmul(l3_n1, W_4) + b_4)
y_n2 = tf.tanh(tf.matmul(l3_n2, W_4) + b_4)
y_n3 = tf.tanh(tf.matmul(l3_n3, W_4) + b_4)
y_n4 = tf.tanh(tf.matmul(l3_n4, W_4) + b_4)

r_p = cosine_similarity(y_q, y_p)
r_n1 = cosine_similarity(y_q, y_n1)
r_n2 = cosine_similarity(y_q, y_n2)
r_n3 = cosine_similarity(y_q, y_n3)
r_n4 = cosine_similarity(y_q, y_n4)

sum_r = tf.exp(r_p) + tf.exp(r_n1) + tf.exp(r_n2) + tf.exp(r_n3) + tf.exp(r_n4)

prob_p = tf.math.divide(tf.exp(r_p), sum_r)
prob_n1 = tf.math.divide(tf.exp(r_n1), sum_r)
prob_n2 = tf.math.divide(tf.exp(r_n2), sum_r)
prob_n3 = tf.math.divide(tf.exp(r_n3), sum_r)
prob_n4 = tf.math.divide(tf.exp(r_n4), sum_r)

logloss = -tf.reduce_sum(tf.log(prob_p))
optimizer = tf.train.AdamOptimizer(0.0001).minimize(logloss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

# First just train on nq
with tf.Session() as sess:
    sess.run(init)

    epoch_accuracies = []
    losses = []

    nq_batch_iterator = NqSQLIterator(
        batch_size=BATCH_SIZE,
        no_of_irrelevant_samples=4)

    for epoch in range(10):
        if epoch > 0:
            nq_batch_iterator.restart()

        ll_overall = 0
        correct = 0
        for batch in tqdm(nq_batch_iterator):
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
            _, ll = sess.run([optimizer, logloss], feed_dict=feed_dict)

            ll_overall += ll
            losses.append(ll)
            correct += correct_guesses_of_dssm(sess, feed_dict, prob_p, prob_n1, prob_n2, prob_n3, prob_n4)

        epoch_accuracies.append(correct / nq_batch_iterator.get_no_of_data_points())

    plt.figure()
    plt.plot(epoch_accuracies)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("accuracies")
    plt.figure()
    plt.plot(losses)
    plt.xlabel("batch")
    plt.title("loss")
    plt.show()
    saver.save(sess, './saved_model', global_step=epoch)
