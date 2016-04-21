#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import mlp_data_helpers
from itertools import islice
from gensim.models import word2vec
from mlp_text_cnn import NewTextCNN

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_size", 3, "Comma-separated filter sizes (default: '3')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_integer("window_size", 3, "Window size to slide on every text. (default: 3)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.iteritems()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

print("Loading data...")
train_x, train_y, test_x, test_y, trsp_vocabulary, trsp_vocabulary_inv, sequence_length = mlp_data_helpers.load_data()

# Randomly shuffle data
np.random.seed(10)
train_shuffle_indices = np.random.permutation(np.arange(len(train_y)))
train_x_shuffled = train_x[train_shuffle_indices]
train_y_shuffled = train_y[train_shuffle_indices]
test_shuffle_indices = np.random.permutation(np.arange(len(test_y)))
test_x_shuffled = test_x[test_shuffle_indices]
test_y_shuffled = test_y[test_shuffle_indices]

print("Loading word2vec...")
model = word2vec.Word2Vec.load_word2vec_format('./data/glove.6B.50d.txt', binary=False)

# embedding matrix
word2vec_matrix = model.syn0
print "len(word2vec_matrix):", len(word2vec_matrix)
# print "word2vec_matrix:", word2vec_matrix

print 'sequence_length:', sequence_length
print("Train Vocabulary Size: {:d}".format(len(trsp_vocabulary)))
print("Train Dataset Size: {:d}".format(len(train_x_shuffled)))
print("Test Dataset Size: {:d}".format(len(test_y_shuffled)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = NewTextCNN(
            sequence_length=sequence_length,
            num_classes=1,
            vocab_size=400001,
            embedding_size=FLAGS.embedding_dim,
            slide_window_size=3,
            filter_size=FLAGS.filter_size,
            batch_size=FLAGS.batch_size,
            num_filters=FLAGS.num_filters)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        # optimizer = tf.train.GradientDescentOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        sess.run(cnn.embedding_matrix.assign(word2vec_matrix))

        rs = sess.run(cnn.embedding_matrix)
        print '----------cnn.embedding_matrix:------------\n', rs

        # one dimension convolution
        def convolution_oneDim(convo_unit):
            feed_dict = {
              cnn.input_x: convo_unit,
              # cnn.input_y: train_y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            convo_unit_rs = sess.run(cnn.relu_rs, feed_dict)
            return convo_unit_rs


        # sliding windows
        def n_grams(text):
            z = (islice(text, start, None) for start in range(FLAGS.window_size))
            return zip(*z)


        # convert text to imageFormat
        def conv_text_to_image(train_x_batch_val, size_val):
            # print '------train_x_batch_val:-------\n', train_x_batch_val
            # print 'len(train_x_batch_val)--------:', len(train_x_batch_val)
            text_comp_result = []
            for text_left, text_right in train_x_batch_val:
                text_left_window = n_grams(text_left)
                text_right_window = n_grams(text_right)

                # composite the slice
                text_comp_list = []
                for text_left_slice in text_left_window:
                    for text_right_slice in text_right_window:
                        convo_unit = text_left_slice + text_right_slice
                        if convo_unit.count(0) < FLAGS.window_size*2:
                            convo_unit = np.array(convo_unit)

                            # 1D Convolution, get a matrix
                            convo_unit = [convo_unit]
                            convo_numpy_list = convolution_oneDim(convo_unit)
                            convo_python_list = convo_numpy_list.tolist()
                            convo_single_value = convo_python_list[0][0][0][0]
                            text_comp_list.append(convo_single_value)
                        else:
                            text_comp_list.append(0)
                text_comp_result.append(text_comp_list)
            train_x_batch_image_format = np.reshape(text_comp_result, [size_val, 45, 45, 1])
            return train_x_batch_image_format


        # A single training step
        def train_step(train_x_batch_image_format, train_y_batch_val):
            feed_dict = {
              cnn.image_matrix: train_x_batch_image_format,
              cnn.input_y: train_y_batch_val,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, score, prediction = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.scores, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, score {}".format(time_str, step, loss, accuracy, score))
            train_summary_writer.add_summary(summaries, step)


        # Evaluates model on a dev set
        def dev_step(x_batch, y_batch, writer=None):
            feed_dict = {
              cnn.image_matrix: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = mlp_data_helpers.batch_iter(
            zip(train_x_shuffled, train_y_shuffled), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            train_x_batch = []
            train_y_batch = []
            for i, j in enumerate(batch):
                train_x_batch.append(j[0].tolist())
                train_y_batch.append(j[1].tolist())

            train_x_batch_imageFormat = conv_text_to_image(train_x_batch, FLAGS.batch_size)
            train_step(train_x_batch_imageFormat, train_y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                print("\nMaking image format matrix...")
                test_x_shuffled_imageFormat = conv_text_to_image(test_x_shuffled, len(test_y_shuffled))
                print("\nImage format matrix is OK.:")
                dev_step(test_x_shuffled_imageFormat, test_y_shuffled, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))