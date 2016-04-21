import tensorflow as tf
import numpy as np


class NewTextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, slide_window_size, filter_size, batch_size, num_filters):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, slide_window_size*2], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.image_matrix = tf.placeholder(tf.float32, [None, sequence_length-filter_size+1,
                                                        sequence_length-filter_size+1, 1], name="image_matrix")

        # Keeping track of l2 regularization loss (optional)
        regularizers = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding_matrix = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                                trainable=False, name="embedding_matrix")
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        # pooled_outputs = []
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # 1D-Convolution
            w_one_dim = tf.Variable(tf.truncated_normal([slide_window_size*2, embedding_size, 1, 1], stddev=0.1),
                                    name="w_one_dim")
            b_oneDim = tf.Variable(tf.constant(0.1, shape=[1]), name="b_oneDim")
            self.conv_oneDim = tf.nn.conv2d(
                self.embedded_chars_expanded,
                w_one_dim,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv_oneDim")
            # Apply nonlinearity
            self.relu_rs = tf.nn.relu(tf.nn.bias_add(self.conv_oneDim, b_oneDim), name="relu_rs")

            # Maxpooling Layer 1
            image_matrix_pooled = tf.nn.max_pool(
                self.image_matrix,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name="image_matrix_pooled")

            # Convolution Layer 2
            w_conv1 = tf.Variable(tf.truncated_normal([slide_window_size, slide_window_size, 1, 32], stddev=0.1),
                                  name="w_conv1")
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv1")
            conv1 = tf.nn.conv2d(
                image_matrix_pooled,
                w_conv1,
                strides=[1, 1, 1, 1],
                padding="SAME",
                name="conv1")

            # Apply nonlinearity
            h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1), name="relu1")

            # Maxpooling over the outputs
            h_pool1 = tf.nn.max_pool(
                h_conv1,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name="h_pool1")

            # Convolution Layer 3
            w_conv2 = tf.Variable(tf.truncated_normal([slide_window_size, slide_window_size, 32, 64], stddev=0.1),
                                  name="w_conv2")
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name="b_conv2")
            conv2 = tf.nn.conv2d(
                h_pool1,
                w_conv2,
                strides=[1, 1, 1, 1],
                padding="SAME",
                name="conv2")

            # Apply nonlinearity
            h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2), name="relu2")

            # Maxpooling over the outputs
            h_pool2 = tf.nn.max_pool(
                h_conv2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name="h_pool2")
            # pooled_outputs.append(pooled)

        # densely connected layer
        with tf.name_scope("densely_connected"):
            w_fc = tf.Variable(tf.truncated_normal([6*6*64, 256], stddev=0.1), name="w_fc")
            b_fc = tf.Variable(tf.constant(0.1, shape=[256]), name="b_fc")
            regularizers += tf.nn.l2_loss(w_fc)
            regularizers += tf.nn.l2_loss(b_fc)
            h_pool2_flat = tf.reshape(h_pool2, [-1, 6*6*64], name="h_pool2_flat")
            self.h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc) + b_fc, name="h_fc")

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_fc, self.dropout_keep_prob, name="dropout")

        # MLP_Layer
        with tf.name_scope("MLP_Layer"):
            w_mlp = tf.Variable(tf.truncated_normal([256, 512], stddev=0.1), name="w_mlp")
            b_mlp = tf.Variable(tf.constant(0.0, shape=[512]), name="b_mlp")
            regularizers += tf.nn.l2_loss(w_mlp)
            regularizers += tf.nn.l2_loss(b_mlp)
            # self.mlp_rs = tf.nn.relu(tf.nn.xw_plus_b(self.h_drop, w_mlp, b_mlp, name="mlp_rs"))
            self.mlp_rs = tf.nn.relu(tf.matmul(self.h_drop, w_mlp) + b_mlp, name="mlp_rs")

        # Add dropout2
        with tf.name_scope("dropout2"):
            self.h_drop2 = tf.nn.dropout(self.mlp_rs, self.dropout_keep_prob, name="dropout2")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # w_output = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.1), name="w_output")
            w_output = tf.get_variable("w_output", shape=[512, 1], initializer=tf.contrib.layers.xavier_initializer())
            b_output = tf.Variable(tf.constant(0.0, shape=[1]), name="b_output")
            regularizers += tf.nn.l2_loss(w_output)
            regularizers += tf.nn.l2_loss(b_output)
            # self.scores = tf.nn.xw_plus_b(self.h_drop2, w_output, b_output, name="scores")
            self.scores = tf.nn.relu(tf.matmul(self.h_drop2, w_output) + b_output, name="scores")
            # self.predictions = tf.sigmoid(self.scores, name="predictions")
            self.predictions = tf.abs(self.scores, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("Loss"):
            # self.loss = tf.nn.l2_loss(self.scores - self.input_y) + 5e-4 * regularizers
            self.loss = tf.nn.l2_loss(self.scores - self.input_y)
            # losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            # self.loss = tf.reduce_mean(losses) + 5e-4 * regularizers

        # Accuracy
        with tf.name_scope("Accuracy"):
            # correct_predictions = tf.equal(self.predictions, self.input_y)
            # self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            zero_list = np.zeros(batch_size, dtype=np.float32)
            zero_tensor = tf.convert_to_tensor(zero_list)
            neg = np.zeros(batch_size, dtype=np.float32)
            neg_tensor = tf.convert_to_tensor(neg)
            pos = np.full((batch_size,), 1.0, dtype=np.float32)
            pos_tensor = tf.convert_to_tensor(pos)
            neg_dis = tf.abs(tf.sub(self.predictions, neg_tensor))
            pos_dis = tf.abs(tf.sub(self.predictions, pos_tensor))
            prediction = tf.cast(tf.greater(tf.sub(neg_dis, pos_dis), zero_tensor), "float")
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, self.input_y), "float"), name="accuracy")
