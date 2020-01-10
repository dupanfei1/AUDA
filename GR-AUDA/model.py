# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from tensorflow.contrib import layers
# from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE
import tensorflow.contrib.slim as slim
import random
import cPickle
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


batch_size = 128


class MNISTModel(object):
    """Simple MNIST domain adaptation model."""

    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X = tf.placeholder(tf.float32, [None, 2, 400, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])
        self.is_training = tf.placeholder(tf.float32, [])

        # X_input = (tf.cast(self.X, tf.float32) - pixel_mean) / 255.
        X_input = self.X
        is_training = self.train
        channel_conv1 = 128
        channel_conv2 = 64
        channel_conv3 = 64
        aa = 390* channel_conv3
#resnet

        # with tf.variable_scope('feature_extractor'):
        #     with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
        #         with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
        #                             activation_fn=tf.nn.relu, is_training=is_training):
        #             # a1 = tf.constant(0.7)
        #             # b1 = tf.constant(0.3)
        #             net00 = slim.conv2d(X_input, 128, [2, 7], scope='conv1',padding='VALID')
        #             # net00 = slim.batch_norm(net00, scope='sgen_bn1')
        #             net0 = slim.conv2d(net00, 128, [1, 5], scope='conv12')
        #             net0 = slim.conv2d(net0, 128, [1, 3], scope='conv13')
        #             # net0 = slim.batch_norm(net0, scope='sgen_bn2')
        #             # netup = tf.multiply(net00, b1) + tf.multiply(net0, a1)
        #             netup = net00 + net0
        #             netup = slim.conv2d(netup, 96, [1, 5], scope='convup')
        #
        #             self.feature = tf.contrib.layers.flatten(netup)
        #             tmp = tf.contrib.layers.flatten(netup)
        #             aa = int(tmp.shape[1])

# x2 = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
# y2 = tf.constant(2.0)
# # 注意这里这里x1,y1要有相同的数据类型，不然就会因为数据类型不匹配而出错
# z2 = tf.multiply(x2, y2)



        # with tf.variable_scope('feature_extractor'):
        #     with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu):
        #         net00 = slim.conv2d(X_input, 128, [2, 7], scope='conv1', padding='VALID')
        #         net0 = slim.conv2d(net00, 128, [1, 3], scope='conv12')
        #         net0 = slim.conv2d(net0, 128, [1, 3], scope='conv13')
        #
        #         net11 = slim.conv2d(X_input, 128, [2, 5], scope='conv2', padding='VALID')
        #         net11 = slim.conv2d(net11, 64, [1, 3], scope='conv20', padding='VALID')
        #         net1 = slim.conv2d(net11, 64, [1, 5], scope='conv21')
        #         net1 = slim.conv2d(net1, 64, [1, 3], scope='conv22')
        #         netup = net00 + net0
        #         netdown = net11 + net1
        #         net = tf.concat([netup, netdown], 3)
        #
        #         self.feature = tf.contrib.layers.flatten(net)


# 双通道
        with tf.variable_scope('feature_extractor'):
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.relu):
                    net0 = slim.conv2d(X_input, 256, [2, 5], scope='conv1')
                    # net0 = slim.batch_norm(net0, scope='sgen_bn1')
                    net1 = slim.conv2d(X_input, 128, [2, 7], scope='conv11')
                    # net1 = slim.max_pool2d(net1, [1,2], stride=1, scope='pool1')
                    net = slim.conv2d(net0, 128, [1,3], scope='conv2')
                    # net = slim.batch_norm(net, scope='sgen_bn2')
                    # net = slim.max_pool2d(net, [1,2], stride=1, scope='pool2')
                    net = tf.concat([net1, net], 3)
                    # slim.batch_norm

                    net = tf.contrib.layers.flatten(net)
                    self.feature = tf.contrib.layers.flatten(net)
                    tmp = tf.contrib.layers.flatten(net)
                    aa = int(tmp.shape[1])

#deep 三通道74%
        # with tf.variable_scope('feature_extractor'):
        #     with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='SAME'):
        #         net0 = slim.conv2d(X_input, 256, [2, 7], scope='conv1')
        #         net0 = tf.nn.dropout(net0, 0.4)
        #         net1 = slim.conv2d(X_input, 256, [2, 5], scope='conv2')
        #         net2 = slim.conv2d(X_input, 128, [2, 3], scope='conv3')
        #         net0 = slim.conv2d(net0, 128, [1, 5], scope='conv12')
        #         net1 = slim.conv2d(net1, 128, [1, 5], scope='conv13')
        #         net1 = slim.conv2d(net1, 128, [1, 3], scope='conv22')
        #         # net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
        #         net = tf.concat([net2,net1, net0], 3)
        #
        #         self.feature = tf.contrib.layers.flatten(net)
        #         aa = self.feature.shape[1]

        # MLP for class prediction
        with tf.variable_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0],
                                               [batch_size // 2, -1])  # size[i]=-1 表示第i维从begin[i]剩余的元素都要被抽取
            classify_feats = tf.cond(self.train, source_features, all_features)

            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size // 2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)

            W_fc0 = weight_variable([aa, 256])
            b_fc0 = bias_variable([256])
            h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)

            # W_fc1 = weight_variable([100, 100])
            # b_fc1 = bias_variable([100])
            # h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

            W_fc2 = weight_variable([256, 10])
            b_fc2 = bias_variable([10])
            logits = tf.matmul(h_fc0, W_fc2) + b_fc2

            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)
            # self.pred_loss = tf.reduce_mean(tf.square(logits - self.classify_labels))

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l)
            # feat = self.feature
            d_W_fc0 = weight_variable([aa, 512])
            d_b_fc0 = bias_variable([512])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_W_fc1 = weight_variable([512, 100])
            d_b_fc1 = bias_variable([100])
            d_h_fc1 = tf.nn.relu(tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1)

            d_W_fc2 = weight_variable([100, 2])
            d_b_fc2 = bias_variable([2])
            d_logits = tf.matmul(d_h_fc1, d_W_fc2) + d_b_fc2

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)

