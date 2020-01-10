# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE

from flip_gradient import flip_gradient
import tensorflow.contrib.slim as slim
from utils import *
import cPickle
import random
import matplotlib.pyplot as plt

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1


def getprotocol(Xd):
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
    X = np.vstack(X)

    np.random.seed(2018)
    n_examples = X.shape[0]
    n_train = n_examples * 0.7
    train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    X_train = X[train_idx]
    X_test =  X[test_idx]

    #产生１２３４５标签=====
    # Y_train = map(lambda x: mods.index(lbl[x][0]), train_idx)
    # Y_test = map(lambda x: mods.index(lbl[x][0]), test_idx)

    #====产生one-hot标签=====
    Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

    in_shp = list(X_train.shape[1:])
    classes = mods

    X_train = np.array(X_train).reshape((X_train.shape[0],2,X_train.shape[2],-1))
    X_test = np.array(X_test).reshape((X_test.shape[0],2,X_test.shape[2],-1))

    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]

    return classes,X_train[0:20000], X_test[0:20000], Y_train[0:20000] ,Y_test[0:20000]

Xs= cPickle.load(open("modulate/data10a0.dat",'rb'))
classes,Xs_train, Xs_test, Ys_train ,Ys_test = getprotocol(Xs)
#index相同
Xt= cPickle.load(open("modulate/data04c0.dat",'rb'))
classes,Xt_train, Xt_test, Yt_train ,Yt_test = getprotocol(Xt)

num_test = 10000
combined_test_imgs = np.vstack([Xs_test[:num_test], Xt_test[:num_test]])
combined_test_labels = np.vstack([Ys_test[:num_test], Yt_test[:num_test]])
combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
                                  np.tile([0., 1.], [num_test, 1])])
#==============================================================================

#画混淆矩阵
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#
def confusion(y_pre,y_true):
    length= y_pre.shape[1]
    nums = y_pre.shape[0]
    conf = np.zeros([length,length])
    confnorm = np.zeros([length,length])
    for i in range(0,nums):
        j = list(y_true[i,:]).index(1)
        k = int(np.argmax(y_pre[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,length):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plot_confusion_matrix(confnorm, labels=classes)


batch_size = 80

class MNISTModel(object):
    """Simple MNIST domain adaptation model."""

    def __init__(self):
        self._build_model()

    def _build_model(self):
        self.X = tf.placeholder(tf.float32, [None, 2, 128, 1])
        self.y = tf.placeholder(tf.float32, [None, 11])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])

        X_input = self.X

        channel_conv1 = 128
        channel_conv2 = 64
        channel_conv3 = 64
        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):

            W_conv0 = weight_variable([2, 5, 1, channel_conv1])  # [filter_height, filter_width, in_channels, out_channels]
            b_conv0 = bias_variable([channel_conv1])
            h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)#[1, height, width, 1]
            # h_conv0 = tf.nn.dropout(h_conv0, keep_prob=0.5)
            # h_pool0 = max_pool_2x2(h_conv0)

            W_conv1 = weight_variable([1, 5, channel_conv1, channel_conv2])
            b_conv1 = bias_variable([channel_conv2])
            h_conv1 = tf.nn.relu(conv2d(h_conv0, W_conv1) + b_conv1)
            # h_conv1 = tf.nn.dropout(h_conv1, keep_prob=0.5)

            # logits = RNN(X, weights, biases)
            # prediction = tf.nn.softmax(logits)

            W_conv2 = weight_variable([1, 3, channel_conv2, channel_conv3])
            b_conv2 = bias_variable([channel_conv3])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
            # h_conv2 = tf.nn.dropout(h_conv2, keep_prob=0.4)

            # h_pool1 = max_pool_2x2(h_conv1)
            print h_conv1
            # width =  h_conv2.shape[2]
            # The domain-invariant feature
            self.feature = tf.reshape(h_conv2, [-1, 118* channel_conv3])
            tmp = tf.contrib.layers.flatten(h_conv2)
            aa = int(tmp.shape[1])
            print aa
            aa = 118* channel_conv3
            print aa


        # with tf.variable_scope('feature_extractor'):
        #     with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):
        #         with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
        #                             activation_fn=tf.nn.relu):
        #             net0 = slim.conv2d(X_input, 256, [2, 5], scope='conv1')
        #             # net0 = slim.batch_norm(net0, scope='sgen_bn1')
        #             net1 = slim.conv2d(X_input, 128, [2, 7], scope='conv11')
        #             # net1 = slim.max_pool2d(net1, [1,2], stride=1, scope='pool1')
        #             net = slim.conv2d(net0, 128, [1,3], scope='conv2')
        #             # net = slim.batch_norm(net, scope='sgen_bn2')
        #             # net = slim.max_pool2d(net, [1,2], stride=1, scope='pool2')
        #             net = tf.concat([net1, net], 3)
        #             # slim.batch_norm
        #
        #             net = tf.contrib.layers.flatten(net)
        #             self.feature = tf.contrib.layers.flatten(net)
        #             tmp = tf.contrib.layers.flatten(net)
        #             aa = int(tmp.shape[1])



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

            W_fc2 = weight_variable([256, 11])
            b_fc2 = bias_variable([11])
            logits = tf.matmul(h_fc0, W_fc2) + b_fc2

            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)

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


# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    model = MNISTModel()

    learning_rate = tf.placeholder(tf.float32, [])

    pred_loss = tf.reduce_mean(model.pred_loss)
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss

    # regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
    # dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

    # loss收敛很快
    regular_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(pred_loss)
    dann_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(total_loss)


    # Evaluation
    truelabel = model.classify_labels
    prelabel = model.pred
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))



#输入数据。修改
def train_and_evaluate(training_mode, graph, model, num_steps=5000, verbose=True):
    """Helper to run the model with different training modes."""

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        # Batch generators
        gen_source_batch = batch_generator(
            [Xs_train, Ys_train], batch_size // 2)
        gen_target_batch = batch_generator(
            [Xt_train, Yt_train], batch_size // 2)
        gen_source_only_batch = batch_generator(
            [Xs_train, Ys_train], batch_size)
        gen_target_only_batch = batch_generator(
            [Xt_train, Yt_train], batch_size)

        domain_labels = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]),
                                   np.tile([0., 1.], [batch_size // 2, 1])])

        # Training loop
        # if training_mode == 'dann':
        #     num_steps = num_steps*10
        target=[]
        for i in range(num_steps):

            # Adaptation param and learning rate schedule as described in the paper
            gama=10    #搜索最好的gamma,alpha tradeoff
            alpha=10
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-gama * float(p))) - 1
            lr = 0.0008 / (1. + alpha * p) ** 0.75
            # lr = 0.001
            # Training step
            if training_mode == 'dann':

                X0, y0 = next(gen_source_batch)
                X1, y1 = next(gen_target_batch)
                X = np.vstack([X0, X1])
                y = np.vstack([y0, y1])

                _, batch_loss, dloss, ploss, d_acc, p_acc = sess.run(  # 返回值看run了几个
                    [dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc],
                    feed_dict={model.X: X, model.y: y, model.domain: domain_labels,
                               model.train: True, model.l: l, learning_rate: lr})
#打印输出，监测
                if verbose and i % 1000 == 0:
                    print('loss: {}  d_acc: {}  p_acc: {}  p: {}  l: {}  lr: {}'.format(
                        batch_loss, d_acc, p_acc, p, l, lr))
                    target_acc = 0
                    for target_test_images_batch, target_test_labels_batch in zip(
                            np.array_split(Xt_test, 2), np.array_split(Yt_test, 2)):
                        target_acc_tmp = sess.run(label_acc,
                                                  feed_dict={model.X: target_test_images_batch,
                                                             model.y: target_test_labels_batch,
                                                             model.train: False})
                        target_acc += target_acc_tmp / 2.
                    print target_acc
                    target.append(target_acc)

            elif training_mode == 'source':
                X, y = next(gen_source_only_batch)
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                         feed_dict={model.X: X, model.y: y, model.train: False,
                                                    model.l: l, learning_rate: lr})
                if verbose and i % 400 == 0:
                    print('loss: {}  p: {}  l: {}  lr: {}'.format(
                        batch_loss, p, l, lr))

            elif training_mode == 'target':
                X, y = next(gen_target_only_batch)
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                         feed_dict={model.X: X, model.y: y, model.train: False,
                                                    model.l: l, learning_rate: lr})
        train_writer = tf.summary.FileWriter('logs', sess.graph)


        # Compute final evaluation on test data
        # source_acc = sess.run(label_acc,
        #                       feed_dict={model.X: Xs_test, model.y: Ys_test,
        #                                  model.train: False})
        # truelabels,prelabels= sess.run(truelabel,prelabel,
        #                       feed_dict={model.X: Xs_test, model.y: Ys_test,
        #                                  model.train: False})

        source_acc = 0
        for test_images_batch, test_labels_batch in zip(
                np.array_split(Xs_test, 10), np.array_split(Ys_test, 10)):
            source_acc_tmp = sess.run(label_acc,
                                  feed_dict={model.X: test_images_batch, model.y: test_labels_batch,
                                             model.train: False})

            source_acc += source_acc_tmp / 10.

        prelabels = sess.run(model.pred,
                              feed_dict={model.X: Xt_test[::3], model.y: Yt_test[::3],
                                         model.train: False})
        confusion(prelabels,Yt_test[::3])
        plt.savefig("noadapta_10-04" + ".png")
        plt.close('all')

        target_acc = 0
        for target_test_images_batch, target_test_labels_batch in zip(
                np.array_split(Xt_test, 10), np.array_split(Yt_test, 10)):
            target_acc_tmp = sess.run(label_acc,
                                  feed_dict={model.X: target_test_images_batch, model.y: target_test_labels_batch,
                                             model.train: False})
            target_acc += target_acc_tmp / 10.

        prelabels_test = sess.run(model.pred,
                              feed_dict={model.X: Xt_test[::3], model.y: Yt_test[::3],
                                         model.train: False})
        confusion(prelabels_test,Yt_test[::3])
        # plt.show()
        plt.savefig("adapta_10-04" + ".png")
        plt.close('all')

        test_domain_acc =0
        for domain_test_images_batch, domain_test_labels_batch in zip(
                np.array_split(combined_test_imgs, 10), np.array_split(combined_test_domain, 10)):
            test_domain_acc_tmp = sess.run(domain_acc,
                                       feed_dict={model.X: domain_test_images_batch,
                                                  model.domain: domain_test_labels_batch, model.train: False,model.l: 1.0})
            test_domain_acc += test_domain_acc_tmp / 10.


    return source_acc, target_acc, test_domain_acc



print('\nSource only training')
source_acc, target_acc, _ = train_and_evaluate('source', graph, model,num_steps=15000)
print('Source (20M) accuracy:', source_acc)
print('Target (18M) accuracy:', target_acc)

print('\nDomain adaptation training')
source_acc, target_acc, d_acc = train_and_evaluate('dann', graph, model,num_steps=100000)
print('Source (20M) accuracy:', source_acc)
print('Target (18M) accuracy:', target_acc)
print('Domain accuracy:', d_acc)


