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
from flip_gradient import flip_gradient
from utils import *
import cPickle
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


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

    X_train = np.array(X_train).reshape((int(n_train), 2, 400, -1))
    X_test = np.array(X_test).reshape((int(n_examples - n_train), 2, 400, -1))
    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]

    return classes, X_train, X_test, Y_train ,Y_test

Xs= cPickle.load(open("/home/lab548/Downloads/protocol2/newdata/protocal20m_indoors_zd.dat",'rb'))
classes, Xs_train, Xs_test, Ys_train ,Ys_test = getprotocol(Xs)
#index相同
Xt= cPickle.load(open("/home/lab548/Downloads/protocol2/newdata/protocal20m_hallway.dat",'rb'))
classes, Xt_train, Xt_test, Yt_train ,Yt_test = getprotocol(Xt)

num_test = 5000
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




batch_size = 128


class MNISTModel(object):
    """Simple MNIST domain adaptation model."""

    def __init__(self):
        self._build_model()

    def AttentionLayer(self, inputs, name):
        hidden_size = 64
        # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,开始随机初始化
            u_context = tf.Variable(tf.truncated_normal([hidden_size * 2]), name='u_context')
            # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, hidden_size * 2, activation_fn=tf.nn.tanh)
            # shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            # reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output


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
        # CNN model for feature extraction
        # with tf.variable_scope('feature_extractor'):
        #
        #     W_conv0 = weight_variable([2, 5, 1, channel_conv1])  # [filter_height, filter_width, in_channels, out_channels]
        #     b_conv0 = bias_variable([channel_conv1])
        #     h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)#[1, height, width, 1]
        #     # h_conv0 = tf.nn.dropout(h_conv0, keep_prob=0.5)
        #     # h_pool0 = max_pool_2x2(h_conv0)
        #
        #     W_conv1 = weight_variable([1, 5, channel_conv1, channel_conv2])
        #     b_conv1 = bias_variable([channel_conv2])
        #     h_conv1 = tf.nn.relu(conv2d(h_conv0, W_conv1) + b_conv1)
        #     h_conv1 = tf.nn.dropout(h_conv1, keep_prob=0.5)
        #
        #     # logits = RNN(X, weights, biases)
        #     # prediction = tf.nn.softmax(logits)
        #
        #     W_conv2 = weight_variable([1, 3, channel_conv2, channel_conv3])
        #     b_conv2 = bias_variable([channel_conv3])
        #     h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
        #     h_conv2 = tf.nn.dropout(h_conv2, keep_prob=0.4)
        #
        #     # h_pool1 = max_pool_2x2(h_conv1)
        #     print h_conv2
        #     # The domain-invariant feature
        #     self.feature = tf.reshape(h_conv2, [-1, 390* channel_conv3])
        #     testsummaries = tf.Summary()

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

    # dann_train_op = tf.train.AdagradOptimizer(learning_rate = learning_rate).minimize(total_loss)
    # loss收敛很快
    regular_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(pred_loss)
    dann_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(total_loss)


    # Evaluation
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
            lr = 0.0006 / (1. + alpha * p) ** 0.75
            lr = 0.0006
            # Training step
            if training_mode == 'dann':
                # lr = 0.0005
                # gama = 10  # 搜索最好的gamma,alpha tradeoff
                # alpha = 10
                # p = float(i) / num_steps
                # l = 2. / (1. + np.exp(-gama * float(p))) - 1
                # lr = 0.001 / (1. + alpha * p) ** 0.75

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
                    # target_acc = sess.run(label_acc,
                    #                       feed_dict={model.X: Xt_test, model.y: Yt_test,
                    #                                  model.train: False})

                    target_acc = 0
                    for target_test_images_batch, target_test_labels_batch in zip(
                            np.array_split(Xt_test, 100), np.array_split(Yt_test, 100)):
                        target_acc_tmp = sess.run(label_acc,
                                                  feed_dict={model.X: target_test_images_batch,
                                                             model.y: target_test_labels_batch,
                                                             model.train: False})
                        target_acc += target_acc_tmp / 100.
                    print target_acc
                    target.append(target_acc)

            elif training_mode == 'source':
                X, y = next(gen_source_only_batch)
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                         feed_dict={model.X: X, model.y: y, model.train: False,
                                                    model.l: l, learning_rate: lr})
                if verbose and i % 100 == 0:
                    print('loss: {}  p: {}  l: {}  lr: {}'.format(
                        batch_loss, p, l, lr))

            elif training_mode == 'target':
                X, y = next(gen_target_only_batch)
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                         feed_dict={model.X: X, model.y: y, model.train: False,
                                                    model.l: l, learning_rate: lr})
        train_writer = tf.summary.FileWriter('logs', sess.graph)


        # Compute final evaluation on test data
        # if training_mode == 'dann':
        #     plt.plot(target)
        #     plt.show()
        source_acc = 0
        for test_images_batch, test_labels_batch in zip(
                np.array_split(Xs_test, 100), np.array_split(Ys_test, 100)):
            source_acc_tmp = sess.run(label_acc,
                                  feed_dict={model.X: test_images_batch, model.y: test_labels_batch,
                                             model.train: False})

            source_acc += source_acc_tmp / 100.

        prelabels = sess.run(model.pred,
                              feed_dict={model.X: Xt_test[::3], model.y: Yt_test[::3],
                                         model.train: False})
        confusion(prelabels,Yt_test[::3])
        plt.savefig("noadapta_in_cor" + str(p) + ".png")
        plt.close('all')

        target_acc = 0
        for target_test_images_batch, target_test_labels_batch in zip(
                np.array_split(Xt_test, 100), np.array_split(Yt_test, 100)):
            target_acc_tmp = sess.run(label_acc,
                                  feed_dict={model.X: target_test_images_batch, model.y: target_test_labels_batch,
                                             model.train: False})
            target_acc += target_acc_tmp / 100.

        prelabels_test = sess.run(model.pred,
                              feed_dict={model.X: Xt_test[::3], model.y: Yt_test[::3],
                                         model.train: False})
        confusion(prelabels_test,Yt_test[::3])
        # plt.show()
        plt.savefig("adapta_in_cor" + str(p) + ".png")
        plt.close('all')



        test_domain_acc =0
        for domain_test_images_batch, domain_test_labels_batch in zip(
                np.array_split(combined_test_imgs, 100), np.array_split(combined_test_domain, 100)):
            test_domain_acc_tmp = sess.run(domain_acc,
                                       feed_dict={model.X: domain_test_images_batch,
                                                  model.domain: domain_test_labels_batch, model.train: False,model.l: 1.0})
            test_domain_acc += test_domain_acc_tmp / 100.


        # test_emb = sess.run(model.feature, feed_dict={model.X: combined_test_imgs})

    return source_acc, target_acc, test_domain_acc


print('\nSource only training')
source_acc, target_acc, _ = train_and_evaluate('source', graph, model,num_steps=7000)
print('Source (20M) accuracy:', source_acc)
print('Target (18M) accuracy:', target_acc)
print('\nout-cor')
print('\nDomain adaptation training')
source_acc, target_acc, d_acc = train_and_evaluate('dann', graph, model,num_steps=80000)
print('Source (20M) accuracy:', source_acc)
print('Target (18M) accuracy:', target_acc)
print('Domain accuracy:', d_acc)

# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
# source_only_tsne = tsne.fit_transform(source_only_emb)
#
# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
# dann_tsne = tsne.fit_transform(dann_emb)
#
# plot_embedding(source_only_tsne, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'Source only')
# plot_embedding(dann_tsne, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'Domain Adaptation')

