# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.io

import sys
import os
import cPickle

import utils
import matplotlib.pyplot as plt
import random
Xs= cPickle.load(open("modulate/data10a0.dat",'rb'))
#index相同
Xt= cPickle.load(open("modulate/data04c0.dat",'rb'))
# Xt = Xs

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
def confusion(classes,y_pre,y_true):
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

class TrainOps(object):
    def __init__(self, model, batch_size=64, pretrain_epochs=40, train_feature_generator_iters=100001,
                 train_DIFA_iters=60001,
                 mnist_dir='./data/mnist', svhn_dir='./data/svhn', log_dir='./logs', model_save_path='./model',
                 pretrained_feature_extractor='feature_extractor', pretrained_feature_generator='feature_generator',
                 DIFA_feature_extractor='DIFA_feature_extractor'):

        self.model = model
        self.batch_size = batch_size
        # self.Xs = Xs
        # self.Xt = Xt
        # Number of iterations for Step 0, 1, 2.
        self.pretrain_epochs = pretrain_epochs
        self.train_feature_generator_iters = train_feature_generator_iters
        self.train_DIFA_iters = train_DIFA_iters

        # Dataset directory
        self.mnist_dir = mnist_dir
        self.svhn_dir = svhn_dir

        self.model_save_path = model_save_path

        self.pretrained_feature_extractor = os.path.join(self.model_save_path, pretrained_feature_extractor)
        self.pretrained_feature_generator = os.path.join(self.model_save_path, pretrained_feature_generator)
        self.DIFA_feature_extractor = os.path.join(self.model_save_path, DIFA_feature_extractor)

        self.log_dir = log_dir

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

#################################################################################################################
    def to_onehot(self,yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    def getprotocol(self,Xd):
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
        X = []
        lbl = []
        for mod in mods:
            for snr in snrs:
                X.append(Xd[(mod, snr)])
                for i in range(Xd[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))
        X = np.vstack(X)

        np.random.seed(2018)
        n_examples = X.shape[0]
        n_train = n_examples * 0.7
        train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
        test_idx = list(set(range(0, n_examples)) - set(train_idx))
        X_train = X[train_idx]
        X_test = X[test_idx]

        # 产生１２３４５标签=====
        Y_train = map(lambda x: mods.index(lbl[x][0]), train_idx)
        Y_test = map(lambda x: mods.index(lbl[x][0]), test_idx)

        # ====产生one-hot标签=====
        # Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
        # Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

        in_shp = list(X_train.shape[1:])
        classes = mods

        X_train = np.array(X_train).reshape((X_train.shape[0], 2, X_train.shape[2], -1))
        X_test = np.array(X_test).reshape((X_test.shape[0], 2, X_test.shape[2], -1))
        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)

        index = [i for i in range(len(X_test))]
        random.shuffle(index)
        X_test = X_test[index]
        Y_test = Y_test[index]

        return classes, X_train[0:20000], X_test[0:10000], Y_train[0:20000], Y_test[0:10000]
#设置test大小，避免内存问题
        # return classes, X_train[0:20000], X_test, Y_train[0:20000], Y_test

#################################################################################################################
    def train_feature_extractor(self):

        print 'Pretraining feature extractor.'

        # images, labels = self.load_svhn(self.svhn_dir, split='train')
        # test_images, test_labels = self.load_svhn(self.svhn_dir, split='test')
        classes,images, test_images, labels, test_labels = self.getprotocol(Xs)

        # build a graph
        model = self.model
        model.build_model()

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()

            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

            t = 0

            for i in range(self.pretrain_epochs):

                print 'Epoch', str(i)

                for start, end in zip(range(0, len(images), self.batch_size),
                                      range(self.batch_size, len(images), self.batch_size)):

                    t += 1

                    feed_dict = {model.images: images[start:end], model.labels: labels[start:end]}

                    sess.run(model.train_op, feed_dict)

                    if t % 100 == 0:
                        rand_idxs1 = np.random.permutation(images.shape[0])[:1000]

                        summary, loss, accuracy = sess.run(fetches=[model.summary_op, model.loss, model.accuracy],
                                                           feed_dict={model.images: images[rand_idxs1],
                                                                      model.labels: labels[rand_idxs1]})
#测试
                        rand_idxs = np.random.permutation(test_images.shape[0])[:1000]
                        summary, val_loss, val_accuracy = sess.run(fetches=[model.summary_op, model.loss, model.accuracy],
                                                           feed_dict={model.images: test_images[rand_idxs],
                                                                      model.labels: test_labels[rand_idxs]})
#prelabel
                        prelabels = sess.run(model.logits,
                                             feed_dict={model.images: test_images[rand_idxs],
                                                        model.labels: test_labels[rand_idxs]})
                        y_true = utils.one_hot(test_labels[rand_idxs], 11)
                        confusion(classes,prelabels, y_true)
                        # plt.show()

                        summary_writer.add_summary(summary, t)

                        print 'Step: [%d/%d] loss: [%.4f] accuracy: [%.4f]' % (
                        t + 1, self.pretrain_epochs * len(images) / self.batch_size, loss, accuracy)

                        print 'Step: [%d] loss: [%.4f] accuracy: [%.4f]' % (
                        t + 1,  val_loss, val_accuracy)

            print 'Saving'
            saver.save(sess, self.pretrained_feature_extractor)

    def train_feature_generator(self):

        print 'Training sampler.'

        # images, labels = self.load_svhn(self.svhn_dir, split='train')


        classes,images, Xs_test, labels, Ys_test = self.getprotocol(Xs)

        labels = utils.one_hot(labels, 11)

        # build a graph
        model = self.model
        model.build_model()

        noise_dim = 100
        epochs = 5000

        with tf.Session(config=self.config) as sess:

            # initialize variables
            tf.global_variables_initializer().run()

            # restore feature extractor trained on Step 0
            print ('Loading pretrained feature extractor.')
            variables_to_restore = slim.get_model_variables(scope='feature_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_feature_extractor)
            print 'Loaded'

            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            for step in range(self.train_feature_generator_iters):

                i = step % int(images.shape[0] / self.batch_size)

                images_batch = images[i * self.batch_size:(i + 1) * self.batch_size]
                labels_batch = labels[i * self.batch_size:(i + 1) * self.batch_size]
                # noise = utils.sample_Z(self.batch_size, noise_dim, 'uniform')
                noise = utils.sample_Z(self.batch_size, noise_dim, 'gaussian')

                feed_dict = {model.noise: noise, model.images: images_batch, model.labels: labels_batch}

                avg_D_fake = sess.run(model.logits_fake, feed_dict)
                avg_D_real = sess.run(model.logits_real, feed_dict)

                sess.run(model.d_train_op, feed_dict)
                sess.run(model.g_train_op, feed_dict)

                if (step + 1) % 1000 == 0:
                    summary, dl, gl = sess.run([model.summary_op, model.d_loss, model.g_loss], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] d_loss: %.6f g_loss: %.6f avg_d_fake: %.2f avg_d_real: %.2f ' \
                           % (
                           step + 1, self.train_feature_generator_iters, dl, gl, avg_D_fake.mean(), avg_D_real.mean()))

            print 'Saving.'
            saver.save(sess, self.pretrained_feature_generator)

    def train_DIFA(self):

        print 'Adapt with DIFA'

        # build a graph
        model = self.model
        model.build_model()

        # source_images, source_labels = self.load_svhn(self.svhn_dir, split='train')
        # target_images, _ = self.load_mnist(self.mnist_dir, split='train')    #无需训练label
        # target_test_images, target_test_labels = self.load_mnist(self.mnist_dir, split='test')

###########协议数据
        # Xs = cPickle.load(open("protocal18m.dat", 'rb'))
        classes_s,source_images, Xs_test, source_labels, Ys_test = self.getprotocol(Xs)

        classes_t,target_images, target_test_images, Yt_train, target_test_labels = self.getprotocol(Xt)


        with tf.Session(config=self.config) as sess:
            # Initialize weights
            tf.global_variables_initializer().run()

            print ('Loading pretrained encoder.')
            variables_to_restore = slim.get_model_variables(scope='feature_extractor')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_feature_extractor)

            print ('Loading pretrained S.')
            variables_to_restore = slim.get_model_variables(scope='feature_generator')
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, self.pretrained_feature_generator)

            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
            saver = tf.train.Saver()

            print ('Start training.')

#先评估目标域
            print 'Evaluating.'
            target_test_acc = 0.
            source_test_acc = 0.
            # 评估源域数据
            # for source_test_labels_batch, source_test_images_batch in zip(
            #         np.array_split(Ys_test, 5), np.array_split(Xs_test, 5)):
            #     source_test_acc_tmp = sess.run(model.trg_accuracy,
            #                                    feed_dict={model.trg_images: source_test_images_batch,
            #                                               model.trg_labels_gt: source_test_labels_batch})
            #     source_test_acc += source_test_acc_tmp / 5.

            source_test_acc = sess.run(model.trg_accuracy,
                                       feed_dict={model.trg_images: Xs_test[::4], model.trg_labels_gt: Ys_test[::4]})

            print 'source test accuracy: [%.3f]' % (source_test_acc)

            # for target_test_labels_batch, target_test_images_batch in zip(
            #         np.array_split(target_test_labels, 5), np.array_split(target_test_images, 5)):
            #     target_test_acc_tmp = sess.run(model.trg_accuracy,
            #                                    feed_dict={model.trg_images: target_test_images_batch,
            #                                               model.trg_labels_gt: target_test_labels_batch})
            #     target_test_acc += target_test_acc_tmp / 5.

            target_test_acc = sess.run(model.trg_accuracy, feed_dict={model.trg_images: target_test_images[::3],
                                                                      model.trg_labels_gt: target_test_labels[::3]})
            print 'target test accuracy: [%.3f]' % (target_test_acc)


#画混淆矩阵
            pred_t = sess.run(model.trg_labels, feed_dict={model.trg_images: target_test_images[1000:3000],
                                                           model.trg_labels_gt: target_test_labels[1000:3000]})
            y_true = utils.one_hot(target_test_labels[1000:3000], 11)
            confusion(classes_t, pred_t, y_true)
            plt.savefig("noadapta.png")
            plt.close('all')
            # plt.show()


            noise_dim = 100
            p=0
            test_acc = []
            for step in range(self.train_DIFA_iters):

                i = step % int(source_images.shape[0] / self.batch_size)
                j = step % int(target_images.shape[0] / self.batch_size)

                source_images_batch = source_images[i * self.batch_size:(i + 1) * self.batch_size]
                target_images_batch = target_images[j * self.batch_size:(j + 1) * self.batch_size]
                labels_batch = utils.one_hot(source_labels[i * self.batch_size:(i + 1) * self.batch_size], 11)
                # noise = utils.sample_Z(self.batch_size, noise_dim, 'uniform')
                noise = utils.sample_Z(self.batch_size, noise_dim, 'gaussian')

                feed_dict = {model.src_images: source_images_batch, model.trg_images: target_images_batch,
                             model.noise: noise, model.labels: labels_batch}

                sess.run(model.e_train_op, feed_dict)
                sess.run(model.d_train_op, feed_dict)

                logits_real, logits_fake = sess.run([model.logits_real, model.logits_fake], feed_dict)


                if (step + 1) % 800 == 0:

                    summary, e_loss, d_loss = sess.run([model.summary_op, model.e_loss, model.d_loss], feed_dict)
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] e_loss: [%.6f] d_loss: [%.6f] e_real: [%.2f] e_fake: [%.2f]' \
                           % (step + 1, self.train_DIFA_iters, e_loss, d_loss, logits_real.mean(), logits_fake.mean()))

                    print 'Evaluating.'
                    target_test_acc = 0.

                    # for target_test_labels_batch, target_test_images_batch in zip(
                    #         np.array_split(target_test_labels, 10), np.array_split(target_test_images, 10)):
                    #     feed_dict[self.model.trg_images] = target_test_images_batch
                    #     feed_dict[self.model.trg_labels_gt] = target_test_labels_batch
                    #     target_test_acc_tmp = sess.run(model.trg_accuracy, feed_dict)
                    #     target_test_acc += target_test_acc_tmp / 10.

                    target_test_acc = sess.run(model.trg_accuracy, feed_dict={model.trg_images: target_test_images[::3],
                                                                              model.trg_labels_gt: target_test_labels[::3]})

                    source_test_acc = sess.run(model.trg_accuracy,
                                               feed_dict={model.trg_images: Xs_test[::4],
                                                          model.trg_labels_gt: Ys_test[::4]})
                    print 'source test accuracy: [%.3f]' % (source_test_acc)

                    pred_t = sess.run(model.trg_labels, feed_dict={model.trg_images: target_test_images[1000:3000],
                                                                   model.trg_labels_gt: target_test_labels[1000:3000]})
                    y_true = utils.one_hot(target_test_labels[1000:3000], 11)
                    confusion(classes_t, pred_t, y_true)
                    p=p+1
                    plt.savefig("pics/adapta"+str(p)+".png")
                    plt.close('all')
                    print 'target test accuracy: [%.3f]' % (target_test_acc)
                    test_acc.append(target_test_acc)

            np.savetxt("resultacc.txt", np.array(test_acc))

            print 'Saving.'
            saver.save(sess, self.DIFA_feature_extractor)
