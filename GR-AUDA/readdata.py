# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
from utils import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Process MNIST
mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

# Load MNIST-M
mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
mnistm_train = mnistm['train']
mnistm_test = mnistm['test']
mnistm_valid = mnistm['valid']

# Compute pixel mean for normalizing data
pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

# Create a mixed dataset for TSNE visualization
num_test = 500
combined_test_imgs = np.vstack([mnist_test[:num_test], mnistm_test[:num_test]])
combined_test_labels = np.vstack([mnist.test.labels[:num_test], mnist.test.labels[:num_test]])
combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
        np.tile([0., 1.], [num_test, 1])])
labels=mnist.train.labels
aa=mnist.train.images

bb=mnist_train[1,:,:,1]
cc=mnistm_train[1,:,:,1]

#import cPickle, random, sys
#
#
#Xd= cPickle.load(open("visual/protocal20m.dat",'rb'))
##==============================================================================
#snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
#
#X = []  
#lbl = []
#for mod in mods:
#    for snr in snrs:
#        X.append(Xd[(mod,snr)])
#        for i in range(Xd[(mod,snr)].shape[0]):  
#            lbl.append((mod,snr))
#X = np.vstack(X)
#
#np.random.seed(2018)
#n_examples = X.shape[0]
#n_train = n_examples * 0.5
#train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
#test_idx = list(set(range(0,n_examples))-set(train_idx))
#X_train = X[train_idx]
#X_test =  X[test_idx]
#
#def to_onehot(yy):
#    yy1 = np.zeros([len(yy), max(yy)+1])
#    yy1[np.arange(len(yy)),yy] = 1
#    return yy1
#Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
#Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
#in_shp = list(X_train.shape[1:])
#print X_train.shape, in_shp
#classes = mods
#X_train = np.array(X_train).reshape((10000,2,400,-1))
#X_test = np.array(X_test).reshape((10000,2,400,-1))