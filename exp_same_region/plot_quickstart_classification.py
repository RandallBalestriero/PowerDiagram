import sys
sys.path.insert(0, "../../Sknet/")

import sknet
import os
import pickle
# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from sknet.dataset import BatchIterator
from sknet import ops,layers
import argparse


PATH = '/mnt/drive1/rbalSpace/regions/'
for DATASET in ['mnist']:
    # Data Loading
    #-------------
    if DATASET=='mnist':
        dataset = sknet.dataset.load_mnist()
    elif DATASET=='fashionmnist':
        dataset = sknet.dataset.load_fashonmnist()
    elif DATASET=='cifar10':
        dataset = sknet.dataset.load_cifar10()
    elif DATASET=='cifar100':
        dataset = sknet.dataset.load_cifar100()
    elif DATASET=='svhn':
        dataset = sknet.dataset.load_svhn()

    if "valid_set" not in dataset.sets:
        dataset.split_set("train_set","valid_set",0.15)
    standardize = sknet.dataset.Standardize().fit(dataset['images/train_set'])
    dataset['images/train_set'] = \
                            standardize.transform(dataset['images/train_set'])
    dataset['images/test_set'] = \
                            standardize.transform(dataset['images/test_set'])
    dataset['images/valid_set'] = \
                            standardize.transform(dataset['images/valid_set'])

    for DATA_AUGMENTATION in ['True','False']:
        for MODEL in ['cnn','resnet']:
            filename = PATH+'save_test_{}_{}_{}.pkl'.format(MODEL,
                                      DATASET,DATA_AUGMENTATION)
            print(filename)
            if not os.path.isfile(filename):
                continue
            f=open(filename,'rb')
            vqs_train,vqs_test=pickle.load(f)
            f.close()
            L = len(vqs_train['0'])
            print("L=",L)
            print("SHAPES=",[np.shape(vq) for vq in vqs_train['0']])
            for epoch in ['0']:
                for l in range(L):
                    print(epoch,l,len(np.unique(vqs_train[epoch][l])))



