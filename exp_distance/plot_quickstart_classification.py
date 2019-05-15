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
import matplotlib.pyplot as plt

PATH = '/mnt/drive1/rbalSpace/distances/'

for DATASET in ['cifar10']:
    # Data Loading
    #-------------
    if DATASET=='mnist':
        dataset = sknet.dataset.load_mnist()
    elif DATASET=='cifar10':
        dataset = sknet.dataset.load_cifar10()
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
        filename = 'save_test_{}_{}.pkl'.format(DATASET,DATA_AUGMENTATION)
        FILENAME = PATH+filename
        if not os.path.isfile(FILENAME):
            continue
        f=open(FILENAME,'rb')
        distances_train,distances_test = pickle.load(f)
        f.close()
        L = len(distances_train['0'])
        print("L=",L)
        print("SHAPES=",[np.shape(d) for d in distances_train['0']])
        for epoch in ['0']:
            for l in range(1,L):
                distances_train[epoch][0]=np.minimum(distances_train[epoch][0],
                                            distances_train[epoch][l])
                distances_test[epoch][0]=np.minimum(distances_test[epoch][0],
                                            distances_test[epoch][l])

            print(np.min(distances_train[epoch][0]),
                    np.max(distances_train[epoch][0]),
                    np.std(distances_train[epoch][0]),
                    np.mean(distances_train[epoch][0]))
            print(np.min(distances_test[epoch][0]),
                    np.max(distances_test[epoch][0]),
                    np.std(distances_test[epoch][0]),
                    np.mean(distances_test[epoch][0]))

            plt.figure(figsize=(3,3))
            plt.hist(distances_train[epoch][0],20)
            plt.savefig('images/histogram_train_'+filename[:-4]+'.pdf')
            plt.close()

            plt.figure(figsize=(3,3))
            plt.hist(distances_test[epoch][0],20)
            plt.savefig('images/histogram_test_'+filename[:-4]+'.pdf')
            plt.close()

            plt.figure(figsize=(3,3))
            plt.hist(np.log(1e-8+distances_train[epoch][0]),20)
            plt.savefig('images/loghistogram_train_'+filename[:-4]+'.pdf')
            plt.close()

            plt.figure(figsize=(3,3))
            plt.hist(np.log(1e-8+distances_test[epoch][0]),20)
            plt.savefig('images/loghistogram_test_'+filename[:-4]+'.pdf')
            plt.close()



