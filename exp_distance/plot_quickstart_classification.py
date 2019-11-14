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

plt.style.use('ggplot')
import matplotlib as mpl
label_size = 18
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size                                    

PATH = '/mnt/drive1/rbalSpace/distances/'
BINS = 80
EPOCHS = ['0','20','40','60','80']

for DATASET in ['svhn','cifar10']:
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
        filename = 'save_largedense_test_v2_{}_{}.pkl'.format(DATASET,DATA_AUGMENTATION)
        FILENAME = PATH+filename
        if not os.path.isfile(FILENAME):
            continue
        f=open(FILENAME,'rb')
        distances_train,distances_test,accus = pickle.load(f)
        f.close()
        L = len(distances_train['0'])
        print("L=",L)
        print("SHAPES=",[np.shape(d) for d in distances_train['0']])
        for epoch in EPOCHS:
            for l in range(1,L):
                print(epoch,l)
                distances_train[epoch][l]=np.minimum(distances_train[epoch][l-1],
                                            distances_train[epoch][l])
                distances_test[epoch][l]=np.minimum(distances_test[epoch][l-1],
                                            distances_test[epoch][l])

        # evolution of the distance at fixed epoch and moving layers
        PAS = 100
        COLOR = 'blue'
        ALPHA = 1
        STEP = None
        BINS = 40
        LAYERS = np.asarray(range(L)).astype('str')
        LAYERS2 = np.asarray(range(L))*PAS

        name = 'images/histogram_layers_train_'+filename[:-4]\
                                            +'_e{}.pdf'

        def do_plot(train_list, test_list, log, name, ytickslabels):
            plt.figure(figsize=(6,3))

            cmap = plt.get_cmap('Blues')
            COLOR = lambda t:cmap(0.25+0.75*t/(len(train_list)-1))
            for i,train in enumerate(train_list):
                if log:
                    values, bins = np.histogram(np.log(1e-7+train), BINS)
                else:
                    values, bins = np.histogram(train, BINS)
                plt.fill_between(bins[1:], PAS*i+values, PAS*i, step=STEP,
                        color=COLOR(i), alpha=ALPHA, zorder=20-i,
                        edgecolor='w', lw=0.2)

            cmap = plt.get_cmap('Reds')
            COLOR = lambda t:cmap(0.25+0.75*t/(len(test_list)-1))
            OFFSET = (4+len(train_list))*PAS
            for i,test in enumerate(test_list):
                if log:
                    values,bins = np.histogram(np.log(1e-7+test),BINS)
                else:
                    values,bins = np.histogram(test,BINS)
                plt.fill_between(bins[1:],OFFSET+PAS*i+values*2.5,OFFSET+PAS*i,
                        step=STEP, color=COLOR(i), alpha=ALPHA, zorder=10-i,
                        edgecolor='w', lw=0.2)

            ylim = plt.gca().get_ylim()
            plt.ylim([-20,ylim[1]])
            plt.xlim([-18,-2])

            ticks = np.asarray(range(len(train_list)))*PAS
            ticks = np.concatenate([ticks,ticks+OFFSET])
            plt.yticks(ticks, ytickslabel+ytickslabel)
            plt.tight_layout()
            plt.savefig(name)
            plt.close()


        ytickslabel = ['1']+['' for i in range(1,L-1)]+[str(L)]
        for epoch in EPOCHS:

            train_list = distances_train[epoch]
            test_list = distances_test[epoch]

            name = 'images/loghistogram_layers_'+filename[:-4]\
                                                +'_e{}.pdf'.format(epoch)
            do_plot(train_list, test_list, True, name, ytickslabel)

            name = 'images/histogram_layers_'+filename[:-4]\
                                                +'_e{}.pdf'.format(epoch)
            do_plot(train_list, test_list, False, name, ytickslabel)

        ytickslabel = [EPOCHS[0]]+['' for i in range(len(EPOCHS)-2)]\
                        +[EPOCHS[-1]]
        ytickslabels = EPOCHS
        for l in range(L):

            train_list = [distances_train[e][l] for e in ytickslabels]
            test_list = [distances_test[e][l] for e in ytickslabels]

            name = 'images/loghistogram_epochs_'+filename[:-4]\
                                                +'_l{}.pdf'.format(l)
            do_plot(train_list, test_list, True, name, ytickslabel)

            name = 'images/histogram_epochs_'+filename[:-4]\
                                                +'_l{}.pdf'.format(l)
            do_plot(train_list, test_list, False, name, ytickslabel)

# evolution of the distance at fixed layer and moving epochs


#            print(np.shape(distances_train[epoch][0]))
#            print(np.min(distances_train[epoch][0]),
#                    np.max(distances_train[epoch][0]),
#                    np.std(distances_train[epoch][0]),
#                    np.mean(distances_train[epoch][0]))
#            print(np.min(distances_test[epoch][0]),
#                    np.max(distances_test[epoch][0]),
#                    np.std(distances_test[epoch][0]),
#                    np.mean(distances_test[epoch][0]))


