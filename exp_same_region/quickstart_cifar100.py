import sys
sys.path.insert(0, "../Sknet")

import sknet
import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from sknet.dataset import BatchIterator
from sknet import ops

import argparse import parser
import h5py


# Data Loading
#-------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset to use", type=str)
parser.add_argument("--model", help="modeul to use 'cnn' or 'resnet'",type=str)
parser.add_argument("--data_augmentation", help="using data augmentation",
                                      type=str2bool)

args              = parser.parse_args()
DATASET           = args.dataset
MODEL             = args.model
DATA_AUGMENTATION = args.data_augmentation

if MODEL=='cifar10':
    dataset = sknet.dataset.load_cifar10()
elif MODEL=='svhn':
    dataset = sknet.dataset.load_svhn()
elif MODEL=='mnist':
    dataset = sknet.dataset.load_mnist()
elif MODEL=='cifar100':
    dataset = sknet.dataset.load_cifar100()

dataset.preprocess(sknet.dataset.Standardize,data="images",axis=[0])

dataset.create_placeholders(batch_size=64,
        iterators_dict={'train_set':BatchIterator("random_see_all"),
                       'valid_set':BatchIterator('continuous'),
                       'test_set':BatchIterator('continuous')},device="/cpu:0")

# Create Network
#---------------

dnn       = sknet.network.Network(name='simple_model')

if DATA_AUGMENTATION:
    dnn.append(ops.RandomAxisReverse(dataset.images,axis=[-1]))
    dnn.append(ops.RandomCrop(dnn[-1],(28,28)))

if MODEL=='resnet':
    sknet.networks.Resnet(dnn,dataset.n_classes,D=2,W=1)
else:
    sknet.networks.LargeConv(dnn,dataset.n_classes)

prediction = dnn[-1]





all_layers = [layer for layer in dnn[3:-1] if type(layer)==sknet.ops.Merge]
mus  = [tf.gradients(layer,dnn[2],layer)[0] for layer in all_layers]
loss = crossentropy_logits(p=dataset.labels,q=prediction)
accu = accuracy(dataset.labels,prediction)

B  = dataset.N('train_set')//64
lr = sknet.optimize.PiecewiseConstant(0.001,
                  {100*B:0.0005,200*B:0.0001,250*B:0.00005})
optimizer = Adam(loss,lr,params=dnn.params)
minimizer = tf.group(optimizer.updates+dnn.updates)

# Workers
#---------

min_worker  = sknet.Worker(op_name='minimizer',context='train_set',
        op=minimizer, instruction='execute every batch', deterministic=False)

loss_worker = sknet.Worker(op_name='loss',context='train_set',op= loss,
        instruction='save & print every 100 batch', deterministic=False)

accu_worker = sknet.Worker(op_name='accu',context='test_set', op=accu,
        instruction='execute every batch and save & print & average',
        deterministic=True, description='standard classification accuracy')



# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)

mus_train = list()
mus_test  = list()
for i in range(300):
    mus_train=[workplace.execute_op(mus,feed_dict=
                {dataset.images:dataset['images']['train_set'][64*i:64*(i+1)]},
                deterministic=True,batch_size=64) for i in range(4)]
    mus_test=[workplace.execute_op(mus, feed_dict=
                {dataset.images:dataset['images']['test_set'][64*i:64*(i+1)]},
                deterministic=True,batch_size=64) for i in range(4)]
    workplace.execute_worker(min_worker)
    workplace.execute_worker(accu_worker)
    print(accu_worker.data[-1])
    f = h5py.File('saved_mus.h5','a')
    saved_mus = np.asarray([mus_train,mus_test])
    f.create_dataset('mus_'+str(i),saved_mus.shape,compression='gzip')
    f['mus_'+str(i)][...] = saved_mus
    if i==0:
        images    = np.asarray([dataset['images']['train_set'][:64*4],
                dataset['images']['test_set'][:64*4]])
        f.create_dataset('images',images.shape,compression='gzip')
        f['images'][...] = images
    f.close()


