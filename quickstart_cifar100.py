import sys
sys.path.insert(0, "../Sknet")

import sknet
from sknet.optimize import Adam
from sknet.optimize.loss import *
from sknet.optimize import schedule
import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pylab as pl
import time
import tensorflow as tf
from sknet.dataset import BatchIterator
from sknet import ops,layers

import h5py


# Data Loading
#-------------
dataset = sknet.dataset.load_cifar10()

dataset.preprocess(sknet.dataset.Standardize,data="images",axis=[0])

dataset.create_placeholders(batch_size=64,
        iterators_dict={'train_set':BatchIterator("random_see_all"),
                        'valid_set':BatchIterator('continuous'),
                        'test_set':BatchIterator('continuous')},device="/cpu:0")

# Create Network
#---------------

# we use a batch_size of 64 and use the dataset.datum shape to
# obtain the shape of 1 observation and create the input shape

dnn       = sknet.network.Network(name='simple_model')

dnn.append(ops.RandomAxisReverse(dataset.images,axis=[-1]))
dnn.append(ops.RandomCrop(dnn[-1],(28,28)))
dnn.append(ops.GaussianNoise(dnn[-1],noise_type='additive',sigma=0.005))

dnn = sknet.network.Resnet(dnn,dataset.n_classes,D=2,W=2)
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


