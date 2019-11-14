import sys
sys.path.insert(0, "../../Sknet")

import sknet
import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from sknet.dataset import BatchIterator
from sknet import ops,layers

import h5py

MODEL = sys.argv[-1]


# Data Loading
#-------------
dataset = sknet.dataset.load_cifar10()

dataset['images/train_set'] -= dataset['images/train_set'].mean((1,2,3),
                                                                keepdims=True)
dataset['images/train_set'] /= dataset['images/train_set'].max((1,2,3),
                                                               keepdims=True)
dataset['images/test_set'] -= dataset['images/test_set'].mean((1,2,3),
                                                                keepdims=True)
dataset['images/test_set'] /= dataset['images/test_set'].max((1,2,3),
                                                               keepdims=True)


iterator = BatchIterator(64, {'train_set': 'random_see_all',
                         'test_set': 'continuous'})

dataset.create_placeholders(iterator, device="/cpu:0")

# Create Network
#---------------

# we use a batch_size of 64 and use the dataset.datum shape to
# obtain the shape of 1 observation and create the input shape

dnn = sknet.Network(name='simple_model')

dnn.append(ops.RandomAxisReverse(dataset.images, axis=[-1]))
dnn.append(ops.RandomCrop(dnn[-1], (28, 28)))

if MODEL == 'smallcnn':
    sknet.networks.ConvSmall(dnn, dataset.n_classes)
    all_layers = dnn[2:-1].as_list()
elif MODEL == 'largecnn':
    sknet.networks.ConvLarge(dnn, dataset.n_classes)
    all_layers = dnn[2:-1].as_list()
else:
    sknet.networks.Resnet(dnn, dataset.n_classes, D=2, W=2)
    all_layers = [layer for layer in dnn[2:-1] if type(layer)==sknet.ops.Merge]

prediction = dnn[-1]

mus = [tf.gradients(layer, dnn[1], layer)[0] for layer in all_layers]
loss = sknet.losses.crossentropy_logits(p=dataset.labels, q=prediction)
accu = sknet.losses.StreamingAccuracy(dataset.labels, prediction)

B = dataset.N('train_set')//64
lr = sknet.schedules.PiecewiseConstant(0.001, {100*B:0.0005,
                                      200*B:0.0001, 250*B:0.00005})
optimizer = sknet.optimizers.Adam(loss, dnn.variables(trainable=True), lr)
minimizer = tf.group(optimizer.updates + dnn.updates)

# Workers
#---------

min_worker = sknet.Worker(name='minimizer', context='train_set',
                          op=[minimizer, loss]+mus+[dataset.images],
                          deterministic=False, 
                          period=[1, 100]+[5000]*(len(mus)+1))

accu_worker = sknet.Worker(name='accu', context='test_set',
                           op=[accu]+mus+[dataset.images], deterministic=True,
                           period=[1]+[5]*(len(mus)+1))

PATH = '/mnt/drive1/rbalSpace/centroids/'
queue = sknet.Queue((min_worker, accu_worker),
                      filename= PATH+'saved_mus_'+MODEL+'.h5')


# Pipeline
#---------
workplace = sknet.Workplace(dnn, dataset=dataset)

workplace.execute_queue(queue, repeat = 150)


