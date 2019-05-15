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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help="the dataset to train on, can be"\
           +"mnist, fasionmnist, cifar10, ...",type=str, default='mnist')
parser.add_argument("--data_augmentation", help="using data augmentation",
                     type=sknet.utils.str2bool,default='0')

args              = parser.parse_args()
DATASET           = args.dataset
DATA_AUGMENTATION = args.data_augmentation

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

dataset.create_placeholders(batch_size=32,
        iterators_dict={'train_set':BatchIterator("random_see_all"),
                        'valid_set':BatchIterator('continuous'),
                        'test_set':BatchIterator('continuous')},device="/cpu:0")

# Utility function
#-----------------

#c_p = tf.placeholder(tf.int32)
i_p = tf.placeholder(tf.int32)
j_p = tf.placeholder(tf.int32)

def get_distance(input,tensor):
    def doit(c):
        gradient = tf.gradients(tensor[:,c,i_p,j_p],input)[0]
        norm     = tf.sqrt(tf.reduce_sum(tf.square(gradient),[1,2,3]))
        return tf.abs(tensor[:,c,i_p,j_p])/norm
    distances = tf.map_fn(doit,tf.range(tensor.shape.as_list()[1]),
                                        dtype=tf.float32)
    return tf.reduce_min(distances,0)



# Create Network
#---------------

dnn       = sknet.Network(name='simple_model')

if DATA_AUGMENTATION:
    dnn.append(ops.RandomAxisReverse(dataset.images,axis=[-1]))
    dnn.append(ops.RandomCrop(dnn[-1],(28,28),seed=10))
    start_op = 2
else:
    dnn.append(dataset.images)
    start_op = 1

sknet.networks.ConvSmall(dnn,dataset.n_classes)

prediction = dnn[-1]
if DATA_AUGMENTATION:
    distances  = [get_distance(dnn[1],op.inner_ops[1]) for op in dnn[2:-1]]
else:
    distances  = [get_distance(dnn[0],op.inner_ops[1]) for op in dnn[1:-1]]
loss       = sknet.losses.crossentropy_logits(p=dataset.labels,q=prediction)
accu       = sknet.losses.accuracy(dataset.labels,prediction)

B         = dataset.N('train_set')//32
lr        = sknet.schedules.PiecewiseConstant(0.002,
                                    {100*B:0.002,200*B:0.001,250*B:0.0005})
optimizer = sknet.optimizers.Adam(loss,lr,params=dnn.variables(trainable=True))
minimizer = tf.group(optimizer.updates+dnn.updates)

# Pipeline
#---------
workplace       = sknet.utils.Workplace(dnn,dataset=dataset)
feed_dict       = dict()
accuracies      = list()
distances_train = dict()
distances_test  = dict()

for epoch in range(150):
    distances_train[str(epoch)] = [[] for i in range(len(distances))]
    distances_test[str(epoch)]  = [[] for i in range(len(distances))]

    #--Train Set--
    dataset.set_set('train_set',session=workplace.session)
    feed_dict.update(dnn.deter_dict(True))

    # VQs
    print('distance train set')
    if epoch%20==0:
        for batch in range(300):
            dataset.next(session=workplace.session)
            for LAYER in range(len(distances)):
                ii,jj=dnn[start_op+LAYER].shape.as_list()[2:]
                feed_dict.update({i_p:0,j_p:0})
                mini_distances=workplace.session.run(distances[LAYER],feed_dict)
                for i in range(ii):
                    for j in range(jj):
                        feed_dict.update({i_p:i,j_p:j})
                        mini_distances = np.minimum(mini_distances,
                             workplace.session.run(distances[LAYER],feed_dict))
                distances_train[str(epoch)][LAYER].append(mini_distances)
        for i in range(len(distances)):
             distances_train[str(epoch)][i]=np.concatenate(
                                                distances_train[str(epoch)][i])
        dataset.reset()
    # Training
    print('training')
    feed_dict.update(dnn.deter_dict(False))
    while dataset.next(session=workplace.session):
        workplace.session.run(minimizer, feed_dict=feed_dict)

    #--Test Set--
    dataset.set_set('test_set',session=workplace.session)
    feed_dict.update(dnn.deter_dict(True))

    # VQs
    print('distances test')
    if epoch%20==0:
        for batch in range(132):
            dataset.next(session=workplace.session)
            for LAYER in range(len(distances)):
                ii,jj=dnn[start_op+LAYER].shape.as_list()[2:]
                feed_dict.update({i_p:0,j_p:0})
                mini_distances=workplace.session.run(distances[LAYER],feed_dict)
                for i in range(ii):
                    for j in range(jj):
                        feed_dict.update({i_p:i,j_p:j})
                        mini_distances = np.minimum(mini_distances,
                             workplace.session.run(distances[LAYER],feed_dict))
                distances_test[str(epoch)][LAYER].append(mini_distances)
        for i in range(len(distances)):
             distances_test[str(epoch)][i]=np.concatenate(
                                                distances_test[str(epoch)][i])
        dataset.reset()
    # Accuracy
    print('accuracy')
    feed_dict.update(dnn.deter_dict(True))
    batch        = 0
    accuracy_out = 0
    while dataset.next(session=workplace.session):
        accuracy_out+=workplace.session.run(accu,feed_dict=feed_dict)
        batch+=1
    accuracies.append(accuracy_out/batch)
    print(accuracies[-1]*100)

    f=open('/mnt/drive1/rbalSpace/distances/save_test_v2_{}_{}.pkl'.format(
                                      DATASET,DATA_AUGMENTATION),'wb')
    pickle.dump([distances_train,distances_test,accuracies],f)
    f.close()


