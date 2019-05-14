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
from sknet.utils.geometry import states2values



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help="the dataset to train on, can be"\
           +"mnist, fasionmnist, cifar10, ...",type=str, default='mnist')
parser.add_argument('--model', help="the model to use: cnn or resnet",
                      type=str,choices=['cnn','resnet'], default='cnn')
parser.add_argument("--data_augmentation", help="using data augmentation",
                     type=sknet.utils.str2bool,default='0')

args              = parser.parse_args()
DATASET           = args.dataset
MODEL             = args.model
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

#standardize = sknet.dataset.Standardize().fit(dataset['images/train_set'])
#dataset['images/train_set'] = \
#                        standardize.transform(dataset['images/train_set'])
#dataset['images/test_set'] = \
#                        standardize.transform(dataset['images/test_set'])
#dataset['images/valid_set'] = \
#                        standardize.transform(dataset['images/valid_set'])

dataset.create_placeholders(batch_size=32,
        iterators_dict={'train_set':BatchIterator("random_see_all"),
                        'valid_set':BatchIterator('continuous'),
                        'test_set':BatchIterator('continuous')},device="/cpu:0")

# Create Network
#---------------

dnn       = sknet.Network(name='simple_model')

if DATA_AUGMENTATION:
    dnn.append(ops.RandomAxisReverse(dataset.images,axis=[-1]))
    dnn.append(ops.RandomCrop(dnn[-1],(28,28),seed=10))
else:
    dnn.append(dataset.images)

if MODEL=='cnn':
    sknet.networks.ConvLarge(dnn,dataset.n_classes)
elif MODEL=='resnet':
    sknet.networks.Resnet(dnn,dataset.n_classes,D=4,W=1)

prediction = dnn[-1]
VQs     = [op.VQ for op in dnn[1:-2] if op.VQ is not None]
loss    = sknet.losses.crossentropy_logits(p=dataset.labels,q=prediction)
accu    = sknet.losses.accuracy(dataset.labels,prediction)

B         = dataset.N('train_set')//32
lr        = sknet.schedules.PiecewiseConstant(0.002,
                                    {100*B:0.002,200*B:0.001,250*B:0.0005})
optimizer = sknet.optimizers.Adam(loss,lr,params=dnn.variables(trainable=True))
minimizer = tf.group(optimizer.updates+dnn.updates)

# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)

feed_dict = dict()
accuracies= list()
vqs_train = dict()
vqs_test  = dict()

for epoch in range(150):
    vqs_train[str(epoch)] = [[] for i in range(len(VQs))]
    vqs_test[str(epoch)]  = [[] for i in range(len(VQs))]

    #--Train Set--
    dataset.set_set('train_set',session=workplace.session)
    feed_dict.update(dnn.deter_dict(True))

    # VQs
    if epoch%20==0:
        print('VQ train set')
        mapping_dict = [dict() for i in range(len(VQs))]
        while dataset.next(session=workplace.session):
            vqs = workplace.session.run(VQs,feed_dict=feed_dict)
            for i,vq in enumerate(vqs):
                vqs_train[str(epoch)][i].append(states2values(
                                  np.concatenate(vqs[:i+1],1),mapping_dict[i]))
        for i in range(len(VQs)):
             vqs_train[str(epoch)][i]=np.concatenate(vqs_train[str(epoch)][i])
             print(len(np.unique(vqs_train[str(epoch)][i])))

    # Training
    print('training')
    feed_dict.update(dnn.deter_dict(False))
    while dataset.next(session=workplace.session):
        workplace.session.run(minimizer,feed_dict=feed_dict)

    #--Test Set--
    dataset.set_set('test_set',session=workplace.session)
    feed_dict.update(dnn.deter_dict(True))

    # VQs
    if epoch%20==0:
        print('VQ test')
        mapping_dict = [dict() for i in range(len(VQs))]
        while dataset.next(session=workplace.session):
            vqs = workplace.session.run(VQs,feed_dict=feed_dict)
            for i,vq in enumerate(vqs):
                vqs_test[str(epoch)][i].append(states2values(
                                 np.concatenate(vqs[:i+1],1), mapping_dict[i]))
        for i in range(len(VQs)):
             vqs_test[str(epoch)][i] = np.concatenate(vqs_test[str(epoch)][i])

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

    f=open('/mnt/drive1/rbalSpace/regions/save_test_{}_{}_{}.pkl'.format(MODEL,
                                      DATASET,DATA_AUGMENTATION),'wb')
    pickle.dump([vqs_train,vqs_test],f)
    f.close()


