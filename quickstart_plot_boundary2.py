import sys
sys.path.insert(0, "../")

import sknet
from sknet.optimize import Adam
from sknet.optimize.loss import *
from sknet.optimize import schedule
import matplotlib
matplotlib.use('Agg')
import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pylab as pl
import time
import tensorflow as tf
from sknet.dataset import BatchIterator
from sknet import ops,layers

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Data Loading
#-------------
N = 500
TIME = np.linspace(-2,2,N)
X = np.meshgrid(TIME,TIME)
X = np.stack([X[0].flatten(),X[1].flatten()],1).astype('float32')

dataset = sknet.dataset.Dataset()
dataset.add_variable({'input':{'train_set':X}})

dataset.create_placeholders(batch_size=50,
       iterators_dict={'train_set':BatchIterator("continuous")},device="/cpu:0")

# Create Network
#---------------

# we use a batch_size of 64 and use the dataset.datum shape to
# obtain the shape of 1 observation and create the input shape

# DN for the layer case

# RANK 1 PARALLEL
opt = int(sys.argv[-1])
b1 = np.asarray([-2.1,-1,-0.3,1,1.6,2]).astype('float32')/5
if opt==0:
    dnn = sknet.network.Network(name='simple_model')
    np.random.seed(111)
    W1 = (np.random.randn(2,1)/4).astype('float32')
    W1 = np.repeat(W1,6,1)
    dnn.append(ops.Dense(dataset.input, 6, W=W1, b=b1))
    output = dnn[0]
elif opt==1:
    # RANK 2 ORTHOGONAL
    dnn  = sknet.network.Network(name='simple_model')
    np.random.seed(10)
    W1 = (np.random.randn(2,2)/4).astype('float32')
    W1[:,1]-=W1[:,0]*(W1[:,1]*W1[:,0]).sum()/(W1[:,0]**2).sum()
    W1[:,1]*=2
    W1 = np.repeat(W1,3,1)
    dnn.append(ops.Dense(dataset.input, 6, W=W1, b=b1[[0,5,1,2,4,3]]))
    output = dnn[0]
else:
    # RANK 2 ARBITRARY
    dnn  = sknet.network.Network(name='simple_model')
    np.random.seed(111)
    W1 = (np.random.randn(2,6)/4).astype('float32')
    dnn.append(ops.Dense(dataset.input, 6, W=W1, b=b1))
    output = dnn[0]



output = sknet.Worker(op_name='poly',context='train_set',op= output,
        instruction='save every batch', deterministic=False)

# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)
workplace.execute_worker(output)

fig = plt.figure(figsize=(6,9))
#
mask     = output.data[0]>0
boundary = sknet.utils.geometry.get_input_space_partition(mask,N,N,2).astype('bool')
boundary = boundary.astype('float32')

#
poly = np.prod(output.data[0],1)

def plotit(poly,previous,name):
    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111,projection='3d')
    ax.plot_trisurf(X[:,0], X[:,1], np.abs(poly)**0.2,
                    linewidth=0.2, antialiased=True)
    ax.grid(False)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    #
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(1)
    ax.dist=9
    plt.savefig(name+'1.png', bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111,projection='3d')
    ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                        previous,3, zdir='z', offset=0.4,cmap='Greys')

    ax.set_zlim((0.36,0.44))
    ax.grid(False)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.dist=7
    plt.savefig(name+'2.png', bbox_inches='tight')
    plt.close()

    #


plotit(poly,boundary,'singlelayer'+str(opt))



