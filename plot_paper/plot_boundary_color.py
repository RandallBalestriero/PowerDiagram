import sys
sys.path.insert(0, "../../Sknet")

import sknet
from sknet.utils.geometry import get_input_space_partition
import matplotlib
matplotlib.use('Agg')
import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pylab as pl
import time
import tensorflow as tf
from sknet import ops,layers

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


# Data Loading
#-------------

# the run number
RUN = sys.argv[-1]
# number of points for the grid
N = 350
# grid
TIME = np.linspace(-7,7,N)
X = np.meshgrid(TIME,TIME)
X = np.stack([X[0].flatten(),X[1].flatten()],1).astype('float32')

# load the dataset
dataset = sknet.Dataset()
dataset['images/train_set'] = X
dataset.create_placeholders(50, {'train_set': "continuous"},
                            device="/cpu:0")

# Create Network
#---------------

dnn = sknet.Network()
np.random.seed(18+int(RUN))

# first layer going from 2d to 9 units
W = np.random.randn(2,4).astype('float32')/np.sqrt(3)
b = np.random.randn(4).astype('float32')/3
dnn.append(ops.Dense(dataset.images, 4, W=W,b=b))
dnn.append(ops.Activation(dnn[-1], 0.05))

# following layers with same input and output dimension
for i in range(1):
    W = np.random.randn(4,4).astype('float32')/np.sqrt(3)
    b = np.random.randn(4).astype('float32')/3
    dnn.append(ops.Dense(dnn[-1], 4, W=W,b=b))
    dnn.append(ops.Activation(dnn[-1], 0.05))

# last layer going to 1d for binary classification
dnn.append(ops.Dense(dnn[-1], 1, W=1*np.random.randn(4,1).astype('float32'),
                    b=np.array([0.5]).astype('float32')))

# extracts the outputs prior nonlinearity
outputs = dnn[0::2].as_list()

# Workers
#---------

outputs = tf.concat(outputs,-1)
output = sknet.Worker(outputs=outputs, context='train_set',
                      feed_dict=dnn.deter_dict(False))

# Pipeline
#---------

# now get the feature maps prior activation for all the points of the 2D grid
workplace = sknet.Workplace(dataset=dataset)
workplace.execute_worker(output)

# format those data for plotting the partitioning
features = output.epoch_data['outputs'][0]
features = features.reshape((-1, features.shape[-1]))

masks = [features[:, :4] > 0, features[:, 4:8] > 0, features[:, [8]] > 0]
boundarys_l = [get_input_space_partition(mask, N, N, 3).astype('float32')
                for mask in masks]


def plotit(previous, name):
    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111,projection='3d')
    previous2=previous[0]-previous[0]*previous[1]+2*previous[1]
    colors = [(0.9, 0.9, 0.9), (0.1, 0.7, 0.4), (0.1, 0.1, 1)]
    cm = LinearSegmentedColormap.from_list('name', colors, N=3)
    ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                previous2, 3, zdir='z', offset=0.4,
                vmin=0, vmax=2, cmap=cm)

    colors = [(1, 1, 1), (0.9, 0.1, 0.1)]
    cm = LinearSegmentedColormap.from_list('name', colors, N=3)
    ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                previous[-1], 3, zdir='z', offset=0.4,
                vmin=0, vmax=1, cmap=cm, alpha=0.5)

    ax.set_zlim((0.23,0.47))
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
    plt.savefig(name+'.png', bbox_inches='tight')
    plt.close()





plotit(boundarys_l, 'layer_color'+str(RUN))






