import sys
sys.path.insert(0, "../Sknet")

import sknet
from sknet.utils.geometry import get_input_space_partition
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
N = 350
TIME = np.linspace(-7,7,N)
X = np.meshgrid(TIME,TIME)
X = np.stack([X[0].flatten(),X[1].flatten()],1).astype('float32')

dataset = sknet.dataset.Dataset()
dataset.add_variable({'input':{'train_set':X}})

dataset.create_placeholders(batch_size=50,
       iterators_dict={'train_set':BatchIterator("continuous")},device="/cpu:0")

# Create Network
#---------------

dnn       = sknet.network.Network(name='simple_model')
np.random.seed(18)

W = np.random.randn(2,9).astype('float32')/np.sqrt(7)
b = np.random.randn(9).astype('float32')/6
dnn.append(ops.Dense(dataset.input, 9, W=W,b=b))
dnn.append(ops.Activation(dnn[-1],0.1))

for i in range(3):
    W = np.random.randn(9,9).astype('float32')/np.sqrt(12)
    b = np.random.randn(9).astype('float32')/3
    dnn.append(ops.Dense(dnn[-1], 9, W=W,b=b))
    dnn.append(ops.Activation(dnn[-1],0.1))

#np.random.seed(108)
dnn.append(ops.Dense(dnn[-1], 1,W=0.01*np.random.randn(9,1).astype('float32'),
                    b=np.array([0.2-0.09-0.13+0.020194676]).astype('float32')))

outputs = dnn[0::2].as_list()

# Workers
#---------

outputs = tf.concat(outputs,-1)
output = sknet.Worker(op_name='poly',context='train_set',op= outputs,
        instruction='save every batch', deterministic=False)

# Pipeline
#---------
workplace = sknet.utils.Workplace(dnn,dataset=dataset)
workplace.execute_worker(output)

#
print(output.data[0][:,[-1]].min(),output.data[0][:,[-1]].max())
#exit()
masks   = [output.data[0][:,9*i:9*(i+1)]>0 for i in range(4)]
boundarys_l  = [get_input_space_partition(mask,N,N,2).astype('bool')
                    for mask in masks]
boundarys_lk = [[get_input_space_partition(mask[:,k],N,N,2).astype('bool')
                    for k in range(9)] for mask in masks]
boundarys_1l = np.cumsum(boundarys_l,axis=0,dtype='bool')
mask           = output.data[0][:,[-1]]>0
boundary_last  = get_input_space_partition(mask,N,N,2).astype('bool')
boundary_1last = boundarys_1l[-1]+boundary_last



def plotit(previous,b1,b2,name,last=False):
    if not np.isscalar(previous):
        previous = previous.astype('float32')
    b1 = b1.astype('float32')
    b2 = b2.astype('float32')
    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111,projection='3d')
    if not last:
        ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                        previous+0.7*b1,3, zdir='z', offset=0.4,cmap='Greys',
                        vmin=0,vmax=(1+0.7))
        ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                        b2,3, zdir='z', offset=0.01,cmap='Greys',
                        vmin=0,vmax=b2.max())
    else:
        colors1 = plt.cm.Greys(np.linspace(0, 1, 3))
        # Red colormap which takes values from 
        colors2 = plt.cm.hsv(np.linspace(0, 1, 100))
        colors  = np.vstack((colors1[:2], colors2[[-1]]))
        # generating a smoothly-varying LinearSegmentedColormap
        cmap = mcolors.LinearSegmentedColormap.from_list('colormap', colors)
        ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                        previous,3, zdir='z', offset=0.4,cmap='Greys',
                        vmin=0,vmax=previous.max())
        ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                        b1,3, zdir='z', offset=0.4,cmap=cmap,
                        vmin=0,vmax=b1.max(),alpha=0.5)
        ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                        b2,3, zdir='z', offset=0.01,cmap=cmap,
                        vmin=0,vmax=b2.max())
    WHERE  = np.where(b2>0)
    A0  = np.where(WHERE[0]<=2)[0]
    B0  = np.where(WHERE[0]>=(N-2))[0]
    A1  = np.where(WHERE[1]<=2)[0]
    B1  = np.where(WHERE[1]>=(N-2))[0]
    POINTS = list()
    if len(A0)>0:
        POINTS.append(A0[0])
    if len(A1)>0:
        POINTS.append(A1[0])
    if len(B0)>0:
        POINTS.append(B0[0])
    if len(B1)>0:
        POINTS.append(B1[0])
    print(POINTS)
    for i in POINTS:

        X1 = TIME[WHERE[0][i]]
        Y1 = TIME[WHERE[1][i]]
        ax.plot([Y1,Y1],[X1,X1],[0.01,0.4],color='k',linestyle='--',zorder=100)

    ax.set_zlim((-0.08,0.47))
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
    plt.savefig(name+'2bis.pdf', bbox_inches='tight')
    plt.close()



for ii,b in enumerate(boundarys_lk[0]):
    plotit(0., boundarys_l[0], b,
                 'layer0_'+str(ii))

for layer in range(1,4):
    for ii,b in enumerate(boundarys_lk[layer]):
        plotit(boundarys_1l[layer], boundarys_l[layer], b,
                        'layer'+str(layer)+'_'+str(ii))

plotit(boundarys_1l[-1],boundary_last,boundary_last,'layer4',last=True)




