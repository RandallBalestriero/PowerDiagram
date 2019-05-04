import sys
sys.path.insert(0, "../Sknet")

import sknet
from sknet.optimize import Adam
from sknet.optimize.loss import *
from sknet.optimize import schedule
import matplotlib
matplotlib.use('Agg')
import os

# Make Tensorflow quiet.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import mpl_toolkits.mplot3d.art3d as art3d

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

np.random.seed(11)
W = np.random.rand(6,2)*3-1.5
np.random.seed(13)
b = np.random.rand(6)

argmaxes = (((X[:,None]-W)**2).sum(2)-b**2).argmin(1)
mask     = np.zeros((X.shape[0],6))
mask[range(X.shape[0]),argmaxes]=1
boundary = sknet.utils.geometry.get_input_space_partition(mask,N,N,2).astype('bool')
boundary = boundary.astype('float32')


fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111,projection='3d')
ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                 boundary,3, zdir='z', offset=0.4,cmap='Greys')

for i,b_ in enumerate(b):
    ax.plot([W[i,0]],[W[i,1]],[0.40],marker='o',markersize=4,
                            zorder=100,color='k')
    p = plt.Circle((W[i,0], W[i,1]), b_,fill=False,color='k')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0.4, zdir="z")
    p = plt.Circle((W[i,0], W[i,1]), b_,color='k',alpha=0.25)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=0.4, zdir="z")

for kk in range(6):
    ww = np.random.randn(2)
    ww/=np.sqrt((ww**2).sum())
    point = W[kk]+ww* b[kk]
    ax.plot([W[kk,0],point[0]],[W[kk,1],point[1]],
             [0.4,0.4],linestyle='dotted',linewidth=2,zorder=100,
            color='b',alpha=0.90)

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
plt.savefig('representation_1.png', bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111,projection='3d')
ax.contourf(X[:,0].reshape((N,N)), X[:,1].reshape((N,N)),
                 boundary,3, zdir='z', offset=0.4,cmap='Greys')

ax.text(W[0,0]+0.2,W[0,1],0.4,r'$[\mu]_{k,.}$', fontsize=15)
ax.text(W[0,0],W[0,1],0.41+b[0]**2,r'$([\mu]_{k,.},[{\rm rad}]_k)$',fontsize=15,
                horizontalalignment='center')

for w,b_ in zip(W,b**2):
    ax.plot([w[0]],[w[1]],[0.4+b_],marker='o',color='b',markersize=4,
                        zorder=102)
    ax.plot([w[0]],[w[1]],[0.4],marker='o',color='k',markersize=4,
                        zorder=100)
    ax.plot([w[0],w[0]],[w[1],w[1]],[0.4,0.4+b_],linestyle='dotted',
                color='b',zorder=101)

#ax.set_zlim((0.36,1.14))
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
plt.savefig('representation_2.png', bbox_inches='tight')
plt.close()




