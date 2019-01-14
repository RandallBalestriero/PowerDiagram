from pylab import *
import glob
import cPickle
import tabulate

import os
SAVE_DIR = os.environ['SAVE_DIR']








def get_stat(name):
        all_files = glob.glob(name)
        TRAIN_LOSS,TRAIN_ACCU,TEST_ACCU=[],[],[]
        for FILE in all_files:
                f = open(FILE,'rb')
                train_loss,train_accu,test_accu = cPickle.load(f)
                f.close()
                TRAIN_LOSS.append(train_loss)
                TEST_ACCU.append(asarray(test_accu))
                TRAIN_ACCU.append(asarray(train_accu))
	if(len(all_files)==0):
		TRAIN_LOSS = zeros((3,3))
                TRAIN_ACCU = zeros((3,3))
                TEST_ACCU  = zeros((3,3))
        return asarray(TRAIN_LOSS),asarray(TEST_ACCU),asarray(TRAIN_ACCU)






lrs = ['0.001','0.0005','0.0001']
models = ['SmallDENSE','LargeDENSE']
DATASET = ['FASHION','SVHN','CIFAR','CIFAR100']

setting = 'unconstrained'

MEANS = []
STDS  = []
for init in [0]:
	for lr in lrs:
		MEAN = [[] for i in xrange(3)]
		STD  = [[] for i in xrange(3)]
		print ' '
		for dataset in DATASET:
			for model in models:

				name  = SAVE_DIR+'VORONOI/distance_nob_'+dataset+'_'+model+'_lr'+lr+'_alpha1e-07_'+setting+'_run*.pkl'
				train_loss,test_accu,train_accu = get_stat(name)
				MEAN[0].append(mean(test_accu[:,-1]))
				STD[0].append(std(test_accu[:,-1]))
	
                                name  = SAVE_DIR+'VORONOI/distance_nob_'+dataset+'_'+model+'_lr'+lr+'_alpha1e-05_'+setting+'_run*.pkl'
                                train_loss,test_accu,train_accu = get_stat(name)
                                MEAN[1].append(mean(test_accu[:,-1]))
                                STD[1].append(std(test_accu[:,-1]))

                                name  = SAVE_DIR+'VORONOI/distance_nob_'+dataset+'_'+model+'_lr'+lr+'_alpha0.001_'+setting+'_run*.pkl'
                                train_loss,test_accu,train_accu = get_stat(name)
                                MEAN[2].append(mean(test_accu[:,-1]))
                                STD[2].append(std(test_accu[:,-1]))
		MEANS.append(MEAN)
		STDS.append(STD)
	
MEANS = asarray(MEANS)
STDS  = asarray(STDS)
	
ARG   = MEANS.argmax(0)

MEAN  = zeros((3,8))
STD   = zeros((3,8))
for i in xrange(3):
	for j in xrange(8):
		MEAN[i,j]=MEANS[ARG[i,j],i,j]
                STD[i,j]=STDS[ARG[i,j],i,j]


MEAN = asarray(MEAN)*100
STD  = asarray(STD)*100

MEAN = ((MEAN*10).astype('int32').astype('float32')/10.).astype('str')
STD  = ((STD*10).astype('int32').astype('float32')/10.).astype('str')

for i in xrange(MEAN.shape[0]):
	for j in xrange(STD.shape[1]):
		MEAN[i,j]+=' ABCD '+STD[i,j]

MEAN = tabulate.tabulate(MEAN,tablefmt='latex')

print 'INIT',init
print MEAN




