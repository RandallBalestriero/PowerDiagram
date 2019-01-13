from pylab import *
import glob
import cPickle
import tabulate




import matplotlib as mpl
label_size = 14
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

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
		TRAIN_LOSS = ones((3,3))
                TRAIN_ACCU = ones((3,3))
                TEST_ACCU  = ones((3,3))
        TRAIN_LOSS,TEST_ACCU,TRAIN_ACCU= asarray(TRAIN_LOSS),asarray(TEST_ACCU),asarray(TRAIN_ACCU)
	return TRAIN_LOSS,TEST_ACCU,TRAIN_ACCU




lrs = ['0.001','0.0005','0.0001']
models = ['allCNN1','allDENSE1','allCNN2','allDENSE2','allCNN3','allDENSE3']
DATASET = ['FASHION','SVHN','CIFAR','CIFAR100']


MEANS_TRAIN = []
STDS_TRAIN  = []
MEANS_TEST = []
STDS_TEST  = []

for _ in [0]:
	for lr in lrs:
		MEAN_TRAIN = [[] for i in xrange(len(models))]
		STD_TRAIN  = [[] for i in xrange(len(models))]
                MEAN_TEST = [[] for i in xrange(len(models))]
                STD_TEST  = [[] for i in xrange(len(models))]
                for model,model_nb in zip(models,xrange(len(models))):
			for dataset in DATASET:
				name  = SAVE_DIR+'VORONOI/cnnvsmlp_'+dataset+'_'+model+'_lr'+lr+'_run*.pkl'
				train_loss,test_accu,train_accu = get_stat(name)
#				print shape(train_loss)
				MEAN_TRAIN[model_nb].append(train_accu.mean(0))
				STD_TRAIN[model_nb].append(train_accu.std(0))
                                MEAN_TEST[model_nb].append(test_accu.mean(0))
                                STD_TEST[model_nb].append(test_accu.std(0))
				print shape(train_accu)
		MEANS_TRAIN.append(MEAN_TRAIN)
		STDS_TRAIN.append(STD_TRAIN)
                MEANS_TEST.append(MEAN_TEST)
                STDS_TEST.append(STD_TEST)
	
MEANS_TRAIN_ = asarray(MEANS_TRAIN) # (#LR,#models,#dataset,#epochs)
STDS_TRAIN_  = asarray(STDS_TRAIN)
MEANS_TEST_ = asarray(MEANS_TEST) # (#LR,#models,#dataset,#epochs)
STDS_TEST_  = asarray(STDS_TEST)


print shape(MEANS_TRAIN_)

#MEANS = MEANS_TEST_[:,:,:,-1]
#STDS  = STDS_TEST_[:,:,:,-1]
MEANS = MEANS_TRAIN_[:,:,:,-1]
STDS  = STDS_TRAIN_[:,:,:,-1]
	
print 

ARG   = MEANS.argmax(0)

DATASETS = ['Fashion','SVHN','CIFAR10','CIFAR100']


figure(figsize=(8,2))
for dataset_nb in xrange(4):
        subplot(1,4,1+dataset_nb)
        plot(MEANS_TRAIN_[ARG[0,dataset_nb],0,dataset_nb],color='b',lw=3)
        plot(MEANS_TRAIN_[ARG[1,dataset_nb],1,dataset_nb],color='k',lw=3)
        ylim([0.1,1])
tight_layout()
savefig('cnn_vs_mlp_1.png')
close()


figure(figsize=(8,2))
for dataset_nb in xrange(4):
	subplot(1,4,1+dataset_nb)
	plot(MEANS_TRAIN_[ARG[2,dataset_nb],2,dataset_nb],color='b',lw=3)
        plot(MEANS_TRAIN_[ARG[3,dataset_nb],3,dataset_nb],color='k',lw=3)
	ylim([0.1,1])
tight_layout()
savefig('cnn_vs_mlp_2.png')
close()

figure(figsize=(8,2))
for dataset_nb in xrange(4):
        subplot(1,4,1+dataset_nb)
        semilogx(xrange(1,151),log(MEANS_TRAIN_[ARG[4,dataset_nb],4,dataset_nb]),color='b',lw=3)
        semilogx(xrange(1,151),log(MEANS_TRAIN_[ARG[5,dataset_nb],5,dataset_nb]),color='k',lw=3)
        ylim([-1.2,0])
tight_layout()
savefig('cnn_vs_mlp_3.png')
close()

figure(figsize=(8,2))
for dataset_nb in xrange(4):
        subplot(1,4,1+dataset_nb)
        semilogx(xrange(1,151),log(MEANS_TRAIN_[ARG[6,dataset_nb],6,dataset_nb]),color='b',lw=3)
        semilogx(xrange(1,151),log(MEANS_TRAIN_[ARG[7,dataset_nb],7,dataset_nb]),color='k',lw=3)
        ylim([-1.2,0])
tight_layout()
savefig('cnn_vs_mlp_4.png')
close()







MEAN  = zeros((len(models),4))
STD   = zeros((len(models),4))
for i in xrange(len(models)):
	for j in xrange(4):
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

#print 'INIT',init
print MEAN




