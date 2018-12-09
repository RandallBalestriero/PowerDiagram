from pylab import *
import glob
import cPickle
import matplotlib as mpl

label_size = 15
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

files    = glob.glob('/mnt/project2/rb42Data/SMASO/new*swish_ortho*.pkl')
datasets = unique([f.split('_')[0].split('/')[-1][3:] for f in files])

print "files",files
print "datasets",datasets

def formatting(files):
	print "FILESFILES",files
	train_loss0 = []
	test_accu0  = []
	train_accu0 = []
	temp0 = []
	pred0 = []
	W0    = []
        train_loss1 = []
        test_accu1  = []
        train_accu1 = []
        temp1 = []
        pred1 = []
        W1    = []
	for f in files:
		print "HERE",f
		fi=open(f,'rb')
		b=cPickle.load(fi)
		fi.close()
		if(1):#b[0][2][-1]>0.2):
			train_loss0.append(b[0][0])
			train_accu0.append(b[0][1])
			test_accu0.append(b[0][2])
			temp0.append(b[0][3])
			pred0.append(b[0][4])
			W0.append(b[0][5])
	                train_loss1.append(b[1][0])
	                train_accu1.append(b[1][1])
	                test_accu1.append(b[1][2])
	                temp1.append(b[1][3])
	                pred1.append(b[1][4])
	                W1.append(b[1][5])
	print shape(train_loss0),shape(train_accu0),shape(test_accu0),shape(temp0),shape(pred0),shape(W0)
	return [stack(train_loss0,0),stack(train_accu0,0),stack(test_accu0,0),concatenate(temp0,0),concatenate(pred0,0),W0[0]],[stack(train_loss1,0),stack(train_accu1,0),stack(test_accu1,0),concatenate(temp1,0),concatenate(pred1,0),W1[0]]





def plotW(w,m,n):
	print shape(w)
	W = w-w.min()
	W/=W.max()
	shape_ = shape(w)
	indexing = argsort((W*W).sum(axis=(0,1,2)))
	for i in xrange(m):
		for j in xrange(n):
			subplot(m,n,1+i*n+j)
			if(shape(W)[2]==1):
				imshow(W[:,:,0,indexing[i*n+j]],aspect='auto',interpolation='nearest',cmap='gray',vmin=0,vmax=1)
			else:
                		imshow(W[:,:,:,indexing[i*n+j]],aspect='auto',interpolation='nearest',cmap='jet',vmin=0,vmax=1)
			xticks([])
			yticks([])












fss = 18
for datas in datasets:
	files  = glob.glob('/mnt/project2/rb42Data/SMASO/new'+datas+'*swish_ortho*.pkl')
	models = unique([f.split('_')[1] for f in files])
	print "models",models
	for m in models:
		print "MODEL:",m
	        files_m       = glob.glob('/mnt/project2/rb42Data/SMASO/new'+datas+'_'+m+'*swish_ortho*.pkl')
		print files_m
		learning_rate = unique([f.split('_')[2] for f in files_m])
		for lr in learning_rate:
			print "LEARNING RATE",lr
		        files_m_lr  = glob.glob('/mnt/project2/rb42Data/SMASO/new'+datas+'_'+m+'_'+lr+'*swish_ortho*.pkl')
			bns    = unique([f.split('_')[3] for f in files_m_lr])
			print files_m_lr
			if(1):
	                        files_m_lr_bn = glob.glob('/mnt/project2/rb42Data/SMASO/new'+datas+'_'+m+'_'+lr+'*swish_ortho*.pkl')
		                data      = formatting(files_m_lr_bn)
				train_loss0,train_accu0,test_accu0,temp0,preds0,W0 = data[0]
                                train_loss1,train_accu1,test_accu1,temp1,preds1,W1 = data[1]
				figure()
				subplot(121)
				plot(train_accu0.mean(0),'b')
		                plot(train_accu1.mean(0),'k')
				grid('on')
				xlabel('Epochs',fontsize=fss)
				ylabel('Train Accuracy',fontsize=fss)
				legend(['Unconstrained','Centered and Ortho.'],fontsize=fss,loc='lower right')
				subplot(122)
                                plot(test_accu0.mean(0),'b')
                                plot(test_accu1.mean(0),'k')
                                grid('on')
                                xlabel('Epochs',fontsize=fss)
                                ylabel('Test Accuracy',fontsize=fss)
				tight_layout()
				figure()
	                        plotW(W0[0],5,5)
                                title('Init. Standard')
                                tight_layout()
				figure()
                                plotW(W1[0],5,5)
				title('Init Centered and orth.')
                                tight_layout()
                                figure()
                                plotW(W0[-1],5,5)
                                title('Learned Standard')
                                tight_layout()
                                figure()
                                plotW(W1[-1],5,5)
                                title('Learned Centered and orth.')
                                tight_layout()
				show()

