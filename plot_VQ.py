from pylab import *
import glob
import cPickle


files = glob.glob('/mnt/project2/rb42Data/SMASO/*positive.pkl')
datasets = unique([f.split('_')[0].split('/')[-1] for f in files])
print "files",files
print "datasets",datasets

def formatting(B):
	train_loss = []
	test_accu = []
	train_accu = []
	temp = []
	pred = []
	for b in B:
		train_loss.append(b[0])
		train_accu.append(b[1])
		test_accu.append(b[2])
		temp.append(b[3])
		pred.append(b[4])
	return stack(train_loss,0),stack(train_accu,0),stack(test_accu,0),concatenate(temp,0),concatenate(pred,0)


for data in datasets:
	files  = glob.glob('/mnt/project2/rb42Data/SMASO/'+data+'*positive.pkl')
	models = unique([f.split('_')[1] for f in files])
	print "models",models
	for m in models:
	        files  = glob.glob('/mnt/project2/rb42Data/SMASO/'+data+'_'+m+'*positive.pkl')
		for f in files:
			fil = open(f,'rb')
			B0,B01,B1=cPickle.load(fil)
			fil.close()
			train_loss0,train_accu0,test_accu0,temp0,preds0=formatting(B0)
                        train_loss01,train_accu01,test_accu01,temp01,preds01=formatting(B01)
			train_loss1,train_accu1,test_accu1,temp1,preds1=formatting(B1)
			print shape(train_loss0),shape(test_accu0)
			figure()
			subplot(121)
			plot(train_accu0.mean(0),'b')
                        plot(test_accu0.mean(0),'--b')
                        plot(train_accu1.mean(0),'k')
                        plot(test_accu1.mean(0),'--k')
			subplot(122)
                        plot(train_loss0.mean(0)[:,1],'b')
                        plot(train_loss1.mean(0)[:,1],'k')
#                               tempk =temp- temp.min(axis=(1,2,3,4),keepdims=True)
#                               tempk/= temp.max(axis=(1,2,3,4),keepdims=True)
#                               for k in xrange(9):
#                                       for i in xrange(10):
#                                               subplot(9,10,1+i+k*10)
#                                               imshow(tempk[k][i].mean(2),aspect='auto',cmap='gray',vmin=0,vmax=1)
#                                               title(str(sum(temp[k][i]*x_train[k]))+','+str(preds[k][i]))
#                                               xticks([])
#                                               yticks([])
#                               tight_layout()
#                               show()
			show()

