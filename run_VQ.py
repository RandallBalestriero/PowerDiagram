from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import cPickle
execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

DATASET = sys.argv[-1]
lr      = 0.0002#float(sys.argv[-3])



bn=0
for model,model_name in zip([DenseCNN,largeCNN],['SmallCNN','LargeCNN']):
	if('Small' in model_name):
		ne=250
	else:
		ne=150
	if(1):#for nonlinearity in ['relu']:
		if(1):#for use_beta in [0,1]:
                        name = DATASET+'_'+model_name+'_positive.pkl'
			ALL_b0 = []
                        ALL_b01 = []
			ALL_b1 = []
                        for k in xrange(5):
				x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)
				m = model(bn=bn,n_classes=10,global_beta=1,pool_type='MAX',use_beta=1)
				model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=0)
				updates   = set_betas(float32(0))
				model1.session.run(updates)
				train_loss,train_accu,test_accu = model1.fit(x_train,y_train,x_test,y_test,n_epochs=ne,return_train_accu=1)
				temp = model1.get_templates(x_train[:200])
				preds = model1.predict(x_train[:200])
                                ALL_b0.append([train_loss,train_accu,test_accu,temp,preds])
####
                                m = model(bn=bn,n_classes=10,global_beta=1,pool_type='MAX',use_beta=0)
                                model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=0)
                                train_loss,train_accu,test_accu = model1.fit(x_train,y_train,x_test,y_test,n_epochs=ne,return_train_accu=1)
                                temp = model1.get_templates(x_train[:200])
                                preds = model1.predict(x_train[:200])
                                ALL_b01.append([train_loss,train_accu,test_accu,temp,preds])
####
				m = model(bn=bn,n_classes=10,global_beta=1,pool_type='MAX',use_beta=0)
				model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=0,e=0)
				updates   = set_betas(float32(0))
				model1.session.run(updates)
                                train_loss,train_accu,test_accu = model1.fit(x_train,y_train,x_test,y_test,n_epochs=ne,return_train_accu=1)
                                temp = model1.get_templates(x_train[:200])
                                preds = model1.predict(x_train[:200])
                                ALL_b1.append([train_loss,train_accu,test_accu,temp,preds])
		        f = open('/mnt/project2/rb42Data/SMASO/'+name,'wb')
		        cPickle.dump([ALL_b0,ALL_b01,ALL_b1],f)
		        f.close()
		
	
	


