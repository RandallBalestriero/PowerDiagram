from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import cPickle
execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

DATASET = sys.argv[-1]
lr      = 0.0005#float(sys.argv[-3])





x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)



vs = linspace(-3,0,10)

for model,model_name in zip([DenseCNN],['denseCNN']):
	for nonlinearity in ['relu','lrelu','abs']:
		for bn in [0,1]:
                        name = DATASET+'_'+model_name+'_bn'+str(bn)+'_'+nonlinearity+'_molly.pkl'
			ALL_TRAIN0 = []
			ALL_TEST0 = []
			ALL_TRAIN1 = []
			ALL_TEST1 = []
                        for k in xrange(2):
				m = model(bn=bn,n_classes=10,nonlinearity=nonlinearity,use_beta=0,global_beta=0,pool_type='MAX')
				all_train = []
				all_test  = []
				model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,l2=0.0000001,learn_beta=0)
				for i in xrange(10):
					updates = set_betas(float32(vs[i]))
					model1.session.run(updates)
					train_loss_pre,test_loss_pre = model1.fit(x_train,y_train,x_test,y_test,n_epochs=1)
					all_train.append(train_loss_pre)
					all_test.append(test_loss_pre)
				updates = set_betas(0)
				model1.session.run(updates)
				train_loss_pre,test_loss_pre = model1.fit(x_train,y_train,x_test,y_test,n_epochs=100)
				all_train.append(train_loss_pre)
				all_test.append(test_loss_pre)
				ALL_TRAIN0.append(concatenate(all_train))
				ALL_TEST0.append(concatenate(all_test))
			# DO BETA CASES
				m = model(bn=bn,n_classes=10,nonlinearity=nonlinearity,use_beta=1,global_beta=0,pool_type='BETA')
				all_train = []
				all_test  = []
				model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,l2=0.0000001,learn_beta=0)
				for i in xrange(10):
					updates = set_betas(float32(vs[i]))
					model1.session.run(updates)
					train_loss_pre,test_loss_pre = model1.fit(x_train,y_train,x_test,y_test,n_epochs=1)
					all_train.append(train_loss_pre)
					all_test.append(test_loss_pre)
				updates = set_betas(0)
				model1.session.run(updates)
				train_loss_pre,test_loss_pre = model1.fit(x_train,y_train,x_test,y_test,n_epochs=100)
				all_train.append(train_loss_pre)
				all_test.append(test_loss_pre)
				ALL_TRAIN1.append(concatenate(all_train))
				ALL_TEST1.append(concatenate(all_test))
		        f = open('/mnt/project2/rb42Data/SMASO/'+name,'wb')
		        cPickle.dump([ALL_TRAIN0,ALL_TEST0,ALL_TRAIN1,ALL_TEST1],f)
		        f.close()
		
	
	


