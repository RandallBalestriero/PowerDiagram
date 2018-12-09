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




for model,model_name in zip([DenseCNN],['denseCNN']):
	for nonlinearity in ['relu']:
		for bn in [0,1]:
                        name = DATASET+'_'+model_name+'_bn'+str(bn)+'_'+nonlinearity+'_learn.pkl'
			ALL_TRAIN00 = []
			ALL_TEST00 = []
                        ALL_TRAIN11 = []
                        ALL_TEST11 = []
			ALL_TRAIN01 = []
			ALL_TEST01 = []
                        ALL_TRAIN10 = []
                        ALL_TEST10 = []
                        for k in xrange(3):
				m = model(bn=bn,n_classes=10,nonlinearity=nonlinearity,use_beta=0,global_beta=1,pool_type='MAX')
				model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=0)
				updates   = set_betas(float32(0))
				model1.session.run(updates)
				train_loss_pre,test_loss_pre = model1.fit(x_train,y_train,x_test,y_test,n_epochs=80)
				ALL_TRAIN00.append(train_loss_pre)
				ALL_TEST00.append(test_loss_pre)
                                m = model(bn=bn,n_classes=10,nonlinearity=nonlinearity,use_beta=1,global_beta=1,pool_type='BETA')
                                model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=1)
                                updates   = set_betas(float32(0))
                                model1.session.run(updates)
                                train_loss_pre,test_loss_pre = model1.fit(x_train,y_train,x_test,y_test,n_epochs=80)
                                ALL_TRAIN01.append(train_loss_pre)
                                ALL_TEST01.append(test_loss_pre)
	##############################################################
				m = model(bn=bn,n_classes=10,nonlinearity=nonlinearity,use_beta=1,global_beta=0,pool_type='MAX')
				model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=1)
				updates   = set_betas(float32(0))
				model1.session.run(updates)
				train_loss_pre,test_loss_pre = model1.fit(x_train,y_train,x_test,y_test,n_epochs=80)
				ALL_TRAIN10.append(train_loss_pre)
				ALL_TEST10.append(test_loss_pre)
		        f = open('/mnt/project2/rb42Data/SMASO/'+name,'wb')
		        cPickle.dump([asarray(ALL_TRAIN00),asarray(ALL_TEST00),asarray(ALL_TRAIN01),asarray(ALL_TEST01),asarray(ALL_TRAIN10),asarray(ALL_TEST10)],f)
		        f.close()
		
	
	


