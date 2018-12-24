from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import cPickle

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')


import os
SAVE_DIR = os.environ['SAVE_DIR']

models       = [smallCNN,largeCNN]
models_names = ['SmallCNN','LargeCNN'] 

init_ws       = [tf.uniform_unit_scaling_initializer(),tf.contrib.layers.xavier_initializer(uniform=True),tf.contrib.layers.xavier_initializer(uniform=False)]
init_ws_names = ['UniformUnitScaling','XavierUniform','XavierNormal']
lrs           = [0.001,0.0005,0.0001]

DATASET          = sys.argv[-1]
bias_constraint  = 'zero'
#training_option  = sys.argv[-2]
lr_nb            = int(sys.argv[-2])
model_nb         = int(sys.argv[-3])

batch_norm       = int(sys.argv[-4])
ne               = 50
batch_size       = 50

model,model_name = models[model_nb],models_names[model_nb]



for k in xrange(10):
    for lr in [lrs[lr_nb]]:
        for init,init_name in zip(init_ws,init_ws_names):
	    for ALPHA in [0, 0.0001,0.001,0.01,0.1,1]:
		if(ALPHA==0):
		    for training_opt in ['during','pretrain']:
	    		if(batch_norm):
	    		    name   = 'VORONOI/pretraining_'+DATASET+'_'+model_name+'_lr'+str(lr)+'_init'+init_name+'_'+training_opt+'_'+str(ALPHA)+'_run'+str(k)
	    		else:
			    name   = 'VORONOI/pretraining_nob_'+DATASET+'_'+model_name+'_lr'+str(lr)+'_init'+init_name+'_'+training_opt+'_'+str(ALPHA)+'_run'+str(k)
	    		x_train,y_train,x_test,y_test,input_shape,c=load_data(DATASET,batch_size=batch_size)
####
			m      = model(bn=batch_norm,bias_option=bias_constraint,init_W=init,n_classes=c)
			model1 = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,ALPHA=float32(ALPHA),train_option=training_opt)
	    		train_loss,train_accu,test_accu = model1.fit(x_train,y_train,x_test,y_test,n_epochs=ne)
####
#	    		f = open(SAVE_DIR+name+'.pkl','wb')
#	    		cPickle.dump([train_loss,train_accu,test_accu],f)
#	    		f.close()
		else:
		    for training_opt in ['during','both']:
	    		if(batch_norm):
	    		    name   = 'VORONOI/pretraining_'+DATASET+'_'+model_name+'_lr'+str(lr)+'_init'+init_name+'_'+training_opt+'_'+str(ALPHA)+'_run'+str(k)
	    		else:
			    name   = 'VORONOI/pretraining_nob_'+DATASET+'_'+model_name+'_lr'+str(lr)+'_init'+init_name+'_'+training_opt+'_'+str(ALPHA)+'_run'+str(k)
	    		x_train,y_train,x_test,y_test,input_shape,c=load_data(DATASET,batch_size=batch_size)
####
			m      = model(bn=batch_norm,bias_option=bias_constraint,init_W=init,n_classes=c)
			model1 = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,ALPHA=float32(ALPHA),train_option=training_opt)
	    		train_loss,train_accu,test_accu = model1.fit(x_train,y_train,x_test,y_test,n_epochs=ne)
####
#	    		f = open(SAVE_DIR+name+'.pkl','wb')
#	    		cPickle.dump([train_loss,train_accu,test_accu],f)
#                        f.close()
	



