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
#models       = [smallDENSE,largeDENSE]
#models_names = ['SmallDENSE','LargeDENSE'] 

init_ws       = [tf.uniform_unit_scaling_initializer(),tf.contrib.layers.xavier_initializer(uniform=True),tf.contrib.layers.xavier_initializer(uniform=False)]
init_ws_names = ['XavierNormal']
lrs           = [0.001,0.0005,0.0001]

DATASET          = sys.argv[-1]
bias_constraint  = sys.argv[-2]
model_nb         = int(sys.argv[-3])

batch_norm       = int(sys.argv[-4])
ne               = 150
batch_size       = 50

model,model_name = models[model_nb],models_names[model_nb]

for k in xrange(10):
    for lr in lrs:
        for init,init_name in zip(init_ws,init_ws_names):
	    for alpha in [0.0000001,0.00001,0.001]:
                if(batch_norm):
	            name   = 'VORONOI/distance_'+DATASET+'_'+model_name+'_lr'+str(lr)+'_init'+init_name+'_'+bias_constraint+'_run'+str(k)
	    	else:
		    name   = 'VORONOI/distance_nob_'+DATASET+'_'+model_name+'_lr'+str(lr)+'_init'+init_name+'_'+bias_constraint+'_run'+str(k)
	    	x_train,y_train,x_test,y_test,input_shape,c=load_data(DATASET,batch_size=batch_size)
####
	    	m      = model(bn=batch_norm,bias_option=bias_constraint,init_W=init,n_classes=c)
	    	model1 = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,distance_coeff=float32(alpha))
	    	train_loss,train_accu,test_accu = model1.fit(x_train,y_train,x_test,y_test,n_epochs=ne)
####
	    	f = open(SAVE_DIR+name+'.pkl','wb')
	    	cPickle.dump([train_loss,train_accu,test_accu],f)
	    	f.close()
		
	
	


