from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import cPickle

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')


import os

SAVE_DIR = os.environ['SAVE_DIR']

models       = [allDENSE,allCNN]
models_names = ['allDENSE','allCNN'] 

init_ws       = [tf.contrib.layers.xavier_initializer(uniform=False)]
init_ws_names = ['XavierNormal']
lrs           = [0.001,0.0005,0.0001]

DATASET          = sys.argv[-1]

batch_norm       = True
ne               = 150
batch_size       = 50


for k in xrange(10):
    for lr in lrs:
        for init,init_name in zip(init_ws,init_ws_names):
	    for model,model_name in zip(models,models_names):
	        name   = 'VORONOI/cnnvsmlp_'+DATASET+'_'+model_name+'_lr'+str(lr)+'_run'+str(k)
	        x_train,y_train,x_test,y_test,input_shape,c=load_data(DATASET,batch_size=batch_size)
####
	        m      = model(bn=batch_norm,bias_option=bias_constraint,init_W=init,n_classes=c)
	        model1 = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr)
	        train_loss,train_accu,test_accu = model1.fit(x_train,y_train,x_test,y_test,n_epochs=ne)
####
	        f = open(SAVE_DIR+name+'.pkl','wb')
	        cPickle.dump([train_loss,train_accu,test_accu],f)
	        f.close()
		
	
	


