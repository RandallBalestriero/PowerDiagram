execfile('lasagne_tf.py')
execfile('utils.py')
import random

class DNNClassifier(object):
	def __init__(self,input_shape,model_class,lr=0.0001,optimizer = tf.train.AdamOptimizer):
		#setting = {base,pretrainlinear,}
		tf.reset_default_graph()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.log_device_placement     = True
                self.n_classes  =  model_class.n_classes
		self.session    = tf.Session(config=config)
		self.batch_size = input_shape[0]
		self.lr         = lr
		opt             = optimizer(lr)
		with tf.device('/device:GPU:0'):
			self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
			optimizer          = optimizer(self.learning_rate)
        		self.x             = tf.placeholder(tf.float32, shape=input_shape,name='x')
        	        self.y_            = tf.placeholder(tf.int32, shape=[input_shape[0]],name='y')
        	        self.training      = tf.placeholder(tf.bool,name='training_phase')
        	        self.layers        = model_class.get_layers(self.x,input_shape,training=self.training)
                        self.prediction    = self.layers[-1].output
			count_number_of_params()
                        self.loss          = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=self.prediction, name='cross_entropy'))
        	        self.variables     = tf.trainable_variables()
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            self.apply_updates = opt.minimize(self.loss,var_list=self.variables)
        	        self.accuracy      = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.prediction,1),tf.int32), self.y_),tf.float32))
		self.session.run(tf.global_variables_initializer())
	def _fit(self,X,y,indices,update_time=50):
		self.e    += 1
        	n_train    = X.shape[0]/self.batch_size
        	train_loss = []
        	for i in xrange(n_train):
			if(self.batch_size<self.n_classes):
				here = [random.sample(k,1) for k in indices]
				here = [here[i] for i in permutation(self.n_classes)[:self.batch_size]]
			else:
				here = [random.sample(k,self.batch_size/self.n_classes) for k in indices]
			here = concatenate(here)
                        self.session.run(self.apply_updates,feed_dict={self.x:X[here],self.y_:y[here],self.training:True,self.learning_rate:float32(self.lr)})
			if(i%update_time==0):
                                train_loss.append(self.session.run(self.loss,feed_dict={self.x:X[here],self.y_:y[here],self.training:False}))
                                print i,n_train,train_loss[-1]
        	return train_loss
        def fit(self,X,y,X_test,y_test,n_epochs=5):
		train_loss = []
		train_accu = []
		test_accu  = []
		self.e     = 0
                n_test     = X_test.shape[0]/self.batch_size
                indices    = [find(y==k) for k in xrange(self.n_classes)]
                print [len(i) for i in indices]
		for i in xrange(n_epochs):
                        if(i==90): self.lr/=5
                        elif(i==130): self.lr/=5
			print "epoch",i
			train_loss.append(self._fit(X,y,indices))
                	acc1 = 0.0
                	for j in xrange(n_test):
                	        acc1+=self.session.run(self.accuracy,feed_dict={self.x:X_test[self.batch_size*j:self.batch_size*(j+1)],
						self.y_:y_test[self.batch_size*j:self.batch_size*(j+1)],self.training:False})
                	test_accu.append(acc1/n_test)
		        n_train = X.shape[0]/self.batch_size
	                acc1    = 0.0
	                for j in xrange(n_train):
	                        acc1+=self.session.run(self.accuracy,feed_dict={self.x:X[self.batch_size*j:self.batch_size*(j+1)],
	                                                self.y_:y[self.batch_size*j:self.batch_size*(j+1)],self.training:False})
	                train_accu.append(acc1/n_train)
			print train_accu[-1]
                	print 'test accu',test_accu[-1]
	        return concatenate(train_loss),train_accu,test_accu
	def predict(self,X):
		n = X.shape[0]/self.batch_size
		preds = []
		for j in xrange(n):
                    preds.append(self.session.run(self.prediction,feed_dict={self.x:X[self.batch_size*j:self.batch_size*(j+1)],self.training:False}))
                return concatenate(preds,axis=0)


class largeCNN:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],64,5,pad='SAME',training=training,bn=False,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(ConvLayer(layers[-1],96,3,pad='FULL',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(ConvLayer(layers[-1],96,3,pad='FULL',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],192,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(ConvLayer(layers[-1],192,3,pad='FULL',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(ConvLayer(layers[-1],192,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],192,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(ConvLayer(layers[-1],192,1,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
		self.layers = layers
                return self.layers



class smallCNN:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],64,5,pad='SAME',training=training,bn=False,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],128,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],192,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(Pool2DLayer(layers[-1],2,pool_type='MAX'))
                layers.append(ConvLayer(layers[-1],192,1,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
		self.layers = layers
                return self.layers







