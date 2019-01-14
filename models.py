execfile('lasagne_tf.py')
execfile('utils.py')
import random
import time

class DNNClassifier(object):
	def __init__(self,input_shape,model_class,lr=0.0001,optimizer = tf.train.AdamOptimizer,distance_coeff=0):
		#train_option = {none,pretrain,during,both}
		tf.reset_default_graph()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.log_device_placement     = True
                self.n_classes  =  model_class.n_classes
		self.session    = tf.Session(config=config)
		self.batch_size = input_shape[0]
		self.lr         = lr
		self.distance_coeff = distance_coeff
		with tf.device('/device:GPU:0'):
			self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
        		self.x             = tf.placeholder(tf.float32, shape=input_shape,name='x')
        	        self.y_            = tf.placeholder(tf.int32, shape=[input_shape[0]],name='y')
        	        self.training      = tf.placeholder(tf.bool,name='training_phase')
        	        self.layers        = model_class.get_layers(self.x,input_shape,training=self.training)
                        self.prediction    = self.layers[-1].output
			count_number_of_params()
			time.sleep(2)
                        self.variables     = tf.trainable_variables()
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			# TRAINING LOSS AND UPDATES
	                training_opt             = optimizer(self.lr)
                        self.crossentropy_loss   = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=self.prediction, name='cross_entropy'))
			self.all_distance_loss   = tf.stack([l.distance_loss for l in self.layers[1:] if l.distance_loss is not None],0)
			self.distance_loss       = -tf.reduce_sum(self.all_distance_loss)/input_shape[0]
			if(distance_coeff>0):
				self.training_loss = self.crossentropy_loss+distance_coeff*self.distance_loss
			else:
                                self.training_loss = self.crossentropy_loss
                        with tf.control_dependencies(update_ops):
                            	self.training_updates  = training_opt.minimize(self.training_loss,var_list=self.variables)
			# ACCURACY
        	        self.accuracy      = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.prediction,1),tf.int32), self.y_),tf.float32))
		self.session.run(tf.global_variables_initializer())
	def _fit(self,X,y,indices,update_time=50):
		self.e    += 1
        	n_train    = X.shape[0]/self.batch_size
        	train_loss = []
        	for I in xrange(n_train):
			# SAMPLE THE BATCH
			if(self.batch_size<self.n_classes):
				here = [random.sample(k,1) for k in indices]
				here = [here[i] for i in permutation(self.n_classes)[:self.batch_size]]
			else:
				here = [random.sample(k,self.batch_size/self.n_classes) for k in indices]
			here = concatenate(here)
			# TRAIN OP
                        self.session.run(self.training_updates,feed_dict={self.x:X[here],self.y_:y[here],self.training:True,self.learning_rate:float32(self.lr)})
			# GET TRAIN LOSS
			if(I%update_time==0):
				if(self.distance_coeff==0):
                                	train_loss.append(self.session.run(self.crossentropy_loss,feed_dict={self.x:X[here],self.y_:y[here],self.training:False}))
				else:
                                        train_loss.append(self.session.run([self.crossentropy_loss,self.distance_loss],feed_dict={self.x:X[here],self.y_:y[here],self.training:False}))
                                print I,n_train,train_loss[-1]
        	return train_loss
        def fit(self,X,y,X_test,y_test,n_epochs=5):
		train_loss = []
		train_accu = []
		test_accu  = []
		self.e     = 0
                n_test     = X_test.shape[0]/self.batch_size
                indices    = [find(y==k) for k in xrange(self.n_classes)]
                print [len(i) for i in indices]
		for II in xrange(n_epochs):
                        if(II==90): self.lr/=5
                        elif(II==130): self.lr/=5
			print "epoch",II
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


class SpecialDNNClassifier(object):
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
			self.gamma_set     = tf.assign(self.layers[1].gamma,self.layers[1].gamma*1.2)
			self.masks          = [l.state for l in self.layers[1:-1]]
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            self.apply_updates = opt.minimize(self.loss,var_list=self.variables)
        	        self.accuracy      = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.prediction,1),tf.int32), self.y_),tf.float32))
		self.session.run(tf.global_variables_initializer())
	def _fit(self,X,y,indices,update_time=50):
		self.e    += 1
		self.session.run(self.gamma_set)
        	n_train    = X.shape[0]/self.batch_size
        	train_loss = []
                print self.session.run(self.layers[1].W_)
        	for I in xrange(n_train):
			if(self.batch_size<self.n_classes):
				here = [random.sample(k,1) for k in indices]
				here = [here[i] for i in permutation(self.n_classes)[:self.batch_size]]
			else:
				here = [random.sample(k,self.batch_size/self.n_classes) for k in indices]
			here = concatenate(here)
                        self.session.run(self.apply_updates,feed_dict={self.x:X[here],self.y_:y[here],self.training:True,self.learning_rate:float32(self.lr)})
			if(I%update_time==0):
                                train_loss.append(self.session.run(self.loss,feed_dict={self.x:X[here],self.y_:y[here],self.training:False}))
                                print I,n_train,train_loss[-1]
        	return train_loss
        def fit(self,X,y,X_test,y_test,n_epochs=5):
		train_loss = []
		train_accu = []
		test_accu  = []
		self.e     = 0
                n_test     = X_test.shape[0]/self.batch_size
                indices    = [find(y==k) for k in xrange(self.n_classes)]
                print [len(i) for i in indices]
		for II in xrange(n_epochs):
                        if(II==90): self.lr/=5
                        elif(II==130): self.lr/=5
			print "epoch",II
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
	def get_states(self,X):
                n     = X.shape[0]/self.batch_size
                masks = [[] for i in xrange(len(self.masks))]
                for j in xrange(n):
                    mask=self.session.run(self.masks,feed_dict={self.x:X[self.batch_size*j:self.batch_size*(j+1)],self.training:False})
		    for i in xrange(len(self.masks)):
			masks[i].append(mask[i])
                return [concatenate(m,axis=0) for m in masks]


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
                layers.append(ConvLayer(layers[-1],64,5,pad='SAME',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
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
                layers.append(ConvLayer(layers[-1],64,5,pad='SAME',training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
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













class smallDENSE:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],1024,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],1024,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers


class largeDENSE:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],4096,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],4096,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers

























class allCNN1:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],5,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
		# (5 28 28) : 3920
                layers.append(ConvLayer(layers[-1],7,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (7 24 24) : 4032
                layers.append(ConvLayer(layers[-1],9,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (9 22 22) : 4356
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 128
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers


class allCNN2:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],7,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
		# (7 28 28)  : 5488
                layers.append(ConvLayer(layers[-1],11,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (11 24 24) : 6336
                layers.append(ConvLayer(layers[-1],15,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (15 22 22) : 7260
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 128
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers


class allCNN3:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],12,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
		# (12 28 28) : 9408
                layers.append(ConvLayer(layers[-1],16,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (16 24 24) : 9216
                layers.append(ConvLayer(layers[-1],20,3,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (20 22 22) : 9680
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 128
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers

class allCNN4:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(ConvLayer(layers[-1],20,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
		# (20 28 28) : 15680
                layers.append(ConvLayer(layers[-1],32,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (32 24 24) : 18432
                layers.append(ConvLayer(layers[-1],48,5,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# (48 20 20) : 19200
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 2048
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
		# 256
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers





class allDENSE1:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],3920,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],4032,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],4356,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers

class allDENSE2:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],5488,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],6336,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],7260,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers


class allDENSE3:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],9408,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],9216,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],9680,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],128,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers

class allDENSE4:
        def __init__(self,bn=1,n_classes=10,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.bn          = bn
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.bias_option = bias_option
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(DenseLayer(layers[-1],15680,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,first=True))
                layers.append(DenseLayer(layers[-1],18432,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],19200,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],256,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=training,bn=self.bn,init_W=self.init_W,init_b=self.init_b,bias_option=self.bias_option,nonlinearity=False))
                self.layers = layers
                return self.layers






class SpecialDense:
        def __init__(self,n_classes=10,constraint='dt',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
                self.n_classes   = n_classes
                self.layers      = 0
                self.init_W      = init_W
                self.init_b      = init_b
                self.constraint  = constraint
        def get_layers(self,input_variable,input_shape,training):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(SpecialDenseLayer(layers[-1],16,constraint=self.constraint,training=training,first=True))
#                layers.append(SpecialDenseLayer(layers[-1],64,constraint=self.constraint,training=training,first=False))
#                layers.append(DenseLayer(layers[-1],6,nonlinearity='relu',training=training))
                layers.append(DenseLayer(layers[-1],self.n_classes,nonlinearity=None,training=training))
		self.layers = layers
                return self.layers












