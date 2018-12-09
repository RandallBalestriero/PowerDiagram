execfile('lasagne_tf.py')
execfile('utils.py')
import random

def onehot(n,k):
        z=zeros(n,dtype='float32')
        z[k]=1
        return z




	



class DNNClassifier(object):
	def __init__(self,input_shape,model_class,lr=0.0001,optimizer = tf.train.AdamOptimizer,l2=0,learn_beta=0,e=1):
		#setting = {base,pretrainlinear,}
		tf.reset_default_graph()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.n_classes=  model_class.n_classes
		config.log_device_placement=True
		self.session = tf.Session(config=config)
		self.batch_size = input_shape[0]
		self.lr = lr
		opt = adam(lr)
		with tf.device('/device:GPU:0'):
			self.learning_rate = tf.placeholder(tf.float32,name='learning_rate')
			optimizer          = optimizer(self.learning_rate)
        		self.x             = tf.placeholder(tf.float32, shape=input_shape,name='x')
        	        self.y_            = tf.placeholder(tf.int32, shape=[input_shape[0]],name='y')
#                        self.gamma         = tf.placeholder(tf.float32,name='gamma')
        	        self.test_phase    = tf.placeholder(tf.bool,name='phase')
        	        self.prediction,self.layers        = model_class.get_layers(self.x,input_shape,test=self.test_phase)
                        self.templates     = tf.stack([tf.gradients(self.prediction,self.x,tf.one_hot(tf.fill([input_shape[0]],c),self.n_classes))[0] for c in xrange(self.n_classes)])
			count_number_of_params()
#			ww = tf.reshape(myortho(tf.trainable_variables()[4],(3,3,32,64)),(-1,64))
			self.W_ = self.layers[1].W_
                        self.crossentropy_loss = tf.reduce_mean(categorical_crossentropy(self.prediction,self.y_))
			self.loss = self.crossentropy_loss
        	        self.variables     = tf.trainable_variables()
			if(learn_beta):
				self.variables += tf.get_collection('beta')
        	        print "VARIABLES",self.variables
                        self.apply_updates      = opt.apply(self.loss,self.variables)#.values()+tf.get_collection(tf.GraphKeys.UPDATE_OPS))#+updates.values()
#                return tf.group(*final)
        	        self.accuracy      = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(self.prediction,1),tf.int32), self.y_),tf.float32))
		self.session.run(tf.global_variables_initializer())
	def _fit(self,X,y,indices,update_time=50):
		self.e+=1
        	n_train    = X.shape[0]/self.batch_size
        	train_loss = []
        	for i in xrange(n_train):
			if(self.batch_size<self.n_classes):
				here = [random.sample(k,1) for k in indices]
				here = [here[i] for i in permutation(self.n_classes)[:self.batch_size]]
			else:
				here = [random.sample(k,self.batch_size/self.n_classes) for k in indices]
			here = concatenate(here)
                        self.session.run(self.apply_updates,feed_dict={self.x:X[here],self.y_:y[here],self.test_phase:True,self.learning_rate:float32(self.lr)})#float32(self.lr/sqrt(self.e))})
#			print self.session.run(self.corr)
			if(i%update_time==0):
                                train_loss.append(self.session.run(self.crossentropy_loss,feed_dict={self.x:X[here],self.y_:y[here],self.test_phase:True}))
#                                train_loss.append(self.session.run([self.return_loss,self.crossentropy_loss],feed_dict={self.x:X[here],self.y_:y[here],self.test_phase:True}))
#                        	train_loss.append(self.session.run(self.extra_loss,feed_dict={self.gamma:self.e*1}))
                        if(i%100 ==0):
                            print i,n_train,train_loss[-1]
        	return train_loss
        def fit(self,X,y,X_test,y_test,n_epochs=5,return_train_accu=0):
		if(n_epochs==0):
			return [0],[0],[]
		train_loss = []
		train_accu = []
		test_loss  = []
		self.e     = 0
		W          = []
                n_test     = X_test.shape[0]/self.batch_size
                indices    = [find(y==k) for k in xrange(self.n_classes)]
		for i in xrange(n_epochs):
			print "epoch",i
			train_loss.append(self._fit(X,y,indices))
			# NOW COMPUTE TEST SET ACCURACY
                	acc1 = 0.0
			W.append(self.session.run(self.W_))
                	for j in xrange(n_test):
                	        acc1+=self.session.run(self.accuracy,feed_dict={self.x:X_test[self.batch_size*j:self.batch_size*(j+1)],
						self.y_:y_test[self.batch_size*j:self.batch_size*(j+1)],self.test_phase:False})
                	test_loss.append(acc1/n_test)
			if(1):#return_train_accu):
		                n_train    = X.shape[0]/self.batch_size
	                        acc1 = 0.0
	                        for j in xrange(n_train):
	                                acc1+=self.session.run(self.accuracy,feed_dict={self.x:X[self.batch_size*j:self.batch_size*(j+1)],
	                                                self.y_:y[self.batch_size*j:self.batch_size*(j+1)],self.test_phase:False})
	                        train_accu.append(acc1/n_train)
				print train_accu[-1]
                	print 'test accu',test_loss[-1]
		if(return_train_accu):
	                return concatenate(train_loss),train_accu,test_loss,W
        	return concatenate(train_loss),test_loss,W
	def predict(self,X):
		n = X.shape[0]/self.batch_size
		preds = []
		for j in xrange(n):
                    preds.append(self.session.run(self.prediction,feed_dict={self.x:X[self.batch_size*j:self.batch_size*(j+1)],self.test_phase:False}))
                return concatenate(preds,axis=0)
        def get_templates(self,X):
                templates = []
                n_batch = X.shape[0]/self.batch_size
                for i in xrange(n_batch):
                        t=self.session.run(self.templates,feed_dict={self.x:X[i*self.batch_size:(i+1)*self.batch_size].astype('float32'),self.test_phase:False})
                        if(len(X.shape)>2):
                                templates.append(transpose(t,[1,0,2,3,4]))
                        else:
                                templates.append(transpose(t,[1,0,2]))
                return concatenate(templates,axis=0)



class largeCNN:
        def __init__(self,bn=1,n_classes=10,global_beta=1,pool_type='BETA',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.),use_beta=1,nonlinearity=tf.nn.relu,centered=0,ortho=0):
                self.nonlinearity = nonlinearity
		self.ortho = ortho
                self.bn          = bn
		self.centered=0
                self.n_classes   = n_classes
                self.global_beta = global_beta
                self.pool_type   = pool_type
                self.layers = 0
                self.use_beta    = use_beta
                self.init_W = init_W
                self.init_b = init_b
        def get_layers(self,input_variable,input_shape,test):
                layers = [InputLayer(input_shape,input_variable)]
                layers.append(Conv2DLayer(layers[-1],64,5,pad='same',test=test,bn=self.bn,first=True,centered=self.centered,ortho=self.ortho))
		layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
                layers.append(Conv2DLayer(layers[-1],96,3,pad='full',test=test,bn=self.bn,centered=self.centered,ortho=self.ortho))
                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
                layers.append(Conv2DLayer(layers[-1],96,3,pad='full',test=test,bn=self.bn,centered=self.centered,ortho=self.ortho))
                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
                layers.append(Pool2DLayer(layers[-1],2,pool_type=self.pool_type))
                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn,centered=self.centered,ortho=self.ortho))
                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
                layers.append(Conv2DLayer(layers[-1],192,3,pad='full',test=test,bn=self.bn,ortho=self.ortho))
                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn,centered=self.centered,ortho=self.ortho))
                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
                layers.append(Pool2DLayer(layers[-1],2,pool_type=self.pool_type))
                layers.append(Conv2DLayer(layers[-1],192,3,test=test,bn=self.bn,centered=self.centered,ortho=self.ortho))
                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
                layers.append(Conv2DLayer(layers[-1],192,1,test=test,centered=self.centered,ortho=self.ortho))
                layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
                layers.append(GlobalPoolLayer(layers[-1]))
                layers.append(DenseLayer(layers[-1],self.n_classes,training=test,bn=0,ortho=0))
		self.layers = layers
                return self.layers[-1].output,self.layers



class DenseCNN:
        def __init__(self,bn=1,n_classes=10,global_beta=1,pool_type='BETA',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.),use_beta=1,nonlinearity=tf.nn.relu,centered=0,ortho=0):
                self.nonlinearity = nonlinearity
                self.bn          = bn
		self.centered=centered
                self.n_classes   = n_classes
		self.global_beta = global_beta
		self.ortho = ortho
		self.pool_type   = pool_type
		self.layers = 0
		self.use_beta    = use_beta
		self.init_W = init_W
		self.init_b = init_b
        def get_layers(self,input_variable,input_shape,test):
		if(self.layers==0):
			extra_layers = []
	                layers = [InputLayer(input_shape,input_variable)]
	                layers.append(Conv2DLayer(layers[-1],32,5,test=test,bn=self.bn,init_W=self.init_W,init_b=self.init_b,first=True,centered=self.centered,ortho=self.ortho))
                        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
	                layers.append(Pool2DLayer(layers[-1],2,pool_type=self.pool_type))
	                layers.append(Conv2DLayer(layers[-1],64,3,test=test,bn=self.bn,init_W=self.init_W,init_b=self.init_b,centered=self.centered,ortho=self.ortho))
                        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
	                layers.append(Pool2DLayer(layers[-1],2,pool_type=self.pool_type))
	                layers.append(Conv2DLayer(layers[-1],128,3,test=test,bn=self.bn,init_W=self.init_W,init_b=self.init_b,centered=self.centered,ortho=self.ortho))
                        layers.append(NonlinearityLayer(layers[-1],nonlinearity=self.nonlinearity,use_beta=self.use_beta,global_beta=self.global_beta,training=test,bn=self.bn))
			layers.append(GlobalPoolLayer(layers[-1]))
                        layers.append(DenseLayer(layers[-1],self.n_classes,training=test,bn=0,init_W=self.init_W,init_b=self.init_b,ortho=0))
			self.layers = layers
			self.extra_layers = layers[1::2]
	                return self.layers[-1].output,self.layers
		else:
			return self.layers[-1].output,self.layers





