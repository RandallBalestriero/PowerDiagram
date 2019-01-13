import tensorflow as tf

############################################################################################
#
# OPTIMIZER and LOSSES
#
#
############################################################################################


def categorical_crossentropy(logits, labels):
	labels = tf.cast(labels, tf.int32)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
	return cross_entropy



def count_number_of_params():
	print np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


def cosine_distance(x1, x2,axis):
    x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1),axis=axis))#+0.00000000000000001
    x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2),axis=axis))#+0.00000000000000001
    return 1-tf.reduce_sum(x1*x2,axis=axis)/(x1_norm*x2_norm)


###########################################################################################
#
#
#		Layers
#
#
###########################################################################################
class Pool2DLayer:
    def __init__(self,incoming,window,pool_type='MAX'):
        self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]/window,incoming.output_shape[2]/window,incoming.output_shape[3])
	self.output = tf.nn.pool(incoming.output,(window,window),pool_type,padding='VALID',strides=(window,window))
        self.distance_loss       = float32(0)
        self.reconstruction_loss = float32(0)




class InputLayer:
    def __init__(self,input_shape,x):
	self.output = x
	self.output_shape = input_shape



class SpecialDenseLayer:
    def __init__(self,incoming,n_output,constraint='none',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.random_normal,training=None,first=False):
        # bias_option : {unconstrained,constrained,zero}
	if(len(incoming.output_shape)>2): reshape_input = tf.layers.flatten(incoming.output)
	else:                             reshape_input = incoming.output
	if(first==False):	renorm_input = tf.layers.batch_normalization(reshape_input,training=training)
	else: renorm_input = reshape_input
        in_dim      = prod(incoming.output_shape[1:])
	self.gamma  = tf.Variable(ones(1,float32),trainable=False)
	if(constraint=='none'):
        	self.W      = tf.Variable(init_W((in_dim,n_output)),name='W_dense',trainable=True)
		self.W_     = self.W
	elif(constraint=='dt'):
                self.W      = tf.Variable(init_W((in_dim,n_output)),name='W_dense',trainable=True)
		self.alpha  = tf.Variable(randn(1,n_output).astype('float32'),trainable=True)
		self.W_     = self.alpha*tf.nn.softmax(self.gamma*self.W,axis=0)
	elif(constraint=='diag_dt'):
                self.W      = tf.Variable(init_W((2,n_output)),name='W_dense',trainable=True)
                self.sign   = tf.Variable(ones((2,n_output),float32),trainable=True)
                self.alpha  = tf.Variable(ones((1,n_output),float32),trainable=True)
                self.W_     = self.alpha*tf.nn.tanh(self.gamma*self.sign)*tf.nn.sigmoid(self.gamma*self.W)
	elif(constraint=='diag'):
                self.sign   = tf.Variable(randn(1,n_output).astype('float32'),trainable=True)
                self.alpha  = tf.Variable(randn(1,n_output).astype('float32'),trainable=True)
                self.W_     = tf.concat([tf.nn.tanh(self.gamma*self.sign)*self.alpha,self.alpha],axis=0)
	elif(constraint=='rot'):
                self.sign   = tf.Variable(randn(1,n_output).astype('float32'),trainable=True)
                self.alpha  = tf.Variable(randn(1,n_output).astype('float32'),trainable=True)
		self.theta  = tf.Variable(randn(1,).astype('float32'),trainable=True)
		self.rot_mat = tf.reshape(tf.concat([tf.cos(self.theta),-tf.sin(self.theta),tf.sin(self.theta),tf.cos(self.theta)],0),[2,2])
                self.W_     = tf.matmul(self.rot_mat,tf.concat([tf.nn.tanh(self.gamma*self.sign)*self.alpha,self.alpha],axis=0))
	self.output_shape = (incoming.output_shape[0],n_output)
        self.b      = tf.Variable(init_b((1,n_output)),name='b_dense',trainable=True)
	output      = tf.matmul(renorm_input,self.W_)+self.b
	self.state  = tf.greater(output,0)
	self.output = tf.cast(self.state,tf.float32)*output




class DenseLayer:
    def __init__(self,incoming,n_output,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.),bn=True,training=None,nonlinearity='relu',first=False):
        # bias_option : {unconstrained,constrained,zero}
	if(len(incoming.output_shape)>2): reshape_input = tf.layers.flatten(incoming.output)
	else:                             reshape_input = incoming.output
        in_dim = prod(incoming.output_shape[1:])
        self.W = tf.Variable(init_W((in_dim,n_output)),name='W_dense',trainable=True)
	print (in_dim,n_output)
	self.output_shape = (incoming.output_shape[0],n_output)
	print reshape_input
	if(first==False):    renorm_input = tf.layers.batch_normalization(reshape_input,training=training,center=bn)
	else: renorm_input = reshape_input
        if(bias_option=='unconstrained'):
            self.b = tf.Variable(init_b((1,n_output)),name='b_dense',trainable=True)
	elif(bias_option=='constrained'):
	    self.b = -tf.reduce_sum(tf.square(self.W),axis=0,keep_dims=True)*0.5
	elif(bias_option=='explicit'):
	    self.radius = tf.Variable(init_b((1,n_output)),name='b_dense',trainable=True)
            self.b = -tf.reduce_sum(tf.square(self.W),axis=0,keep_dims=True)*0.5+tf.abs(self.radius)
	else:
            self.b = tf.zeros((1,n_output))
	output = tf.matmul(renorm_input,self.W)+self.b
        if(nonlinearity=='relu'):
		self.state  = tf.greater(output,0)
		self.output = tf.cast(self.state,tf.float32)*output
	elif(nonlinearity=='lrelu'):
                self.state  = tf.greater(output,0)
                self.output = tf.cast(self.state,tf.float32)*output+0.01*(1-tf.cast(self.state,tf.float32))*output
	elif(nonlinearity=='abs'):
                self.state  = tf.greater(output,0)
                self.output = tf.cast(self.state,tf.float32)*output-(1-tf.cast(self.state,tf.float32))*output
        else:	self.output = output
	self.distance_loss       = tf.reduce_min(2*tf.log(tf.abs(output)+0.0001)-tf.log(tf.reduce_sum(tf.square(self.W),[0],keepdims=True)+0.0001),3)#tf.reduce_min(tf.abs(output)/tf.sqrt(tf.reduce_sum(tf.square(self.W),0,keepdims=True)+0.0001),1)
	reconstruction           = tf.gradients(self.output,renorm_input,self.output)[0]
	self.reconstruction_loss = cosine_distance(reconstruction,renorm_input,axis=[1])






class ConvLayer:
    def __init__(self,incoming,n_filters,filter_shape,bias_option='unconstrained',training=None,bn=True,init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.),pad='VALID',first=False,nonlinearity='relu'):
	if(first): renorm_input = incoming.output
        else:	   renorm_input = tf.layers.batch_normalization(incoming.output,training=training,center=bn)
        if(pad=='VALID'):
            padded_input      = renorm_input
            self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]-filter_shape+1),(incoming.output_shape[1]-filter_shape+1),n_filters)
        elif(pad=='SAME'):
            assert(filter_shape%2 ==1)
            p                 = (filter_shape-1)/2
            padded_input      = tf.pad(renorm_input,[[0,0],[p,p],[p,p],[0,0]],mode='constant')
            self.output_shape = (incoming.output_shape[0],incoming.output_shape[1],incoming.output_shape[2],n_filters)
        elif(pad=='FULL'):
            p                 = filter_shape-1
            padded_input      = tf.pad(renorm_input,[[0,0],[p,p],[p,p],[0,0]],mode='constant')
            self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]+filter_shape-1),(incoming.output_shape[1]+filter_shape-1),n_filters)
        self.W     = tf.Variable(init_W((filter_shape,filter_shape,incoming.output_shape[3],n_filters)),name='W_conv2d',trainable=True)
        if(bias_option=='unconstrained'):
	    self.b = tf.Variable(init_b((1,1,1,n_filters)),name='b_conv',trainable=True)
        elif(bias_option=='constrained'):
            self.b = -tf.reduce_sum(tf.square(self.W),axis=[0,1,2],keepdims=True)*0.5
	elif(bias_option=='explicit'):
	    self.radius = tf.Variable(init_b((1,1,1,n_filters)),name='b_conv',trainable=True)
            self.b = -tf.reduce_sum(tf.square(self.W),axis=[0,1,2],keepdims=True)*0.5+tf.abs(self.radius)
        else:
            self.b = tf.zeros((1,1,1,n_filters))
	output     = tf.nn.conv2d(padded_input,self.W,strides=[1,1,1,1],padding='VALID')+self.b
	if(nonlinearity=='relu'):
		self.state  = tf.greater(output,0)
		self.output = tf.cast(self.state,tf.float32)*output
	elif(nonlinearity=='lrelu'):
                self.state  = tf.greater(output,0)
                self.output = tf.cast(self.state,tf.float32)*output+0.01*(1-tf.cast(self.state,tf.float32))*output
	elif(nonlinearity=='abs'):
                self.state  = tf.greater(output,0)
                self.output = tf.cast(self.state,tf.float32)*output-(1-tf.cast(self.state,tf.float32))*output
	else:	self.output = output
        self.distance_loss       = tf.reduce_min(2*tf.log(tf.abs(output)+0.0001)-tf.log(tf.reduce_sum(tf.square(self.W),[0,1,2],keepdims=True)+0.0001),3)#tf.reduce_min(tf.abs(output)/tf.sqrt(tf.reduce_sum(tf.square(self.W),[0,1,2],keepdims=True)+0.0001),3)
        reconstruction           = tf.gradients(self.output,renorm_input,self.output)[0]
        self.reconstruction_loss = cosine_distance(reconstruction,renorm_input,axis=[1,2,3])



class GlobalPoolLayer:
    def __init__(self,incoming,pool_type='AVG',global_beta=1):
        self.output = tf.reduce_mean(incoming.output,[1,2],keep_dims=True)
        self.output_shape = [incoming.output_shape[0],1,1,incoming.output_shape[3]]
	self.reconstruction_loss = float32(0)
        self.distance_loss       = float32(0)



