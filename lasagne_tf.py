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


class InputLayer:
    def __init__(self,input_shape,x):
	self.output = x
	self.output_shape = input_shape

class DenseLayer:
    def __init__(self,incoming,n_output,bias_option='unconstrained',init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.),bn=True,training=None,nonlinearity=True):
        # bias_option : {unconstrained,constrained,zero}
	if(len(incoming.output_shape)>2): reshape_input = tf.layers.flatten(incoming.output)
	else:                             reshape_input = incoming.output
        in_dim = prod(incoming.output_shape[1:])
        self.W = tf.Variable(init_W((in_dim,n_output)),name='W_dense',trainable=True)
	self.output_shape = (incoming.output_shape[0],n_output)
        if(bn): renorm_input = tf.layers.batch_normalization(reshape_input,training=training)
        else:           renorm_input = reshape_input
        if(bias_option=='unconstrained'):
            self.b = tf.Variable(init_b((1,n_output)),name='b_dense',trainable=True)
	elif(bias_option=='constrained'):
	    self.b = -tf.reduce_sum(tf.square(self.W),axis=0,keep_dims=True)*0.5
	else:
            self.b = tf.zeros((1,n_output))
        if(nonlinearity):self.output = tf.nn.relu(tf.matmul(renorm_input,self.W)+self.b)
        else            :self.output = tf.matmul(renorm_input,self.W)+self.b


class ConvLayer:
    def __init__(self,incoming,n_filters,filter_shape,bias_option='unconstrained',training=None,bn=True,init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.),pad='VALID'):
        if(bn): renorm_input = tf.layers.batch_normalization(incoming.output,training=training)
        else:           renorm_input = incoming.output
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
        else:
            self.b = tf.zeros((1,1,1,n_filters))
	self.output = tf.nn.relu(tf.nn.conv2d(padded_input,self.W,strides=[1,1,1,1],padding='VALID')+self.b)




class GlobalPoolLayer:
    def __init__(self,incoming,pool_type='AVG',global_beta=1):
        self.output = tf.reduce_mean(incoming.output,[1,2],keep_dims=True)
        self.output_shape = [incoming.output_shape[0],1,1,incoming.output_shape[3]]





