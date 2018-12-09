from pylab import *
import matplotlib as mpl
import tensorflow as tf

label_size = 15
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size



def trainit(X_,Y_,n_epochs=22000,cut=0,K=15):
	train_losses = dict()
	preds = dict()
	
	bbb = linspace(0.5,0.99,5)
	for bb in bbb:
	        beta_            = float32(bb)
		train_losses[str(cut)+'_'+str(bb)] = []
		preds[str(cut)+'_'+str(bb)]        = []
		for i in xrange(10):
			DN = DNN1(bs,K,activ,cut=cut)
			train_losses[str(cut)+'_'+str(bb)].append(DN.train(X_,Y_,beta_,n_epochs))
			preds[str(cut)+'_'+str(bb)].append(DN.pred(X_,beta_))
		train_losses[str(cut)+'_'+str(bb)]=asarray(train_losses[str(cut)+'_'+str(bb)])
	        preds[str(cut)+'_'+str(bb)]=asarray(preds[str(cut)+'_'+str(bb)])
	        beta_            = float32(bb)
	bb=str(cut)+'_trainlocal'
	train_losses[bb] = []
	preds[bb]        = []
	for i in xrange(10):
		DN = DNN1(bs,K,activ,learn_beta=1,global_beta=0,cut=cut)
	        train_losses[bb].append(DN.train(X_,Y_,beta_,n_epochs))
	        preds[bb].append(DN.pred(X_,beta_))
	train_losses[bb]=asarray(train_losses[bb])
	preds[bb]=asarray(preds[bb])
	
	bb=str(cut)+'_trainglobal'
	train_losses[bb] = []
	preds[bb]        = []
	for i in xrange(10):
	        DN = DNN1(bs,K,activ,learn_beta=1,global_beta=1,cut=cut)
	        train_losses[bb].append(DN.train(X_,Y_,beta_,n_epochs))
	        preds[bb].append(DN.pred(X_,beta_))
	train_losses[bb]=asarray(train_losses[bb])
	preds[bb]=asarray(preds[bb])
	return list(bbb)+[str(cut)+'_trainlocal',str(cut)+'_trainglobal'],train_losses,preds






def activatioon(x,nonlinearity,beta,cut=0):
	coeff=beta/(1-beta)
	if(nonlinearity=='relu'):
		if(cut):
			mask = tf.nn.sigmoid(coeff*tf.stop_gradient(x))
		else:
                        mask = tf.nn.sigmoid(coeff*x)
        	output = mask*x
        elif(nonlinearity=='lrelu'):
		if(cut):
                        coeff1 = tf.exp(tf.clip_by_value(coeff*0.01*tf.stop_gradient(x),-10,10))
                        coeff2 = tf.exp(tf.clip_by_value(coeff*tf.stop_gradient(x),-10,10))
		else:
	                coeff1 = tf.exp(tf.clip_by_value(coeff*0.01*x,-10,10))
	                coeff2 = tf.exp(tf.clip_by_value(coeff*x,-10,10))
                mask1  = coeff1/(coeff1+coeff2)
                mask2  = coeff2/(coeff1+coeff2)
	        output = x*(mask1*0.01+mask2)
        elif(nonlinearity=='abs'):
                if(cut):
                        coeff1 = tf.exp(-coeff*tf.stop_gradient(x))
                        coeff2 = tf.exp(coeff*tf.stop_gradient(x))
		else:
	        	coeff1 = tf.exp(-coeff*x)
                	coeff2 = tf.exp(coeff*x)
                output = x*(-coeff1+coeff2)/(coeff1+coeff2)
	return output






bs=1000
activ= 'relu'

class DNN1:
	def __init__(self,bs,K,activ,learn_beta=0,global_beta=0,cut=0):
		with tf.device('/device:GPU:0'):
			self.X = tf.placeholder(tf.float32, shape = [bs])
			self.Y = tf.placeholder(tf.float32, shape = [bs])
			if(learn_beta==0):
				self.beta = tf.placeholder(tf.float32)
				self.beta_= self.beta
			else:
				if(global_beta):
	                                self.beta = tf.Variable(tf.zeros(1),trainable=True)
                                        self.beta_= tf.sigmoid(self.beta)
				else:
                                	self.beta = tf.Variable(tf.zeros((1,K)),trainable=True)
                                        self.beta_= tf.sigmoid(self.beta)
			self.W1   = tf.Variable(tf.random_normal((1,K)),trainable=True)
			self.B1   = tf.Variable(tf.zeros((1,K)),trainable=True)
			self.W2   = tf.Variable(tf.random_normal((K,1)),trainable=True)
			self.B2   = tf.Variable(tf.zeros((1,1)),trainable=True)
			#
			self.h1   = activatioon(tf.matmul(tf.expand_dims(self.X,-1),self.W1)+self.B1,activ,self.beta_,cut=cut)
			self.h2   = (tf.matmul(self.h1,self.W2)+self.B2)[:,0]
			self.error= tf.reduce_sum(tf.pow(self.h2-self.Y,2))
			self.opt  = tf.train.AdamOptimizer(0.1).minimize(self.error)
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.learn_beta=learn_beta
		self.global_beta=global_beta
	def train(self,X_,Y_,beta_,n):
		error = []
                for _ in xrange(n):
			if(self.learn_beta==0):
	                	self.sess.run(self.opt, feed_dict = {self.X : X_,self.Y:Y_,self.beta:beta_}) #initial guess = 5.0
	                        error.append(self.sess.run(self.error,feed_dict = {self.X : X_,self.Y:Y_,self.beta:beta_}))
			else:
                                self.sess.run(self.opt, feed_dict = {self.X : X_,self.Y:Y_}) #initial guess = 5.0
                                error.append(self.sess.run(self.error,feed_dict = {self.X : X_,self.Y:Y_}))
                        print error[-1]
		return array(error)
	def pred(self,X_,beta_):
		if(self.learn_beta==0):
			return self.sess.run(self.h2,feed_dict = {self.X : X_,self.beta:beta_})
		else:
                        return self.sess.run(self.h2,feed_dict = {self.X : X_})




if(int(sys.argv[-1])==0):
	X_=linspace(-6,6,bs).astype('float32')
	Y_=cos(X_*2).astype('float32')

if(int(sys.argv[-1])==1):
        X_=linspace(-6,6,bs).astype('float32')
        Y_=cos(X_**1.105).astype('float32')

if(int(sys.argv[-1])==2):
        X_=linspace(-6,6,bs).astype('float32')
        Y_=exp(-X_**2).astype('float32')



results = dict()
for K in [15,25,35]:
	for cut in [0,1]:
		results[str(K)+'_'+str(cut)]=trainit(X_,Y_,K=K,cut=cut)


f=file('approx1D.pkl','rb')
cPickle.dump(results,f)
f.close()

#figure()
#for bb,ii in zip(tags,range(len(tags))):
#	plot([ii],train_losses[bb].min(1).mean(),'or')
#	plot([ii],train_losses[bb].min(1).mean()+train_losses[bb].min(1).std(),'xk')
#	plot([ii],train_losses[bb].min(1).mean()-train_losses[bb].min(1).std(),'xr')
#	plot([ii,ii],[train_losses[bb].min(1).mean()-train_losses[bb].min(1).std(),train_losses[bb].min(1).mean()+train_losses[bb].min(1).std()],'k')
#	print tags
#	print len(tags)
#	xticks(range(len(tags)),array(tags).astype('str'))
#
#
##figure()
#plot(X_,Y_,'r')
#plot(X_,preds[tags[0]][0],'g')
#plot(X_,preds[tags[-3]][0],'b')
#plot(X_,preds[tags[-1]][0],'k')
#
#show()
#









