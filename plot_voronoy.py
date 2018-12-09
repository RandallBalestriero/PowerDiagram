from pylab import *

eps=0.1

def VQ(x,W,ppi,sigma):
	print shape(dot(x,W.T))
	vq = argmax(dot(x,W.T)+log(ppi).reshape((1,-1))*sigma-0.5*(W*W).sum(1).reshape((1,-1)),axis=1)
	print unique(vq)
	return vq


def MVQ(x,W,b):
        print shape(dot(x,W.T))
        vq = argmax(dot(x,W.T)+b.reshape((1,-1)),axis=1)
        print unique(vq)
        return vq

def fVQ(x,W,ppi,sigma):
#	print shape(dot(x,W.T))
	H = dot(x,W.T)+log(ppi).reshape((1,-1))*sigma-0.5*(W*W).sum(1).reshape((1,-1))
	vq = argmax(H,axis=1)
#	print unique(vq)
	return H[range(len(vq)),vq]


def fMVQ(x,W,b):
#        print shape(dot(x,W.T))
	H  = dot(x,W.T)+b.reshape((1,-1))
        vq = argmax(H,axis=1)
#        print unique(vq)
        return H[range(len(vq)),vq]








def get_boundary(X):
	K=abs(X[:-1,:-1]-X[1:,:-1])+abs(X[:-1,:-1]-X[:-1,1:])+abs(X[1:,:-1]-X[:-1,1:])+abs(X[1:,1:]-X[:-1,:-1])#diff(X,axis=0))+abs(diff(X,axis=0))
	K=K+roll(K,1,1)+roll(K,0,1)
	return pad(K,[[1,0],[1,0]],'constant')>0 



def doplot(W,ppi,sigma):
	xx, yy = np.meshgrid(linspace(W[:,0].min()-eps, W[:,0].max()+eps, h),linspace(W[:,1].min()-eps,W[:,1].max()+eps, h))
	xxx=xx.flatten()
	yyy=yy.flatten()
	DD = asarray([xxx,yyy]).astype('float32').T
	vq = VQ(DD,W,ppi,sigma)
	vq=vq.reshape(xx.shape)
	contourf(xx, yy, 1-get_boundary(vq),cmap='gray')
	for w in W:
		plot(w[0],w[1],'ko')

def doplotm(W,b):
	xx, yy = np.meshgrid(linspace(W[:,0].min()-eps, W[:,0].max()+eps, h),linspace(W[:,1].min()-eps,W[:,1].max()+eps, h))
	xxx=xx.flatten()
	yyy=yy.flatten()
	DD = asarray([xxx,yyy]).astype('float32').T
	vq = MVQ(DD,W,b)
	vq=vq.reshape(xx.shape)
	contourf(xx, yy, 1-get_boundary(vq),cmap='gray')
	for w in W:
		plot(w[0],w[1],'ko')



def fdoplot(W,ppi,sigma):
	xx, yy = np.meshgrid(linspace(W[:,0].min()-eps, W[:,0].max()+eps, h),linspace(W[:,1].min()-eps,W[:,1].max()+eps, h))
	xxx=xx.flatten()
	yyy=yy.flatten()
	DD = asarray([xxx,yyy]).astype('float32').T
	vq = fVQ(DD,W,ppi,sigma)
	vq=vq.reshape(xx.shape)
	contourf(xx, yy, vq,20,cmap='jet',alpha=0.5)
	for w in W:
		plot(w[0],w[1],'ko')

def fdoplotm(W,b):
	xx, yy = np.meshgrid(linspace(W[:,0].min()-eps, W[:,0].max()+eps, h),linspace(W[:,1].min()-eps,W[:,1].max()+eps, h))
	xxx=xx.flatten()
	yyy=yy.flatten()
	DD = asarray([xxx,yyy]).astype('float32').T
	vq = fMVQ(DD,W,b)
	vq=vq.reshape(xx.shape)
	contourf(xx, yy, vq,20,cmap='jet',alpha=0.5)
	for w in W:
		plot(w[0],w[1],'ko')







h=100
K=16

W=randn(K,2)#asarray([[0.5,0.5],[0,0],[-0.5,0.5]])
ppi=ones(K)/K#asarray([0.33,0.33,0.33])



figure(figsize=(25,5))
for i in xrange(5):
	subplot(1,5,i+1)
	doplot(W,ppi,0+i*2)
        fdoplot(W,ppi,0+i*2)
	title(r'$\sigma='+str(i*2)+'$',fontsize=18)
	xticks([])
	yticks([])

tight_layout()
savefig('plotsimplepartition1.png')
close()

figure(figsize=(25,5))
for i in xrange(5):
        subplot(1,5,i+1)
	b=randn(K)/10
        doplotm(W,i*b)
        fdoplotm(W,i*b)
	xticks([])
	yticks([])
        title(r'$B \sim \mathcal{N}(0;'+str(i*0.1)+')$',fontsize=18)

tight_layout()
savefig('plotsimplepartition2.png')
close()


#
