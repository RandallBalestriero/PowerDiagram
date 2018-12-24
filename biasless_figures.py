from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from numpy import gradient as npgrad





class Voronoi:
    def __init__(self,mu,r):
        self.mu = mu
        self.b  = -(mu**2).sum(1)+r
        self.r  = r
        self.hyperplanes = [Hyperplane(2*mu,b) for mu,b in zip(self.mu,self.b)]
    def cluster(self,x):
        distances = ((x[:,newaxis,:]-self.mu[newaxis,:,:])**2).sum(2)-self.r[newaxis,:]
        return distances.argmin(1).astype('float32')
    def project(self,x):
        return stack([hyperplane.project(x) for hyperplane in self.hyperplanes])


class Hyperplane:
    def __init__(self,slope,bias):
        self.slope = slope
        self.bias  = bias
    def project(self,x):
        return dot(x,self.slope)+self.bias


class Space:
    def __init__(self,MIN_X,MAX_X,MIN_Y,MAX_Y,N):
        self.x,self.y = meshgrid(linspace(MIN_X,MAX_X,N),linspace(MIN_Y,MAX_Y,N))
        self.X        = concatenate([self.x.reshape((-1,1)),self.y.reshape((-1,1))],axis=1)
 

def plot3(w,b,name='noname.png'):
        K = len(w)
        V = Voronoi(w,b)
        C = V.cluster(space.X).reshape((N,N))
        C/=C.max()
        #PLOT THE VQ
        imshow(C,aspect='auto',interpolation='nearest',cmap='RdYlGn',extent=[MIN,MAX,MIN,MAX])#,levels = levels*0.000001,cmap='Greys')
#        for k in range(K):
#            ax.plot([w[k,0]],[w[k,1]],'ok',ms=6)
#            if(V.r[k]>0):
#                theta = linspace(-1,1,100)
#                for tm,t in zip(theta[:-1],theta[1:]):
#                    ax.plot([w[k,0]+sqrt(V.r[k])*tm,w[k,0]+sqrt(V.r[k])*t],[w[k,1]-sqrt(V.r[k]-V.r[k]*tm**2),w[k,1]-sqrt(V.r[k]-V.r[k]*t**2)],'--k',alpha=0.5)
#                    ax.plot([w[k,0]+sqrt(V.r[k])*tm,w[k,0]+sqrt(V.r[k])*t],[w[k,1]+sqrt(V.r[k]-V.r[k]*tm**2),w[k,1]+sqrt(V.r[k]-V.r[k]*t**2)],'--k',alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        tight_layout()
        if(name is not None):
                savefig(name)




N        = 300
K        = 3
MIN,MAX  = -2.,2.
DISTANCE = MAX-MIN
START    = -3.1
offset   = 2.6
space    = Space(MIN,MAX,MIN,MAX,N)

seed(10)
w = randn(3,2)
K = len(w)
V = Voronoi(w,b)
C = V.cluster(space.X).reshape((N,N))
C/=C.max()


# WITH BIAS
#figure(figsize=(10,12))#,dpi=20)
#ax = subplot(111,projection='3d')
#plot(w,array([0.1,0.05,0.3,0.2,0.3]),'VQ_bias.png')
#close()










# CASE 1 : arbitrary








