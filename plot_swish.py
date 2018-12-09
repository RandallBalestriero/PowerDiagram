from pylab import *
import matplotlib as mpl
label_size = 19
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size




def smoothrelu(x,beta):
        coeff = beta/(1-beta)
	return log(1+exp(x*coeff))/coeff

def smoothlrelu(x,beta):
        coeff = beta/(1-beta)
        return log(exp(-0.01*coeff*x)+exp(x*coeff))/coeff

def smoothabs(x,beta):
        coeff = beta/(1-beta)
        return log(exp(-coeff*x)+exp(x*coeff))/coeff


def maso(W,b,x):
	return (W.reshape((1,-1))*x.reshape((-1,1))+b.reshape((1,-1))).max(1)

def sigmoid(x):
	return 1/(1+exp(-x))

def dsigmoid(x):
        return sigmoid(x)*(1-sigmoid(x))#1/(1+exp(-x))


def swish(x,beta):
        coeff = beta/(1-beta)
	return sigmoid(coeff*x)*x

def abs(x,beta):
	coeff = beta/(1-beta)
	return (-exp(-coeff*x)+exp(coeff*x))/(exp(-coeff*x)+exp(coeff*x))*x


def lrelu(x,beta):
        coeff = beta/(1-beta)
        return (0.01*exp(0.01*coeff*x)+exp(coeff*x))/(exp(0.01*coeff*x)+exp(coeff*x))*x


def dswish(x,beta):
        coeff = beta/(1-beta)
	return sigmoid(coeff*x)

def dabs(x,beta):
	coeff = beta/(1-beta)
	return (-exp(-coeff*x)+exp(coeff*x))/(exp(-coeff*x)+exp(coeff*x))


def dlrelu(x,beta):
        coeff = beta/(1-beta)
        return (0.01*exp(0.01*coeff*x)+exp(coeff*x))/(exp(0.01*coeff*x)+exp(coeff*x))

def ddswish(x,beta):
        coeff = beta/(1-beta)
        return dsigmoid(coeff*x)*x+sigmoid(coeff*x)


def gg(t,coeff,alpha):
	p1 = (alpha*exp(alpha*coeff*t)+exp(coeff*t))/(exp(alpha*coeff*t)+exp(coeff*t))
	p2 = ((alpha**2*exp(alpha*coeff*t)+exp(coeff*t))*(exp(coeff*alpha*t)+exp(coeff*t))+(alpha*exp(alpha*coeff*t)+exp(coeff*t))**2)/(exp(alpha*coeff*t)+exp(coeff*t))**2
	return t*p2+p1

def ddabs(x,beta):
        coeff = beta/(1-beta)
        return gg(x,coeff,-1)


def ddlrelu(x,beta):
        coeff = beta/(1-beta)
        return gg(x,coeff,0.01)








x=linspace(-3,3,1000)

fs = 1.5

fig=figure(figsize=(18,5))

subplot(131)
for b in linspace(0.1,0.8,10):
	plot(x,swish(x,b),color=(1-b,0,b),linewidth=fs)
        plot(x,smoothrelu(x,1-b),color=(b,0,1-b),linewidth=fs)

plot(x,swish(x,0.5),'--k',linewidth=fs*2)
plot(x,smoothrelu(x,0.5),'--k',linewidth=fs*2)
plot(x,x*(x>0),color='k',linewidth=fs*2)
grid('on')
xlim([-3,3])
title('ReLU',fontsize=18)


subplot(132)
for b in linspace(0.1,0.8,10):
        plot(x,lrelu(x,b),color=(1-b,0,b),linewidth=fs)
        plot(x,smoothlrelu(x,1-b),color=(b,0,1-b),linewidth=fs)

plot(x,x*(x>0)+0.01*x*(x<=0),color='k',linewidth=fs*2)
plot(x,lrelu(x,0.5),'--k',linewidth=fs*2)
plot(x,smoothlrelu(x,0.5),'--k',linewidth=fs*2)
grid('on')
xlim([-3,3])
title('leaky-ReLU',fontsize=18)


subplot(133)
for b in linspace(0.1,0.8,10):
        plot(x,abs(x,b),color=(1-b,0,b),linewidth=fs)
        plot(x,smoothabs(x,1-b),color=(b,0,1-b),linewidth=fs)

plot(x,x*(x>0)-1*x*(x<=0),color='k',linewidth=fs*2)
plot(x,abs(x,0.5),'--k',linewidth=fs*2)
plot(x,smoothabs(x,0.5),'--k',linewidth=fs*2)
grid('on')
xlim([-3,3])
title('Abs. Value',fontsize=18)

tight_layout()

#show()

#tight_layout()

savefig('plot_smooth.png')



#show()







#show()



