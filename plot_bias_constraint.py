from pylab import *
import glob
import cPickle


import os
SAVE_DIR = os.environ['SAVE_DIR']


files = glob.glob(SAVE_DIR+'VORONOI/bias_constraint*')
FILES = sort(unique([f.split('run')[0] for f in files]))

for name in FILES:
	all_files = glob.glob(name+'*.pkl')
	for FILE in all_files:
		f = open(FILE,'rb')
		train_loss,train_accu,test_accu = cPickle.load(f)
		f.close()
		print name,test_accu[-3:]

def do_plots(dataset):
	files = glob.glob('/mnt/project2/rb42Data/SMASO/'+dataset+'*bn1*_molly.pkl')
	print files
	for f in files:
		file1 = open(f,'rb')
		data = cPickle.load(file1)
		data =asarray(data)
		data = [d.mean(0) for d in data]
		print shape(data[0])
		file1.close()
		subplot(121)
		plot(data[0],'k')
		plot(data[2])
		subplot(122)
		plot(data[1],'k')
		plot(data[3])

	
		





