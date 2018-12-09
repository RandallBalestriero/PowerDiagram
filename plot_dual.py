from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import glob
import cPickle


execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')


def do_plots(dataset):
	files = glob.glob('/mnt/project2/rb42Data/SMASO/'+dataset+'*_dual.pkl')
	print files
	for f in files:
		file1 = open(f,'rb')
		data = cPickle.load(file1)
#		data =asarray(data)
		print shape(data[1][0])
		data = [d.mean(0) for d in data]
		print shape(data[0])
		file1.close()
		subplot(121)
		plot(data[0],'k')
		plot(data[2])
		subplot(122)
		plot(data[1],'k')
		plot(data[3])

	
		




files = glob.glob('/mnt/project2/rb42Data/SMASO/*_dual.pkl')
print files
DATASETS = unique([f.split('_')[0].split('/')[-1] for f in files])
print DATASETS


for dataset in DATASETS:
	do_plots(dataset)
	show()	




