import pylab as pl
import h5py
import sys
sys.path.insert(0, "../Sknet")
import sknet


f=h5py.File('saved_mus.h5','r')

for epoch in ['0','12']:
    mus=f['mus_'+epoch][...]
    images=f['images'][...]
    n_layer  = pl.shape(mus)[2]
    for batch_nb in range(4):
        for set_nb,set_name in enumerate(['train','test']):
            for image_nb in range(64):
                pl.figure(figsize=(2*(n_layer+1),2))
                pl.subplot(1,n_layer,1)
                sknet.utils.plotting.imshow(images[set_nb,64*batch_nb+image_nb])
                for layer in range(n_layer):
                    pl.subplot(1,n_layer,layer+1)
                    sknet.utils.plotting.imshow(mus[set_nb,batch_nb,layer,image_nb])
                pl.savefig('images/centroids_'+set_name+'_image'\
                        +str(64*batch_nb+image_nb)+'_epoch'+str(epoch)+'.pdf')
                pl.close()
f.close()
