import pylab as pl
import h5py
import sys
sys.path.insert(0, "../../Sknet")
import sknet

dataset = sknet.dataset.load_cifar10()

dataset['images/train_set'] -= dataset['images/train_set'].mean((1,2,3),
                                                                keepdims=True)
dataset['images/train_set'] /= dataset['images/train_set'].max((1,2,3),
                                                               keepdims=True)
dataset['images/test_set'] -= dataset['images/test_set'].mean((1,2,3),
                                                              keepdims=True)
dataset['images/test_set'] /= dataset['images/test_set'].max((1,2,3),
                                                             keepdims=True)
dataset['images/test_set'] = dataset['images/test_set'][:,:,2:-2,2:-2]


for MODEL in ['smallcnn', 'largecnn', 'resnet']:
    f = h5py.File('/mnt/drive1/rbalSpace/centroids/saved_mus_'+MODEL+'.h5', 'r', swmr=True)
    mus = {'train': [], 'test': []}

    keys = sorted([i for i in f['train_set/minimizer']], key=float)
    print(keys)
    for key in keys[2:]:
        print(pl.shape(f['train_set/minimizer/'+key][...]))
        mus['train'].append(f['train_set/minimizer/'+key][...][[0, -1]])

    keys = sorted([i for i in f['test_set/accu']], key=float)
    for key in keys[1:]:
        mus['test'].append(f['test_set/accu/'+key][...][[0, -1]])
    f.close()

    n_layer = len(mus['train'])

    # test the distance thing
    print('model before training', MODEL)
    for layer in range(n_layer-1):
        distribution = list()
        for batch in range(mus['test'][layer].shape[1]):
            for n in range(64):
                tester1 = mus['test'][layer][0, batch, n]
                d1 = (tester1-dataset['images/test_set'])**2
                index = pl.where(pl.argsort(d1.sum((1, 2, 3))) == ((64*5*batch)+n))[0]
                distribution.append(index)
        print(pl.mean(distribution), end=', ')
    print(';;',len(distribution))


    print('model after training', MODEL)
    for layer in range(n_layer-1):
        distribution = list()
        for batch in range(mus['test'][layer].shape[1]):
            for n in range(64):
                tester1 = mus['test'][layer][1, batch, n]
                d1 = (tester1-dataset['images/test_set'])**2
                index = pl.where(pl.argsort(d1.sum((1, 2, 3))) == ((64*5*batch)+n))[0]
                distribution.append(index)
        print(layer, pl.mean(distribution),end=', ')
    print('')

#    continue

    # now to the plotting
    for set_name in ['train', 'test']:
        for image_nb in [13]:
            for layer in range(n_layer):
                name = 'images/centroids_{}_init_{}_{}_{}.pdf'.format(MODEL,
                                                                      set_name,
                                                                      image_nb,
                                                                      layer)
                pl.figure(figsize=(2, 2))
                image = mus[set_name][layer][0, 0, image_nb]
                sknet.utils.plotting.imshow(image, interpolation='nearest')
                pl.tight_layout()
                pl.savefig(name)
                pl.close()

                name = 'images/centroids_{}_end_{}_{}_{}.pdf'.format(MODEL,
                                                                     set_name,
                                                                     image_nb,
                                                                     layer)
                pl.figure(figsize=(2, 2))
                image = mus[set_name][layer][1, 0, image_nb]
                sknet.utils.plotting.imshow(image, interpolation='nearest')
                pl.tight_layout()
                pl.savefig(name)
                pl.close()


