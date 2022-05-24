from torch import tensor, as_tensor, float32, long, int32, int64, randperm, Tensor, cat, cuda, device as device_t
from numpy import concatenate
from torch.utils import data
from torch.jit import script
from pytorch_h5dataset.dataset.metaDataset import H5MetaDataset
from pandas import concat
import numpy as np
from math import ceil
import types
import itertools
from operator import itemgetter


def collate_samples_tensor(sample):
    sample, meta = zip(*sample)
    cl, meta = zip(*meta)

    if isinstance(meta[0], Tensor) or isinstance(meta[0], np.ndarray):
        meta = cat(meta)
        out = None
        ### code from
        # https://github.com/pytorch/pytorch/blob/be2dc8f2940d3c95941516a811be8c504910d1ea/torch/utils/data/_utils/collate.py
        if data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in sample])
            storage = sample[0].storage()._new_shared(numel)
            out = sample[0].new(storage)

        return cat(sample, dim=0, out = out), (cat(cl), meta)
    else:
        meta = concat(meta)
        return cat(sample, dim=0), (cat(cl), meta)

def collate_samples_list(sample):
    sample, meta = zip(*sample)
    sample = list(itertools.chain(*sample))
    cl, meta = zip(*meta)
    if isinstance(meta[0], Tensor):
        return sample, (cat(cl), cat(meta))
    elif isinstance(meta[0], np.ndarray):
        return sample, (concatenate(cl), concatenate(meta))
    else:
        return sample, (cat(cl), concat(meta))

def collate_samples(sample):
    _sample = sample[0][0]
    if isinstance(_sample, Tensor) or isinstance(_sample, np.ndarray):
        return collate_samples_tensor(sample)
    else:
        return collate_samples_list(sample)

class DataLoader(object):

    def __init__(self, dataset: H5MetaDataset, batch_size=1, device=device_t('cpu'), num_batches_buffered=10,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None, normalize=None,
                 return_meta_indices = True, meta_filter=None):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_sub_batches_buffered = dataset.sub_batch_size
        self.device = device_t(device) if cuda.is_available() else device_t('cpu')
        self.normalize = normalize
        self.dataset = dataset
        self.return_meta_indices = return_meta_indices
        self.pin_memory = pin_memory if cuda.is_available() else False
        assert collate_fn is None or isinstance(collate_fn, types.FunctionType), "Collate Function must be a function"
        self.collate_fn = collate_fn if collate_fn is not None else collate_samples
        self.shuffle = shuffle
        assert meta_filter in [None, 'indices', 'labels']
        self.meta_filter = meta_filter
        self.dataloader = data.DataLoader(dataset, batch_size=num_batches_buffered, shuffle=self.shuffle, sampler=sampler,
                                     batch_sampler=batch_sampler, num_workers=self.num_workers, collate_fn=self.collate_fn,
                                     pin_memory=self.pin_memory, drop_last=drop_last, timeout=timeout,
                                     worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context)


    def __iter__(self):



        for sample, (meta_class, meta_indices) in self.dataloader:
            if self.pin_memory and isinstance(sample, Tensor):
                sample = sample.pin_memory().to(self.device)
            meta = as_tensor(meta_class, dtype=int64, device=self.device).view(-1)#.requires_grad_(True)
            meta_indices = as_tensor(meta_indices, dtype=int64, device=self.device).view(-1)#.requires_grad_(True)
            if self.normalize is not None:
                sample = self.normalize(sample)
            if self.shuffle:
                perm = randperm(len(sample))
            else:
                perm = tensor(range(len(sample)))
            for i in range(0, perm.size(0), self.batch_size):
                rand_batch_indexes = perm[i:i + self.batch_size]
                if isinstance(sample, list):
                    if len(rand_batch_indexes) <2:
                        x = [sample[rand_batch_indexes[0]]] # case where only one sample is selected # keep type list
                    else:
                        x = itemgetter(*rand_batch_indexes)(sample) # in last test faster than map
                else:
                    x = sample[rand_batch_indexes]
                    if len(rand_batch_indexes) == 1:
                        x =  np.expand_dims(x, axis=0)
                    x = as_tensor(x, device=self.device)
                y = meta[rand_batch_indexes]
                if self.meta_filter is not None:
                    if self.meta_filter == 'labels':
                        yield x, y
                    else:
                        yield x, meta_indices[rand_batch_indexes]
                if self.return_meta_indices:
                    yield x,(y, meta_indices[rand_batch_indexes])
                else:
                    yield x,(y, self.dataset.get_meta_data_from_indices(np.asarray(meta_indices[rand_batch_indexes].cpu())))

    def __len__(self):
        return ceil(self.dataset.num_samples / self.batch_size)
