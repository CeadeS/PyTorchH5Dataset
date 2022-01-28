from torch import tensor, as_tensor, float32, randperm, Tensor, cat
from torch.utils import data
from torch.jit import script
from . import H5Dataset
from utils import NormImage
from pandas import concat
from math import ceil
import types

def collate_samples(sample):
    sample, meta = zip(*sample)
    cl, meta = zip(*meta)
    if isinstance(meta[0], Tensor):
        out = None
        ### code from
        # https://github.com/pytorch/pytorch/blob/be2dc8f2940d3c95941516a811be8c504910d1ea/torch/utils/data/_utils/collate.py
        if data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in sample])
            storage = sample[0].storage()._new_shared(numel)
            out = sample[0].new(storage)
        return cat(sample, dim=0, out = out), (cat(cl), cat(meta))
    else:
        return cat(sample, dim=0), (cat(cl), concat(meta))

class H5DataLoader(object):

    def __init__(self, dataset: H5Dataset, batch_size=1, device='cpu', num_batches_buffered=10,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None, normalize=script(NormImage()).cuda(),
                 return_meta_indices = True):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_sub_batches_buffered = dataset.sub_batch_size
        self.device = device
        self.normalize = normalize
        self.dataset = dataset
        self.return_meta_indices = return_meta_indices
        self.pin_memory = pin_memory
        assert collate_fn is None or isinstance(collate_fn, types.FunctionType), "Collate Function must be a function"
        self.collate_fn = collate_fn if collate_fn is not None else collate_samples
        self.shuffle = shuffle
        self.dataloader = data.DataLoader(dataset, batch_size=num_batches_buffered, shuffle=self.shuffle, sampler=sampler,
                                     batch_sampler=batch_sampler, num_workers=self.num_workers, collate_fn=self.collate_fn,
                                     pin_memory=self.pin_memory, drop_last=drop_last, timeout=timeout,
                                     worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context)


    def __iter__(self):

        for sample, (meta_class, meta_indices) in self.dataloader:
            meta_indices = meta_indices.view(-1)
            if self.pin_memory:
                sample = sample.pin_memory()
            sample =  sample.to(self.device)
            meta = as_tensor(meta_class.view(-1), dtype=float32, device=self.device).requires_grad_(True)
            sample = self.normalize(sample)
            if self.shuffle:
                perm = randperm(len(sample))
            else:
                perm = tensor(range(len(sample)))
            for i in range(0, perm.size(0), self.batch_size):
                rand_batch_indexes = perm[i:i + self.batch_size]
                x = sample[rand_batch_indexes]
                y = meta[rand_batch_indexes]
                if self.return_meta_indices:
                    yield x,(y,meta_indices[rand_batch_indexes])
                else:
                    yield x,(y, self.dataset.get_meta_data_from_indices(meta_indices[rand_batch_indexes]))
        self.dataset.script_transform = None

    def __len__(self):
        return ceil(len(self.dataset)/self.batch_size)