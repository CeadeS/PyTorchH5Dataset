from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy

from torch import tensor, as_tensor, int64, randperm, device as device_t

from math import ceil

import numpy as np



class collate_fn:
    def __init__(self, batch_size, dataset):
        self.batch_size = batch_size
        self.dataset = dataset
        self.full_iterations = len(self.dataset) // batch_size

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration()
        img_bytes, (label, index) = self.dataset[sample_idx]
        encoded_img = np.frombuffer(img_bytes, dtype=np.uint8)
        label = np.int32(label)
        index = np.int32(index)
        return encoded_img, label, index


def get_decode_crop_transform_fn(random_area=(0.08,1.0), random_aspect_ratio=(0.75,1.33), output_size=(244,244)):
    def decode_crop_transform_fn(jpegs):
        decode = fn.decoders.image_random_crop(jpegs,random_area=random_area,random_aspect_ratio=random_aspect_ratio, device="mixed")
        decode = fn.resize(decode, size=output_size)
        return decode
    return decode_crop_transform_fn

def get_wrapper_pipeline(dataset,transform_fn=get_decode_crop_transform_fn, batch_size = 10):
    def wrapper_pipeline():
        jpegs, labels, indexes = fn.external_source(source=collate_fn(batch_size,dataset), num_outputs=3, batch=False, parallel=True,
                                                    dtype=[types.UINT8,types.INT32, types.INT32])
        decode = transform_fn(jpegs)
        return decode, labels, indexes
    return wrapper_pipeline

def get_random_erase_function(nregions=1, ankor_range=(0.,1.), shape_range=(20.,50), fill_value=(0,0,0)):
    args_shape=(2*nregions,)
    def random_erase_function(images):
        random_anchor = fn.random.uniform(range=ankor_range, shape=args_shape, device='cpu')
        random_shape = fn.random.uniform(range=shape_range, shape=args_shape, device='cpu')
        erased = fn.erase(
            images,
            device="gpu",
            anchor=random_anchor,
            shape=random_shape,
            axis_names='WH',
            fill_value=fill_value,
            normalized_anchor=True,
            normalized_shape=False)
        return erased

    return  random_erase_function

def augment_function(jpegs):
    im = fn.decoders.image_random_crop(jpegs,random_area=(0.08,1.0),random_aspect_ratio=(0.75,1.33), device="mixed")#.as_cpu()
    im = im.gpu()
    im = fn.jitter(im, nDegree=5)
    im = im.gpu()
    rnd_angle = fn.random.uniform(range = (-45.0,45.0), device='cpu')
    im = fn.rotate(im, angle=rnd_angle, keep_size =True, device='gpu')
    im = im.gpu()
    im = fn.resize(im, size=(244, 244))
    im = im.gpu()
    im = get_random_erase_function(shape_range=(20,50))(im)
    return im


def get_pipeline(dataset, transform_fn = get_decode_crop_transform_fn(), batch_size=10, num_threads=5, device_id=0, num_workers=5):
    prep = pipeline_def(batch_size=10, num_threads=5, device_id=0, py_num_workers=5, py_start_method='spawn')
    return prep(get_wrapper_pipeline(dataset, transform_fn ,batch_size))


class DaliDataLoader(object):

    def __init__(self,
                 dataset,
                 dali_transform_fn=get_decode_crop_transform_fn(),
                 batch_size=1,
                 num_batches_buffered=10,
                 shuffle=False,
                 num_workers=5,
                 num_threads=5,
                 return_meta_indices = True,
                 normalize = None,
                 device= device_t('cpu')):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset = dataset
        self.num_sub_batches_buffered = dataset.sub_batch_size
        self.return_meta_indices = return_meta_indices
        self.device = device_t(device)
        self.shuffle = shuffle
        self.normalize=normalize

        did = device.index if device.index else 0

        pipeline = get_pipeline(dataset, dali_transform_fn)
        pipeline = pipeline()
        pipeline.build()

        self.dataloader = DALIGenericIterator(pipelines=pipeline,
                                              output_map=['data','label','index'],
                                              size=-1,
                                              reader_name=None,
                                              auto_reset=True,
                                              fill_last_batch=None,
                                              dynamic_shape=False,
                                              last_batch_padded=False,
                                              last_batch_policy=LastBatchPolicy.PARTIAL,
                                              prepare_first_batch=True)


    def __iter__(self):
        for sample in self.dataloader:
            sample, meta_class, meta_indices = sample[0]['data'].movedim(-1,1),sample[0]['label'],sample[0]['index']
            meta_indices = meta_indices.view(-1)
            meta = as_tensor(meta_class.view(-1), dtype=int64, device=self.device)#.requires_grad_(True)
            if self.normalize is not None:
                sample = self.normalize(sample)
            if self.shuffle:
                perm = randperm(len(sample))
            else:
                perm = tensor(range(len(sample)))
            for i in range(0, perm.size(0), self.batch_size):
                rand_batch_indexes = perm[i:i + self.batch_size]
                x = sample[rand_batch_indexes]
                if len(rand_batch_indexes) == 1:
                    x =  np.expand_dims(x, axis=0)
                y = meta[rand_batch_indexes]
                if self.return_meta_indices:
                    yield x,(y,meta_indices[rand_batch_indexes])
                else:
                    yield x,(y, self.dataset.get_meta_data_from_indices(meta_indices[rand_batch_indexes]))
        self.dataset.script_transform = None

    def __len__(self):
        return ceil(self.dataset.num_samples / self.batch_size)

batch_size = 10


