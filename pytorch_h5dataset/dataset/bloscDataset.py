import numpy as np
from torch import device, version, from_numpy, as_tensor, dtype
import torch

import logging
from functools import partial
from ..fn.blosc import BloscInterface
from pytorch_h5dataset.fn.transforms import Transform


from pytorch_h5dataset.dataset.metaDataset import H5MetaDataset

def get_batch_shapes():
    pass


class BloscDataset(H5MetaDataset):


    def __getitem__(self, sub_batch_idx):

        sample_reference, meta = super(BloscDataset, self).__getitem__(sub_batch_idx=sub_batch_idx)
        sub_batch_idx, sub_batch_slice = (sub_batch_idx[0],sub_batch_idx[1]) if isinstance(sub_batch_idx, tuple) else (sub_batch_idx, slice(None,None,None))
        batch_height, batch_width = self.batch_shapes[sub_batch_idx][-2:]
        original_shapes = None

        if self._BloscDataset__crop_strategy == 'original':
            original_shapes = self.shapes[sub_batch_idx][sub_batch_slice]


        if self.crop_function:
            sample = self.crop_function(sample_reference,
                                    batch_height = batch_height,
                                    batch_width = batch_width,
                                    shapes = original_shapes)

        else:
            sample = sample_reference[()] ## dereference i.e. read from file system

        sample = self.transforms(sample)

        if self.tensor_transforms is not None:
            if isinstance(sample, list):
                sample = [self.tensor_transforms(s) for s in sample]
            else:
                sample = self.tensor_transforms(sample)


        return sample, as_tensor(meta)

    @staticmethod
    def convert_samples_to_dataset(dataset_dataframe,
                                   dataset_destination_h5_file='./data/test_dataset.h5',
                                   sub_batch_size=50, max_n_group= 10):

        BloscDataset._convert_samples_to_dataset(dataset_dataframe=dataset_dataframe,
                                                 dataset_destination_h5_file=dataset_destination_h5_file,
                                                 sub_batch_size=sub_batch_size, data_mode='blosc',
                                                 max_n_group=max_n_group)

    # def initiate_crop_function(self):
    #     """
    #     Is called after loading the first batch. Fixes problem with worker processes that are unable to pickle self.crop_function
    #     :param loading_crop_size:
    #     :param loading_crop_area_ratio_range:
    #     :return:
    #     """
    #     random_location = 'center' not in self.crop_strategy
    #     logging.info("called initiate crop function")
    #     self.crop_function = self.data_interface.get_random_crop_function(
    #         crop_size=self.crop_size,
    #         crop_area_ratio_range=self.crop_area_ratio_range,
    #         random_location=random_location)


    def __init__(self,
                 dataset_name='dataset_name',
                 dataset_root='/path/to/dataset',
                 split_mode = 'full',
                 split_ratio = 1.0,
                 split_number = 0,
                 tr_crop_strategy = None,
                 tr_crop_size = None,
                 tr_crop_area_ratio_range = None,
                 output_device = device('cpu'), #cpu or cuda
                 tensor_transforms = None,
                 force_output_dtype = None
                 ):

        output_device = device(output_device)
        assert tr_crop_strategy in ['random', 'center','original', None]

        tr_crop_strategy = tr_crop_strategy.lower() if tr_crop_strategy else None

        self.crop_function = None
        self.__crop_strategy = tr_crop_strategy

        super(BloscDataset, self).__init__(dataset_name, dataset_root, split_mode, split_ratio, split_number)
        assert self.data_dtype != str(bytes)
        #BLOSC Transforms

        if tr_crop_strategy is not None:
            if tr_crop_strategy == 'original':
                self.crop_function = self.data_interface.crop_original_samples_from_batch
            else:
                self.crop_function = BloscInterface.get_random_crop_function(
                    random_location=tr_crop_strategy == 'random',
                    crop_size =tr_crop_size,
                    crop_area_ratio_range = tr_crop_area_ratio_range)

        transforms = []

        ## ugly as hell,

        data_dtype, np_dtype = (torch.int32, np.dtype('int32')) if self.data_dtype == 'uint16' else (torch.__dict__[self.data_dtype],np.dtype(self.data_dtype))
        if force_output_dtype is not None:
            np_dtype = np.dtype(force_output_dtype)
        if tr_crop_strategy == 'original':
            transforms.append(partial(BloscInterface.sub_batch_list_as_tensor, torch_dtype=data_dtype, np_dtype=np_dtype, device=device(output_device)))
        else:
            transforms.append(partial(BloscInterface.sub_batch_array_as_tensor, torch_dtype=data_dtype, np_dtype=np_dtype, device=device(output_device)))

        self.transforms = Transform(transforms=transforms)
        self.tensor_transforms = tensor_transforms
