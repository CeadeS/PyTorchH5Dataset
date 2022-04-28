from torch.utils.data import Dataset
import numpy as np
import h5py, hdf5plugin, tables  # fixes plugins not found Exceptions
import torch
from math import sqrt
import imghdr
import sndhdr
import os
import pandas as pd
import random
import logging
import difflib
from .fn.blosc import BloscInterface
from .fn.image import ImageInterface
from math import floor

#TODO
#handlers = [logging.FileHandler(filename='data_import.log'),logging.StreamHandler(sys.stdout) ]
#logging.basicConfig(format='%(asctime)s %(message)s', level=log_level , handlers=handlers)
import tarfile
from random import shuffle
from time import time

from pytorch_h5dataset.dataset.metaDataset import H5MetaDataset


class H5Dataset(H5MetaDataset):

    def __getitem__(self, sub_batch_idx):
        sub_batch_idx = self.group_number_mapping[sub_batch_idx]
        group_no = str(sub_batch_idx//self.max_n_group)
        dataset_no = str(sub_batch_idx%self.max_n_group)
        if self.crop_function is None:
            if 'random' in self.crop_strategy:
                if 'center' in self.crop_strategy:
                    self.crop_function = self.data_interface.get_random_center_crop_function(crop_size=self.crop_size, crop_area_ratio_range=self.crop_area_ratio_range)
                else:
                    self.crop_function = self.data_interface.get_random_crop_function(
                        crop_size=self.crop_size, crop_area_ratio_range=self.crop_area_ratio_range)
            elif self.crop_strategy == 'center':
                self.crop_function = self.data_interface.get_center_crop_function(crop_size=self.crop_size)
            elif self.crop_strategy == 'original':
                self.crop_function = self.data_interface.crop_original_samples_from_batch()
            else:
                self.crop_function = 'none'
        if self.h5_file is None:
            self.h5_file = h5py.File(self.dataset_h5_file_path, "r")

        if self.transforms is not None and isinstance(self.transforms, torch.nn.Module):
            self.script_transform = torch.jit.script(self.transforms)

        sub_batch = self.h5_file[group_no][f'samples/{dataset_no}']
        batch_shape = self.batch_shapes[sub_batch_idx][-2:]
        if self.crop_strategy == 'original':
            sample = self.crop_function(sub_batch, self.batch_shapes[sub_batch_idx])  ## FIX for Issue with TIFF uint16 dtypes
        elif self.crop_function == 'none':
            pass
        else:
            sample = self.crop_function(sub_batch, *batch_shape)  ## FIX for Issue with TIFF uint16 dtypes
        if isinstance(sample, list):
            for i in range(len(sample)):
                pass
                #sample[i] = torch.from_numpy(np.frombuffer(sample[i], dtype=np.uint8))
        else:
            if sample.dtype == np.uint16:
                sample = torch.from_numpy(sample.astype(int))
            else:
                sample = torch.as_tensor(sample)

        if self.script_transform is not None:
            self.script_transform = self.script_transform
            sample = self.script_transform(sample)

        meta_data = (torch.as_tensor(self.classes[sub_batch_idx]),
                     torch.as_tensor(self.indices[sub_batch_idx]))

        return sample, meta_data


    def initiate_crop_function(self):
        """
        Is called after loading the first batch. Fixes problem with worker processes that are unable to pickle self.crop_function
        :param loading_crop_size:
        :param loading_crop_area_ratio_range:
        :return:
        """
        random_location = 'center' not in self.crop_strategy
        logging.info("called initiate crop function")
        self.crop_function = self.data_interface.get_random_crop_function(
            crop_size=self.crop_size,
            crop_area_ratio_range=self.crop_area_ratio_range,
            random_location=random_location)


    def __init__(self,
                 dataset_name='dataset_name',
                 dataset_root='/path/to/dataset',
                 crop_strategy = 'random',
                 loading_crop_size=(0.73, 1.33),
                 loading_crop_area_ratio_range=244 * 244,
                 transforms=None,
                 split_mode = 'full',
                 split_ratio = 1.0,
                 split_number = 0,
                 ):
        """
        Constructor of the Dataset.

        :param dataset_name: Name of the datset files.
        :param dataset_root: Path to the dataset folder.
        :param crop_strategy: Crop Strategy can be center, random, random_center or original.
        Center returns a center crop, random a random sized crop and
        random_center a random sized center crop - original performs no cropping.
        :param loading_crop_size: Crop crop_size aspect ratio range - see random_located_sized_crop_function().
        :param loading_crop_area_ratio_range: Number of values cropped from sample OR range of cropped values proportion.
        :param transforms: torchvision.transforms instance, Module or Transform. Module is not pickable after first iteration if Modules are used. Instantiate Dataloaders before first iteration!!
        :param split_mode: Determine the data split. ;must be in ['full', 'montecarlo', 'cross_val', split']
        :param split_ratio: float setting the percentage that is split for training if split_mode is full or montecarlo - int if split_mode is cross_val determining the number of splits, tuple of floats if mode is split
        :param split_number: int. for montecarlo this is the random seed
        """
        dataset_h5_file_path = os.path.join(dataset_root, dataset_name + '.h5')
        metadata_file_path = os.path.join(dataset_root, dataset_name + '.csv')
        self.h5_file = None

        assert crop_strategy is None or crop_strategy.lower() in ['random', 'random_center', 'center', 'original', 'none']

        self.transforms = transforms
        self.script_transform = None
        self.crop_function = None
        self.crop_strategy = crop_strategy.lower()

        with h5py.File(self.dataset_h5_file_path, "r") as h5_file:
            self.data_mode = h5_file.attrs['data_mode']


        super(H5Dataset, self).__init__()


    def __del__(self):
        logging.info("called del")
        if self.h5_file is not None and isinstance(self.h5_file, h5py._hl.files.File):
            logging.info('closing File')
            self.h5_file.close()
        logging.info("Deletion Complete")
