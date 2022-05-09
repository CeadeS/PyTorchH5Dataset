from unittest import TestCase

import os

from pytorch_h5dataset.dataset import ImageDataset, BloscDataset
from pytorch_h5dataset.h5dataloader import H5DataLoader

import os
import pathlib
if  pathlib.Path(os.getcwd()).name == 'test':
    os.chdir('../')


class TestH5Dataloader(TestCase):

    def test___init__(self):
        import pandas as pd
        import shutil as sh
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        BloscDataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = BloscDataset('test_dataset', './test/data/tmp/dataset/h5/')
        with self.assertRaises(AssertionError):
            dataLoader = H5DataLoader(dataset=dataset,
                                         device='cpu:0', batch_size=1,
                                         return_meta_indices=True,
                                         pin_memory=True,
                                         num_workers=0, collate_fn="dd")


    def test___len__(self):
        import pandas as pd
        import shutil as sh
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        BloscDataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = BloscDataset('test_dataset', './test/data/tmp/dataset/h5/')
        dataloader = H5DataLoader(dataset=dataset,
                                  device='cpu:0', batch_size=1,
                                  return_meta_indices=True,
                                  pin_memory=True,
                                  num_workers=0, collate_fn=None)
        self.assertEqual(len(dataloader), 2)

        dataloader = H5DataLoader(dataset=dataset,
                                  device='cpu:0', batch_size=2,
                                  return_meta_indices=True,
                                  pin_memory=True,
                                  num_workers=0, collate_fn=None)


        self.assertEqual(len(dataloader), 1)

        dataframe = pd.read_csv('./test/data/test_dataset_multichannel.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset_multichannel.csv','./test/data/tmp/dataset/h5/test_dataset_multichannel.csv')
        BloscDataset.convert_samples_to_dataset(dataframe,
                                                             './test/data/tmp/dataset/h5/test_dataset_multichannel.h5'
                                                             ,sub_batch_size=2)

        self.assertEqual(len(dataloader), 1)


    def test_collate_samples(self):
        import pandas as pd
        import shutil as sh
        from torch import Tensor
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        ImageDataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = ImageDataset('test_dataset', './test/data/tmp/dataset/h5/',
                               tr_crop_strategy='center',  decode='cuda',
                               tr_crop_size=(244,244))
        dataloader = H5DataLoader(dataset=dataset,
                                  device='cpu:0', batch_size=2,
                                  return_meta_indices=True,
                                  pin_memory=True,
                                  num_workers=0, collate_fn=None)
        for sample, (meta, meta_data) in dataloader:
            print(sample.shape)
            self.assertEqual(len(sample),1)
            self.assertIsInstance(meta_data, Tensor)

        dataloader = H5DataLoader(dataset=dataset,
                                  device='cpu:0', batch_size=1,
                                  return_meta_indices=False,
                                  pin_memory=True,
                                  num_workers=0, collate_fn=None)
        for sample, (meta, meta_data) in dataloader:
            print(sample.shape)
            self.assertEqual(len(sample),2)
            self.assertIsInstance(meta_data, pd.DataFrame)


        dataframe = pd.read_csv('./test/data/test_dataset_multichannel.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset_multichannel.csv','./test/data/tmp/dataset/h5/test_dataset_multichannel.csv')
        BloscDataset.convert_samples_to_dataset(dataframe,
                                             './test/data/tmp/dataset/h5/test_dataset_multichannel.h5'
                                             ,sub_batch_size=2)

        from torchvision.transforms import Resize
        from torch.jit import script
        r = script(Resize((244,244)))
        dataset = BloscDataset('test_dataset_multichannel',
                               './test/data/tmp/dataset/h5/',
                               tr_crop_strategy='original', tensor_transforms=r)

        dataloader = H5DataLoader(dataset=dataset,
                                  device='cpu:0', batch_size=2,
                                  return_meta_indices=True,
                                  pin_memory=True,
                                  num_workers=0, collate_fn=None)

        for sample, (meta, meta_data) in dataloader:
            print(sample.shape)
            self.assertEqual(len(sample),2)
            self.assertIsInstance(meta_data, Tensor)


