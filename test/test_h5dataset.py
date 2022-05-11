import os
import pathlib
from unittest import TestCase

from pytorch_h5dataset.dataset import imageDataset, bloscDataset
from pytorch_h5dataset.dataset.bloscDataset import BloscDataset as _H5Dataset

# TODO refactor tests to test jpeg as well
if  pathlib.Path(os.getcwd()).name == 'test':
    os.chdir('../')

andromeda_sample = {
    'path': './test/data/images/rgb/a/andromeda_0.png',
    'index': 1,
    'class': 0,
    'type': 'jpg',
    'height': 3694,
    'width': 6652,
    'crop_size': 3694 * 6652,
    'shape': (3, 3694, 6652)
}

pollen_sample = {
    'path': './test/data/images/multichannel/a/pollen_0.tif',
    'index': 1,
    'class': 0,
    'type': 'tif',
    'height': 188,
    'width': 85,
    'crop_size': 188 * 85,
    'shape': (24, 188, 85)
}

pano_sample = {
    'path': './test/data/images/rgb/b/pano_1.jpg',
    'index': 1,
    'class': 1,
    'type': 'jpg',
    'height': 3762,
    'width': 12706,
    'crop_size': 3762 * 12706,
    'shape': (3, 3762, 12706)
}

something_sample = {
    'path': './test/data/images/multichannel/b/something_1.tif',
    'index': 1,
    'class': 1,
    'type': 'tif',
    'height': 22,
    'width': 24,
    'crop_size': 22 * 24,
    'shape': (24, 22, 24)
}

class TestH5Dataset(TestCase):

    def test_load_sample(self):
        # jpg/png

        sample = {
            'FilePath': andromeda_sample['path'],
        }
        image, shape = _H5Dataset.load_sample(sample)
        self.assertEqual(shape, andromeda_sample['shape']), 'Loaded Image has the wrong shape'

        sample = {
            'FilePath': andromeda_sample['path'],
            'FileType': 'png'
        }
        image, shape = _H5Dataset.load_sample(sample)
        self.assertEqual(shape, andromeda_sample['shape']), 'Loaded Image has the wrong shape'

        sample = {
            'FilePath': pano_sample['path'],
            'FileType': 'jpg'
        }
        image, shape = _H5Dataset.load_sample(sample)
        self.assertEqual(shape, pano_sample['shape']), 'Loaded Image has the wrong shape'

        # tif
        sample = {
            'FilePath': something_sample['path'],
            'FileType': 'tif'
        }
        image, shape = _H5Dataset.load_sample(sample)
        self.assertEqual(shape, something_sample['shape']), 'Loaded Image has the wrong shape'

        sample = {
            'FilePath': something_sample['path'],
        }
        image, shape = _H5Dataset.load_sample(sample)
        self.assertEqual(shape, something_sample['shape']), 'Loaded Image has the wrong shape'

        sample = {
            'FilePath': pollen_sample['path'],
        }

        image, shape = _H5Dataset.load_sample(sample)
        self.assertEqual(shape, pollen_sample['shape']), 'Loaded Image has the wrong shape'

    def test_stack_batch_data_padded(self):
        from pytorch_h5dataset.fn.blosc import BloscInterface
        # jpg/png
        l = [andromeda_sample, pano_sample]
        batch = BloscInterface.stack_batch_data_padded(l)
        self.assertEqual(len(batch),2)
        self.assertEqual(batch[0].shape, (3,3762,12706), batch[1].shape)

        l = [pano_sample, andromeda_sample]
        batch = BloscInterface.stack_batch_data_padded(l)
        self.assertEqual(len(batch),2)
        self.assertEqual(batch[0].shape, (3,3762,12706), batch[1].shape)

        # tifs
        l = [something_sample, pollen_sample]
        batch = BloscInterface.stack_batch_data_padded(l)
        self.assertEqual(len(batch),2)
        self.assertEqual(batch[0].shape, (24,188,85), batch[1].shape)

        l = [pollen_sample, something_sample]
        batch = BloscInterface.stack_batch_data_padded(l)
        self.assertEqual(len(batch),2)
        self.assertEqual(batch[0].shape, (24,188,85), batch[1].shape)

    def test_batchify_sorted_sample_data_list(self):
        from pytorch_h5dataset.fn.blosc import BloscInterface
        # jpg/png
        sample_list = [andromeda_sample, pano_sample]
        batch_generator = BloscInterface.batchify_sorted_sample_data_list(sample_list)
        for batch_array, classes, shapes, indices in batch_generator:
            self.assertEqual(batch_array.shape, (2,3,3762,12706))
            (self.assertEqual(a,b) for a,b in zip(classes,[0,1]))
            (self.assertEqual(a,b) for a,b in zip(((3, 3694, 6652),(3, 3762, 12706))))

        # tif
        sample_list = [something_sample, pollen_sample]
        batch_generator = BloscInterface.batchify_sorted_sample_data_list(sample_list)
        for batch_array, classes, shapes, indices in batch_generator:
            self.assertEqual(batch_array.shape, (2,24,188,85))
            (self.assertEqual(a,b) for a,b in zip(classes,[0,1]))
            (self.assertEqual(a,b) for a,b in zip(((24, 22, 24),(24, 188, 85))))

    def test_get_central_crop_args(self):
        from pytorch_h5dataset.fn.data_interface import DataInterface

        h_begin, w_begin, h_end, w_end = DataInterface.get_central_crop_args(batch_width=244, batch_height=217, crop_area_ratio_range=(0.5,0.5))
        self.assertEqual(round(((-h_begin+h_end) * (-w_begin+w_end))/(244*217),1), 0.5)

        h_begin, w_begin, h_end, w_end = DataInterface.get_central_crop_args(batch_width=244, batch_height=217, crop_area_ratio_range=(0.2,0.2))
        self.assertEqual(round(((-h_begin+h_end) * (-w_begin+w_end))/(244*217),1), 0.2)

        h_begin, w_begin, h_end, w_end = DataInterface.get_central_crop_args(batch_width=244, batch_height=217, crop_area_ratio_range=(0.9,0.9))
        self.assertEqual(round(((-h_begin+h_end) * (-w_begin+w_end))/(244*217),1), 0.9)

        h_begin, w_begin, h_end, w_end = DataInterface.get_central_crop_args(batch_width=244, batch_height=217, crop_area_ratio_range=(0.1,0.9))
        for _ in range(10):
            self.assertLessEqual(round(((-h_begin+h_end) * (-w_begin+w_end))/(244*217),1), 0.9)
            self.assertGreaterEqual(round(((-h_begin+h_end) * (-w_begin+w_end))/(244*217),1), 0.1)

        h_begin, w_begin, h_end, w_end = DataInterface.get_central_crop_args(batch_width=244, batch_height=217, crop_area_ratio_range=(1.1,1.1))
        self.assertEqual(round(((-h_begin+h_end) * (-w_begin+w_end)),1), 244*217)

        kwargs = {'batch_width':244,
                  'batch_height':217,
                  'crop_area_ratio_range':(-1,-.5)}

        with self.assertRaises(ValueError):
            DataInterface.get_central_crop_args(**kwargs)

    def test_get_random_crop_within_ratio_range_given_crop_size_ratio(self):
        from pytorch_h5dataset.fn.data_interface import DataInterface
        crop_size_aspect_ratio = (0.7, 0.7)
        crop_area_ratio_range = (0.5, 0.5)
        original_ratio = 244. / 217
        batch_height = 217
        batch_width = 244
        h_begin, w_begin, h_end, w_end = DataInterface.get_random_crop_args_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        ratio = res_width/ res_height
        self.assertGreaterEqual(crop_size_aspect_ratio[0], round(ratio,2))
        self.assertLessEqual(crop_size_aspect_ratio[1], round(ratio,2))

        area_ratio = (res_height * res_width) / (batch_height * batch_width)

        self.assertGreaterEqual(crop_area_ratio_range[0], round(area_ratio,2))
        self.assertLessEqual(crop_area_ratio_range[1], round(area_ratio,2))

        crop_size_aspect_ratio = (0.3, 0.7)
        crop_area_ratio_range = (0.3, 0.3)
        batch_height = 217
        batch_width = 244
        h_begin, w_begin, h_end, w_end = DataInterface.get_random_crop_args_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        ratio = res_width / res_height

        self.assertGreaterEqual(crop_size_aspect_ratio[1], round(ratio,1))
        self.assertLessEqual(crop_size_aspect_ratio[0], round(ratio,1))

        area_ratio = (res_height * res_width) / (batch_height * batch_width)

        self.assertGreaterEqual(crop_area_ratio_range[0], round(area_ratio,1))
        self.assertLessEqual(crop_area_ratio_range[1], round(area_ratio,1))


        crop_size_aspect_ratio = (0.5, 0.7)
        crop_area_ratio_range = (0.1, 0.7)
        batch_height = 217
        batch_width = 244
        h_begin, w_begin, h_end, w_end = DataInterface.get_random_crop_args_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        ratio = res_width / res_height

        self.assertGreaterEqual(crop_size_aspect_ratio[1], round(ratio,1))
        self.assertLessEqual(crop_size_aspect_ratio[0], round(ratio,1))

        area_ratio = (res_height * res_width) / (batch_height * batch_width)

        self.assertGreaterEqual(crop_area_ratio_range[1], round(area_ratio,1))
        self.assertLessEqual(crop_area_ratio_range[0], round(area_ratio,1))

        crop_size_aspect_ratio = (0.73, 1.33)
        crop_area_ratio_range = (0.08, 0.69)
        batch_height = 217
        batch_width = 244
        h_begin, w_begin, h_end, w_end = DataInterface.get_random_crop_args_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        ratio = res_width / res_height

        self.assertGreaterEqual(crop_size_aspect_ratio[1], round(ratio,1))
        self.assertLessEqual(crop_size_aspect_ratio[0], round(ratio,1))

        area_ratio = (res_height * res_width) / (batch_height * batch_width)

        self.assertGreaterEqual(crop_area_ratio_range[1], round(area_ratio,1))
        self.assertLessEqual(crop_area_ratio_range[0], round(area_ratio,1))


        crop_size_aspect_ratio = (0.5, 0.5)
        crop_area_ratio_range = (0.5, 0.5)
        batch_height = 217
        batch_width = 244
        h_begin, w_begin, h_end, w_end = DataInterface.get_random_crop_args_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        ratio = res_width / res_height

        self.assertEqual(round(original_ratio,1), round(ratio,1))

        area_ratio = (res_height * res_width) / (batch_height * batch_width)

        self.assertGreaterEqual(crop_area_ratio_range[1], round(area_ratio,1))
        self.assertLessEqual(crop_area_ratio_range[0], round(area_ratio,1))

    def test_get_random_crop_within_ratio_range_given_target_area(self):
        from pytorch_h5dataset.fn.data_interface import DataInterface
        crop_size_aspect_ratio = (0.7, 0.7)
        crop_area_ratio_range = (0.5, 0.5)

        batch_height = 217
        batch_width = 244
        target_area = 217 * 244 * 0.476

        h_begin, w_begin, h_end, w_end = DataInterface.get_random_crop_args_within_ratio_range_given_target_area(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width, target_area)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        ratio = res_width / res_height
        self.assertGreaterEqual(crop_size_aspect_ratio[1], round(ratio,2))
        self.assertLessEqual(crop_size_aspect_ratio[0], round(ratio,2))

        area_ratio = (res_height * res_width) / target_area
        self.assertEqual(1.0, round(area_ratio,1))

        crop_size_aspect_ratio = (0.7, 1.3)
        crop_area_ratio_range = (0.5, 0.7)

        batch_height = 217
        batch_width = 244
        target_area = 217 * 244 * 0.33

        h_begin, w_begin, h_end, w_end = DataInterface.get_random_crop_args_within_ratio_range_given_target_area(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width, target_area)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        ratio = res_width / res_height
        self.assertGreaterEqual(crop_size_aspect_ratio[1], round(ratio,2))
        self.assertLessEqual(crop_size_aspect_ratio[0], round(ratio,2))

        area_ratio = (res_height * res_width) / target_area
        self.assertEqual(1.0, round(area_ratio,1))


        area_ratio_in = 1/3.
        crop_size_aspect_ratio = (0.1, .1)
        crop_area_ratio_range = (area_ratio_in, area_ratio_in)
        original_ratio = 244. / 217


        batch_height = 217
        batch_width = 244
        target_area = 217 * 244 * area_ratio

        h_begin, w_begin, h_end, w_end = DataInterface.get_random_crop_args_within_ratio_range_given_target_area(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width, target_area)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        ratio = res_width / res_height

        self.assertEqual(round(original_ratio,1), round(ratio,1))

        area_ratio = (res_height * res_width) / target_area
        self.assertEqual(round(area_ratio_in,1), round(area_ratio,1))

    def test_get_random_crop_within_ratio_range_given_fixed_side(self):
        from pytorch_h5dataset.fn.blosc import DataInterface
        crop_height, crop_width = 133, None
        crop_area_ratio_range = (0.5, 0.5)

        batch_height = 217
        batch_width = 244

        h_begin, w_begin, h_end, w_end = DataInterface.get_random_crop_args_within_ratio_range_given_fixed_side(crop_height, crop_width, crop_area_ratio_range,
                                                                                                       batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        self.assertEqual(res_height , crop_height)


        area_ratio = (res_height * res_width) / (batch_height * batch_width)

        self.assertGreaterEqual(crop_area_ratio_range[1], round(area_ratio,1))
        self.assertLessEqual(crop_area_ratio_range[0], round(area_ratio,1))

        crop_height, crop_width = None, 122
        crop_area_ratio_range = (0.5, 0.5)

        batch_height = 217
        batch_width = 244

        h_begin, w_begin, h_end, w_end = DataInterface.get_random_crop_args_within_ratio_range_given_fixed_side(crop_height, crop_width, crop_area_ratio_range,
                                                                                                       batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        self.assertEqual(res_width, crop_width)


        area_ratio = (res_height * res_width) / (batch_height * batch_width)

        self.assertGreaterEqual(crop_area_ratio_range[1], round(area_ratio,1))
        self.assertLessEqual(crop_area_ratio_range[0], round(area_ratio,1))


        crop_height, crop_width = 133, 133
        crop_area_ratio_range = (0.5, 0.5)

        batch_height = 217
        batch_width = 244

        h_begin, w_begin, h_end, w_end = DataInterface.get_random_crop_args_within_ratio_range_given_fixed_side(crop_height, crop_width, crop_area_ratio_range,
                                                                                                       batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        self.assertEqual(res_height , crop_height)
        self.assertEqual(res_width, crop_width)

        crop_height, crop_width = None, 450
        crop_area_ratio_range = (0.7, 1.0)

        batch_height = 217
        batch_width = 244

        h_begin, w_begin, h_end, w_end = DataInterface.get_random_crop_args_within_ratio_range_given_fixed_side(crop_height, crop_width, crop_area_ratio_range,
                                                                                                       batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)


        area_ratio = (res_height * res_width) / (batch_height * batch_width)

        self.assertGreaterEqual(crop_area_ratio_range[1], round(area_ratio,1))
        self.assertLessEqual(crop_area_ratio_range[0], round(area_ratio,1))

    def test_random_crop(self):
        import numpy as np
        from pytorch_h5dataset.fn.blosc import BloscInterface
        h5_group_dummy = np.random.rand(50,5,400,512)
        batch_height, batch_width = 400, 512
        batch_area = batch_height * batch_width
        crop_size = (None, 480)
        crop_area_ratio_range = 0.7
        crop = BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]
        crop_area=np.prod(crop_shape)
        self.assertEqual(round(crop_area_ratio_range,1), round(crop_area/batch_area,1))

        h5_group_dummy = np.random.rand(50,5,144,244)
        batch_height, batch_width = 144, 244
        batch_area =batch_height * batch_width
        crop_size = (120, None)
        crop_area_ratio_range = 0.8
        crop = BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]
        crop_area=np.prod(crop_shape)
        self.assertEqual(round(crop_area_ratio_range,1), round(crop_area/batch_area,1))

        h5_group_dummy = np.random.rand(50,5,144,244)
        batch_height, batch_width = 144, 244
        batch_area = batch_height * batch_width
        crop_size = (250, None)
        crop_area_ratio_range = 0.66
        crop = BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]
        crop_area=np.prod(crop_shape)
        self.assertEqual(round(crop_area_ratio_range,1), round(crop_area/batch_area,1))

        h5_group_dummy = np.random.rand(50,5,144,244)
        batch_height, batch_width = 144, 244
        crop_size = (250, 250)
        crop_area_ratio_range = 0.8
        crop = BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]

        self.assertTrue(all(a==b for a,b in zip(crop_shape,(batch_height, batch_width))))

        h5_group_dummy = np.random.rand(50,5,144,244)
        batch_height, batch_width = 144, 244
        crop_size = (110, 120)
        crop_area_ratio_range = 0.8
        crop = BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]
        self.assertTrue(all(a==b for a,b in zip(crop_shape,crop_size)))

        ## crop crop_size is given as float
        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        batch_area = batch_height * batch_width
        crop_size = (0.7)
        crop_area_ratio_range = 0.2
        crop = BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]
        crop_ratio = crop_shape[1]/crop_shape[0]
        crop_area = np.prod(crop_shape)
        self.assertEqual(round(crop_ratio, 1), round(crop_size ,1))
        self.assertEqual(round(crop_area/batch_area,1), crop_area_ratio_range)

        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        batch_area = batch_height * batch_width
        crop_size = (1.3)
        crop_area_ratio_range = 0.6
        crop = BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]
        crop_ratio = crop_shape[1]/crop_shape[0]
        crop_area = np.prod(crop_shape)
        self.assertEqual(round(crop_ratio, 1), round(crop_size ,1))
        self.assertEqual(round(crop_area/batch_area,1), crop_area_ratio_range)

        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        batch_area = batch_height * batch_width
        crop_size = (0.7, 1.3)
        crop_area_ratio_range = (0.6, 0.7)
        crop = BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]
        crop_ratio = crop_shape[1]/crop_shape[0]
        crop_area = np.prod(crop_shape)
        self.assertGreaterEqual(round(crop_ratio, 1), round(crop_size[0] ,1))
        self.assertLessEqual(round(crop_ratio, 1), round(crop_size[1] ,1))
        self.assertGreaterEqual(round(crop_area/batch_area,1), crop_area_ratio_range[0])
        self.assertLessEqual(round(crop_area/batch_area,1), crop_area_ratio_range[1])

        ## crop_area is given as number of pixels
        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        crop_size = (0.7, 1.3)
        crop_area_ratio_range = 150*200
        crop = BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]
        crop_ratio = crop_shape[1]/crop_shape[0]
        crop_area = np.prod(crop_shape)
        self.assertGreaterEqual(round(crop_ratio, 1), round(crop_size[0] ,1))
        self.assertLessEqual(round(crop_ratio, 1), round(crop_size[1] ,1))
        self.assertAlmostEqual(crop_area, crop_area_ratio_range, delta=int(crop_area*.05))

        ## Failure Cases

        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244

        crop_size = (0.7, 1.3)
        crop_area_ratio_range = (0.6, 1.7)

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

        crop_size = (0.7, 1.3)
        crop_area_ratio_range = (0.6, 1)

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

        crop_size = (0.7, 1.3)
        crop_area_ratio_range = (1, 0.7)

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

        crop_size = (0.7, 1.3)
        crop_area_ratio_range = (None, 0.7)

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

        crop_size = (0.7, 1.3)
        crop_area_ratio_range = (1.0, None)

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

        crop_size = (0.7, 1.3)
        crop_area_ratio_range = (None, None)

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

        crop_size = (0.7, 1.3)
        crop_area_ratio_range = (None)

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

        crop_size = (0.7, 1.3)
        crop_area_ratio_range = None

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)


        crop_size = (0.7, None)
        crop_area_ratio_range = (.8, 1.0)

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

        crop_size = (None, None)
        crop_area_ratio_range = (.8, 1.0)

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

        crop_size = (None, 0.7)
        crop_area_ratio_range = (.8, 1.0)

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

        crop_size = (None)
        crop_area_ratio_range = (.8, 1.0)

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

        crop_size = None
        crop_area_ratio_range = (.8, 1.0)

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

        crop_size = 127
        crop_area_ratio_range = (.8, 1.0)

        with self.assertRaises(AssertionError):
            BloscInterface.random_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)



    def test_center_crop(self):
        import numpy as np
        from pytorch_h5dataset.fn.blosc import BloscInterface
        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        crop_height, crop_width = (150, 180)
        crop = BloscInterface.center_crop(h5_group_dummy, batch_height, batch_width, crop_height, crop_width)
        cropped_height, cropped_width = crop.shape[-2:]

        self.assertEqual((cropped_height, cropped_width), (crop_height, crop_width))

        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        crop_height, crop_width = (240, 180)
        crop = BloscInterface.center_crop(h5_group_dummy, batch_height, batch_width, crop_height, crop_width)
        cropped_height, cropped_width = crop.shape[-2:]
        self.assertEqual((cropped_height, cropped_width), (batch_height, crop_width))

        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        crop_height, crop_width = (180, 250)
        crop = BloscInterface.center_crop(h5_group_dummy, batch_height, batch_width, crop_height, crop_width)
        cropped_height, cropped_width = crop.shape[-2:]
        self.assertEqual((cropped_height, cropped_width), (crop_height, batch_width))

        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        crop_height, crop_width = (240, 250)
        crop = BloscInterface.center_crop(h5_group_dummy, batch_height, batch_width, crop_height, crop_width)
        cropped_height, cropped_width = crop.shape[-2:]
        self.assertEqual((cropped_height, cropped_width), (batch_height, batch_width))


    def test_center_crop_as_tensor(self):
        import numpy as np
        import torch
        from pytorch_h5dataset.fn.blosc import BloscInterface
        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        crop_height, crop_width = (150, 180)
        crop = BloscInterface.center_crop_as_tensor(h5_group_dummy, batch_height, batch_width, crop_height, crop_width)
        self.assertIsInstance(crop, torch.Tensor)

    def test_crop_original_sample_from_batch(self):
        import numpy as np
        from pytorch_h5dataset.fn.blosc import BloscInterface
        batch = np.random.rand(3,5,200,244)
        shapes = ((50,144),(177,34),(190,234))
        crops = BloscInterface.crop_original_samples_from_batch(batch, shapes)
        crop_shapes = tuple(crop.shape[-2:] for crop in crops)
        self.assertEqual(crop_shapes, shapes)

        batch = np.random.rand(3,5,200,244)
        shapes = ((50,144),(177,34),(190,1000))
        crops = BloscInterface.crop_original_samples_from_batch(batch, shapes)
        crop_shapes = tuple(crop.shape[-2:] for crop in crops)
        right_shapes = ((50,144),(177,34),(190,244))
        self.assertEqual(crop_shapes, right_shapes)

    def test_convert_samples_to_dataset(self):
        import pandas as pd
        import shutil as sh
        import numpy as np

        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        _H5Dataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5',
                                              sub_batch_size=1)
        dataset = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/',
                             tr_crop_strategy='random',
                             tr_crop_size=(0.8,1.3),
                             tr_crop_area_ratio_range=244*244)
        im, (cl, idx) = dataset[0]

        self.assertAlmostEqual(np.prod(im[0].shape)/10000, 3*244*244/10000,0)


        del dataset
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)

        _H5Dataset._convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset_image.h5',
                                              data_mode='image', sub_batch_size=1)

        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset_image.csv')
        dataset = imageDataset.ImageDataset('test_dataset_image', './test/data/tmp/dataset/h5/',
                                            decode='cpu',
                            tr_crop_strategy='random',
                            tr_crop_size=(0.8,1.3),
                            tr_crop_area_ratio_range=244*244)
        im, (cl, idx) = dataset[0]
        self.assertAlmostEqual(np.prod(im[0].shape)/10000, 3*244*244/10000,0)
        self.assertEqual(cl.item(), 1)
        del dataset

        imageDataset.ImageDataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset_image.h5',
                                              sub_batch_size=2)

        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset_image.csv')
        dataset = imageDataset.ImageDataset('test_dataset_image', './test/data/tmp/dataset/h5/',
                                            decode='cpu',
                                            tr_crop_strategy='random',
                                            tr_crop_size=(0.8,1.3),
                                            tr_crop_area_ratio_range=244*244)
        im, (cl, idx) = dataset[0]

        self.assertAlmostEqual(np.prod(im[0].shape)/10000, 3*244*244/10000,0)
        self.assertListEqual(cl.tolist(), [1,0])
        del dataset


    def test_create_metadata_for_dataset(self):

        dataframe = _H5Dataset.create_metadata_for_dataset('./test/data/images/rgb/')
        self.assertEqual(len(dataframe),2)
        self.assertTrue(all(dataframe.columns.values == ['FilePath', 'ClassNo', 'Index', 'ClassFolderName', 'FileType']))

        dataframe = _H5Dataset.create_metadata_for_dataset('./test/data/images/rgb/', lambda a : dict(zip(['Filename', 'class'], a[:-4].split('_'))))
        self.assertEqual(len(dataframe),2)
        self.assertEqual(dataframe.loc[1]['ClassNo'],1)
        self.assertTrue(all(dataframe.columns.values == ['FilePath', 'ClassNo', 'Index', 'ClassFolderName', 'FileType', 'Filename', 'class']))


        dataframe = _H5Dataset.create_metadata_for_dataset('./test/data/images/rgb/', lambda a : dict(zip(['Filename', 'class'], a[:-4].split('_'))), True)
        self.assertEqual(dataframe.loc[1]['ClassNo'],0)
        self.assertEqual(len(dataframe),2)
        self.assertTrue(all(dataframe.columns.values == ['FilePath', 'ClassNo', 'Index', 'ClassFolderName', 'FileType', 'Filename', 'class']))


        dataframe = _H5Dataset.create_metadata_for_dataset('./test/data/images/', lambda a : dict(zip(['Filename', 'class'], a[:-4].split('_'))), True)
        self.assertEqual(dataframe.loc[1]['ClassNo'],0)
        self.assertEqual(len(dataframe),4)
        self.assertTrue(all(dataframe.columns.values == ['FilePath', 'ClassNo', 'Index', 'ClassFolderName', 'FileType', 'Filename', 'class']))

        dataframe = _H5Dataset.create_metadata_for_dataset('./test/data/images/rgb/a/', lambda a : dict(zip(['Filename', 'class'], a[:-4].split('_'))), True)
        self.assertEqual(dataframe.loc[0]['ClassNo'],0)
        self.assertEqual(len(dataframe),1)





    def test___len__(self):
        import pandas as pd
        import shutil as sh
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        _H5Dataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        self.assertEqual(len(dataset),1)

        import pandas as pd
        import shutil as sh
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        _H5Dataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5', 1)
        dataset = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        self.assertEqual(len(dataset),2)


    def test___getitem__(self):
        import pandas as pd
        import shutil as sh
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        _H5Dataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        sample, (meta_class, meta_indices) = dataset[0]
        self.assertEqual(len(sample), 2)
        self.assertTrue(all(a is not None for a in [sample, meta_class, meta_indices]))
        self.assertEqual(len(dataset),1)
        del dataset

    def test_get_meta_data_from_indices(self):
        import pandas as pd
        import shutil as sh
        import numpy as np
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        _H5Dataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        sample, (meta_class, meta_indices) = dataset[0]
        meta_indices = np.array(meta_indices,dtype=int)
        test_meta = dataset.get_meta_data_from_indices(meta_indices)
        gt_meta = dataset.metadata[dataset.metadata['Index'].isin(meta_indices)]
        pd.testing.assert_frame_equal(test_meta, gt_meta)
        del dataset

        dataframe = pd.read_csv('./test/data/test_dataset_multichannel.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset_multichannel.csv','./test/data/tmp/dataset/h5/test_dataset_multichannel.csv')
        _H5Dataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset_multichannel.h5')
        dataset = _H5Dataset('test_dataset_multichannel', './test/data/tmp/dataset/h5/')
        sample, (meta_class, meta_indices) = dataset[0]
        meta_indices = np.array(meta_indices,dtype=int)
        test_meta = dataset.get_meta_data_from_indices(meta_indices)
        gt_meta = dataset.metadata[dataset.metadata['Index'].isin(meta_indices)]
        pd.testing.assert_frame_equal(test_meta, gt_meta)
        del dataset

    # def test_initiate_crop_function(self):
    #     import pandas as pd
    #     import shutil as sh
    #     import numpy as np
    #     dataframe = pd.read_csv('./test/data/test_dataset.csv')
    #     os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
    #     sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
    #     _H5Dataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
    #     dataset = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
    #     dataset.initiate_crop_function()
    #     self.assertIsNotNone(dataset.crop_function)
    #     sample , _ = dataset[0]
    #     self.assertAlmostEqual(np.prod(sample.shape[-2:])/100000., 244*244/100000., 2)
    #     del dataset

    def test___init__(self):
        import pandas as pd
        import shutil as sh
        import pickle as pkl
        from torchvision.transforms import Resize
        from torch import nn
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        _H5Dataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', tensor_transforms=Resize)
        # Following must be None to be picklable
        self.assertIsNone(dataset.h5_file)
        self.assertIsNone(dataset.crop_function)
        self.assertIsNotNone(dataset.transforms)
        self.assertIsNotNone(dataset.tensor_transforms)

        _ = pkl.dumps(dataset)

        dataset = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', tensor_transforms=nn.Sequential(Resize((12, 12))))
        sample, (meta_class, meta_indices) = dataset[0]
        with self.assertRaises(TypeError):
            pkl.dumps(dataset)



        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='montecarlo', split_ratio=1.0, split_number=-1)
        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='montecarlo', split_ratio=1.0, split_number=0.1)

        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='montecarlo', split_ratio=1, split_number=0)
        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='montecarlo', split_ratio=-.1, split_number=0)
        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='montecarlo', split_ratio=1.1, split_number=0)


        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='cross_val', split_ratio=1.1, split_number=5)
        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='cross_val', split_ratio=5, split_number=6)
        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='cross_val', split_ratio=0, split_number=0)

        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='split', split_ratio=0, split_number=0)
        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='split', split_ratio=0.1, split_number=0)
        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='split', split_ratio=[1, 2, 3], split_number=0)
        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='split', split_ratio=[0.1, 0.5, 0.5], split_number=0)
        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='split', split_ratio=[0.1, 0.5, 0.4], split_number=0)
        with self.assertRaises(AssertionError):
            _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='split', split_ratio=[0.1, 0.5, 0.4], split_number=3)


        d = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='montecarlo', split_ratio=.2, split_number=0)
        d.max_idx=99
        r = d.get_group_number_mapping()
        self.assertEqual(len(r), 20)

        d = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='cross_val', split_ratio=8, split_number=7)
        d.max_idx=99
        r = d.get_group_number_mapping() # lengths should be [12,13,13,12,12,13,12]
        self.assertEqual(len(r), 12)
        d = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='cross_val', split_ratio=8, split_number=6)
        d.max_idx=99
        r = d.get_group_number_mapping() # lengths should be [12,13,13,12,12,13,12]
        self.assertEqual(len(r), 13)

        d = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/', split_mode='split', split_ratio=(0.1, 0.4, 0.2, 0.2, 0.1), split_number=3)
        d.max_idx=99
        r = d.get_group_number_mapping()
        self.assertEqual(len(r), 20)




    def test___del__(self):
        import pandas as pd
        import shutil as sh

        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        _H5Dataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        del dataset

        dataset = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        sample, (meta_class, meta_indices) = dataset[0]
        del dataset

    def test_num_samples(self):
        import pandas as pd
        import shutil as sh

        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        _H5Dataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5', sub_batch_size=1)
        dataset = _H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        self.assertEqual(dataset.num_sub_batches, 2)


    def tearDown(self):
        '''
        Destructor cleans up tmp files after tests are done.
        :return:
        '''
        import shutil as sh
        try:
            sh.rmtree('./test/data/tmp')
        except:
            pass

    def test_imageDataset(self):
        import pandas as pd
        import shutil as sh

        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')


        imageDataset.ImageDataset.convert_samples_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset_image.h5',
                                               sub_batch_size=1)

        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset_image.csv')
        dataset = imageDataset.ImageDataset('test_dataset_image', './test/data/tmp/dataset/h5/',
                                            tr_crop_strategy='center',
                                            tr_crop_size= (0.3,3.0),
                                            tr_crop_area_ratio_range = (0.01,.1),
                                            tr_random_scale_range=(0.8,1.3),
                                            tr_random_rotation_angles=(-90,180,0,270,90),
                                            tr_output_size=(244,244),
                                            tr_random_flip='hv',
                                            decode='cuda',
                                            output_device='cuda')


        dataframe = pd.read_csv('./test/data/test_dataset_multichannel.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset_multichannel.csv','./test/data/tmp/dataset/h5/test_dataset_multichannel.csv')
        bloscDataset.BloscDataset.convert_samples_to_dataset(dataframe,
                                                             './test/data/tmp/dataset/h5/test_dataset_multichannel.h5'
                                                             ,sub_batch_size=2)
        from torchvision.transforms import Resize
        from torch.jit import script
        r = script(Resize((244,244)))
        dataset = bloscDataset.BloscDataset('test_dataset_multichannel',
                                            './test/data/tmp/dataset/h5/',
                                            tr_crop_strategy='original', tensor_transforms=r)

        im, meta = dataset[0,:]
        #self.assertAlmostEqual(np.prod(im[0].shape)/10000, 3*244*244/10000,0)

