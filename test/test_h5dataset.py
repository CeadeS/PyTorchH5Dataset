from unittest import TestCase

from pytorch_h5dataset import H5Dataset

import os
import pathlib
if  pathlib.Path(os.getcwd()).name == 'test':
    os.chdir('../')

andromeda_sample = {
    'path': './test/data/images/rgb/a/andromeda_0.png',
    'index': 1,
    'class': 0,
    'type': 'jpg',
    'height': 3694,
    'width': 6652,
    'size': 3694 * 6652,
    'shape': (3, 3694, 6652)
}

pollen_sample = {
    'path': './test/data/images/multichannel/a/pollen_0.tif',
    'index': 1,
    'class': 0,
    'type': 'tif',
    'height': 188,
    'width': 85,
    'size': 188 * 85,
    'shape': (24, 188, 85)
}

pano_sample = {
    'path': './test/data/images/rgb/b/pano_1.jpg',
    'index': 1,
    'class': 1,
    'type': 'jpg',
    'height': 3762,
    'width': 12706,
    'size': 3762 * 12706,
    'shape': (3, 3762, 12706)
}

something_sample = {
    'path': './test/data/images/multichannel/b/something_1.tif',
    'index': 1,
    'class': 1,
    'type': 'tif',
    'height': 22,
    'width': 24,
    'size': 22 * 24,
    'shape': (24, 22, 24)
}




class TestH5Dataset(TestCase):
    def test_load_image_sample(self):
        # jpg/png

        sample = {
            'FilePath': andromeda_sample['path'],
        }
        image = H5Dataset.load_image_sample(sample)
        self.assertEqual(image.shape, andromeda_sample['shape']), 'Loaded Image has the wrong shape'

        sample = {
            'FilePath': andromeda_sample['path'],
            'FileType': 'png'
        }
        image = H5Dataset.load_image_sample(sample)
        self.assertEqual(image.shape, andromeda_sample['shape']), 'Loaded Image has the wrong shape'

        sample = {
            'FilePath': pano_sample['path'],
            'FileType': 'jpg'
        }
        image = H5Dataset.load_image_sample(sample)
        self.assertEqual(image.shape, pano_sample['shape']), 'Loaded Image has the wrong shape'

        # tif
        sample = {
            'FilePath': something_sample['path'],
            'FileType': 'tif'
        }
        image = H5Dataset.load_image_sample(sample)
        self.assertEqual(image.shape, something_sample['shape']), 'Loaded Image has the wrong shape'

        sample = {
            'FilePath': something_sample['path'],
        }
        image = H5Dataset.load_image_sample(sample)
        self.assertEqual(image.shape, something_sample['shape']), 'Loaded Image has the wrong shape'

        sample = {
            'FilePath': pollen_sample['path'],
        }

        image = H5Dataset.load_image_sample(sample)
        self.assertEqual(image.shape, pollen_sample['shape']), 'Loaded Image has the wrong shape'

    def test_stack_batch_data_padded(self):
        # jpg/png
        l = [andromeda_sample, pano_sample]
        batch = H5Dataset.stack_batch_data_padded(l)
        self.assertEqual(len(batch),2)
        self.assertEqual(batch[0].shape, (3,3762,12706), batch[1].shape)

        l = [pano_sample, andromeda_sample]
        batch = H5Dataset.stack_batch_data_padded(l)
        self.assertEqual(len(batch),2)
        self.assertEqual(batch[0].shape, (3,3762,12706), batch[1].shape)

        # tifs
        l = [something_sample, pollen_sample]
        batch = H5Dataset.stack_batch_data_padded(l)
        self.assertEqual(len(batch),2)
        self.assertEqual(batch[0].shape, (24,188,85), batch[1].shape)

        l = [pollen_sample, something_sample]
        batch = H5Dataset.stack_batch_data_padded(l)
        self.assertEqual(len(batch),2)
        self.assertEqual(batch[0].shape, (24,188,85), batch[1].shape)

    def test_batchify_sorted_sample_data_list(self):
        # jpg/png
        sample_list = [andromeda_sample, pano_sample]
        batch_generator = H5Dataset.batchify_sorted_sample_data_list(sample_list)
        for batch_array, classes, shapes, indices in batch_generator:
            self.assertEqual(batch_array.shape, (2,3,3762,12706))
            (self.assertEqual(a,b) for a,b in zip(classes,[0,1]))
            (self.assertEqual(a,b) for a,b in zip(((3, 3694, 6652),(3, 3762, 12706))))

        # tif
        sample_list = [something_sample, pollen_sample]
        batch_generator = H5Dataset.batchify_sorted_sample_data_list(sample_list)
        for batch_array, classes, shapes, indices in batch_generator:
            self.assertEqual(batch_array.shape, (2,24,188,85))
            (self.assertEqual(a,b) for a,b in zip(classes,[0,1]))
            (self.assertEqual(a,b) for a,b in zip(((24, 22, 24),(24, 188, 85))))

    def test_get_central_crop_args(self):

        h_begin, w_begin, h_end, w_end = H5Dataset.get_central_crop_args(batch_width=244, batch_height=217, crop_area_ratio_range=(0.5,0.5))
        self.assertEqual(round(((-h_begin+h_end) * (-w_begin+w_end))/(244*217),1), 0.5)

        h_begin, w_begin, h_end, w_end = H5Dataset.get_central_crop_args(batch_width=244, batch_height=217, crop_area_ratio_range=(0.2,0.2))
        self.assertEqual(round(((-h_begin+h_end) * (-w_begin+w_end))/(244*217),1), 0.2)

        h_begin, w_begin, h_end, w_end = H5Dataset.get_central_crop_args(batch_width=244, batch_height=217, crop_area_ratio_range=(0.9,0.9))
        self.assertEqual(round(((-h_begin+h_end) * (-w_begin+w_end))/(244*217),1), 0.9)

        h_begin, w_begin, h_end, w_end = H5Dataset.get_central_crop_args(batch_width=244, batch_height=217, crop_area_ratio_range=(0.1,0.9))
        for _ in range(10):
            self.assertLessEqual(round(((-h_begin+h_end) * (-w_begin+w_end))/(244*217),1), 0.9)
            self.assertGreaterEqual(round(((-h_begin+h_end) * (-w_begin+w_end))/(244*217),1), 0.1)

        h_begin, w_begin, h_end, w_end = H5Dataset.get_central_crop_args(batch_width=244, batch_height=217, crop_area_ratio_range=(1.1,1.1))
        self.assertEqual(round(((-h_begin+h_end) * (-w_begin+w_end)),1), 244*217)

        kwargs = {'batch_width':244,
                  'batch_height':217,
                  'crop_area_ratio_range':(-1,-.5)}

        with self.assertRaises(ValueError):
            H5Dataset.get_central_crop_args(**kwargs)

    def test_get_random_crop_within_ratio_range_given_crop_size_ratio(self):
        crop_size_aspect_ratio = (0.7, 0.7)
        crop_area_ratio_range = (0.5, 0.5)
        original_ratio = 244. / 217
        batch_height = 217
        batch_width = 244
        h_begin, w_begin, h_end, w_end = H5Dataset.get_random_crop_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        ratio = res_width/ res_height
        self.assertGreaterEqual(crop_size_aspect_ratio[0], round(ratio,2))
        self.assertLessEqual(crop_size_aspect_ratio[1], round(ratio,2))

        area_ratio = (res_height * res_width) / (batch_height * batch_width)

        self.assertGreaterEqual(crop_area_ratio_range[0], round(area_ratio,2))
        self.assertLessEqual(crop_area_ratio_range[1], round(area_ratio,2))

        crop_size_aspect_ratio = (0.3, 0.7)
        crop_area_ratio_range = (0.5, 0.5)
        batch_height = 217
        batch_width = 244
        h_begin, w_begin, h_end, w_end = H5Dataset.get_random_crop_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width)
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
        h_begin, w_begin, h_end, w_end = H5Dataset.get_random_crop_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        ratio = res_width / res_height

        self.assertGreaterEqual(crop_size_aspect_ratio[1], round(ratio,1))
        self.assertLessEqual(crop_size_aspect_ratio[0], round(ratio,1))

        area_ratio = (res_height * res_width) / (batch_height * batch_width)

        self.assertGreaterEqual(crop_area_ratio_range[1], round(area_ratio,1))
        self.assertLessEqual(crop_area_ratio_range[0], round(area_ratio,1))

        crop_size_aspect_ratio = (0.73, 1.33)
        crop_area_ratio_range = (0.4, 0.6)
        batch_height = 217
        batch_width = 244
        h_begin, w_begin, h_end, w_end = H5Dataset.get_random_crop_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width)
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
        h_begin, w_begin, h_end, w_end = H5Dataset.get_random_crop_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        ratio = res_width / res_height

        self.assertEqual(round(original_ratio,1), round(ratio,1))

        area_ratio = (res_height * res_width) / (batch_height * batch_width)

        self.assertGreaterEqual(crop_area_ratio_range[1], round(area_ratio,1))
        self.assertLessEqual(crop_area_ratio_range[0], round(area_ratio,1))

    def test_get_random_crop_within_ratio_range_given_target_area(self):
        crop_size_aspect_ratio = (0.7, 0.7)
        crop_area_ratio_range = (0.5, 0.5)

        batch_height = 217
        batch_width = 244
        target_area = 217 * 244 * 0.476

        h_begin, w_begin, h_end, w_end = H5Dataset.get_random_crop_within_ratio_range_given_target_area(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width, target_area)
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

        h_begin, w_begin, h_end, w_end = H5Dataset.get_random_crop_within_ratio_range_given_target_area(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width, target_area)
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

        h_begin, w_begin, h_end, w_end = H5Dataset.get_random_crop_within_ratio_range_given_target_area(crop_size_aspect_ratio, crop_area_ratio_range, batch_height, batch_width, target_area)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        ratio = res_width / res_height

        self.assertEqual(round(original_ratio,1), round(ratio,1))

        area_ratio = (res_height * res_width) / target_area
        self.assertEqual(round(area_ratio_in,1), round(area_ratio,1))

    def test_get_random_crop_within_ratio_range_given_fixed_side(self):
        crop_height, crop_width = 133, None
        crop_area_ratio_range = (0.5, 0.5)

        batch_height = 217
        batch_width = 244

        h_begin, w_begin, h_end, w_end = H5Dataset.get_random_crop_within_ratio_range_given_fixed_side(crop_height, crop_width, crop_area_ratio_range,
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

        h_begin, w_begin, h_end, w_end = H5Dataset.get_random_crop_within_ratio_range_given_fixed_side(crop_height, crop_width, crop_area_ratio_range,
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

        h_begin, w_begin, h_end, w_end = H5Dataset.get_random_crop_within_ratio_range_given_fixed_side(crop_height, crop_width, crop_area_ratio_range,
                                                                                                       batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)
        self.assertEqual(res_height , crop_height)
        self.assertEqual(res_width, crop_width)

        crop_height, crop_width = None, 450
        crop_area_ratio_range = (0.7, 1.0)

        batch_height = 217
        batch_width = 244

        h_begin, w_begin, h_end, w_end = H5Dataset.get_random_crop_within_ratio_range_given_fixed_side(crop_height, crop_width, crop_area_ratio_range,
                                                                                                       batch_height, batch_width)
        res_height, res_width = (-h_begin+h_end), (-w_begin+w_end)


        area_ratio = (res_height * res_width) / (batch_height * batch_width)

        self.assertGreaterEqual(crop_area_ratio_range[1], round(area_ratio,1))
        self.assertLessEqual(crop_area_ratio_range[0], round(area_ratio,1))

    def test_random_located_sized_crop(self):
        import numpy as np
        ## crop size given as ints
        h5_group_dummy = np.random.rand(50,5,144,244)
        batch_height, batch_width = 144, 244
        batch_area =batch_height * batch_width
        crop_size = (None, 300)
        crop_area_ratio_range = 0.7
        crop = H5Dataset.random_located_sized_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]
        crop_area=np.prod(crop_shape)
        self.assertEqual(round(crop_area_ratio_range,1), round(crop_area/batch_area,1))

        h5_group_dummy = np.random.rand(50,5,144,244)
        batch_height, batch_width = 144, 244
        batch_area =batch_height * batch_width
        crop_size = (120, None)
        crop_area_ratio_range = 0.9
        crop = H5Dataset.random_located_sized_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]
        crop_area=np.prod(crop_shape)
        self.assertEqual(round(crop_area_ratio_range,1), round(crop_area/batch_area,1))

        h5_group_dummy = np.random.rand(50,5,144,244)
        batch_height, batch_width = 144, 244
        batch_area =batch_height * batch_width
        crop_size = (250, None)
        crop_area_ratio_range = 0.66
        crop = H5Dataset.random_located_sized_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]
        crop_area=np.prod(crop_shape)
        self.assertEqual(round(crop_area_ratio_range,1), round(crop_area/batch_area,1))

        h5_group_dummy = np.random.rand(50,5,144,244)
        batch_height, batch_width = 144, 244
        crop_size = (250, 250)
        crop_area_ratio_range = 0.8
        crop = H5Dataset.random_located_sized_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]

        self.assertTrue(all(a==b for a,b in zip(crop_shape,(batch_height, batch_width))))

        h5_group_dummy = np.random.rand(50,5,144,244)
        batch_height, batch_width = 144, 244
        crop_size = (110, 120)
        crop_area_ratio_range = 0.8
        crop = H5Dataset.random_located_sized_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]
        self.assertTrue(all(a==b for a,b in zip(crop_shape,crop_size)))

        ## crop size is given as float
        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        batch_area = batch_height * batch_width
        crop_size = (0.7)
        crop_area_ratio_range = 0.2
        crop = H5Dataset.random_located_sized_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
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
        crop = H5Dataset.random_located_sized_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
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
        crop = H5Dataset.random_located_sized_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
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
        crop = H5Dataset.random_located_sized_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)
        crop_shape = crop.shape[-2:]
        crop_ratio = crop_shape[1]/crop_shape[0]
        crop_area = np.prod(crop_shape)
        self.assertGreaterEqual(round(crop_ratio, 1), round(crop_size[0] ,1))
        self.assertLessEqual(round(crop_ratio, 1), round(crop_size[1] ,1))
        self.assertAlmostEqual(crop_area, crop_area_ratio_range, delta=int(crop_area*.05))

        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        crop_size = (0.7, 1.3)
        crop_area_ratio_range = (0.6, 1.7)

        with self.assertRaises(AssertionError):
            H5Dataset.random_located_sized_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        crop_size = (0.7, 1.3)
        crop_area_ratio_range = (0.6, 1.7)

        with self.assertRaises(AssertionError):
            H5Dataset.random_located_sized_crop(h5_group_dummy, batch_height, batch_width, crop_size, crop_area_ratio_range)

    def test_center_crop(self):
        import numpy as np
        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        crop_height, crop_width = (150, 180)
        crop = H5Dataset.center_crop(h5_group_dummy, batch_height, batch_width, crop_height, crop_width)
        cropped_height, cropped_width = crop.shape[-2:]

        self.assertEqual((cropped_height, cropped_width), (crop_height, crop_width))

        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        crop_height, crop_width = (240, 180)
        crop = H5Dataset.center_crop(h5_group_dummy, batch_height, batch_width, crop_height, crop_width)
        cropped_height, cropped_width = crop.shape[-2:]
        self.assertEqual((cropped_height, cropped_width), (batch_height, crop_width))

        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        crop_height, crop_width = (180, 250)
        crop = H5Dataset.center_crop(h5_group_dummy, batch_height, batch_width, crop_height, crop_width)
        cropped_height, cropped_width = crop.shape[-2:]
        self.assertEqual((cropped_height, cropped_width), (crop_height, batch_width))

        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        crop_height, crop_width = (240, 250)
        crop = H5Dataset.center_crop(h5_group_dummy, batch_height, batch_width, crop_height, crop_width)
        cropped_height, cropped_width = crop.shape[-2:]
        self.assertEqual((cropped_height, cropped_width), (batch_height, batch_width))


    def test_center_crop_as_tensor(self):
        import numpy as np
        import torch
        h5_group_dummy = np.random.rand(50,5,200,244)
        batch_height, batch_width = 200, 244
        crop_height, crop_width = (150, 180)
        crop = H5Dataset.center_crop_as_tensor(h5_group_dummy, batch_height, batch_width, crop_height, crop_width)
        self.assertIsInstance(crop, torch.Tensor)

    def test_crop_original_image_from_batch(self):
        import numpy as np
        batch = np.random.rand(3,5,200,244)
        shapes = ((50,144),(177,34),(190,234))
        crops = H5Dataset.crop_original_image_from_batch(batch, shapes)
        crop_shapes = tuple(crop.shape[-2:] for crop in crops)
        self.assertEqual(crop_shapes, shapes)

        batch = np.random.rand(3,5,200,244)
        shapes = ((50,144),(177,34),(190,1000))
        crops = H5Dataset.crop_original_image_from_batch(batch, shapes)
        crop_shapes = tuple(crop.shape[-2:] for crop in crops)
        right_shapes = ((50,144),(177,34),(190,244))
        self.assertEqual(crop_shapes, right_shapes)

    def test_convert_images_to_dataset(self):
        import pandas as pd
        import shutil as sh
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        H5Dataset.convert_images_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dastaset.h5')


    def test_create_metadata_for_dataset(self):

        dataframe = H5Dataset.create_metadata_for_dataset('./test/data/images/rgb/')
        self.assertEqual(len(dataframe),2)
        self.assertTrue(all(dataframe.columns.values == ['FilePath', 'ClassNo', 'Index', 'ClassFolderName', 'FileType']))

        dataframe = H5Dataset.create_metadata_for_dataset('./test/data/images/rgb/', lambda a : dict(zip(['Filename','class'], a[:-4].split('_'))))
        self.assertEqual(len(dataframe),2)
        self.assertEqual(dataframe.loc[1]['ClassNo'],1)
        self.assertTrue(all(dataframe.columns.values == ['FilePath', 'ClassNo', 'Index', 'ClassFolderName', 'FileType', 'Filename', 'class']))


        dataframe = H5Dataset.create_metadata_for_dataset('./test/data/images/rgb/', lambda a : dict(zip(['Filename','class'], a[:-4].split('_'))), True)
        self.assertEqual(dataframe.loc[1]['ClassNo'],0)
        self.assertEqual(len(dataframe),2)
        self.assertTrue(all(dataframe.columns.values == ['FilePath', 'ClassNo', 'Index', 'ClassFolderName', 'FileType', 'Filename', 'class']))


        dataframe = H5Dataset.create_metadata_for_dataset('./test/data/images/', lambda a : dict(zip(['Filename','class'], a[:-4].split('_'))), True)
        self.assertEqual(dataframe.loc[1]['ClassNo'],0)
        self.assertEqual(len(dataframe),4)
        self.assertTrue(all(dataframe.columns.values == ['FilePath', 'ClassNo', 'Index', 'ClassFolderName', 'FileType', 'Filename', 'class']))

        dataframe = H5Dataset.create_metadata_for_dataset('./test/data/images/rgb/a/', lambda a : dict(zip(['Filename','class'], a[:-4].split('_'))), True)
        self.assertEqual(dataframe.loc[0]['ClassNo'],0)
        self.assertEqual(len(dataframe),1)

    def test___len__(self):
        import pandas as pd
        import shutil as sh
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        H5Dataset.convert_images_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        self.assertEqual(len(dataset),1)

        import pandas as pd
        import shutil as sh
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        H5Dataset.convert_images_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5',1)
        dataset = H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        self.assertEqual(len(dataset),2)


    def test___getitem__(self):
        import pandas as pd
        import shutil as sh
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        H5Dataset.convert_images_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        sample, (meta_class, meta_indices) = dataset[0]
        self.assertEqual(len(sample), 2)
        self.assertTrue(all(a is not None for a in [sample, meta_class, meta_indices]))
        self.assertEqual(len(dataset),1)
        del dataset

    def test_get_meta_data_from_indices(self):
        import pandas as pd
        import shutil as sh
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        H5Dataset.convert_images_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        sample, (meta_class, meta_indices) = dataset[0]
        test_meta = dataset.get_meta_data_from_indices(meta_indices)
        gt_meta = dataset.metadata[dataset.metadata['Index'].isin(meta_indices)]
        pd.testing.assert_frame_equal(test_meta, gt_meta)
        del dataset

    def test_initiate_crop_function(self):
        import pandas as pd
        import shutil as sh
        import numpy as np
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        H5Dataset.convert_images_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        dataset.initiate_crop_function()
        self.assertIsNotNone(dataset.crop_function)
        sample , _ = dataset[0]
        self.assertAlmostEqual(np.prod(sample.shape[-2:])/100000., 244*244/100000., 2)
        del dataset

    def test___init__(self):
        import pandas as pd
        import shutil as sh
        import pickle as pkl
        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        H5Dataset.convert_images_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        # Following must be None to be picklable
        self.assertIsNone(dataset.h5_file)
        self.assertIsNone(dataset.crop_function)
        self.assertIsNone(dataset.script_transform)
        _ = pkl.dumps(dataset)

        sample, (meta_class, meta_indices) = dataset[0]
        with self.assertRaises(TypeError):
            pkl.dumps(dataset)
        del dataset


    def test___del__(self):
        import pandas as pd
        import shutil as sh

        dataframe = pd.read_csv('./test/data/test_dataset.csv')
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        sh.copy('./test/data/test_dataset.csv','./test/data/tmp/dataset/h5/test_dataset.csv')
        H5Dataset.convert_images_to_dataset(dataframe, './test/data/tmp/dataset/h5/test_dataset.h5')
        dataset = H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        del dataset

        dataset = H5Dataset('test_dataset', './test/data/tmp/dataset/h5/')
        sample, (meta_class, meta_indices) = dataset[0]
        del dataset


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
