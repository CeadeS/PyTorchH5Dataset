from unittest import TestCase

from h5dataset import H5Dataset

andromeda_sample = {
    'path': './test/data/images/andromeda.png',
    'index': 1,
    'class': 0,
    'type': 'jpg',
    'height': 3694,
    'width': 6652,
    'size': 3694 * 6652,
    'shape': (3, 3694, 6652)
}

pano_sample = {
    'path': './test/data/images/pano.jpg',
    'index': 1,
    'class': 1,
    'type': 'jpg',
    'height': 3762,
    'width': 12706,
    'size': 3762 * 12706,
    'shape': (3, 3762, 12706)
}

something_sample = {
    'path': './test/data/images/something.tif',
    'index': 1,
    'class': 0,
    'type': 'tif',
    'height': 22,
    'width': 24,
    'size': 22 * 24,
    'shape': (24, 22, 24)
}

pollen_sample = {
    'path': './test/data/images/pollen.tif',
    'index': 1,
    'class': 1,
    'type': 'tif',
    'height': 188,
    'width': 85,
    'size': 188 * 85,
    'shape': (24, 188, 85)
}


class TestH5Dataset(TestCase):
    def test_load_image_sample(self):
        # jpg/png

        sample = {
            'FilePath': './test/data/images/andromeda.png',
        }
        image = H5Dataset.load_image_sample(sample)
        self.assertEqual(image.shape, andromeda_sample['shape']), 'Loaded Image has the wrong shape'

        sample = {
            'FilePath': './test/data/images/andromeda.png',
            'FileType': 'png'
        }
        image = H5Dataset.load_image_sample(sample)
        self.assertEqual(image.shape, andromeda_sample['shape']), 'Loaded Image has the wrong shape'

        sample = {
            'FilePath': './test/data/images/pano.jpg',
            'FileType': 'jpg'
        }
        image = H5Dataset.load_image_sample(sample)
        self.assertEqual(image.shape, pano_sample['shape']), 'Loaded Image has the wrong shape'

        # tif
        sample = {
            'FilePath': './test/data/images/something.tif',
            'FileType': 'tif'
        }
        image = H5Dataset.load_image_sample(sample)
        self.assertEqual(image.shape, something_sample['shape']), 'Loaded Image has the wrong shape'

        sample = {
            'FilePath': './test/data/images/something.tif',
        }
        image = H5Dataset.load_image_sample(sample)
        self.assertEqual(image.shape, something_sample['shape']), 'Loaded Image has the wrong shape'

        sample = {
            'FilePath': './test/data/images/pollen.tif',
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

        self.assertGreaterEqual(crop_area_ratio_range[1], round(area_ratio,2))
        self.assertLessEqual(crop_area_ratio_range[0], round(area_ratio,2))

        crop_size_aspect_ratio = (0.1, 0.7)
        crop_area_ratio_range = (0.5, 0.5)
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


        crop_size_aspect_ratio = (0.1, 0.7)
        crop_area_ratio_range = (0.1, 0.8)
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
        crop_area_ratio_range = (0.1, 0.6)
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
