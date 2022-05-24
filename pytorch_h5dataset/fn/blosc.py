import torch
import numpy as np
from math import sqrt
import imghdr
from .data_interface import DataInterface
from torch import as_tensor, uint8


class BloscInterface(DataInterface):

    @staticmethod
    def blosc_opts(complevel=9, complib='blosc:lz4hc', shuffle: any = 'bit'):
        """
        Code from https://github.com/h5py/h5py/issues/611#issuecomment-497834183 for issue https://github.com/h5py/h5py/issues/611
        :param complevel:
        :param complib:
        :param shuffle:
        :return:
        """
        shuffle = 2 if shuffle == 'bit' else 1 if shuffle else 0
        compressors = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
        complib = ['blosc:' + c for c in compressors].index(complib)
        args = {
            'compression': 32001,
            'compression_opts': (0, 0, 0, 0, complevel, shuffle, complib)
        }
        if shuffle:
            args['shuffle'] = False
        return args

    @staticmethod
    def load_sample(sample):
        from PIL import Image
        from skimage import io as skio
        import io

        path_key = 'path' if 'path' in sample.keys() else 'FilePath'
        type_key = 'type' if 'type' in sample.keys() else 'FileType'
        try:
            file_type = sample[type_key]
        except KeyError:
            file_type = imghdr.what(sample[path_key])

        if file_type in ['tif', 'tiff']:
            spl = skio.imread(sample[path_key])
            spl = np.moveaxis(np.array(spl), -1, 0)
            return spl, spl.shape
        elif file_type in ['numpy', 'np']:
            spl = np.load(sample[path_key])
            return spl, spl.shape
        else:
            with open(sample[path_key], 'rb') as f:
                im = Image.open(io.BytesIO(f.read()))
                if len(im.getbands()) < 3:
                    rgbimg = Image.new("RGB", im.size)
                    rgbimg.paste(im)
                    im = rgbimg
                im = np.array(im)
                im = np.moveaxis(np.array(im), -1, 0)
            return im, im.shape

    @staticmethod
    def stack_batch_data_padded(batch_list):
        max_height = max(batch_list, key=lambda x: x['height'])['height']
        max_width = max(batch_list, key=lambda x: x['width'])['width']
        batch_array = None
        batch_dimensions = None
        for sample_idx, sample in enumerate(batch_list):
            sample_shape = sample['shape']
            im, im_shape = BloscInterface.load_sample(sample)  #########################
            if batch_array is None:
                batch_dimensions = ([len(batch_list)] + [im_shape[0], max_height, max_width])
                batch_array = np.zeros(batch_dimensions, dtype=np.array(im).dtype)
            begin_idx_1 = (batch_dimensions[2] - sample_shape[1]) // 2
            end_idx_1 = begin_idx_1 + sample_shape[1]
            begin_idx_2 = (batch_dimensions[3] - sample_shape[2]) // 2
            end_idx_2 = begin_idx_2 + sample_shape[2]
            batch_array[sample_idx, :, begin_idx_1:end_idx_1, begin_idx_2:end_idx_2] = im
        return np.array(batch_array)

    @staticmethod
    def batchify_sorted_sample_data_list(sample_data_list, batch_size=50):

        for start_idx in range(0, len(sample_data_list), batch_size):
            batch_list = sample_data_list[start_idx:start_idx + batch_size]
            classes, shapes, indices = zip(*((s['class'], s['shape'], s['index']) for s in batch_list))
            batch_array = BloscInterface.stack_batch_data_padded(batch_list)
            shapes = np.array(shapes, dtype=np.uint16)
            yield batch_array, np.array(classes, dtype=np.uint16), \
                  np.array(shapes, dtype=np.uint16), np.array(indices, dtype=np.uint32)


    @staticmethod
    def _get_fixed_crop_function(crop_size, random_location=True):
        crop_height, crop_width = crop_size

        def crop_func(sub_batch, batch_height, batch_width, shapes=None):
            # Issue with uint16 dtype when loading shape from dataset
            if isinstance(batch_width, np.uint16):
                batch_height, batch_width = int(batch_height), int(batch_width)
            if random_location == True:
                h_offset = max(0, int((batch_height - crop_height) * torch.rand(1).item()))
                w_offset = max(0, int((batch_width - crop_width) * torch.rand(1).item()))
            else:
                h_offset = max(0, int(batch_height - crop_height ) // 2)
                w_offset = max(0, int(batch_width - crop_width) // 2)
            w_end = min((batch_width, w_offset + crop_width))
            h_end = min((batch_height, h_offset + crop_height))
            return sub_batch[..., h_offset:h_end, w_offset:w_end]

        return crop_func

    @staticmethod
    def _get_random_one_side_fixed_crop_function(crop_height,
                                                 crop_width,
                                                 crop_area_ratio_range, random_location=True):
        def crop_func(sub_batch, batch_height, batch_width, shapes=None):
            # Issue with uint16 dtype when loading shape from dataset
            if isinstance(batch_width, np.uint16):
                batch_height, batch_width = int(batch_height), int(batch_width)
            h_offset, w_offset, h_end, w_end = \
                BloscInterface.get_random_crop_args_within_ratio_range_given_fixed_side(crop_height,
                                                                                        crop_width,
                                                                                        crop_area_ratio_range,
                                                                                        batch_height,
                                                                                        batch_width,
                                                                                        random_location)
            return sub_batch[..., h_offset:h_end, w_offset:w_end]

        return crop_func

    @staticmethod
    def _get_random_fixed_area_crop(crop_area_ratio_range, crop_size, random_location=True):
        def crop_func(sub_batch, batch_height, batch_width, shapes=None):
            # Issue with uint16 dtype when loading shape from dataFset
            if isinstance(batch_width, np.uint16):
                batch_height, batch_width = int(batch_height), int(batch_width)
            side = sqrt(crop_area_ratio_range)
            h_offset, w_offset, h_end, w_end = \
                BloscInterface.get_random_crop_args_within_ratio_range_given_target_area(crop_size, (side, side),
                                                                                         batch_height,
                                                                                         batch_width,
                                                                                         crop_area_ratio_range,
                                                                                         random_location)
            return sub_batch[..., h_offset:h_end, w_offset:w_end]

        return crop_func

    @staticmethod
    def _get_crop_by_range_or_value_function(crop_area_ratio_range, crop_size, random_location=True):
        if isinstance(crop_area_ratio_range, tuple):
            if len(crop_area_ratio_range) < 2:
                crop_area_ratio_range = crop_area_ratio_range[0], crop_area_ratio_range[0]
        else:
            crop_area_ratio_range = (crop_area_ratio_range, crop_area_ratio_range)

        def crop_func(sub_batch, batch_height, batch_width, shapes=None):
            # Issue with uint16 dtype when loading shape from dataset
            if isinstance(batch_width, np.uint16):
                batch_height, batch_width = int(batch_height), int(batch_width)
            h_offset, w_offset, h_end, w_end = \
                BloscInterface.get_random_crop_args_within_ratio_range_given_crop_size_ratio(crop_size,
                                                                                             crop_area_ratio_range,
                                                                                             batch_height,
                                                                                             batch_width, random_location)
            return sub_batch[..., h_offset:h_end, w_offset:w_end]

        return crop_func

    @staticmethod
    def center_crop(sub_batch, batch_height, batch_width, crop_height, crop_width, shapes=None):
        beg_idx_1 = max(0, (batch_height - crop_height) // 2)
        end_idx_1 = beg_idx_1 + crop_height
        beg_idx_2 = max(0, (batch_width - crop_width) // 2)
        end_idx_2 = beg_idx_2 + crop_width
        return sub_batch[..., beg_idx_1:end_idx_1, beg_idx_2:end_idx_2]

    @staticmethod
    def center_crop_as_tensor(sub_batch, batch_width, batch_height, crop_width, crop_height, shapes=None):
        ret = BloscInterface.center_crop(sub_batch, batch_width, batch_height, crop_width, crop_height, shapes=None)
        return torch.as_tensor(ret)

    @staticmethod
    def crop_original_samples_from_batch(batch, shapes, batch_height = None, batch_width = None):
        batch_shape = batch[0].shape
        if not isinstance(shapes[0], list) and not isinstance(shapes[0], tuple):
            shapes = [shapes]
            batch = batch[None,...]
        imlist = []
        for idx, sample_shape in enumerate(shapes):
            beg_idx_1 = (batch_shape[-2] - sample_shape[-2]) // 2
            end_idx_1 = beg_idx_1 + sample_shape[-2]
            beg_idx_2 = (batch_shape[-1] - sample_shape[-1]) // 2
            end_idx_2 = beg_idx_2 + sample_shape[-1]
            imlist.append(batch[idx, :, beg_idx_1:end_idx_1, beg_idx_2:end_idx_2])
        return imlist

    from torch import from_numpy
    @staticmethod
    def sub_batch_list_as_tensor(sub_batch: [np.array], device=torch.device('cpu'), torch_dtype = uint8, np_dtype=np.dtype('uint8')):
        for i in range(len(sub_batch)):
            sub_batch[i] = as_tensor(sub_batch[i].astype(np_dtype, copy=False),
                                     dtype=torch_dtype,
                                     device=torch.device(device))
        return sub_batch

    @staticmethod
    def sub_batch_array_as_tensor(sub_batch: [np.array], device=torch.device('cpu'), torch_dtype = uint8, np_dtype=np.dtype('uint8')):
        return as_tensor(sub_batch.astype(np_dtype, copy=False),
                         dtype=torch_dtype,
                         device=torch.device(device))
