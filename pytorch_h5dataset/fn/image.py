## TODO USE LIBJPEG TURBO for LOADING
## TODO USE JPEGTRAN for augmentation
## TODO Store Lists of vlen strings of jpegs directly into hdf5
from jpegtran import lib

from simplejpeg import encode_jpeg as encode
from simplejpeg import decode_jpeg as jpeg_decode
from torchvision.io import decode_image as torch_decode
from torchvision.transforms.functional import resize
from torch import as_tensor, uint8

import torch

import numpy as np
from math import sqrt
import imghdr

from .data_interface import DataInterface


class ImageInterface(DataInterface):

    @staticmethod
    def load_sample(sample, jpeg_quality=83):
        from skimage import io as skio

        path_key = 'path' if 'path' in sample.keys() else 'FilePath'
        type_key = 'type' if 'type' in sample.keys() else 'FileType'

        try:
            file_type = sample[type_key]
        except KeyError:
            file_type = imghdr.what(sample[path_key])
        if file_type in ['tif', 'tiff']:
            im_bytes = encode(skio.imread(sample[path_key]), quality=jpeg_quality)
            return im_bytes
        elif file_type in ['numpy', 'np']:
            im_bytes = encode(np.load(sample[path_key]), quality=jpeg_quality)
            return im_bytes
        else:
            with open(sample[path_key], 'rb') as f:
                im_bytes = f.read()            
            return np.frombuffer(im_bytes, dtype=f'S{len(im_bytes)}'), lib.Transformation(im_bytes).get_dimensions()

    @staticmethod
    def batchify_sample_data_list(sample_data_list, batch_size=50):
        for start_idx in range(0, len(sample_data_list), batch_size):
            batch_list = sample_data_list[start_idx:start_idx + batch_size]
            classes, shapes, indices = zip(*((s['class'], s['shape'], s['index']) for s in batch_list))
            im_list = []
            for sample_idx, sample in enumerate(batch_list):
                im, _ = ImageInterface.load_sample(sample)
                im_list.append(im)
            shapes = np.array(shapes, dtype=np.uint16)
            yield im_list, np.array(classes, dtype=np.uint16), \
                  np.array(shapes, dtype=np.uint16), np.array(indices, dtype=np.uint32)

    @staticmethod
    def _get_crop_by_range_or_value_function(crop_area_ratio_range, crop_size, random_location):
        if isinstance(crop_area_ratio_range, tuple):
            if len(crop_area_ratio_range) < 2:
                crop_area_ratio_range = crop_area_ratio_range[0], crop_area_ratio_range[0]
        else:
            crop_area_ratio_range = (crop_area_ratio_range, crop_area_ratio_range)

        def crop_func(sub_batch, batch_height=None, batch_width=None):
            heights, widths = ImageInterface.get_sub_batch_shapes(sub_batch)
            h_offsets, h_ends, w_offsets, w_ends = [], [], [], []
            for i in range(len(sub_batch)):
                batch_width, batch_height = int(widths[i]), int(heights[i])
                h_offset, w_offset, h_end, w_end = \
                    ImageInterface.get_random_crop_args_within_ratio_range_given_crop_size_ratio(crop_size,
                                                                                                 crop_area_ratio_range,
                                                                                                 batch_height,
                                                                                                 batch_width,
                                                                                                 random_location)
                h_offsets.append(h_offset)
                w_offsets.append(w_offset)
                h_ends.append(h_end)
                w_ends.append(w_end)
                return ImageInterface.crop(sub_batch, h_offsets, h_ends, w_offsets, w_ends)

        return crop_func

    @staticmethod
    def get_sub_batch_shapes(sub_batch):
        heights, widths = [], []
        for i in range(len(sub_batch)):
            width, height = lib.Transformation(sub_batch[i][0]).get_dimensions()
            heights.append(int(height))
            widths.append(int(width))
        return heights, widths

    @staticmethod
    def _get_random_fixed_area_crop(crop_area_ratio_range, crop_size, random_location=True):
        def crop_func(sub_batch, batch_height=None, batch_width=None):
            side = sqrt(crop_area_ratio_range)
            heights, widths = ImageInterface.get_sub_batch_shapes(sub_batch)
            h_offsets, h_ends, w_offsets, w_ends = [], [], [], []
            for i in range(len(sub_batch)):
                batch_width, batch_height = int(widths[i]), int(heights[i])

                h_offset, w_offset, h_end, w_end = \
                    ImageInterface.get_random_crop_args_within_ratio_range_given_target_area(crop_size, (side, side),
                                                                                             batch_height,
                                                                                             batch_width,
                                                                                             crop_area_ratio_range,
                                                                                             random_location)
                h_offsets.append(h_offset)
                w_offsets.append(w_offset)
                h_ends.append(h_end)
                w_ends.append(w_end)
            return ImageInterface.crop(sub_batch, h_offsets, h_ends, w_offsets, w_ends)

        return crop_func

    @staticmethod
    def _get_random_one_side_fixed_crop_function(crop_height,
                                                 crop_width,
                                                 crop_area_ratio_range, random_location=True):
        def crop_func(sub_batch, batch_height=None, batch_width=None):
            heights, widths = ImageInterface.get_sub_batch_shapes(sub_batch)
            h_offsets, h_ends, w_offsets, w_ends = [], [], [], []
            for i in range(len(sub_batch)):
                batch_width, batch_height = widths[i], heights[i]
                h_offset, w_offset, h_end, w_end = \
                    ImageInterface.get_random_crop_args_within_ratio_range_given_fixed_side(crop_height,
                                                                                            crop_width,
                                                                                            crop_area_ratio_range,
                                                                                            batch_height,
                                                                                            batch_width,
                                                                                            random_location)
                h_offsets.append(h_offset)
                w_offsets.append(w_offset)
                h_ends.append(h_end)
                w_ends.append(w_end)
            return ImageInterface.crop(sub_batch, h_offsets, h_ends, w_offsets, w_ends)

        return crop_func

    @staticmethod
    def _get_fixed_crop_function(crop_size, random_location=True):
        # both sides are fixed
        crop_height, crop_width = crop_size

        def crop_func(sub_batch, batch_height=None, batch_width=None):
            heights, widths = ImageInterface.get_sub_batch_shapes(sub_batch)
            if random_location:
                rh, rw = torch.rand(1).item(), torch.rand(1).item()
            else:
                rh, rw = 0.5, 0.5
            h_offsets = tuple(max((0, int((batch_height - crop_height) * rh))) for batch_height in heights)
            w_offsets = tuple(max(0, int((batch_width - crop_width) * rw)) for batch_width in widths)
            w_end = tuple(min((batch_width, w_offset + crop_width)) for batch_width, w_offset in zip(widths, w_offsets))
            h_end = tuple(
                min((batch_height, h_offset + crop_height)) for batch_height, h_offset in zip(heights, h_offsets))
            return ImageInterface.crop(sub_batch, h_offsets, h_end, w_offsets, w_end)

        return crop_func

    @staticmethod
    def crop(sub_batch: [bytes], h_offset: [int], h_end: [int], w_offset: [int], w_end: [int], width=None, height=None):
        result = list(range(len(sub_batch)))
        if width is None or height is None:
            width, height = [], []
            for i in range(len(sub_batch)):
                _width, _height = lib.Transformation(sub_batch[i][0]).get_dimensions()
                width.append(int(_width))
                height.append(int(_height))
        for i in range(len(sub_batch)):
            crop_width = int(w_end[i] - w_offset[i])
            crop_height = int(h_end[i] - h_offset[i])
            result[i] = [lib.Transformation(sub_batch[i][0]).crop(w_offset[i], h_offset[i], min(crop_width, width[i]),
                                                              min(crop_height, height[i]))]
        return result

    @staticmethod
    def random_rotation(sub_batch: [bytes], angles=(90,)):
        rands = torch.randint(len(angles), (len(sub_batch),))

        result = list(range(len(sub_batch)))
        for i in range(len(sub_batch)):
            angle = angles[rands[i].item()]
            result[i] =[lib.Transformation(sub_batch[i][0]).rotate(angle)]

        return result

    @staticmethod
    def random_h_flip(sub_batch: [bytes]):
        result = list(range(len(sub_batch)))
        r = torch.randint(2, size=(len(sub_batch),))
        for i in range(len(sub_batch)):
            result[i] = [lib.Transformation(sub_batch[i][0]).flip(direction='horizontal')]
        return result

    @staticmethod
    def random_v_flip(sub_batch: [bytes]):
        result = list(range(len(sub_batch)))
        r = torch.randint(2, size=(len(sub_batch),))
        for i in range(len(sub_batch)):
            result[i] = [lib.Transformation(sub_batch[i][0]).flip(direction='vertical')]
        return result

    @staticmethod
    def center_crop(sub_batch: [bytes], batch_height, batch_width, crop_height, crop_width):
        assert isinstance(crop_height, int) and isinstance(crop_width,
                                                           int), "Wrong Datatype for crop_height or crop_width"
        assert isinstance(sub_batch, list) and len(sub_batch) > 0 and (isinstance(sub_batch[0], bytes) or isinstance(sub_batch[0], np.ndarray)), "" \
                                                                                                       "Sub Batch must be a List containing bytes"

        result = list(range(len(sub_batch)))
        for i in range(len(sub_batch)):
            beg_idx_1 = max(0, (100 - crop_height) // 2)
            beg_idx_2 = max(0, (100 - crop_width) // 2)
            width, height = lib.Transformation(sub_batch[i][0]).get_dimensions()
            result[i] = [lib.Transformation(sub_batch[i][0]).crop(beg_idx_2, beg_idx_1, min(crop_width, width),
                                                              min(crop_height, height))]
        return result

    @staticmethod
    def random_scale(sub_batch: [bytes], scale_range=(0.8, 1.33), quality=83):
        result = list(range(len(sub_batch)))
        rands = torch.FloatTensor(len(sub_batch)).uniform_(*scale_range)
        width, height = [], []
        for i in range(len(sub_batch)):
            _width, _height = lib.Transformation(sub_batch[i]).get_dimensions()
            width.append(int(rands[i].item() * _width))
            height.append(int(rands[i].item() * _height))
        for i in range(len(sub_batch)):
            result[i] = [lib.Transformation(sub_batch[i][0]).scale(width[i], height[i], quality)]
        return result

    @staticmethod
    def scale(sub_batch: [bytes], heights: [int], widths: [int], quality=83):
        if isinstance(widths, int) and isinstance(widths, int):
            widths = [widths] * len(sub_batch)
            heights = [heights] * len(sub_batch)
        result = list(range(len(sub_batch)))
        for i in range(len(sub_batch)):         
            result[i] = [lib.Transformation(sub_batch[i][0]).scale(widths[i], heights[i], quality)]
        return result

    @staticmethod
    def scale_torch(sub_batch: [bytes], heights: [int], widths: [int]):
        if isinstance(widths, int) and isinstance(widths, int):
            widths = [widths] * len(sub_batch)
            heights = [heights] * len(sub_batch)
        result = list(range(len(sub_batch)))
        for i in range(len(sub_batch)):     
            result[i] = resize(sub_batch[i][0],(widths[i], heights[i]))
        return result

    @staticmethod
    def sub_batch_as_tensor(sub_batch: [bytes], device=torch.device('cpu')):
        result = list(range(len(sub_batch)))
        for i in range(len(sub_batch)):
            result[i] = as_tensor(sub_batch[i][0], dtype=uint8, device=torch.device(device))
        return result

    @staticmethod
    def decode(bytes_obj, device='cpu'):
        if torch.device('cuda').type == torch.device(device).type:
            return torch_decode(bytes_obj)
        else:
            return torch.from_numpy(np.moveaxis(jpeg_decode(bytes_obj), -1, 0))

    @staticmethod
    def sub_batch_decode(sub_batch: [bytes], device='cpu'):
        result = list(range(len(sub_batch)))
        for i in range(len(sub_batch)):
            result[i] = [ImageInterface.decode(sub_batch[i][0], device=device)]
        return result
