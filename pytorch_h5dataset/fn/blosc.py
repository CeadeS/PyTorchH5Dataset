import torch
import numpy as np
from math import sqrt
import imghdr
import sndhdr


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
        return spl
    elif file_type in ['numpy', 'np']:
        spl = np.load(sample[path_key])
        return spl
    else:
        with open(sample[path_key ], 'rb') as f:
            im = Image.open(io.BytesIO(f.read()))
            if len(im.getbands()) < 3:
                rgbimg = Image.new("RGB", im.size)
                rgbimg.paste(im)
                im = rgbimg
            im = np.array(im)
            im = np.moveaxis(np.array(im), -1, 0)
        return im


def stack_batch_data_padded(batch_list):
    max_height = max(batch_list, key=lambda x: x['height'])['height']
    max_width = max(batch_list, key=lambda x: x['width'])['width']
    batch_array = None
    batch_dimensions = None
    for sample_idx, sample in enumerate(batch_list):
        sample_shape = sample['shape']
        im = load_sample(sample)
        if batch_array is None:
            batch_dimensions = ([len(batch_list)] + [im.shape[0], max_height, max_width])
            batch_array = np.zeros(batch_dimensions, dtype=np.array(im).dtype)
        begin_idx_1 = (batch_dimensions[2] - sample_shape[1]) // 2
        end_idx_1 = begin_idx_1 + sample_shape[1]
        begin_idx_2 = (batch_dimensions[3] - sample_shape[2]) // 2
        end_idx_2 = begin_idx_2 + sample_shape[2]
        batch_array[sample_idx, :, begin_idx_1:end_idx_1, begin_idx_2:end_idx_2] = im
    return np.array(batch_array)

def batchify_sorted_sample_data_list(sample_data_list, batch_size=50):

    for start_idx in range(0, len(sample_data_list), batch_size):
        batch_list = sample_data_list[start_idx:start_idx + batch_size]
        classes, shapes, indices = zip(*((s['class'], s['shape'], s['index']) for s in batch_list))
        batch_array = stack_batch_data_padded(batch_list)
        shapes = np.array(shapes, dtype=np.uint16)
        yield batch_array, np.array(classes, dtype=np.uint16), \
              np.array(shapes, dtype=np.uint16), np.array(indices, dtype=np.uint32)


def get_central_crop_args(batch_width, batch_height, crop_area_ratio_range):
    random_ratio = torch.empty(1).uniform_(*crop_area_ratio_range).item()
    if 0 < random_ratio <= 1 :
        h = int(sqrt(random_ratio) * batch_height)
        w = int(sqrt(random_ratio) * batch_width)
    elif random_ratio > 1.0:
        h = batch_width
        w = batch_height
    else:
        raise ValueError("Ratio must be in (0,1]")
    h_begin = (batch_height - h) // 2
    w_begin = (batch_width - w) // 2
    return h_begin, w_begin, h_begin + h, w_begin + w


def get_random_crop_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio_range_range, crop_area_ratio_range,
                                                             batch_height, batch_width):
    """
    Code from https://github.com/pytorch/vision/blob/d367a01a18a3ae6bee13d8be3b63fd6a581ea46f/torchvision/transforms/transforms.py
    :param crop_size_aspect_ratio:
    :param crop_area_ratio_range:
    :param batch_height:
    :param batch_width:
    :return:
    """
    log_ratio = torch.log(torch.tensor(crop_size_aspect_ratio_range_range))
    for _ in range(10):
        random_number = torch.empty(1).uniform_(*crop_area_ratio_range).item()
        target_area = batch_height * batch_width * random_number
        aspect_ratio = torch.exp(torch.empty(1).uniform_(*log_ratio)).item()
        crop_height = int(round(sqrt(target_area / aspect_ratio)))
        crop_width = int(round(sqrt(target_area * aspect_ratio)))


        if 0 < crop_width <= batch_width and 0 < crop_height <= batch_height:
            h_begin = torch.randint(0, batch_height - crop_height + 1, size=(1,)).item()
            w_begin = torch.randint(0, batch_width - crop_width + 1, size=(1,)).item()
            return h_begin, w_begin, h_begin + crop_height, w_begin + crop_width
    # Fallback to central crop

    return get_central_crop_args(batch_width, batch_height, crop_area_ratio_range)


def get_random_crop_within_ratio_range_given_target_area(crop_size_aspect_ratio, crop_area_ratio_range,
                                                         batch_height, batch_width, target_area):
    """
    Code from https://github.com/pytorch/vision/blob/d367a01a18a3ae6bee13d8be3b63fd6a581ea46f/torchvision/transforms/transforms.py
    :param crop_size_aspect_ratio:
    :param crop_area_ratio_range:
    :param batch_height:
    :param batch_width:
    :param target_area:
    :return:
    """
    log_ratio = torch.log(torch.tensor(crop_size_aspect_ratio))
    for _ in range(10):
        aspect_ratio = torch.exp(torch.empty(1).uniform_(*log_ratio)).item()
        crop_height = int(round(sqrt(target_area / aspect_ratio)))
        crop_width = int(round(sqrt(target_area * aspect_ratio)))
        if 0 < crop_width <= batch_width and 0 < crop_height <= batch_height:
            h_begin = torch.randint(0, batch_height - crop_height + 1, size=(1,)).item()
            w_begin = torch.randint(0, batch_width - crop_width + 1, size=(1,)).item()
            return h_begin, w_begin, h_begin + crop_height, w_begin + crop_width
    # Fallback to central crop
    return get_central_crop_args(batch_width, batch_height, crop_area_ratio_range)


def get_random_crop_within_ratio_range_given_fixed_side(crop_height, crop_width, crop_area_ratio_range,
                                                        batch_height, batch_width):
    """
    Stolen from https://github.com/pytorch/vision/blob/d367a01a18a3ae6bee13d8be3b63fd6a581ea46f/torchvision/transforms/transforms.py
    Gives the random crop parameters with at least one given length, crop_height or crop_width.
    :param crop_height:
    :param crop_width:
    :param crop_area_ratio_range:
    :param batch_height:
    :param batch_width:
    :return:
    """
    for _ in range(10):
        target_area = batch_height * batch_width * torch.empty(1).uniform_(*crop_area_ratio_range).item()
        if crop_height is None:
            crop_height = int(target_area / crop_width)
        elif crop_width is None:
            crop_width = int(target_area / crop_height)
        if 0 < crop_height <= batch_height and 0 < crop_width <= batch_width:
            h_begin = torch.randint(0, batch_height - crop_height + 1, size=(1,)).item()
            w_begin = torch.randint(0, batch_width - crop_width + 1, size=(1,)).item()
            return h_begin, w_begin, h_begin + crop_height, w_begin + crop_width
    # Fallback to central crop
    return get_central_crop_args(batch_width, batch_height, crop_area_ratio_range)


def random_located_sized_crop(h5_group,
                              batch_height,
                              batch_width,
                              crop_size=(400,None),
                              crop_area_ratio_range=(.8,.9)):

    """
    Function crops a sub batch i.e a h5_group.
    The sample Tensor is expected to have [..., H, W] shape.
    The original batch pixel area is reduced matching the crop_area_ratio.
    The pixel aspect ratio is given by crop_size

    :parameters
    h5_group
        The selected h5_group containing the datasets'
    batch_width
        The Width of the samples in the sub batch
    batch_height
        The height of the samples in the sub batch
    crop_size
        If crop size is a integer tuple the given side(s) is/are fixed.
        If crop size is given as a float value the width/height aspect ratio is fixed.
        If crop size is given as a 2-tuple of floats the height/width aspect ratio is in range of the crop size.
        all floats must be in [0.1,10]
        all uints must be in (0, maxint)
    crop_area_ratio_range
        Float tuple determining the cropped area of the batched samples.
        If its length is 1, the ratio is fixed.
        If its a tuple of len 2 the ratio is chosen in between both numbers.
        If its a number it is a fixed area that is cropped from the samples
        Any floats must be in (0,2]

    :param h5_group: h5group of samples
    :param batch_width: uint16
    :param batch_height: uint16
    :param crop_size: one of (float), (float, float), (None,int),(int,None), (uint16,int),
    :param crop_area_ratio_range: one of (float), (float, float)
    :return: cropped sub batch of samples of type np.ndarray
    """
    func = random_located_sized_crop_function(crop_size=crop_size,
                                                        crop_area_ratio_range=crop_area_ratio_range)
    return func(h5_group=h5_group, batch_height=batch_height, batch_width=batch_width)


def get_crop_by_range_or_value_function(crop_area_ratio_range, crop_size):
    if isinstance(crop_area_ratio_range, tuple):
        if len(crop_area_ratio_range) < 2:
            crop_area_ratio_range = crop_area_ratio_range[0], crop_area_ratio_range[0]
    else:
        crop_area_ratio_range = (crop_area_ratio_range, crop_area_ratio_range)

    def crop_func(h5_group, batch_height, batch_width):
        # Issue with uint16 dtype when loading shape from dataset
        if isinstance(batch_width, np.uint16):
            batch_height, batch_width = int(batch_height), int(batch_width)
        h_offset, w_offset, h_end, w_end = \
            get_random_crop_within_ratio_range_given_crop_size_ratio(crop_size,
                                                                               crop_area_ratio_range,
                                                                               batch_height,
                                                                               batch_width)
        return h5_group[..., h_offset:h_end, w_offset:w_end]

    return crop_func


def random_located_sized_crop_function(crop_size=(400,),
                                       crop_area_ratio_range=(.8,)):

    """
    Returns a Function that crops a group given the group and the batch heights and width

    The original batch pixel area is reduced matching the crop_area_ratio.
    The pixel aspect ratio is given by crop_size

    parameters:
        crop_size:
            If crop size is a integer tuple the given side(s) is/are fixed.
            If crop size is given as a float value the width/height aspect ratio is fixed.
            If crop size is given as a 2-tuple of floats the height/width aspect ratio is in range of the crop size.
            all floats must be in [0.1,10]
            all uints must be in (0, maxint)
        crop_area_ratio_range:
            Float tuple determining the cropped area of the batched samples.
            If its length is 1, the ratio is fixed.
            If its a tuple of len 2 the ratio is chosen in between both numbers.
            If its a number it is a fixed area that is cropped from the samples
            Any floats must be in (0,2]


    :param crop_size: one of (float), (float, float), (None,int),(int,None), (int,int),
    :param crop_area_ratio_range: one of float, (float), (float, float) (int)
    :return: function that crops a h5 group
    """

    assert isinstance(crop_area_ratio_range, (float, int)) or isinstance(crop_area_ratio_range, tuple) and \
           len(
               crop_area_ratio_range) <= 2, "Crop area ratio must be a int or float value or a tuple of length shorter then 3."

    if isinstance(crop_area_ratio_range, tuple):
        assert all((isinstance(a, float) and 0. < a <= 1. for a in crop_area_ratio_range))
        if len(crop_area_ratio_range) == 2:
            assert crop_area_ratio_range[0] <= crop_area_ratio_range[1], \
                f"Upper bound (first value) of crop_area_ratio_range {crop_area_ratio_range[0]} must me smaller " \
                f"then lower bound (second value) of {crop_area_ratio_range[1]}."
    assert isinstance(crop_size, float) or isinstance(crop_size, tuple), "crop_size must be a float or a tuple."
    if isinstance(crop_size, tuple):
        assert len(crop_size) <= 2, "Shape of crop size must have 1 or 2 dimensions."
        assert all((isinstance(a, float) for a in crop_size)) or \
               all((isinstance(a, (int, type(None))) for a in crop_size)), "All values of crop_size must be float " \
                                                                           "or they must be int or None."
        if len(crop_size) == 1:
            assert crop_size[0] is not None, "Non is not allowed if you wish a fixed side or a fixed aspect ratio."
            crop_size = (crop_size[0], crop_size[0])
        elif None in crop_size:
            assert (crop_size[0] is not None or crop_size[1] is not None), "None is only allowed once in crop_size"
        elif len(crop_size) == 2 and all((isinstance(a, float) for a in crop_size)):
            assert crop_size[0] <= crop_size[1], f"Lower bound of crop_size (first value) {crop_size[0]} must be" \
                                                 f"lower then the upper bound (second value) {crop_size[1]}."
            assert (all((a > 0 for a in crop_size))), "Negative Values are not allowed"
        elif all((isinstance(a, int) for a in crop_size)):
            pass
        else:
            raise ValueError(f"Something unexpected happened to the crop_size parameter {crop_size}")

    else:
        assert crop_size is not None, 'Crop_size is not allowed to be None'
        crop_size = (crop_size, crop_size)

    # crop size completely determined
    if all(isinstance(a, int) for a in crop_size) and len(crop_size) == 2:
        # both sides are fixed
        crop_height, crop_width = crop_size
        def crop_func(h5_group, batch_height, batch_width):
            # Issue with uint16 dtype when loading shape from dataset
            if isinstance(batch_width, np.uint16):
                batch_height, batch_width = int(batch_height), int(batch_width)
            h_offset =  max((0,int((batch_height - crop_height) * torch.rand(1).item())))
            w_offset = max(0,int((batch_width - crop_width) * torch.rand(1).item()))
            w_end = min((batch_width, w_offset + crop_width))
            h_end = min((batch_height, h_offset + crop_height))
            return h5_group[..., h_offset:h_end, w_offset:w_end]

        return crop_func

    if isinstance(crop_area_ratio_range, float) or (not isinstance(crop_area_ratio_range, int) and
                                                    all(isinstance(a, float) for a in crop_area_ratio_range)):
        # one crop size is unknown and a crop area ratio is given
        if any(isinstance(a, int) for a in crop_size):
            # one side is fixed
            crop_height, crop_width = crop_size
            if isinstance(crop_area_ratio_range, tuple):
                if len(crop_area_ratio_range) < 2:
                    crop_area_ratio_range = crop_area_ratio_range[0], crop_area_ratio_range[0]
            else:
                crop_area_ratio_range = (crop_area_ratio_range, crop_area_ratio_range)
            def crop_func(h5_group, batch_height, batch_width):
                # Issue with uint16 dtype when loading shape from dataset
                if isinstance(batch_width, np.uint16):
                    batch_height, batch_width = int(batch_height), int(batch_width)
                h_offset, w_offset, h_end, w_end = \
                    get_random_crop_within_ratio_range_given_fixed_side(crop_height,
                                                                                  crop_width,
                                                                                  crop_area_ratio_range,
                                                                                  batch_height,
                                                                                  batch_width)
                return h5_group[..., h_offset:h_end, w_offset:w_end]

            return crop_func

        # crop size is given by a range of ints or a single int value
        # and the crop_area_ratio_range is given as float or tuple of floats
        elif all(isinstance(a, int) for a in crop_size) or isinstance(crop_size, int):
            return get_crop_by_range_or_value_function(crop_area_ratio_range, crop_size)

        # crop size is given by a range of floats or a single float value
        # and the crop_area_ratio_range is given as float value or tuple of floats
        elif all(isinstance(a, float) for a in crop_size) or isinstance(crop_size, float):
            return get_crop_by_range_or_value_function(crop_area_ratio_range, crop_size)
        else:
            raise NotImplementedError
    elif isinstance(crop_area_ratio_range, int):
        # crop size is given by a range of floats or a single float value
        # and the crop_area_ratio_range is given as int value
        if all(isinstance(a, float) for a in crop_size) or isinstance(crop_size, float):
            def crop_func(h5_group, batch_height, batch_width):
                # Issue with uint16 dtype when loading shape from dataset
                if isinstance(batch_width, np.uint16):
                    batch_height, batch_width = int(batch_height), int(batch_width)
                side = sqrt(crop_area_ratio_range)
                h_offset, w_offset, h_end, w_end = \
                    get_random_crop_within_ratio_range_given_target_area(crop_size, (side, side),
                                                                                   batch_height,
                                                                                   batch_width,
                                                                                   crop_area_ratio_range)
                return h5_group[..., h_offset:h_end, w_offset:w_end]

            return crop_func
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def center_crop(h5_group, batch_height, batch_width, crop_height, crop_width):
    beg_idx_1 = max(0, (batch_height - crop_height) // 2)
    end_idx_1 = beg_idx_1 + crop_height
    beg_idx_2 = max(0, (batch_width - crop_width) // 2)
    end_idx_2 = beg_idx_2 + crop_width
    return h5_group[..., beg_idx_1:end_idx_1, beg_idx_2:end_idx_2]


def center_crop_as_tensor(h5_group, batch_width, batch_height, crop_width, crop_height):
    ret = center_crop(h5_group, batch_width, batch_height, crop_width, crop_height)
    return torch.as_tensor(ret)


def crop_original_samples_from_batch(batch, shapes):
    batch_shape = batch[0].shape
    imlist = []
    for idx, sample_shape in enumerate(shapes):
        beg_idx_1 = (batch_shape[-2] - sample_shape[-2]) // 2
        end_idx_1 = beg_idx_1 + sample_shape[-2]
        beg_idx_2 = (batch_shape[-1] - sample_shape[-1]) // 2
        end_idx_2 = beg_idx_2 + sample_shape[-1]
        imlist.append(batch[idx, :, beg_idx_1:end_idx_1, beg_idx_2:end_idx_2])
    return imlist
