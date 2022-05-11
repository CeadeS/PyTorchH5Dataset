import torch
from math import sqrt

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
    return w_begin, h_begin, w, h

def get_random_crop_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio_range_range, crop_area_ratio_range,
                                                             batch_height, batch_width, random_location = True):
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

        if random_location:
            if 0 < crop_width <= batch_width and 0 < crop_height <= batch_height:
                h_begin = torch.randint(0, batch_height - crop_height + 1, size=(1,)).item()
                w_begin = torch.randint(0, batch_width - crop_width + 1, size=(1,)).item()
                return h_begin, w_begin, h_begin + crop_height, w_begin + crop_width
        else:
            h_begin = min(0, batch_height - crop_height//2 )
            w_begin = min(0, batch_width - crop_width//2 )
            return h_begin, w_begin, max(h_begin + batch_height), crop_height, max(w_begin + crop_width, batch_width)

    # Fallback to central crop

    return get_central_crop_args(batch_width, batch_height, crop_area_ratio_range, random_location)

# def get_random_crop_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio_range_range, crop_area_ratio_range,
#                                                              batch_height, batch_width):
#     """
#     Code from https://github.com/pytorch/vision/blob/d367a01a18a3ae6bee13d8be3b63fd6a581ea46f/torchvision/transforms/transforms.py
#     :param crop_size_aspect_ratio:
#     :param crop_area_ratio_range:
#     :param batch_height:
#     :param batch_width:
#     :return:
#     """
#     log_ratio = torch.log(torch.tensor(crop_size_aspect_ratio_range_range))
#     for _ in range(10):
#         random_number = torch.empty(1).uniform_(*crop_area_ratio_range).item()
#         target_area = batch_height * batch_width * random_number
#         aspect_ratio = torch.exp(torch.empty(1).uniform_(*log_ratio)).item()
#         crop_height = int(round(sqrt(target_area / aspect_ratio)))
#         crop_width = int(round(sqrt(target_area * aspect_ratio)))
#
#
#         if 0 < crop_width <= batch_width and 0 < crop_height <= batch_height:
#             h_begin = torch.randint(0, batch_height - crop_height + 1, size=(1,)).item()
#             w_begin = torch.randint(0, batch_width - crop_width + 1, size=(1,)).item()
#             return h_begin, w_begin, h_begin + crop_height, w_begin + crop_width
#     # Fallback to central crop
#
#     return get_central_crop_args(batch_width, batch_height, crop_area_ratio_range)


def get_random_crop_within_ratio_range_given_target_area(crop_size_aspect_ratio, crop_area_ratio_range,
                                                         batch_height, batch_width, target_area, random_location=True):
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
            if random_location:
                h_begin = torch.randint(0, batch_height - crop_height + 1, size=(1,)).item()
                w_begin = torch.randint(0, batch_width - crop_width + 1, size=(1,)).item()
            else:
                h_begin, w_begin = (batch_height - crop_height)//2, (batch_width - crop_width)//2
            return h_begin, w_begin, h_begin + crop_height, w_begin + crop_width
    # Fallback to central crop
    return get_central_crop_args(batch_width, batch_height, crop_area_ratio_range)


def get_random_crop_within_ratio_range_given_fixed_side(crop_height, crop_width, crop_area_ratio_range,
                                                        batch_height, batch_width, random_location = True):
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
            if random_location:
                h_begin = torch.randint(0, batch_height - crop_height + 1, size=(1,)).item()
                w_begin = torch.randint(0, batch_width - crop_width + 1, size=(1,)).item()
            else:
                h_begin, w_begin = (batch_height - crop_height)//2, (batch_width - crop_width)//2
            return h_begin, w_begin, h_begin + crop_height, w_begin + crop_width
    # Fallback to central crop
    return get_central_crop_args(batch_width, batch_height, crop_area_ratio_range)



# def get_random_located_sized_crop_function(sub_batch,
#                                            batch_height,
#                                            batch_width,
#                                            crop_size=(400,None),
#                                            crop_area_ratio_range=(.8,.9)):
#
#     """
#     Function crops a sub batch i.e a sub_batch.
#     The sample Tensor is expected to have [..., H, W] shape.
#     The original batch pixel area is reduced matching the crop_area_ratio.
#     The pixel aspect ratio is given by crop_size
#
#     :parameters
#     sub_batch
#         The selected sub_batch containing the datasets'
#     batch_width
#         The Width of the samples in the sub batch
#     batch_height
#         The height of the samples in the sub batch
#     crop_size
#         If crop crop_size is a integer tuple the given side(s) is/are fixed.
#         If crop crop_size is given as a float value the width/height aspect ratio is fixed.
#         If crop crop_size is given as a 2-tuple of floats the height/width aspect ratio is in range of the crop crop_size.
#         all floats must be in [0.1,10]
#         all uints must be in (0, maxint)
#     crop_area_ratio_range
#         Float tuple determining the cropped area of the batched samples.
#         If its length is 1, the ratio is fixed.
#         If its a tuple of len 2 the ratio is chosen in between both numbers.
#         If its a number it is a fixed area that is cropped from the samples
#         Any floats must be in (0,2]
#
#     :param sub_batch: sub_batch of samples
#     :param batch_width: uint16
#     :param batch_height: uint16
#     :param crop_size: one of (float), (float, float), (None,int),(int,None), (uint16,int),
#     :param crop_area_ratio_range: one of (float), (float, float)
#     :return: cropped sub batch of samples of type np.ndarray
#     """
#     func = get_random_located_sized_crop_function(crop_size=crop_size,
#                                                   crop_area_ratio_range=crop_area_ratio_range)
#     return func(sub_batch=sub_batch, batch_height=batch_height, batch_width=batch_width)