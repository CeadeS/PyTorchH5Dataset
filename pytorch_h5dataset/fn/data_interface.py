from abc import ABC, abstractmethod
from typing import final
from math import sqrt
from torch import log, tensor, randint, empty, exp


class DataInterface(ABC):

    @classmethod
    @final
    def random_crop(mcs, sub_batch,
                   batch_height,
                   batch_width,
                   crop_size=(400, None),
                   crop_area_ratio_range=(.8, .9), random_location=True):

        """
        Function crops a sub batch i.e a h5_dataset.
        The sample Tensor is expected to have [..., H, W] shape.
        The original batch pixel area is reduced matching the crop_area_ratio.
        The pixel aspect ratio is given by crop_size

        :parameters
        sub_batch
            The selected sub_batch containing the samples
        batch_width
            The Width of the samples in the sub batch
        batch_height
            The height of the samples in the sub batch
        crop_size
            If crop crop_size is a integer tuple the given side(s) is/are fixed.
            If crop crop_size is given as a float value the width/height aspect ratio is fixed.
            If crop crop_size is given as a 2-tuple of floats the height/width aspect ratio is in range of the crop crop_size.
            all floats must be in [0.1,10]
            all uints must be in (0, maxint)
        crop_area_ratio_range
            Float tuple determining the cropped area of the batched samples.
            If its length is 1, the ratio is fixed.
            If its a tuple of len 2 the ratio is chosen in between both numbers.
            If its a number it is a fixed area that is cropped from the samples
            Any floats must be in (0,2]

        :param sub_batch: batch of samples
        :param batch_width: uint16
        :param batch_height: uint16
        :param crop_size: one of (float), (float, float), (None,int),(int,None), (uint16,int),
        :param crop_area_ratio_range: one of (float), (float, float)
        :return: cropped sub batch of samples of type np.ndarray
        """
        func = mcs.get_random_crop_function(crop_size=crop_size, crop_area_ratio_range=crop_area_ratio_range,
                                                                random_location=random_location)
        return func(sub_batch=sub_batch, batch_height=batch_height, batch_width=batch_width)

    @classmethod
    @final
    def get_random_crop_function(mcs, crop_size=(400,),
                                           crop_area_ratio_range=(.8,), random_location=True):
        """
        Returns a Function that crops a sub_batch

        The original batch pixel area is reduced matching the crop_area_ratio.
        The pixel aspect ratio is given by crop_size

        parameters:
            crop_size:
                If crop crop_size is a integer tuple the given side(s) is/are fixed.
                If crop crop_size is given as a float value the width/height aspect ratio is fixed.
                If crop crop_size is given as a 2-tuple of floats the height/width aspect ratio is in range of the crop crop_size.
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
        :return: function that crops a sub_batch
        """

        if crop_area_ratio_range is (None):
            crop_area_ratio_range = None

        assert crop_area_ratio_range is not None or isinstance(crop_area_ratio_range, (float, int)) or isinstance(crop_area_ratio_range, tuple) and \
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
            assert len(crop_size) <= 2, "Shape of crop crop_size must have 1 or 2 dimensions."
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

        # crop crop_size completely determined
        if all(isinstance(a, int) for a in crop_size) and len(crop_size) == 2:
            crop_func = mcs._get_fixed_crop_function(crop_size, random_location)
            return crop_func


        # one crop crop_size is unknown and a crop area ratio is given
        if isinstance(crop_area_ratio_range, float) or (not isinstance(crop_area_ratio_range, int) and
                                                        all(isinstance(a, float) for a in crop_area_ratio_range)):

            if any(isinstance(a, int) for a in crop_size):
                # one side is fixed
                crop_height, crop_width = crop_size
                if isinstance(crop_area_ratio_range, tuple):
                    if len(crop_area_ratio_range) < 2:
                        crop_area_ratio_range = crop_area_ratio_range[0], crop_area_ratio_range[0]
                else:
                    crop_area_ratio_range = (crop_area_ratio_range, crop_area_ratio_range)
                crop_func = mcs._get_random_one_side_fixed_crop_function(crop_height,
                                                                         crop_width,
                                                                         crop_area_ratio_range,
                                                                         random_location)
                return crop_func

            # crop crop_size is given by a range of ints or a single int value
            # and the crop_area_ratio_range is given as float or tuple of floats
            elif all(isinstance(a, int) for a in crop_size) or isinstance(crop_size, int):
                return mcs._get_crop_by_range_or_value_function(crop_area_ratio_range, crop_size, random_location)

            # crop crop_size is given by a range of floats or a single float value
            # and the crop_area_ratio_range is given as float value or tuple of floats
            elif all(isinstance(a, float) for a in crop_size) or isinstance(crop_size, float):
                return mcs._get_crop_by_range_or_value_function(crop_area_ratio_range, crop_size, random_location)
            else:
                raise NotImplementedError
        elif isinstance(crop_area_ratio_range, int):
            # crop crop_size is given by a range of floats or a single float value
            # and the crop_area_ratio_range is given as int value
            if all(isinstance(a, float) for a in crop_size) or isinstance(crop_size, float):
                crop_func = mcs._get_random_fixed_area_crop(crop_area_ratio_range, crop_size,
                                                                       random_location)
                return crop_func
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    @classmethod
    @final
    def get_center_crop_function(mcs, crop_size=(100, 100)):
        """

        :param crop_size: crop crop_size can be a tuple of ints or a float. Tuple of ints -> cropped crop_size. Float -> cropped area ratio
        :return: crop function
        """
        if all(isinstance(a, int) for a in crop_size):
            def func(sub_batch):
                return mcs.center_crop(sub_batch, batch_height=sub_batch.shape[-2],
                                                 batch_width=sub_batch.shape[-1],
                                                 crop_height=crop_size[0], crop_width=crop_size[1])

        elif isinstance(crop_size, float):
            def func(sub_batch):
                batch_height = sub_batch.shape[-2]
                batch_width = sub_batch.shape[-1]
                crop_height = sqrt(crop_size) * batch_height
                crop_width = sqrt(crop_size) * batch_width
                return mcs.center_crop(sub_batch, batch_height=batch_height, batch_width=batch_width,
                                                 crop_height=crop_height, crop_width=crop_width)
        else:
            raise NotImplementedError
        return func

    @classmethod
    @final
    def get_random_center_crop_function(mcs, crop_size=(400,), crop_area_ratio_range=(.8,)):
        """
        Returns a Function that crops a sub_batch given the  and the batch heights and width
    
        The original batch pixel area is reduced matching the crop_area_ratio.
        The pixel aspect ratio is given by crop_size
    
        parameters:
            crop_size:
                If crop crop_size is a integer tuple the given side(s) is/are fixed.
                If crop crop_size is given as a float value the width/height aspect ratio is fixed.
                If crop crop_size is given as a 2-tuple of floats the height/width aspect ratio is in range of the crop crop_size.
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
        :return: function that crops a sub_batch
        """
        return mcs.get_random_crop_function(crop_size, crop_area_ratio_range,
                                                                    random_location=False)

    @staticmethod
    @final
    def get_central_crop_args(batch_width, batch_height, crop_area_ratio_range):
        random_ratio = empty(1).uniform_(*crop_area_ratio_range).item()
        ratio = sqrt(random_ratio)

        if 0 < random_ratio <= 1:
            crop_width, crop_height = int(batch_width * ratio),  int(batch_height * ratio)
        elif random_ratio > 1.0:
            crop_width, crop_height = int(batch_width),  int(batch_height)
        else:
            raise ValueError("Ratio must be in (0,1]")

        h_begin = (batch_height - crop_height) // 2
        w_begin = (batch_width - crop_width) // 2

        return h_begin, w_begin,  h_begin+crop_height, w_begin+crop_width,

    @staticmethod
    @final
    def get_random_crop_args_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio_range_range,
                                                                      crop_area_ratio_range,
                                                                      batch_height, batch_width, random_location=True):
        """
        Code from https://github.com/pytorch/vision/blob/d367a01a18a3ae6bee13d8be3b63fd6a581ea46f/torchvision/transforms/transforms.py
        :param crop_size_aspect_ratio:
        :param crop_area_ratio_range:
        :param batch_height:
        :param batch_width:
        :return:
        """
        log_ratio = log(tensor(crop_size_aspect_ratio_range_range))
        for _ in range(10):
            random_number = empty(1).uniform_(*crop_area_ratio_range).item()
            target_area = batch_height * batch_width * random_number
            aspect_ratio = exp(empty(1).uniform_(*log_ratio)).item()
            crop_height = int(round(sqrt(target_area / aspect_ratio)))
            crop_width = int(round(sqrt(target_area * aspect_ratio)))

            if random_location:
                if 0 < crop_width <= batch_width and 0 < crop_height <= batch_height:
                    h_begin = randint(0, batch_height - crop_height + 1, size=(1,)).item()
                    w_begin = randint(0, batch_width - crop_width + 1, size=(1,)).item()
                    return h_begin, w_begin, h_begin + crop_height, w_begin + crop_width
            else:
                h_begin = max(0, (batch_height - crop_height) // 2)
                w_begin = max(0, (batch_width - crop_width) // 2)
                return h_begin, w_begin, min(h_begin + crop_height, batch_height), min(w_begin + crop_width,
                                                                                           batch_width)
        # Fallback to central crop
        return DataInterface.get_central_crop_args(batch_width, batch_height, crop_area_ratio_range)

    @staticmethod
    @final
    def get_random_crop_args_within_ratio_range_given_target_area(crop_size_aspect_ratio, crop_area_ratio_range,
                                                                  batch_height, batch_width, target_area,
                                                                  random_location=True):
        """
        Code from https://github.com/pytorch/vision/blob/d367a01a18a3ae6bee13d8be3b63fd6a581ea46f/torchvision/transforms/transforms.py
        :param crop_size_aspect_ratio:
        :param crop_area_ratio_range:
        :param batch_height:
        :param batch_width:
        :param target_area:
        :return:
        """
        log_ratio = log(tensor(crop_size_aspect_ratio))
        for _ in range(10):
            aspect_ratio = exp(empty(1).uniform_(*log_ratio)).item()
            crop_height = int(round(sqrt(target_area / aspect_ratio)))
            crop_width = int(round(sqrt(target_area * aspect_ratio)))
            if 0 < crop_width <= batch_width and 0 < crop_height <= batch_height:
                if random_location:
                    h_begin = randint(0, batch_height - crop_height + 1, size=(1,)).item()
                    w_begin = randint(0, batch_width - crop_width + 1, size=(1,)).item()
                else:
                    h_begin, w_begin = (batch_height - crop_height) // 2, (batch_width - crop_width) // 2
                return h_begin, w_begin, h_begin + crop_height, w_begin + crop_width
        # Fallback to central crop
        return DataInterface.get_central_crop_args(batch_width, batch_height, crop_area_ratio_range)

    @staticmethod
    @final
    def get_random_crop_args_within_ratio_range_given_fixed_side(crop_height, crop_width, crop_area_ratio_range,
                                                                 batch_height, batch_width, random_location=True):
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
            target_area = batch_height * batch_width * empty(1).uniform_(*crop_area_ratio_range).item()
            if crop_height is None:
                crop_height = int(target_area / crop_width)
            elif crop_width is None:
                crop_width = int(target_area / crop_height)
            if 0 < crop_height <= batch_height and 0 < crop_width <= batch_width:
                if random_location:
                    h_begin = randint(0, batch_height - crop_height + 1, size=(1,)).item()
                    w_begin = randint(0, batch_width - crop_width + 1, size=(1,)).item()
                else:
                    h_begin, w_begin = (batch_height - crop_height) // 2, (batch_width - crop_width) // 2
                return h_begin, w_begin, h_begin + crop_height, w_begin + crop_width
        # Fallback to central crop
        return DataInterface.get_central_crop_args(batch_width, batch_height, crop_area_ratio_range)

    @staticmethod
    @abstractmethod
    def center_crop(sub_batch, batch_height, batch_width, crop_height, crop_width):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _get_fixed_crop_function(crop_size, random_location):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _get_random_one_side_fixed_crop_function(crop_height, crop_width, crop_area_ratio_range, random_location):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _get_crop_by_range_or_value_function(crop_area_ratio_range, crop_size, random_location):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _get_random_fixed_area_crop(crop_area_ratio_range, crop_size, random_location):
        raise NotImplementedError
