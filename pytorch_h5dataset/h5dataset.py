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
from math import floor

#TODO
#handlers = [logging.FileHandler(filename='data_import.log'),logging.StreamHandler(sys.stdout) ]
#logging.basicConfig(format='%(asctime)s %(message)s', level=log_level , handlers=handlers)


class H5Dataset(Dataset):

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

    @staticmethod
    def stack_batch_data_padded(batch_list):
        max_height = max(batch_list, key=lambda x: x['height'])['height']
        max_width = max(batch_list, key=lambda x: x['width'])['width']
        batch_array = None
        batch_dimensions = None
        for sample_idx, sample in enumerate(batch_list):
            sample_shape = sample['shape']
            im = H5Dataset.load_sample(sample)
            if batch_array is None:
                batch_dimensions = ([len(batch_list)] + [im.shape[0], max_height, max_width])
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
            batch_array = H5Dataset.stack_batch_data_padded(batch_list)
            shapes = np.array(shapes, dtype=np.uint16)
            yield batch_array, np.array(classes, dtype=np.uint16), \
                  np.array(shapes, dtype=np.uint16), np.array(indices, dtype=np.uint32)

    @staticmethod
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

    @staticmethod
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

        return H5Dataset.get_central_crop_args(batch_width, batch_height, crop_area_ratio_range)

    @staticmethod
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
        return H5Dataset.get_central_crop_args(batch_width, batch_height, crop_area_ratio_range)

    @staticmethod
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
        return H5Dataset.get_central_crop_args(batch_width, batch_height, crop_area_ratio_range)

    @staticmethod
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
        func = H5Dataset.random_located_sized_crop_function(crop_size=crop_size,
                                                            crop_area_ratio_range=crop_area_ratio_range)
        return func(h5_group=h5_group, batch_height=batch_height, batch_width=batch_width)

    @staticmethod
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
                H5Dataset.get_random_crop_within_ratio_range_given_crop_size_ratio(crop_size,
                                                                                   crop_area_ratio_range,
                                                                                   batch_height,
                                                                                   batch_width)
            return h5_group[..., h_offset:h_end, w_offset:w_end]

        return crop_func

    @staticmethod
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
                        H5Dataset.get_random_crop_within_ratio_range_given_fixed_side(crop_height,
                                                                                      crop_width,
                                                                                      crop_area_ratio_range,
                                                                                      batch_height,
                                                                                      batch_width)
                    return h5_group[..., h_offset:h_end, w_offset:w_end]

                return crop_func

            # crop size is given by a range of ints or a single int value
            # and the crop_area_ratio_range is given as float or tuple of floats
            elif all(isinstance(a, int) for a in crop_size) or isinstance(crop_size, int):
                return H5Dataset.get_crop_by_range_or_value_function(crop_area_ratio_range, crop_size)

            # crop size is given by a range of floats or a single float value
            # and the crop_area_ratio_range is given as float value or tuple of floats
            elif all(isinstance(a, float) for a in crop_size) or isinstance(crop_size, float):
                return H5Dataset.get_crop_by_range_or_value_function(crop_area_ratio_range, crop_size)
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
                        H5Dataset.get_random_crop_within_ratio_range_given_target_area(crop_size, (side, side),
                                                                                       batch_height,
                                                                                       batch_width,
                                                                                       crop_area_ratio_range)
                    return h5_group[..., h_offset:h_end, w_offset:w_end]

                return crop_func
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    @staticmethod
    def center_crop(h5_group, batch_height, batch_width, crop_height, crop_width):
        beg_idx_1 = max(0, (batch_height - crop_height) // 2)
        end_idx_1 = beg_idx_1 + crop_height
        beg_idx_2 = max(0, (batch_width - crop_width) // 2)
        end_idx_2 = beg_idx_2 + crop_width
        return h5_group[..., beg_idx_1:end_idx_1, beg_idx_2:end_idx_2]

    @staticmethod
    def center_crop_as_tensor(h5_group, batch_width, batch_height, crop_width, crop_height):
        ret = H5Dataset.center_crop(h5_group, batch_width, batch_height, crop_width, crop_height)
        return torch.as_tensor(ret)

    @staticmethod
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

    @staticmethod
    def convert_samples_to_dataset(dataset_dataframe,
                                  dataset_destination_h5_file='./data/test_dataset.h5',
                                  sub_batch_size=50):
        from pathlib import Path
        """
        Read all samples from the FilePath read from each row of a pandas dataframe and add them to a h5 dataset file.
        Samples will be padded to equal height and width, batched, compressed and saved to the H5File in batches of
        length sub_batch_size.

        :param dataset_dataframe: pandas dataframe with columns named: Index, FilePath, ClassNo, FileType...
        :param dataset_destination_h5_file: string filepath
        :param sub_batch_size: int number of padded samples stored together in one group
        :return: None
        """
        sample_data_list = []
        for idx in range(len(dataset_dataframe)):

            row = dataset_dataframe.loc()[idx]
            im = H5Dataset.load_sample(row)

            if im.shape == (0, 0, 0):
                sample_path = row['FilePath']
                logging.info(f"Error in {sample_path}")
                continue

            sample = {
                'path': row['FilePath'],
                'index': row['Index'],
                'class': row['ClassNo'],
                'type': row['FileType'],
                'height': im.shape[1],
                'width': im.shape[2],
                'size': np.prod(im.shape),
                'shape': im.shape,
            }
            sample_data_list.append(sample)
        sample_data_list = sorted(sample_data_list, key=lambda x: x['size'], reverse=False)

        directory = Path(dataset_destination_h5_file).parent
        if not os.path.exists(directory):
            os.makedirs(directory)

        with h5py.File(dataset_destination_h5_file, "w") as h5_file:
            idx = 0
            for idx, (batch, classes, shapes, indices) in enumerate(
                    H5Dataset.batchify_sorted_sample_data_list(sample_data_list, batch_size=sub_batch_size)):
                h5_file.create_dataset(f"samples/{idx}", data=batch, dtype=batch.dtype,
                                       **H5Dataset.blosc_opts(9, 'blosc:blosclz', 'bit'))
                h5_file.create_dataset(f"classes/{idx}", data=classes, dtype=np.dtype('uint16'))
                h5_file.create_dataset(f"shapes/{idx}", data=shapes, dtype=np.dtype('uint16'))
                h5_file.create_dataset(f"indices/{idx}", data=indices, dtype=np.dtype('uint32'))
                h5_file.create_dataset(f"batch_shapes/{idx}", data=np.array(batch.shape, dtype=np.dtype('uint16')),
                                       dtype=np.dtype('uint16'))
            h5_file['samples'].attrs['max_idx'] = idx
            h5_file['classes'].attrs['max_idx'] = idx
            h5_file['shapes'].attrs['max_idx'] = idx

    @staticmethod
    def create_metadata_for_dataset(raw_files_dir, filename_to_metadata_func=None, no_classes=False):
        """
        Function returns pandas dataframe with meta_data from a directory recusively crawling sub directories.
        ClassNo are given by first level of subfolders; If no subfolders are provided all samples will belong to class 0;
        :param raw_files_dir:
        :param filename_to_metadata_func:
        :param no_classes: if no_classes is True, sub directory step is skipped
        :return:
        """
        assert os.path.exists(raw_files_dir), f"Directory {raw_files_dir} not found"
        datalist = []
        classes = sorted(os.listdir(raw_files_dir))

        if not no_classes:
            for cl in classes:
                class_dir = os.path.join(raw_files_dir, cl)
                if os.path.isdir(class_dir):
                    break
            else:
                raise NotADirectoryError('No class directories provided')

        i = 0
        for cl_id, cl in enumerate(classes):

            if no_classes:
                class_dir = raw_files_dir
            else:
                class_dir = os.path.join(raw_files_dir, cl)
            if os.path.isdir(class_dir):
                for root, dirs, files in os.walk(class_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        meta_entry_dict = {'FilePath': file_path,
                                           'ClassNo': cl_id,
                                           'Index': i,
                                           'ClassFolderName': cl}
                        f_type = imghdr.what(file_path)
                        if f_type is None and sndhdr.what(file_path) is not None:
                            f_type = sndhdr.whathdr(file_path)
                            raise NotImplementedError("Sound files not supported yet")
                        if f_type is None:
                            try:
                                np.load(file_path, allow_pickle=True)
                                f_type = 'np'
                            except ValueError:
                                pass
                        if f_type is None:
                            logging.info(f'Skipped {file_path}, not a recognized file type.')
                            continue
                        meta_entry_dict['FileType'] = f_type
                        if filename_to_metadata_func is not None:
                            meta_entry_dict.update(filename_to_metadata_func(file))
                        datalist.append(meta_entry_dict)
                        i += 1
            if no_classes:
                break
        return pd.DataFrame(datalist)

    def __len__(self):
        return len(self.group_number_mapping)
        return self.max_idx

    def __getitem__(self, group_no):
        group_no = self.group_number_mapping[group_no]
        if self.crop_function is None:
            self.crop_function = H5Dataset.random_located_sized_crop_function(
                crop_size=self.crop_size, crop_area_ratio_range=self.crop_area_ratio_range)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.dataset_h5_file_path, "r")

        if self.transforms is not None and isinstance(self.transforms, torch.nn.Module):
            self.script_transform = torch.jit.script(self.transforms)

        batch_shape = self.batch_shapes[group_no]
        group = self.h5_file[f'samples/{group_no}']

        sample = self.crop_function(group, *batch_shape)  ## FIX for Issue with TIFF uint16 dtypes
        if sample.dtype == np.uint16:
            #sample = sample.astype(int)
            sample = torch.from_numpy(sample.astype(int))
        else:
            sample = torch.as_tensor(sample)
        #sample = torch.from_numpy(sample) ## from_numpy causes Error when do device is called with cuda

        if self.script_transform is not None:
            self.script_transform = self.script_transform
            sample = self.script_transform(sample)

        meta_data = (torch.as_tensor(self.classes[group_no]), torch.as_tensor(self.indices[group_no], dtype=torch.int64))
        #meta_data = (torch.from_numpy(self.classes[group_no]), torch.from_numpy(self.indices[group_no]))

        return sample, meta_data

    def get_meta_data_from_indices(self, indices):
        """
        Returns the metadata from the dataframe given indices as np.array
        :param indices:
        :return:
        """
        return self.metadata[self.metadata['Index'].isin(np.array(indices,dtype=np.int64))]
        #return self.metadata[self.metadata['Index'].isin(np.array(indices,dtype=int))]

    def initiate_crop_function(self, loading_crop_size=(0.73, 1.33), loading_crop_area_ratio_range=244 * 244):
        """
        Is called after loading the first batch. Fixes problem with worker processes that are unable to pickle self.crop_function
        :param loading_crop_size:
        :param loading_crop_area_ratio_range:
        :return:
        """
        logging.info("called initiate crop function")
        self.crop_function = H5Dataset.random_located_sized_crop_function(
            crop_size=loading_crop_size, crop_area_ratio_range=loading_crop_area_ratio_range)

    @staticmethod
    def create_dataset(
            dataset_name,
            dataset_source_root_files_dir,
            dataset_dest_root_dir,
            dataset_sub_batch_size=50,
            filename_to_metadata_func=lambda s: (zip(('name', 'type'), s.split('.'))),
            overwrite_existing=False,
            no_class_dirs = False
    ):
        """
        Creates a dataset files names dataset_name.<h5/csv>
        :param dataset_name: name of the dataset. Also the created dataset files name.
        :param dataset_source_root_files_dir: Data should be stored in this directory. Should have sub directories for the classes.
        :param dataset_dest_root_dir: Destination for h5 and csv file. If csv file is already existing process is resumed with existing file unless overwrite is set true.
        :param dataset_sub_batch_size: Number of Samples padded and batched together. The smaller the number the slower the loading and vice versa.
        :param filename_to_metadata_func: Function that extracts meta data from samples names.
        :param overwrite_existing: If true, meta data and existing h5 are overwritten. If not meta is reused.
        :param no_class_dirs: Classes are provided in subdirs in dataset_source_files_dir or not.
        :return:
        """

        assert os.path.exists(dataset_source_root_files_dir), "Raw data root directory not found."

        assert overwrite_existing or os.path.exists(dataset_dest_root_dir), \
            f"Dataset destination directory already exists and overwrite_existing is set to {overwrite_existing}."

        if not os.path.exists(dataset_dest_root_dir):
            logging.info("Create dataset destination file directory.")
            os.mkdir(dataset_dest_root_dir)

        dataset_h5_file_path = os.path.join(dataset_dest_root_dir, dataset_name + '.h5')
        metadata_file_path = os.path.join(dataset_dest_root_dir, dataset_name + '.csv')

        if not os.path.exists(metadata_file_path) or overwrite_existing:
            logging.info('Creating meta data file.')
            metadata = H5Dataset.create_metadata_for_dataset(dataset_source_root_files_dir, filename_to_metadata_func, no_class_dirs)
            metadata.to_csv(metadata_file_path)
            logging.info("Finished creating meta data file.")
        else:
            metadata = pd.read_csv(metadata_file_path)
            logging.info("Meta data file found. Proceeding")

        if not os.path.exists(dataset_h5_file_path) or overwrite_existing:
            logging.info('Converting raw data files to h5.')
            H5Dataset.convert_samples_to_dataset(dataset_dataframe=metadata,
                                                dataset_destination_h5_file=dataset_h5_file_path,
                                                sub_batch_size=dataset_sub_batch_size)
            logging.info('Finished converting Files')

    def get_group_number_mapping(self):
        group_indices = list(range(self.max_idx))
        if self.split_mode == 'montecarlo': ## select random subset
            random.Random(self.split_number).shuffle(group_indices)
            split_size = int(self.split_ratio * len(group_indices))
            selected_group_indices = group_indices[0:split_size]
        if self.split_mode == 'full':
            split_size = int(self.split_ratio * len(group_indices))
            selected_group_indices = group_indices[:split_size]
        if self.split_mode in ['split', 'cross_val']:
            split_begin = round(sum(self.split_ratio[0:self.split_number])*len(group_indices))
            split_end = round((1-sum(self.split_ratio[self.split_number+1:]))*len(group_indices))
            selected_group_indices = group_indices[split_begin:split_end]
        return selected_group_indices

    def __init__(self,
                 dataset_name='dataset_name',
                 dataset_root='/path/to/dataset',
                 loading_crop_size=(0.73, 1.33),
                 loading_crop_area_ratio_range=244 * 244,
                 transforms=None,
                 split_mode = 'full',
                 split_ratio = 1.0,
                 split_number =0
                 ):
        """
        Constructor of the Dataset.

        :param dataset_name: Name of the datset files.
        :param dataset_root: Path to the dataset folder.
        :param loading_crop_size: Crop size aspect ratio range - see random_located_sized_crop_function().
        :param loading_crop_area_ratio_range: Number of values cropped from sample OR range of cropped values proportion.
        :param transforms: torchvision.transforms instance, Module or Transform. Module is not pickable after first iteration if Modules are used. Instantiate Dataloaders before first iteration!!
        :param split_mode: Determine the data split. ;must be in ['full', 'montecarlo', 'cross_val', split']
        :param split_ratio: float setting the percentage that is split for training if split_mode is full or montecarlo - int if split_mode is cross_val determining the number of splits, tuple of floats if mode is split
        :param split_number: int. for montecarlo this is the random seed
        """
        dataset_h5_file_path = os.path.join(dataset_root, dataset_name + '.h5')
        metadata_file_path = os.path.join(dataset_root, dataset_name + '.csv')
        self.h5_file = None

        assert split_number >=0 and isinstance(split_number, int), f"split_number musst be a nonzero integer value but it is {split_number}"
        assert os.path.isfile(dataset_h5_file_path) and os.path.isfile(metadata_file_path), f"found in {dataset_root} directory. Call create_dataset first."
        assert split_mode.lower() in ['montecarlo', 'full', 'cross_val', 'split'], f"split_mode is {split_number} but must be in ['montecarlo', 'full', 'cross_val', 'split']. Did you mean {difflib.get_close_matches(split_mode, ['montecarlo', 'full', 'cross_val', 'split'])}?"
        if split_mode.lower() == 'montecarlo' or split_mode == 'full':
            assert isinstance(split_ratio, float) and 0<split_ratio<=1, f"The proportion of train samples must be given as float in {split_mode} mode"
        elif split_mode.lower() == 'cross_val':
            assert isinstance(split_ratio, int) and split_ratio > 0, f"Split_ratio in case if cross_val is the number of splits (n-fold) and therefore must be an integer larger than 1."
            assert isinstance(split_number, int) and split_number in range(split_ratio), f"The 'split_number' must be in range of the n-fold. You selected split no. {split_number} but you have {range(split_ratio)} splits."
            split_ratio = tuple((1./split_ratio for _ in range(split_ratio)))
        elif split_mode.lower() == 'split':
            assert isinstance(split_ratio, tuple) and all((isinstance(a, float)for a in split_ratio)) and all((0<a<1.for a in split_ratio)), "In 'split mode', the split_ratio is given as tuple of floats adding up to one. Every number refers to the percentage of data in that split."
            assert 0.999 <= sum(split_ratio) <= 1.001, f"The split ratios must add up to one. Split ratio sum is {sum(split_ratio)}"
            assert isinstance(split_number, int) and split_number in range(len(split_ratio)), f"The 'split_number' must be an integer selecting a split percentage in 'split_ratio'"
        else:
            raise ValueError(f'Unsupported variable combination. This should not happen.')

        super(H5Dataset, self).__init__()

        self.dataset_h5_file_path = dataset_h5_file_path
        self.metadata_file_path = metadata_file_path
        self.crop_size = loading_crop_size
        self.crop_area_ratio_range = loading_crop_area_ratio_range

        self.crop_function = None
        self.batch_shapes = []
        self.classes = []
        self.indices = []
        self.transforms = transforms
        self.script_transform = None
        self.metadata = pd.read_csv(metadata_file_path)
        self.split_mode = split_mode.lower()
        self.split_ratio = split_ratio
        self.split_number = split_number

        with h5py.File(self.dataset_h5_file_path, "r") as h5_file:
            self.max_idx = h5_file['shapes'].attrs['max_idx']+1
            self.num_samples = 0
            for group_no in range(self.max_idx):
                self.indices.append(np.array(h5_file[f'indices/{group_no}'], dtype=np.dtype('int32')))
                self.num_samples += len(h5_file[f'indices/{group_no}'])
                self.batch_shapes.append(np.array(h5_file[f'batch_shapes/{group_no}'][2:], dtype=np.dtype('int32')))
                self.classes.append(np.array(h5_file[f'classes/{group_no}'], dtype=np.dtype('int32')))

        self.sub_batch_size = self.batch_shapes[0].shape[0]

        self.group_number_mapping =  self.get_group_number_mapping()

    def __del__(self):
        logging.info("called del")
        if self.h5_file is not None and isinstance(self.h5_file, h5py._hl.files.File):
            logging.info('closing File')
            self.h5_file.close()
        logging.info("Deletion Complete")
