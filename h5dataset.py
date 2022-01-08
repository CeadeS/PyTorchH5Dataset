from torch.utils.data import Dataset
import numpy as np
import h5py, hdf5plugin, tables  # fixes plugins not found Exceptions
import torch
from math import sqrt
import imghdr
import sndhdr
import os
import pandas as pd


class H5Dataset(Dataset):

    @staticmethod
    def blosc_opts(complevel=9, complib='blosc:lz4', shuffle: any = True):
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
    def load_image_sample(sample):
        from PIL import Image
        from skimage import io as skio
        import io
        key = 'path' if 'path' in sample.keys() else 'FilePath'
        file_type = sample['type']
        if file_type in ['tif', 'tiff']:
            im = skio.imread(sample[key])
            im = np.moveaxis(np.array(im), -1, 0)
            return im
        else:
            with open(sample[key], 'rb') as f:
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
            im = H5Dataset.load_image_sample(sample)
            if batch_array is None:
                batch_dimensions = ([len(batch_list)] + [im.shape[0], max_height, max_width])
                batch_array = np.zeros(batch_dimensions, dtype=np.uint8)
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
        in_ratio = float(batch_width) / float(batch_height)
        if in_ratio < min(crop_area_ratio_range):
            w = batch_width
            h = int(round(w / min(crop_area_ratio_range)))
        elif in_ratio > max(crop_area_ratio_range):
            h = batch_height
            w = int(round(h * max(crop_area_ratio_range)))
        else:  # whole image
            w = batch_width
            h = batch_height
        h_begin = (batch_height - h) // 2
        w_begin = (batch_width - w) // 2
        return h_begin, w_begin, h_begin + h, w_begin + w

    @staticmethod
    def get_random_crop_within_ratio_range_given_crop_size_ratio(crop_size_aspect_ratio, crop_area_ratio_range,
                                                                 batch_height, batch_width):
        """
        Stolen from https://github.com/pytorch/vision/blob/d367a01a18a3ae6bee13d8be3b63fd6a581ea46f/torchvision/transforms/transforms.py
        :param crop_size_aspect_ratio:
        :param crop_area_ratio_range:
        :param batch_height:
        :param batch_width:
        :return:
        """
        log_ratio = torch.log(torch.tensor(crop_size_aspect_ratio))
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
        Stolen from https://github.com/pytorch/vision/blob/d367a01a18a3ae6bee13d8be3b63fd6a581ea46f/torchvision/transforms/transforms.py
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
                                  crop_size=(400,),
                                  crop_area_ratio_range=(.8,)):

        """
        Function crops a sub batch i.e a h5_group.
        The image Tensor is expected to have [..., H, W] shape.
        The original batch pixel area is reduced matching the crop_area_ratio.
        The pixel aspect ratio is given by crop_size

        :parameters
        h5_group
            The selected h5_group containing the datasets' images
        batch_width
            The Width of the images in the sub batch
        batch_height
            The height of the images in the sub batch
        crop_size
            If crop size is a integer tuple the given side(s) is/are fixed.
            If crop size is given as a float value the width/height aspect ratio is fixed.
            If crop size is given as a 2-tuple of floats the height/width aspect ratio is in range of the crop size.
            all floats must be in [0.1,10]
            all uints must be in (0, maxint)
        crop_area_ratio_range
            Float tuple determining the cropped area of the batched images.
            If its length is 1, the ratio is fixed.
            If its a tuple of len 2 the ratio is chosen in between both numbers.
            If its a number it is a fixed area that is cropped from the image
            Any floats must be in (0,2]

        :param h5_group: h5group of images
        :param batch_width: uint16
        :param batch_height: uint16
        :param crop_size: one of (float), (float, float), (None,uint16),(uint16,None), (uint16,uint16),
        :param crop_area_ratio_range: one of (float), (float, float)
        :return: cropped sub batch of images of type np.ndarray
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

        :parameters
        crop_size
            If crop size is a integer tuple the given side(s) is/are fixed.
            If crop size is given as a float value the width/height aspect ratio is fixed.
            If crop size is given as a 2-tuple of floats the height/width aspect ratio is in range of the crop size.
            all floats must be in [0.1,10]
            all uints must be in (0, maxint)
        crop_area_ratio_range
            Float tuple determining the cropped area of the batched images.
            If its length is 1, the ratio is fixed.
            If its a tuple of len 2 the ratio is chosen in between both numbers.
            If its a number it is a fixed area that is cropped from the image
            Any floats must be in (0,2]
        :param crop_size: one of (float), (float, float), (None,uint16),(uint16,None), (uint16,uint16),
        :param crop_area_ratio_range: one of (float), (float, float) (int)
        :return: function that crops a h5 group
        """

        assert isinstance(crop_area_ratio_range, (float, int)) or isinstance(crop_area_ratio_range, tuple) and \
               len(
                   crop_area_ratio_range) <= 2, "Crop area ratio must be a int or float value or a tuple of length shorter then 3."

        if isinstance(crop_area_ratio_range, tuple):
            assert all((isinstance(a, float) and 0. < a <= 2. for a in crop_area_ratio_range))
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
                if crop_size[0] is None:
                    crop_size = (crop_size[1], crop_size[1])
                else:
                    crop_size = (crop_size[0], crop_size[0])
            elif len(crop_size) == 2:
                assert crop_size[0] <= crop_size[1], f"Lower bound of crop_size (first value) {crop_size[0]} must be" \
                                                     f"lower then the upper bound (second value) {crop_size[1]}."
                assert (all((a > 0 for a in crop_size))), "Negative Values are not allowed"
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
                h_offset = int((batch_height - crop_height) * torch.rand(1).item())
                w_offset = int((batch_width - crop_width) * torch.rand(1).item())
                w_end = w_offset + crop_width
                h_end = h_offset + crop_height
                return h5_group[..., h_offset:h_end, w_offset:w_end]

            return crop_func

        if isinstance(crop_area_ratio_range, float):
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
    def center_crop(h5_group, batch_width, batch_height, crop_width, crop_height):
        beg_idx_1 = max(0, (batch_height - crop_height) // 2)
        end_idx_1 = beg_idx_1 + crop_height
        beg_idx_2 = max(0, (batch_width - crop_width) // 2)
        end_idx_2 = beg_idx_1 + crop_width
        return h5_group[..., beg_idx_1:end_idx_1, beg_idx_2:end_idx_2]

    @staticmethod
    def center_crop_as_tensor(h5_group, batch_width, batch_height, crop_width, crop_height):
        ret = H5Dataset.center_crop(h5_group, batch_width, batch_height, crop_width, crop_height)
        return torch.as_tensor(ret)

    @staticmethod
    def crop_original_image_from_batch(batch, shapes):
        batch_shape = batch[0].shape
        imlist = []
        for idx, sample_shape in enumerate(shapes):
            beg_idx_1 = (batch_shape[1] - sample_shape[1]) // 2
            end_idx_1 = beg_idx_1 + sample_shape[1]
            beg_idx_2 = (batch_shape[2] - sample_shape[2]) // 2
            end_idx_2 = beg_idx_2 + sample_shape[2]
            imlist.append(batch[idx, :, beg_idx_1:end_idx_1, beg_idx_2:end_idx_2])
        return imlist

    @staticmethod
    def write_batch_to_h5(batch, h5_file):
        raise NotImplementedError

    @staticmethod
    def convert_images_to_dataset(dataset_dataframe,
                                  dataset_destination_h5_file='D:/Datasets/benchmarking/h5/bikini_dataset.h5',
                                  sub_batch_size=50):
        """
        Recursively parse dataset_images_root_folder and add all images to a h5 dataset file
        Writes dataset to dataset_destination_h5_file

        :param dataset_images_root_folder:
        :param dataset_dataframe: pandas dataframe with columns named: Index, FilePath, ClassNo...
        :param dataset_destination_h5_file: string filepath
        :param sub_batch_size: int size of sliced stored together in one group
        :return: None
        """
        sample_data_list = []
        for idx in range(len(dataset_dataframe)):
            row = dataset_dataframe.loc()[idx]
            im = H5Dataset.load_image_sample(row)

            if im.shape == (0, 0, 0):
                image_path = row['FilePath']
                print(f"Error in {image_path}")
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
    def create_metadata_for_dataset(raw_files_dir, filename_to_metadata_func=None):
        assert os.path.exists(raw_files_dir), f"Directory {raw_files_dir} not found"
        datalist = []
        classes = sorted(os.listdir(raw_files_dir))

        i = 0
        for cl_id, cl in enumerate(classes):
            class_dir = os.path.join(raw_files_dir, cl)
            for root, dirs, files in os.walk(class_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    meta_entry_dict = {'FilePath': file_path,
                                       'ClassNo': cl_id,
                                       'Index': i,
                                       'ClassFolderName': cl}
                    f_type = imghdr.what(file_path)
                    if f_type is None:
                        f_type = sndhdr.what(file_path)
                        raise NotImplementedError("Sound files not supported yet")
                    if f_type is None:
                        print(f'Skipped {file_path}, not a recognized file type.')
                        continue
                    meta_entry_dict['FileType'] = f_type
                    if filename_to_metadata_func is not None:
                        meta_entry_dict.update(filename_to_metadata_func(file))
                    datalist.append(meta_entry_dict)
                    i += 1
        return pd.DataFrame(datalist)

    def __len__(self):
        return self.max_idx

    def __getitem__(self, group_no):
        if self.crop_function is None:
            self.crop_function = H5Dataset.random_located_sized_crop_function(
                crop_size=self.crop_size, crop_area_ratio_range=self.crop_area_ratio_range)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.dataset_h5_file_path, "r")

        if self.transforms is not None and self.script_transform is None:
            self.script_transform = torch.jit.script(self.transforms)

        batch_shape = self.batch_shapes[group_no]
        group = self.h5_file[f'samples/{group_no}']
        # return self.random_located_sized_crop(group, *batch_shape, self.crop_size, self.crop_area_ratio_range)

        sample = torch.as_tensor(self.crop_function(group, *batch_shape))
        if self.script_transform is not None:
            self.script_transform = self.script_transform
            sample = self.script_transform(sample)

        meta_data = (torch.as_tensor(self.classes[group_no]), torch.as_tensor(self.indices[group_no]))

        return sample, meta_data

    def get_meta_data_from_indices(self, indices):
        return self.metadata[self.metadata['Index'].isin(indices)]

    def initiate_crop_function(self, loading_crop_size=(0.73, 1.33), loading_crop_area_ratio_range=244 * 244):
        """
        Is called after loading the first batch.
        :param loading_crop_size:
        :param loading_crop_area_ratio_range:
        :return:
        """
        print("called initiate crop function")
        self.crop_function = H5Dataset.random_located_sized_crop_function(
            crop_size=loading_crop_size, crop_area_ratio_range=loading_crop_area_ratio_range)

    @staticmethod
    def create_dataset(
            dataset_name,
            dataset_source_root_files_dir,
            dataset_dest_root_dir,
            dataset_sub_batch_size=50,
            filename_to_metadata_func=lambda s: (zip(('name', 'type'), s.split('.'))),
            overwrite_existing=False
    ):

        assert os.path.exists(dataset_source_root_files_dir), "Raw data root directory not found."

        assert overwrite_existing or os.path.exists(dataset_dest_root_dir), \
            f"Dataset destination directory already exists and overwrite_existing is set to {overwrite_existing}."

        if not os.path.exists(dataset_dest_root_dir):
            print("Create dataset destination file directory.")
            os.mkdir(dataset_dest_root_dir)

        dataset_h5_file_path = os.path.join(dataset_dest_root_dir, dataset_name + '.h5')
        metadata_file_path = os.path.join(dataset_dest_root_dir, dataset_name + '.csv')

        if not os.path.exists(metadata_file_path) or overwrite_existing:
            print('Creating meta data file.')
            metadata = H5Dataset.create_metadata_for_dataset(dataset_source_root_files_dir, filename_to_metadata_func)
            metadata.to_csv(metadata_file_path)
            print("Finished creating meta data file.")
        else:
            metadata = pd.read_csv(metadata_file_path)
            print("Meta data file found. Proceeding")

        if not os.path.exists(dataset_h5_file_path) or overwrite_existing:
            print('Converting raw data files to h5.')
            H5Dataset.convert_images_to_dataset(dataset_dataframe=metadata,
                                                dataset_destination_h5_file=dataset_h5_file_path,
                                                sub_batch_size=dataset_sub_batch_size)
            print('Finished converting Files')

    def __init__(self,
                 dataset_name='dataset_name',
                 dataset_root='/path/to/dataset',
                 loading_crop_size=(0.73, 1.33),
                 loading_crop_area_ratio_range=244 * 244,
                 transforms=None
                 ):
        dataset_h5_file_path = os.path.join(dataset_root, dataset_name + '.h5')
        metadata_file_path = os.path.join(dataset_root, dataset_name + '.csv')

        assert os.path.isfile(dataset_h5_file_path) and os.path.isfile(metadata_file_path), \
            print(f"Data File {dataset_h5_file_path} or Meta Data {metadata_file_path} not "
                  f"found in {dataset_root} directory. Call create_dataset first.")

        super(H5Dataset, self).__init__()

        self.dataset_h5_file_path = dataset_h5_file_path
        self.metadata_file_path = metadata_file_path
        self.crop_size = loading_crop_size
        self.crop_area_ratio_range = loading_crop_area_ratio_range
        self.h5_file = None
        self.crop_function = None
        self.batch_shapes = []
        self.classes = []
        self.indices = []
        self.transforms = transforms
        self.script_transform = None
        self.metadata = pd.read_csv(metadata_file_path)

        with h5py.File(self.dataset_h5_file_path, "r") as h5_file:
            self.max_idx = h5_file['shapes'].attrs['max_idx']
            self.num_samples = 0
            for group_no in range(self.max_idx):
                self.indices.append(np.array(h5_file[f'indices/{group_no}'], dtype=np.dtype('int32')))
                self.num_samples += len(h5_file[f'indices/{group_no}'])
                self.batch_shapes.append(np.array(h5_file[f'batch_shapes/{group_no}'][2:], dtype=np.dtype('int32')))
                self.classes.append(np.array(h5_file[f'classes/{group_no}'], dtype=np.dtype('int32')))

    def __del__(self):
        print("called del")
        if self.h5_file is not None and isinstance(self.h5_file, h5py._hl.files.File):
            print('closing File')
            self.h5_file.close()
        print("Deletion Complete")
