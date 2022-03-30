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
from fn import blosc as blosc
from fn import image as image
from math import floor

#TODO
#handlers = [logging.FileHandler(filename='data_import.log'),logging.StreamHandler(sys.stdout) ]
#logging.basicConfig(format='%(asctime)s %(message)s', level=log_level , handlers=handlers)


class H5Dataset(Dataset):

    @staticmethod
    def load_sample(row, data_mode='blosc'): ## returns object with shape
        if data_mode == 'blosc':
            return blosc.load_sample(row)
        elif data_mode == 'image':
            return image.load_sample(row)
        else:
            raise NotImplementedError



    @staticmethod
    def convert_samples_to_dataset(dataset_dataframe,
                                  dataset_destination_h5_file='./data/test_dataset.h5',
                                  sub_batch_size=50, data_mode = 'blosc'):
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


        directory = Path(dataset_destination_h5_file).parent
        if not os.path.exists(directory):
            os.makedirs(directory)

        if data_mode == 'blosc':
            sample_data_list = sorted(sample_data_list, key=lambda x: x['size'], reverse=False)

            with h5py.File(dataset_destination_h5_file, "w") as h5_file:
                idx = 0
                for idx, (batch, classes, shapes, indices) in enumerate(
                        blosc.batchify_sorted_sample_data_list(sample_data_list, batch_size=sub_batch_size)):
                    h5_file.create_dataset(f"samples/{idx}", data=batch, dtype=batch.dtype,
                                           **blosc.blosc_opts(9, 'blosc:blosclz', 'bit'))
                    h5_file.create_dataset(f"classes/{idx}", data=classes, dtype=np.dtype('uint16'))
                    h5_file.create_dataset(f"shapes/{idx}", data=shapes, dtype=np.dtype('uint16'))
                    h5_file.create_dataset(f"indices/{idx}", data=indices, dtype=np.dtype('uint32'))

                    h5_file.create_dataset(f"batch_shapes/{idx}", data=np.array(batch.shape, dtype=np.dtype('uint16')),
                                           dtype=np.dtype('uint16'))
        elif data_mode == 'image':
            with h5py.File(dataset_destination_h5_file, "w") as h5_file:
                idx = 0
                for idx, (batch, classes, shapes, indices) in enumerate(
                        image.batchify_sample_data_list(sample_data_list, batch_size=sub_batch_size)):
                    h5_file.create_dataset(f"samples/{idx}", data=batch, dtype=h5py.vlen_dtype(bytes))
                    h5_file.create_dataset(f"classes/{idx}", data=classes, dtype=np.dtype('uint16'))
                    h5_file.create_dataset(f"shapes/{idx}", data=shapes, dtype=np.dtype('uint16'))
                    h5_file.create_dataset(f"indices/{idx}", data=indices, dtype=np.dtype('uint32'))

                    h5_file.create_dataset(f"batch_shapes/{idx}", data=np.array(batch.shape, dtype=np.dtype('uint16')),
                                           dtype=np.dtype('uint16'))


            h5_file['samples'].attrs['max_idx'] = idx
            h5_file['samples'].attrs['data_mode'] = data_mode
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
            if self.data_mode == 'blosc':
                self.crop_function = blosc.random_located_sized_crop_function(
                    crop_size=self.crop_size, crop_area_ratio_range=self.crop_area_ratio_range)
            else:
                self.crop_function = image.random_located_sized_crop_function(
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
        if self.data_mode == 'blosc':
            self.crop_function = blosc.random_located_sized_crop_function(
                crop_size=self.crop_size, crop_area_ratio_range=self.crop_area_ratio_range)
        else:
            self.crop_function = image.random_located_sized_crop_function(
                crop_size=self.crop_size, crop_area_ratio_range=self.crop_area_ratio_range)

    @staticmethod
    def create_dataset(
            dataset_name,
            dataset_source_root_files_dir,
            dataset_dest_root_dir,
            dataset_sub_batch_size=50,
            filename_to_metadata_func=lambda s: (zip(('name', 'type'), s.split('.'))),
            overwrite_existing=False,
            no_class_dirs = False,
            data_mode = 'blosc'
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
                                                sub_batch_size=dataset_sub_batch_size, data_mode=data_mode)
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
            self.data_mode = h5_file['samples'].attrs['data_mode']
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
