import difflib
import imghdr
import logging
import os
import random
import sndhdr
import tarfile
from abc import ABC
from math import ceil
import hdf5plugin
# TODO
# handlers = [logging.FileHandler(filename='data_import.log'),logging.StreamHandler(sys.stdout) ]
# logging.basicConfig(format='%(asctime)s %(message)s', level=log_level , handlers=handlers)
from random import shuffle
from typing import final
from pathlib import Path

import h5py  # fixes plugins not found Exceptions
import numpy as np
import pandas as pd
from jpegtran import lib

from simplejpeg import decode_jpeg as jpeg_decode
from torch.utils.data import Dataset
from tqdm import tqdm

from ..fn.blosc import BloscInterface
from ..fn.image import ImageInterface

from tar_dir_indexer.tarDirIndexer import TarDirIndexer

logging.info(f"hdf5plugin version == {hdf5plugin._version.version}")

def time_left_str(seconds):
    ##from https://www.adamsmith.haus/python/answers/how-to-convert-seconds-to-days,-hours,-and-minutes-in-python#:~:text=Use subtraction and floor division,each result in a variable.
    seconds_in_day = 60 * 60 * 24
    seconds_in_hour = 60 * 60
    seconds_in_minute = 60
    days = seconds // seconds_in_day
    hours = (seconds - (days * seconds_in_day)) // seconds_in_hour
    minutes = (seconds - (days * seconds_in_day) - (hours * seconds_in_hour)) // seconds_in_minute
    seconds = seconds - days * seconds_in_day - hours * seconds_in_hour - minutes * seconds_in_minute
    return f"Time Left: {int(days):3d} days, {int(hours):2d} hours, {int(minutes):2d} minutes, and {int(seconds):3d} seconds  "

class H5MetaDataset(Dataset, ABC):

    @staticmethod
    @final
    def load_sample(row, data_interface=BloscInterface):  ## returns object with shape
        return data_interface.load_sample(row)

    @staticmethod
    def write_tar_jpeg_file_data_to_hdf5(tar_root_in_dir, tar_files_contents_lists, n_samples,
                                         h5_file_name='imagenet.h5',
                                         sub_batch_size=1, max_n_group=int(1e5),
                                         shuffle_indexes=False):
        from time import time

        n_batches = ceil(n_samples / sub_batch_size)
        meta_cls = np.zeros((n_batches, sub_batch_size, 1), dtype=np.int32)
        meta_shapes = np.zeros((n_batches, sub_batch_size, 2), dtype=np.uint16)
        meta_indexes = np.zeros((n_batches, sub_batch_size, 1), dtype=np.int64)
        meta_max_shapes = np.zeros((n_batches, sub_batch_size, 2), dtype=np.uint16)
        indexes = list(range(n_samples))
        if shuffle_indexes:
            shuffle(indexes)
        idx = 0
        t0 = time()
        iter_p_sec = -1

        with h5py.File(h5_file_name, "w", fs_strategy='fsm', fs_persist='ALL', fs_threshold=1) as h5_file:
            for tar_dict in tar_files_contents_lists:
                tar_file_name, tar_contents = tar_dict['tar_file'], tar_dict['contents']
                file_path = os.path.join(tar_root_in_dir, tar_file_name)
                with tarfile.open(file_path, "r") as tar_file:
                    for tar_content in tar_contents:
                        content_name, tar_file_name, cl, index, _ = tar_content
                        im_bytes = tar_file.extractfile(content_name).read()
                        np_obj = np.frombuffer(im_bytes, dtype=f'S{len(im_bytes)}')
                        try:
                            jpeg_decode(im_bytes)
                        except:
                            print(f"Skipped File {content_name} in {file_path}")
                            continue
                        sample_index = indexes.pop(0)
                        batch_index = sample_index // sub_batch_size
                        group_key = str(int(batch_index) // max_n_group)
                        dataset_key = f"samples/{str(int(batch_index) % max_n_group)}"
                        # print(f"Writing {group_key}{dataset_key}, with {batch_index}, {sample_index}")
                        sub_batch_key = 0
                        if group_key in h5_file.keys():
                            if dataset_key in h5_file[group_key].keys():
                                sample = h5_file[group_key][dataset_key][()]
                                sub_batch_key = len(sample)
                                np_obj = [*sample, np_obj]
                                del h5_file[group_key][dataset_key]
                            else:
                                np_obj = [np_obj]

                        else:
                            h5_file.create_group(group_key)
                            np_obj = [np_obj]

                        h5_file[group_key].create_dataset(dataset_key, data=np_obj)

                        shape = lib.Transformation(im_bytes).get_dimensions()

                        meta_cls[batch_index, sub_batch_key] = cl
                        meta_shapes[batch_index, sub_batch_key] = shape
                        meta_indexes[batch_index, sub_batch_key] = index
                        meta_max_shapes[batch_index, sub_batch_key] = shape
                        idx = idx + 1
                        if iter_p_sec == -1:
                            iter_p_sec = 1 / (time() - t0)
                        else:
                            iter_p_sec = 0.001 * (1 / (time() - t0)) + 0.999 * iter_p_sec
                        print(
                            f"\r{int(idx):7d} of {n_samples:7d} written {iter_p_sec:4.2f} iter/s {time_left_str(int((n_samples - idx) / iter_p_sec))}",
                            end='')
                        t0 = time()
                        if idx % 1000 == 0:
                            logging.info(f"{int(idx):7d} of {n_samples:7d} written")

        return meta_cls, meta_shapes, meta_indexes, meta_max_shapes, n_batches

    @staticmethod
    @final
    def tar_dir_to_hdf5_dataset(tar_root_in_dir='ILSVRC2012_img_train',
                                root_hdf5_out_dir='ILSVRC2012_img_train_h5',
                                dataset_name='imagenet_meta', shuffle_tar_data=False, sub_batch_size=1,
                                max_n_group=int(1e5), lower_file_count_limit=0):

        tar_files_list = os.listdir(tar_root_in_dir)
        Path(root_hdf5_out_dir).mkdir(parents=True, exist_ok=True)
        tar_files_contents_list = []
        classes = []
        index = 0
        all_contents = []

        cl = 0
        for tar_file_name in tqdm(tar_files_list, total=len(tar_files_list)):

            file = os.path.join(tar_root_in_dir, tar_file_name)
            tar_file_contents = {'tar_file': tar_file_name, 'contents': []}

            if tarfile.is_tarfile(file):
                with tarfile.open(file, "r") as tf:
                    nnames = tf.getnames()
                    if lower_file_count_limit > 0 and len(nnames) < lower_file_count_limit:
                        print(f'Skipping {file} with only {len(nnames)} samples')
                        logging.info(f'Skipping {file} with only {len(nnames)} samples')
                        continue
                    tar_file_contents['contents'].extend(
                        ((n, tar_file_name[:-4], cl, index + i, 'jpeg') for i, n in enumerate(nnames)))
                    all_contents.extend(((n, tar_file_name[:-4], cl, index + i, 'jpeg') for i, n in enumerate(nnames)))
                    index = index + len(nnames)
            classes.append((cl, tar_file_name))
            cl = cl+1
            tar_files_contents_list.append(tar_file_contents)



        h5_file_name = f"{os.path.join(root_hdf5_out_dir, dataset_name)}.h5"

        meta = H5MetaDataset.write_tar_jpeg_file_data_to_hdf5(tar_root_in_dir, tar_files_contents_list,
                                                              h5_file_name=h5_file_name, sub_batch_size=sub_batch_size,
                                                              max_n_group=max_n_group, n_samples=index,
                                                              shuffle_indexes=shuffle_tar_data)

        classes_list, shapes_list, indices_list, batch_shapes_list, max_idx = meta

        df = pd.DataFrame(all_contents, columns=['FileName', 'Class', 'ClassNo', 'Index', 'FileType'])
        df.to_csv(f"{os.path.join(root_hdf5_out_dir, dataset_name)}.csv")
        data_dtype = str(bytes)

        with h5py.File(h5_file_name, "a") as h5_file:
            h5_file.create_group('attrs')
            h5_file['attrs'].create_dataset('classes', data=np.array(classes_list[:-1], dtype=np.dtype('int32')))
            h5_file['attrs'].create_dataset('indices', data=np.array(indices_list[:-1], dtype=np.dtype('int64')))
            h5_file['attrs'].create_dataset('batch_shapes',
                                            data=np.array(batch_shapes_list[:-1], dtype=np.dtype('uint16')))
            h5_file['attrs'].create_dataset('shapes', data=np.array(shapes_list[:-1], dtype=np.dtype('uint16')))
            h5_file['attrs'].create_dataset('last_classes', data=np.array(classes_list[-1], dtype=np.dtype('int32')))
            h5_file['attrs'].create_dataset('last_indices', data=np.array(indices_list[-1], dtype=np.dtype('int64')))
            h5_file['attrs'].create_dataset('last_batch_shapes',
                                            data=np.array(batch_shapes_list[-1], dtype=np.dtype('uint16')))
            h5_file['attrs'].create_dataset('last_shapes', data=np.array(shapes_list[-1], dtype=np.dtype('uint16')))
            # h5_file.attrs['classes'] =np.stack(classes_list)
            # h5_file.attrs['shapes'] =np.stack(shapes_list)
            # h5_file.attrs['indices'] =np.stack(indices_list)
            # h5_file.attrs['batch_shapes'] =np.stack(batch_shapes_list)
            h5_file.attrs['max_idx'] = max_idx
            h5_file.attrs['num_samples'] = int(index)
            h5_file.attrs['max_n_group'] = int(max_n_group)
            h5_file.attrs['data_mode'] = str('image')
            h5_file.attrs['data_dtype'] = str(data_dtype)
            h5_file.attrs['sub_batch_size'] = int(sub_batch_size)


    @staticmethod
    def get_class_mappings(tar_dir_index, path_to_metadata_function, test=False):
        for node in tar_dir_index.index:
            node.meta = path_to_metadata_function(str(node.get_path()))
            if test:
                print(node.meta)

        classes = []
        for node in tar_dir_index.index:
            classes.append(node.meta['ClassFolderName'])


        classes = sorted(list(set(classes)))
        if test:
            print(classes)
        classes_mapping = dict((classname, classidx) for classidx, classname in enumerate(classes))
        if test:
            print(classes_mapping)
            raise Exception('Test Finished')

        for node in tar_dir_index.index:
            node.meta['ClassNo'] = classes_mapping[node.meta['ClassFolderName']]

        return classes_mapping


    @staticmethod
    @final
    def _convert_tar_dir_to_dataset(data_root, path_to_metadata_function=lambda x: {'ClassFolderName':str(x).split('/')[0]},
                                    dataset_destination='./data', dataset_name = 'test_dataset',
                                    sub_batch_size=50, data_mode='blosc', max_n_group=10, test = False):
        from pathlib import Path
        """
        Read all samples from the FilePath read from each row of a pandas dataframe and add them to a h5 dataset file.
        Samples will be padded to equal height and width, batched, compressed and saved to the H5File in batches of
        length sub_batch_size.

        :param dataset_dataframe: pandas dataframe with columns named: Index, FilePath, ClassNo, FileType...
        :param dataset_destination_h5_file: string filepath
        :param sub_batch_size: int number of padded samples stored together in one group
        :param max_n_group: Number of datasets per h5group. Value between 1e-5 and 5e-5.
        :param images_from_tar_mode: bool enables loading images from tar balls
        :param data_mode: str. either 'blosc' (lossless multi channel) or 'image' (jpeg)
        :return: None
        """

        assert data_mode.lower() in ['blosc', 'jpeg', 'jpg', 'image'], "data_mode must be 'blosc' or 'image'"

        data_mode = 'blosc' if data_mode.lower() == 'blosc' else 'image'
        data_interface = BloscInterface if data_mode == 'blosc' else ImageInterface

        tar_dir_index = TarDirIndexer(data_root)
        H5MetaDataset.get_class_mappings(tar_dir_index, path_to_metadata_function, test)

        sample_data_list = []
        meta_entry_dict_list = []
        if data_mode == 'blosc':
            tar_dir_index.get_shapes()
        else:
            shuffle(tar_dir_index.index)
        for idx, node in enumerate(tqdm(tar_dir_index.index, desc="Extracting meta data")):

            if data_mode == 'blosc':
                im_shape = tuple((node.shape[-1],node.shape[-3],node.shape[-2]))
            else:
                im_shape = np.array((0,0,0))

            sample = {
                'path': str(node.get_path()),
                'index': idx,
                'class': node.meta['ClassNo'],
                'type': node.dtype,
                'height': im_shape[-2],
                'width': im_shape[-1],
                'crop_size': np.prod(im_shape),
                'shape': im_shape,
                'node': node
            }

            meta_entry_dict = {'FilePath': str(node.get_path()),
                               'ClassNo': node.meta['ClassNo'],
                               'Index': idx,
                               'ClassFolderName': node.meta['ClassFolderName'],
                               'FileType': node.dtype}

            meta_entry_dict.update(node.meta)
            meta_entry_dict_list.append(meta_entry_dict)
            sample_data_list.append(sample)


        dataframe = pd.DataFrame(meta_entry_dict_list)

        Path(dataset_destination).mkdir(parents=True,exist_ok=True)
        dataframe.to_csv(Path(dataset_destination,dataset_name+'.csv'))
        dataset_destination_h5_file = Path(dataset_destination,dataset_name+'.h5')
        directory = Path(dataset_destination_h5_file).parent
        if not os.path.exists(directory):
            os.makedirs(directory)

        with h5py.File(dataset_destination_h5_file, "w") as h5_file:
            idx = 0
            num_samples = 0
            classes_list = []
            shapes_list = []
            indices_list = []
            batch_shapes_list = []

            if data_mode == 'blosc':
                sample_data_list = sorted(sample_data_list, key=lambda x: x['crop_size'], reverse=False)
                for idx, (batch, classes, shapes, indices) in tqdm(enumerate(
                        data_interface.batchify_sorted_sample_data_list(sample_data_list, batch_size=sub_batch_size)),
                        desc="Writing Samples to H5"):
                    if idx % max_n_group == 0:
                        group_key = str(idx // max_n_group)
                        h5_file.create_group(group_key)
                    h5_file[group_key].create_dataset(f"samples/{int(idx % max_n_group)}", data=batch,
                                                      dtype=batch.dtype,
                                                      **data_interface.blosc_opts(9, 'blosc:blosclz', 'bit'))
                    classes_list.append(np.array(classes, dtype=np.dtype('int32')))
                    shapes_list.append(np.array(shapes, dtype=np.dtype('uint16')))
                    indices_list.append(np.array(indices, dtype=np.dtype('int64')))
                    batch_shapes_list.append(np.array(batch.shape, dtype=np.dtype('uint16')))
                    num_samples += len(batch)
                data_dtype = batch[0].dtype
            elif data_mode == 'image':

                for idx, (batch, classes, shapes, indices) in tqdm(enumerate(
                        data_interface.batchify_sample_data_list(sample_data_list, batch_size=sub_batch_size)),
                        desc="Writing Samples to H5"):
                    if idx % max_n_group == 0:
                        group_key = str(idx // max_n_group)
                        h5_file.create_group(group_key)
                    h5_file[group_key].create_dataset(f"samples/{int(idx % max_n_group)}", data=batch)
                    classes_list.append(np.array(classes, dtype=np.dtype('int32')))
                    __shapes = np.array(shapes, dtype=np.dtype('uint16'))
                    shapes_list.append(__shapes)
                    indices_list.append(np.array(indices, dtype=np.dtype('int64')))
                    batch_shapes_list.append(__shapes.max(axis=1))
                    print(__shapes.max(axis=1))
                    num_samples += len(batch)
                data_dtype = str(bytes)

            h5_file.create_group('attrs')
            h5_file['attrs'].create_dataset('classes', data=np.array(classes_list[:-1], dtype=np.dtype('int32')))
            h5_file['attrs'].create_dataset('indices', data=np.array(indices_list[:-1], dtype=np.dtype('int64')))
            h5_file['attrs'].create_dataset('batch_shapes',
                                            data=np.array(batch_shapes_list[:-1], dtype=np.dtype('uint16')))
            h5_file['attrs'].create_dataset('shapes', data=np.array(shapes_list[:-1], dtype=np.dtype('uint16')))
            h5_file['attrs'].create_dataset('last_classes', data=np.array(classes_list[-1], dtype=np.dtype('int32')))
            h5_file['attrs'].create_dataset('last_indices', data=np.array(indices_list[-1], dtype=np.dtype('int64')))
            h5_file['attrs'].create_dataset('last_batch_shapes',
                                            data=np.array(batch_shapes_list[-1], dtype=np.dtype('uint16')))
            h5_file['attrs'].create_dataset('last_shapes', data=np.array(shapes_list[-1], dtype=np.dtype('uint16')))

            # h5_file['attrs/indices'] =np.stack(indices_list)
            # h5_file['attrs/batch_shapes'] =np.stack(batch_shapes_list)
            h5_file.attrs['max_idx'] = int(idx)
            h5_file.attrs['num_samples'] = int(num_samples)
            h5_file.attrs['max_n_group'] = int(max_n_group)
            h5_file.attrs['data_mode'] = str(data_mode)
            h5_file.attrs['data_dtype'] = str(data_dtype)
            h5_file.attrs['sub_batch_size'] = int(sub_batch_size)

    @staticmethod
    @final
    def _convert_samples_to_dataset(dataset_dataframe,
                                    dataset_destination_h5_file='./data/test_dataset.h5',
                                    sub_batch_size=50, data_mode='blosc', max_n_group=10):
        from pathlib import Path
        """
        Read all samples from the FilePath read from each row of a pandas dataframe and add them to a h5 dataset file.
        Samples will be padded to equal height and width, batched, compressed and saved to the H5File in batches of
        length sub_batch_size.

        :param dataset_dataframe: pandas dataframe with columns named: Index, FilePath, ClassNo, FileType...
        :param dataset_destination_h5_file: string filepath
        :param sub_batch_size: int number of padded samples stored together in one group
        :param max_n_group: Number of datasets per h5group. Value between 1e-5 and 5e-5.
        :param images_from_tar_mode: bool enables loading images from tar balls
        :param data_mode: str. either 'blosc' (lossless multi channel) or 'image' (jpeg)
        :return: None
        """

        assert data_mode.lower() in ['blosc', 'jpeg', 'jpg', 'image'], "data_mode must be 'blosc' or 'image'"

        data_mode = 'blosc' if data_mode.lower() == 'blosc' else 'image'
        data_interface = BloscInterface if data_mode == 'blosc' else ImageInterface

        sample_data_list = []
        for idx in tqdm(range(len(dataset_dataframe)), desc="Checking Samples"):

            row = dataset_dataframe.loc()[idx]
            im, im_shape = H5MetaDataset.load_sample(row, data_interface)

            if not (isinstance(im, np.ndarray) or isinstance(im, bytes)):
                sample_path = row['FilePath']
                logging.info(f"Error in {sample_path}")
                continue

            sample = {
                'path': row['FilePath'],
                'index': row['Index'],
                'class': row['ClassNo'],
                'type': row['FileType'],
                'height': im_shape[-2],
                'width': im_shape[-1],
                'crop_size': np.prod(im_shape),
                'shape': im_shape,
            }
            sample_data_list.append(sample)

        directory = Path(dataset_destination_h5_file).parent
        if not os.path.exists(directory):
            os.makedirs(directory)

        with h5py.File(dataset_destination_h5_file, "w") as h5_file:
            idx = 0
            num_samples = 0
            classes_list = []
            shapes_list = []
            indices_list = []
            batch_shapes_list = []

            if data_mode == 'blosc':
                sample_data_list = sorted(sample_data_list, key=lambda x: x['crop_size'], reverse=False)
                for idx, (batch, classes, shapes, indices) in tqdm(enumerate(
                        data_interface.batchify_sorted_sample_data_list(sample_data_list, batch_size=sub_batch_size)),
                        desc="Writing Samples to H5"):
                    if idx % max_n_group == 0:
                        group_key = str(idx // max_n_group)
                        h5_file.create_group(group_key)
                    h5_file[group_key].create_dataset(f"samples/{int(idx % max_n_group)}", data=batch,
                                                      dtype=batch.dtype,
                                                      **data_interface.blosc_opts(9, 'blosc:blosclz', 'bit'))
                    classes_list.append(np.array(classes, dtype=np.dtype('int32')))
                    shapes_list.append(np.array(shapes, dtype=np.dtype('uint16')))
                    indices_list.append(np.array(indices, dtype=np.dtype('int64')))
                    batch_shapes_list.append(np.array(batch.shape, dtype=np.dtype('uint16')))
                    num_samples += len(batch)
                data_dtype = batch[0].dtype
            elif data_mode == 'image':

                for idx, (batch, classes, shapes, indices) in tqdm(enumerate(
                        data_interface.batchify_sample_data_list(sample_data_list, batch_size=sub_batch_size)),
                        desc="Writing Samples to H5"):
                    if idx % max_n_group == 0:
                        group_key = str(idx // max_n_group)
                        h5_file.create_group(group_key)
                    h5_file[group_key].create_dataset(f"samples/{int(idx % max_n_group)}", data=batch)
                    classes_list.append(np.array(classes, dtype=np.dtype('int32')))
                    __shapes = np.array(shapes, dtype=np.dtype('uint16'))
                    shapes_list.append(__shapes)
                    indices_list.append(np.array(indices, dtype=np.dtype('int64')))
                    batch_shapes_list.append(__shapes.max(axis=1))
                    num_samples += len(batch)
                data_dtype = str(bytes)

            h5_file.create_group('attrs')
            h5_file['attrs'].create_dataset('classes', data=np.array(classes_list[:-1], dtype=np.dtype('int32')))
            h5_file['attrs'].create_dataset('indices', data=np.array(indices_list[:-1], dtype=np.dtype('int64')))
            h5_file['attrs'].create_dataset('batch_shapes',
                                            data=np.array(batch_shapes_list[:-1], dtype=np.dtype('uint16')))
            h5_file['attrs'].create_dataset('shapes', data=np.array(shapes_list[:-1], dtype=np.dtype('uint16')))
            h5_file['attrs'].create_dataset('last_classes', data=np.array(classes_list[-1], dtype=np.dtype('int32')))
            h5_file['attrs'].create_dataset('last_indices', data=np.array(indices_list[-1], dtype=np.dtype('int64')))
            h5_file['attrs'].create_dataset('last_batch_shapes',
                                            data=np.array(batch_shapes_list[-1], dtype=np.dtype('uint16')))
            h5_file['attrs'].create_dataset('last_shapes', data=np.array(shapes_list[-1], dtype=np.dtype('uint16')))

            # h5_file['attrs/indices'] =np.stack(indices_list)
            # h5_file['attrs/batch_shapes'] =np.stack(batch_shapes_list)
            h5_file.attrs['max_idx'] = int(idx)
            h5_file.attrs['num_samples'] = int(num_samples)
            h5_file.attrs['max_n_group'] = int(max_n_group)
            h5_file.attrs['data_mode'] = str(data_mode)
            h5_file.attrs['data_dtype'] = str(data_dtype)
            h5_file.attrs['sub_batch_size'] = int(sub_batch_size)


    @staticmethod
    @final
    def create_metadata_for_jpeg_dataset(raw_files_dir, filename_to_metadata_func=None, no_classes=False):
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

        total = 0
        for root, dirs, files in os.walk(raw_files_dir):
            total += len(files)

        if not no_classes:
            for cl in classes:
                class_dir = os.path.join(raw_files_dir, cl)
                if os.path.isdir(class_dir):
                    break
            else:
                raise NotADirectoryError('No class directories provided')

        i = 0
        with tqdm(total=total) as pbar:
            for cl_id, cl in enumerate(classes):

                if no_classes:
                    class_dir = raw_files_dir
                else:
                    class_dir = os.path.join(raw_files_dir, cl)
                if os.path.isdir(class_dir):
                    for root, dirs, files in os.walk(class_dir):
                        for file in files:
                            pbar.set_description(f"Working on {cl}/{file}")
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

                            pbar.update()
                if no_classes:
                    break
        return pd.DataFrame(datalist)

    @final
    def __len__(self):
        return len(self.group_number_mapping)
        #return self.max_idx + 1

    def __getitem__(self, sub_batch_idx):
        """
        Returns a  Tuple of data:h5_file object and (classes:tensor, index:tensor)
        :param sub_batch_idx:
        :return:
        """

        sub_batch_slice = None
        if isinstance(sub_batch_idx, tuple) and (
                isinstance(sub_batch_idx[1], int) or isinstance(sub_batch_idx[1], slice)):
            sub_batch_idx, sub_batch_slice = sub_batch_idx
            if isinstance(sub_batch_slice, int):
                sub_batch_slice = slice(sub_batch_slice, sub_batch_slice + 1)

        if self.h5_file is None:
            self.h5_file = h5py.File(self.dataset_h5_file_path, "r")

        try: ## index == 0 hotfix
            sub_batch_idx = self.group_number_mapping[sub_batch_idx]
            group_no = str(sub_batch_idx // self.max_n_group)
            dataset_no = str(sub_batch_idx % self.max_n_group)

            sub_batch = self.h5_file[group_no][f'samples/{dataset_no}']

            meta_data = (self.classes[sub_batch_idx],
                         self.indices[sub_batch_idx])

            if sub_batch_slice is not None:
                sub_batch, meta_data = sub_batch[sub_batch_slice].squeeze(), \
                                       (np.array(meta_data[0])[sub_batch_slice].squeeze(),
                                        np.array(meta_data[1])[sub_batch_slice].squeeze())

            return sub_batch, np.array(meta_data)
        except:
            sub_batch_idx
            sub_batch_idx = self.group_number_mapping[sub_batch_idx]
            group_no = str(sub_batch_idx // self.max_n_group)
            dataset_no = str(sub_batch_idx % self.max_n_group)

            sub_batch = self.h5_file[group_no][f'samples/{dataset_no}']

            meta_data = (self.classes[sub_batch_idx],
                         self.indices[sub_batch_idx])

            if sub_batch_slice is not None:
                sub_batch, meta_data = sub_batch[sub_batch_slice].squeeze(), \
                                       (np.array(meta_data[0])[sub_batch_slice].squeeze(),
                                        np.array(meta_data[1])[sub_batch_slice].squeeze())

            return sub_batch, np.array(meta_data)


    def reset(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = h5py.File(self.dataset_h5_file_path, "r")

    @final
    def get_meta_data_from_indices(self, indices):
        """
        Returns the metadata from the dataframe given indices as np.array
        :param indices:
        :return:
        """
        return self.metadata.iloc[pd.Index(self.metadata['Index']).get_indexer(np.asarray(indices, dtype=np.int64))]
        #return self.metadata[self.metadata['Index'].isin(np.asarray(indices, dtype=np.int64))]

    @classmethod
    @final
    def create_dataset(cls,
                       dataset_name,
                       dataset_source_root_files_dir,
                       dataset_dest_root_dir,
                       dataset_sub_batch_size=50,
                       path_to_metadata_func=lambda x: {'ClassFolderName':str(x).split('/')[0]},
                       overwrite_existing=False,
                       data_mode=None,
                       max_n_group = 10,
                       ):
        """
        Creates a dataset files names dataset_name.<h5/csv>
        :param dataset_name: name of the dataset. Also the created dataset files name.
        :param dataset_source_root_files_dir: Data should be stored in this directory. Should have sub directories for the classes.
        :param dataset_dest_root_dir: Destination for h5 and csv file. If csv file is already existing process is resumed with existing file unless overwrite is set true.
        :param dataset_sub_batch_size: Number of Samples padded and batched together. The smaller the number the slower the loading and vice versa.
        :param path_to_metadata_func: Function that extracts meta data from samples names.
        :param overwrite_existing: If true, meta data and existing h5 are overwritten. If not meta is reused.
        :param no_class_dirs: Classes are provided in subdirs in dataset_source_files_dir or not.
        :param data_mode: str. either 'blosc' (lossless multi channel) or 'image' (jpeg)
        :return:
        """
        ## TODO add progress Bar
        if data_mode is None:
            data_mode = 'blosc' if str(cls).split('.')[-1][:-2] == 'BloscDataset' else 'image'
            print(f"Creating {data_mode} dataset")

        assert os.path.exists(dataset_source_root_files_dir), "Raw data root directory not found."

        # assert overwrite_existing or not os.path.exists(dataset_dest_root_dir), \
        #    f"Dataset destination directory already exists and overwrite_existing is set to {overwrite_existing}."

        assert data_mode.lower() in ['blosc', 'jpeg', 'jpg', 'image'], "data_mode must be 'blosc' or 'image'"

        data_mode = 'blosc' if data_mode.lower() == 'blosc' else 'image'

        if not os.path.exists(dataset_dest_root_dir):
            logging.info("Create dataset destination file directory.")
            os.makedirs(dataset_dest_root_dir)

        dataset_h5_file_path = os.path.join(dataset_dest_root_dir, dataset_name + '.h5')
        metadata_file_path = os.path.join(dataset_dest_root_dir, dataset_name + '.csv')
        print(os.listdir())

        if not os.path.exists(metadata_file_path) or overwrite_existing:
            print(metadata_file_path)
            logging.info('Creating meta data file.')
            H5MetaDataset._convert_tar_dir_to_dataset(dataset_source_root_files_dir,
                                                     path_to_metadata_function=path_to_metadata_func,
                                                     dataset_destination=dataset_dest_root_dir,
                                                     dataset_name = dataset_name,
                                                     sub_batch_size=dataset_sub_batch_size,
                                                     data_mode=data_mode,
                                                     max_n_group=max_n_group
                                                      )
            logging.info("Finished creating dataset.")
            return
        else:
            metadata = pd.read_csv(metadata_file_path)
            logging.info("Meta data file found. Proceeding")
            if not os.path.exists(dataset_h5_file_path) or overwrite_existing:
                logging.info('Converting raw data files to h5.')
                H5MetaDataset._convert_samples_to_dataset(dataset_dataframe=metadata,
                                                          dataset_destination_h5_file=dataset_h5_file_path,
                                                          sub_batch_size=dataset_sub_batch_size, data_mode=data_mode)
                logging.info('Finished converting Files')

    @final
    def get_group_number_mapping(self):
        sample_index = list(range(self.max_idx + 1))
        if self.split_mode == 'montecarlo':  ## select random subset
            random.Random(self.split_number).shuffle(sample_index)
            split_size = int(self.split_ratio * len(sample_index))
            selected_samples_index_map = sample_index[0:split_size]
        elif self.split_mode == 'full':
            split_size = int(self.split_ratio * len(sample_index))
            selected_samples_index_map = sample_index[:split_size]
        elif self.split_mode in ['split', 'cross_val']:
            from math import floor
            split_begin = floor(sum(self.split_ratio[0:self.split_number]) * len(sample_index))
            split_end = floor(sum(self.split_ratio[0:self.split_number + 1]) * len(sample_index))
            #split_end = floor((1 - sum(self.split_ratio[self.split_number + 1:])) * len(sample_index))
            selected_samples_index_map = sample_index[split_begin:split_end]
        else:
            raise ValueError(f"{self.split_mode} not supported")
        return selected_samples_index_map

    def __init__(self,
                 dataset_name='dataset_name',
                 dataset_root='/path/to/dataset',
                 split_mode='full',
                 split_ratio=1.0,
                 split_number=0,
                 ):
        """
        Constructor of the Dataset.

        :param dataset_name: Name of the datset files.
        :param dataset_root: Path to the dataset folder.

        Center returns a center crop, random a random sized crop and
        random_center a random sized center crop - original performs no cropping.
        :param split_mode: Determine the data split. ;must be in ['full', 'montecarlo', 'cross_val', split']
        :param split_ratio: float setting the percentage that is split for training if split_mode is full or montecarlo - int if split_mode is cross_val determining the number of splits, tuple of floats if mode is split
        :param split_number: int. for montecarlo this is the random seed
        """
        dataset_h5_file_path = os.path.join(dataset_root, dataset_name + '.h5')
        metadata_file_path = os.path.join(dataset_root, dataset_name + '.csv')
        self.h5_file = None

        assert split_number >= 0 and isinstance(split_number, int), f"split_number musst be a nonzero integer value but it is {split_number}"
        assert os.path.isfile(dataset_h5_file_path) and os.path.isfile(
            metadata_file_path), f"found in {dataset_root} directory. Call create_dataset first."
        assert split_mode.lower() in ['montecarlo', 'full', 'cross_val',
                                      'split'], f"split_mode is {split_number} but must be in ['montecarlo', 'full', 'cross_val', 'split']. Did you mean {difflib.get_close_matches(split_mode, ['montecarlo', 'full', 'cross_val', 'split'])}?"
        if split_mode.lower() == 'montecarlo' or split_mode == 'full':
            assert isinstance(split_ratio,
                              float) and 0 < split_ratio <= 1, f"The proportion of train samples must be given as float in {split_mode} mode"
        elif split_mode.lower() == 'cross_val':
            assert isinstance(split_ratio,
                              int) and split_ratio > 0, f"Split_ratio in case if cross_val is the number of splits (n-fold) and therefore must be an integer larger than 1."
            assert isinstance(split_number, int) and split_number in range(
                split_ratio), f"The 'split_number' must be in range of the n-fold. You selected split no. {split_number} but you have {range(split_ratio)} splits."
            split_ratio = tuple((1. / split_ratio for _ in range(split_ratio)))
        elif split_mode.lower() == 'split':
            assert isinstance(split_ratio, tuple) and all((isinstance(a, float) for a in split_ratio)) and all(
                (0 < a < 1. for a in
                 split_ratio)), "In 'split mode', the split_ratio is given as tuple of floats adding up to one. Every number refers to the percentage of data in that split."
            assert 0.999 <= sum(
                split_ratio) <= 1.001, f"The split ratios must add up to one. Split ratio sum is {sum(split_ratio)}"
            assert isinstance(split_number, int) and split_number in range(len(
                split_ratio)), f"The 'split_number' must be an integer selecting a split percentage in 'split_ratio'"
        else:
            raise ValueError(f'Unsupported variable combination. This should not happen.')

        super(H5MetaDataset, self).__init__()

        self.dataset_h5_file_path = dataset_h5_file_path
        self.metadata_file_path = metadata_file_path

        self.shapes = []
        self.batch_shapes = []
        self.classes = []
        self.indices = []

        self.metadata = pd.read_csv(metadata_file_path)
        self.split_mode = split_mode.lower()
        self.split_ratio = split_ratio
        self.split_number = split_number

        with h5py.File(self.dataset_h5_file_path, "r") as h5_file:
            self.data_mode = h5_file.attrs['data_mode']
            self.max_n_group = h5_file.attrs['max_n_group']
            self.sub_batch_size = h5_file.attrs['sub_batch_size']
            self.max_idx = h5_file.attrs['max_idx']
            self.num_samples = h5_file.attrs['num_samples']
            self.data_dtype = h5_file.attrs['data_dtype']

            self.classes = h5_file['attrs/classes'][()].astype(int).tolist()
            self.classes.append(h5_file['attrs/last_classes'][()].astype(int).tolist())
            self.indices = h5_file['attrs/indices'][()].astype(int).tolist()
            self.indices.append(h5_file['attrs/last_indices'][()].astype(int).tolist())
            self.batch_shapes = h5_file['attrs/batch_shapes'][()].astype(int).tolist()
            self.batch_shapes.append(h5_file['attrs/last_batch_shapes'][()].astype(int).tolist())
            self.shapes = h5_file['attrs/shapes'][()].astype(int).tolist()
            self.shapes.append(h5_file['attrs/last_shapes'][()].astype(int).tolist())

            self.num_sub_batches = len(self.indices)

        self.data_interface = BloscInterface if self.data_mode == 'blosc' else ImageInterface

        self.group_number_mapping = self.get_group_number_mapping()

    def reset(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __del__(self):
        logging.info("called del")
        if self.h5_file is not None and isinstance(self.h5_file, h5py._hl.files.File):
            logging.info('closing File')
            self.h5_file.close()
        logging.info("Deletion Complete")
