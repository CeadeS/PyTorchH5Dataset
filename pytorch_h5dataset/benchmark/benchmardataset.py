import os
import zipfile
import pandas as pd
import json
from ..dataset import BloscDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch import nn

import logging, sys
from tqdm import tqdm
import urllib

log_level = logging.ERROR

mode = 'val'


handlers = [logging.FileHandler(filename='data_import.log'),logging.StreamHandler(sys.stdout) ]
logging.basicConfig(format='%(asctime)s %(message)s', level=log_level , handlers=handlers)


class BenchmarkDataset:
    ## code from https://stackoverflow.com/a/53877507
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    @staticmethod
    def download_file(url, output_path):

        logging.info(f"Downloading File from {url} to {output_path}")
        with BenchmarkDataset.DownloadProgressBar(unit='B', unit_scale=True,
                                                  miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

    @staticmethod
    def make_traffic(url):
        logging.info(f"Downloading File from {url}")
        with BenchmarkDataset.DownloadProgressBar(unit='B', unit_scale=True,
                                                  miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename='./tmp.f', reporthook=t.update_to)
            os.remove('./tmp.f')


    def __init__(self, mode = 'val', crop_ratio=(0.73,1.33),crop_size=(244,244), crop_area=(0.1,1.0), dataset_root='../data'):
        im_url = f"http://images.cocodataset.org/zips/{mode}2017.zip"
        an_url = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        download_path = f"{dataset_root}/download"
        destination_path =  f"{dataset_root}/h5"

        im_zip_file_path = os.path.join(download_path, f"coco2017_{mode}_imgs.zip")
        an_zip_file_path = os.path.join(download_path, f"coco2017_{mode}_anns.zip")

        img_data_path = os.path.join(download_path, f"coco2017_{mode}_imgs/")
        ann_data_path = os.path.join(download_path, f"coco2017_trainval_annotations")

        self.ann_data_path = ann_data_path
        self.img_data_path = img_data_path

        annotations_csv_file = os.path.join(destination_path,f'coco2017_{mode}.csv')
        annotations_json_file = os.path.join(ann_data_path, f"annotations/instances_{mode}2017.json")

        logging.info(f"Preparing Benchmark Data in {mode} mode")


        if not os.path.isdir(destination_path):
            os.makedirs(destination_path)
        if not os.path.isdir(download_path):
            os.makedirs(download_path)


        if not os.path.isfile(im_zip_file_path) or not os.path.isfile(an_zip_file_path):
            logging.info("Started Download")
            if not os.path.isfile(im_zip_file_path):
                BenchmarkDataset.download_file(im_url, im_zip_file_path)
            else:
                logging.info('Skipping download of images zip file ... already present')
            if not os.path.isfile(an_zip_file_path):
                BenchmarkDataset.download_file(an_url, an_zip_file_path)
            else:
                logging.info('Skipping download of annotations zip file ... already present')
            logging.info("Ended Download")
        else:
            logging.info("Skipping download ... already present")

        if not os.path.isdir(ann_data_path):
            logging.info("Begin Unpacking Annotations")
            os.makedirs(ann_data_path)
            with zipfile.ZipFile(an_zip_file_path) as zf:
                for member in tqdm(zf.infolist(), desc='Extracting '):
                    try:
                        zf.extract(member, ann_data_path)
                    except zipfile.error as e:
                        pass
            logging.info("Ended Unpacking Annotations")
        else:
            logging.info("Annotation already present ... skipping download")

        if not os.path.isdir(img_data_path):
            logging.info("Begin Unpacking Images")
            os.makedirs(img_data_path)
            with zipfile.ZipFile(im_zip_file_path) as zf:
                for member in tqdm(zf.infolist(), desc='Extracting '):
                    try:
                        zf.extract(member, img_data_path)
                    except zipfile.error as e:
                        pass
            logging.info("Ended Unpacking Images")
        else:
            logging.info("Images already present ... skipping download")

        logging.info('Loading Annotations')

        if not os.path.isfile(annotations_csv_file):
            with open(annotations_json_file) as json_data:
                data = json.load(json_data)

            data_list = []
            for idx,(image, annotation) in enumerate(zip(data['images'],data['annotations'])):
                data_list.append({
                    'Index': idx,
                    'FilePath': os.path.join(img_data_path,f'{mode}2017/',image['file_name']),
                    'ClassNo': annotation['category_id'],
                    'FileType': image['file_name'].split('.')[-1],
                    'im_data': image,
                    'ann_data':annotation,
                })
            logging.info("Finished Loading Annotations")

            df = pd.DataFrame(data_list)
            df.to_csv(annotations_csv_file)
            self.df = df
            logging.info("Wrote Annotations to Disk.")
        else:
            logging.info(f"Read Annotation from csv {annotations_csv_file}")
            self.df = pd.read_csv(annotations_csv_file)

        logging.info("Finished preparing Benchmark Data")

        BloscDataset.create_dataset(f'coco2017_{mode}',
                                 fr'{dataset_root}/download/coco2017_{mode}_imgs',
                                 fr'{dataset_root}/h5',
                                  filename_to_metadata_func=None,
                                  dataset_sub_batch_size = 100
                                  )

        self.h5_transform = nn.Sequential(transforms.Resize(crop_size))
        self.h5dataset = BloscDataset(f'coco2017_{mode}', fr'{dataset_root}/h5', crop_ratio, crop_area, self.h5_transform)



        self.im_folder_transforms = transforms.Compose([
            transforms.RandomResizedCrop(crop_size,crop_area,crop_ratio),
            transforms.ToTensor(),
        ])
        self.imageFolderDataset = ImageFolder(fr'{dataset_root}/download/coco2017_{mode}_imgs', transform=self.im_folder_transforms)

