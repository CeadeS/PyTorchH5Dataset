from .metaDataset import H5MetaDataset
from torch import device, version, cat, jit, nn, stack, as_tensor
from torchvision.transforms import Resize
import warnings
from ..fn import image
from functools import partial
from pytorch_h5dataset.fn.transforms import Transform



class ImageDataset(H5MetaDataset):


    def __getitem__(self, sub_batch_idx):
        sample_reference, meta = super(ImageDataset, self).__getitem__(sub_batch_idx=sub_batch_idx)
        sample = self.image_transforms(sample_reference[()]) ##  [read sample from disk] and transform
        
        if self.tensor_transforms is not None:
            sample = self.tensor_transforms(sample)

        return sample, meta

    @staticmethod
    def convert_samples_to_dataset(dataset_dataframe,
                                   dataset_destination_h5_file='./data/test_dataset.h5',
                                   sub_batch_size=50, max_n_group= 10):

        ImageDataset._convert_samples_to_dataset(dataset_dataframe=dataset_dataframe,
                                                        dataset_destination_h5_file=dataset_destination_h5_file,
                                                        sub_batch_size=sub_batch_size, data_mode='image',
                                                 max_n_group=max_n_group)

    def __init__(self,
                 dataset_name='dataset_name',
                 dataset_root='/path/to/dataset',
                 split_mode = 'full',
                 split_ratio = 1.0,
                 split_number = 0,
                 tr_crop_strategy = None,
                 tr_crop_size = None,
                 tr_crop_area_ratio_range = None,
                 tr_output_size = None,
                 tr_random_rotation_angles = None,
                 tr_random_flip = None,
                 tr_random_scale_range = None,
                 decode = None, ## None, cpu, cuda
                 output_device: device = device('cpu'), #cpu or cuda
                 tensor_transforms = None,
                 quality=83
                 ):
        decode = None if decode is None else device(decode)
        assert decode is not None or all(a is None for a in [tr_crop_strategy,
                                   tr_crop_size,
                                   tr_crop_area_ratio_range,
                                   tr_output_size,
                                   tr_random_rotation_angles,
                                   tr_random_flip,
                                   tr_random_scale_range]), "If image preprocessing is activated, a decode device must be specified"

        assert tr_crop_strategy in ['random', 'center', None]

        output_device = device(output_device)

        super(ImageDataset, self).__init__(dataset_name, dataset_root, split_mode, split_ratio, split_number)
        assert self.data_dtype == str(bytes)

        if float(version.cuda) < 11.6 and decode is not None and decode.type == 'cuda':
            s = str(f"Function not available for cuda version {version.cuda} is < 11.6, using cpu instead")
            warnings.warn(s)
            decode = device('cpu')

        #JPEG Transforms

        transforms = []
        if tr_random_scale_range is not None:
            assert isinstance(tr_random_scale_range, tuple) and len(tr_random_scale_range) < 3 \
                   or isinstance(tr_random_scale_range, float)
            transforms.append(partial(image.ImageInterface.random_scale, scale_range= tr_random_scale_range, quality=quality))

        if tr_crop_strategy is not None:
            assert tr_crop_strategy.lower() in ['random', 'center']
            crop_function = image.ImageInterface.get_random_crop_function(
                                    random_location=tr_crop_strategy.lower() == 'random',
                                    crop_size =tr_crop_size,
                                    crop_area_ratio_range = tr_crop_area_ratio_range)
            transforms.append(crop_function)

        if tr_output_size is not None:
            transforms.append(partial(image.ImageInterface.scale, heights = tr_output_size[0], widths = tr_output_size[1]))

        if tr_random_rotation_angles is not None:
            assert isinstance(tr_random_rotation_angles, tuple) and all((a in [-90,0,90,180,270]) for a in  tr_random_rotation_angles)
            transforms.append(partial(image.ImageInterface.random_rotation, angles = tr_random_rotation_angles))

        if tr_random_flip is not None:
            if 'v' in tr_random_flip.lower():
                transforms.append(image.ImageInterface.random_v_flip)
            if 'h' in tr_random_flip.lower():
                transforms.append(image.ImageInterface.random_h_flip)

        if decode is not None:
            if 'cuda' == decode.type:
                transforms.append(partial(image.ImageInterface.sub_batch_as_tensor, device=device(decode)))
            transforms.append(partial(image.ImageInterface.sub_batch_decode, device=device(decode)))



        if tr_output_size is not None and decode is not None:
            transforms.append(partial(image.ImageInterface.scale_torch, heights = tr_output_size[0], widths = tr_output_size[1]))
            transforms.append(stack)
            transforms.append(partial(as_tensor, device=device(output_device)))

        elif output_device.type == 'cuda' or decode is not None or tensor_transforms is not None:
            transforms.append(partial(image.ImageInterface.sub_batch_as_tensor, device=device(output_device)))







        self.image_transforms = Transform(transforms=transforms)
        self.tensor_transforms = tensor_transforms

