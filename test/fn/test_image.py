from unittest import TestCase



import os
import pathlib

if pathlib.Path(os.getcwd()).name == 'fn':
    os.chdir('../../')


class Test(TestCase):

    def test_load_sample(self):
        import numpy as np
        from pytorch_h5dataset.fn.image import ImageInterface
        sample_bytes = ImageInterface.load_sample({'type': 'jpeg', 'path': './test/data/images/rgb/a/andromeda_0.png'})
        self.assertIsInstance(sample_bytes[0], np.ndarray)
        sample_bytes = ImageInterface.load_sample({'type': 'jpeg', 'path': './test/data/images/rgb/b/pano_1.jpg'})
        self.assertIsInstance(sample_bytes[0], np.ndarray)

    def test_center_crop(self):
        from pytorch_h5dataset.fn.image import ImageInterface
        import h5py
        import simplejpeg
        import numpy as np
        os.makedirs('./test/data/tmp/dataset/h5/',exist_ok=True)
        with open('./test/data/images/rgb/b/pano_1.jpg','rb') as fp1:
            with open('./test/data/images/rgb/a/andromeda_0.png','rb') as fp2:
                with h5py.File('./test/data/tmp/dataset/h5/image_test_dataset.h5', "w") as hdf5_file:
                    im1 = fp1.read()
                    im2 = fp2.read()
                    im_bytes_1 = np.frombuffer(im1, dtype=f'S{len(im1)}')
                    im_bytes_2 = np.frombuffer(im2, dtype=f'S{len(im2)}')
                    im_bytes = tuple([im_bytes_1,im_bytes_2]*2)
                    hdf5_file.create_group('0')
                    hdf5_file.create_group('1')
                    hdf5_file['0'].create_dataset(name='0',data=im_bytes)
                    hdf5_file['0'].create_dataset(name='1',data=im_bytes)
                    hdf5_file['1'].create_dataset(name='0',data=im_bytes)
                    bytes_list = list(hdf5_file['0']['1'])
                    im_out_bytes = ImageInterface.center_crop(sub_batch=bytes_list, crop_height=200, crop_width=200,batch_height=None, batch_width=None)
                    ims = [simplejpeg.decode_jpeg(aaa[0], fastdct=True, fastupsample=True) for aaa in im_out_bytes]
                    self.assertEqual(ims[0].shape, (200,200,3))
                    self.assertEqual(ims[1].shape, (200,200,3))

                    bytes_list = list(hdf5_file['1']['0'])
                    im_out_bytes = ImageInterface.center_crop(sub_batch=bytes_list, crop_height=200, crop_width=200,batch_height=None, batch_width=None)
                    ims = [simplejpeg.decode_jpeg(aaa[0], fastdct=True, fastupsample=True) for aaa in im_out_bytes]
                    self.assertEqual(ims[0].shape, (200,200,3))
                    self.assertEqual(ims[1].shape, (200,200,3))
        with self.assertRaises(AssertionError):
            ImageInterface.center_crop([b'123'],None,None,0.,1)
        with self.assertRaises(AssertionError):
            ImageInterface.center_crop([b'123'],None,None,0,None)
        with self.assertRaises(AssertionError):
            ImageInterface.center_crop([b'123'],None,None,0,1.)
        with self.assertRaises(AssertionError):
            ImageInterface.center_crop([b'123'],None,None,None,1)
        with self.assertRaises(AssertionError):
            ImageInterface.center_crop([],None,None,1,1)
        with self.assertRaises(AssertionError):
            ImageInterface.center_crop([1],None,None,1,1)



        os.remove('./test/data/tmp/dataset/h5/image_test_dataset.h5')
