from setuptools import setup

import unittest
def run_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('test', pattern='test_*.py')
    return test_suite

setup(
    name='h5Dataset',
    version='0.1',
    packages=['h5dataset'],
    url='https://github.com/CeadeS/PyTorchH5Dataset',
    license='BSD-3-Clause License',
    author='Martin Hofmann',
    author_email='Martin.Hofmann@tu-ilmenau.de',
    description='Accelerated data loading H5 dataset module for  PyTorch.',
    install_requires=[
       'pytorch',
       'numpy',
       'h5py',
       'hdf5plugin',
       'pandas',
       'Pillow',
       'tables',
       'torch',
       'torchvision'
       ],
    test_suite='setup.run_tests',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
