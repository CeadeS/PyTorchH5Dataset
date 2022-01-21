from setuptools import setup

setup(
    name='pytorch_h5dataset',
    version='0.2.0',
    packages=['pytorch_h5dataset'],
    url='https://github.com/CeadeS/PyTorchH5Dataset',
    license='BSD-3-Clause License',
    author='Martin Hofmann',
    author_email='Martin.Hofmann@tu-ilmenau.de',
    description='Accelerated data loading H5 dataset module for  PyTorch.',
    install_requires=[
       'numpy',
       'h5py',
       'hdf5plugin',
       'pandas',
       'Pillow',
       'tables',
       'torch',
       'scikit-image',
       'torchvision'
       ],
    classifiers=[
        'Development Status :: 4 - Beta',
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
