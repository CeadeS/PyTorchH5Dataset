from setuptools import setup, find_packages
from distutils.util import convert_path

with open("readme.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = f.read().split()

main_ns = {}
ver_path = convert_path('pytorch_h5dataset/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name='pytorch_h5dataset',
    version=main_ns['__version__'],
    packages=find_packages(),
    url='https://github.com/CeadeS/PyTorchH5Dataset',
    license='BSD-3-Clause License',
    author='Martin Hofmann',
    author_email='Martin.Hofmann@tu-ilmenau.de',
    description='Accelerated data loading H5 dataset module for  PyTorch.',
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=requirements,
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