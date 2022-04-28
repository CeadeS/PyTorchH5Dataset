[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
![PyPI](https://img.shields.io/pypi/v/pytorch-h5dataset)
[![tests](https://github.com/CeadeS/PyTorchH5Dataset/actions/workflows/tests.yml/badge.svg)](https://github.com/CeadeS/PyTorchH5Dataset/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/CeadeS/PyTorchH5Dataset/branch/main/graph/badge.svg?token=8GN4N5TU63)](https://codecov.io/gh/CeadeS/PyTorchH5Dataset)

# This is a Dataset module for PyTorch 
The dataset uses compressed h5 to accelerate data loading.

Create a Dataset from an image directory: Every class is put into a subdir. The directories in the class dirs will be crawled recursively.
Further meta-data can be provided by the file name or using an existing pandas data frame stored as csv file. The data frame must contain the columns FilePath, Index, ClassNo, and FileType.

# How does it work?

The data is stored in stacks of equal shape. Channels
must match, but width and height are zero-padded. The stacks
are generated in increasing width orders. Lossless blosc
compression nihilates padded zeros. Data is loaded
in random or fixed crops minimizing IO and data transfer.
Various cropping techniques are provided, random, center
resized. All data points are cropped in the same way.
Loading in original shape is supported but slower.

Since the order of images in a loaded batch is fixed,
several random batches are loaded and shuffled. Please look
at the example below.

Classical color jpg/png and multi-channel tif files
are crawled and stored in the dataset while metadata is stored into a pandas data frame. Metadata
can be provided by a data frame or the file name.

# Installation
pip install
```bash
pip install pytorch-h5dataset==0.2.2
```

dev intall
```bash

git clone https://github.com/CeadeS/PyTorchH5Dataset.git
cd PyTorchH5Dataset
pip install -e .
```


# Creating a Dataset

```python
import pytorch_h5dataset

d = pytorch_h5dataset._H5Dataset.create_dataset(
    "some_name",
    "some/dir/containing/data",
    "output/dir",
    overwrite_existing=False)
```

# Example Usage

```python

from torchvision.transforms import Resize
from torch import nn, float32, as_tensor
from torch.nn import MSELoss
from torch.jit import script
from time import time
from pytorch_h5dataset.utils import NormImage
from numpy import prod

norm = script(NormImage()).cuda()

from pytorch_h5dataset import _H5Dataset
from pytorch_h5dataset import H5DataLoader

batch_size = 100
epochs = 100
device = 'cuda:0'
dataset = _H5Dataset(dataset_name="test_dataset",
                    dataset_root="path/to/dataset",  # empl: '../test/data/tmp/dataset/h5',
                    transforms=Resize((244, 244)),
                    loading_crop_size=(0.73, 1.33),  # cropped aspect ratio
                    loading_crop_area_ratio_range=244 * 244)  # number of cropped px read more at definition of random_located_sized_crop_function

dataloader = H5DataLoader(dataset=dataset,
                          device='cpu:0', batch_size=1,
                          return_meta_indices=True,
                          pin_memory=True,
                          num_workers=0)

model = nn.Linear(3 * 244 * 244, 1000).to(device)
criterion = MSELoss()
sum_loss = 0
t0 = time()
num_out = 0
for e in range(epochs):
    for sample, label in dataloader:
        if isinstance(label, tuple):
            label = label[0]
        x = sample.to(device).view(sample.size(0), -1)
        y = as_tensor(label.view(-1), dtype=float32, device=device).requires_grad_(True)
        y_out = model(x).argmax(1).float()
        num_out += prod(y_out.shape)
        loss = criterion(y, y_out)
        loss = loss.sum()
        sum_loss += loss.item()
        loss.backward()
print(f"Time for {epochs} epochs was {time() - t0}")

del dataloader
print(loss, num_out)
```

# Benchmark

The Benchmark will download the coco dataset, convert it and run some time measurements.
./scripts provides also a jupyter notebook collecting more information.
```shell
git clone https://github.com/CeadeS/PyTorchH5Dataset.git
cd PyTorchH5Dataset
pip install -e .

cd scripts/
python run_benchmark.py

```
