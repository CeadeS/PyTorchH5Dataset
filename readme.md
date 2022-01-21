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
pip install pytorch-h5dataset==0.1.0
```

dev intall
```bash

git clone https://github.com/CeadeS/PyTorchH5Dataset.git
pip install -e .
```


# Creating a Dataset

```python
import pytorch_h5dataset

d = pytorch_h5dataset.H5Dataset.create_dataset(
    "some_name",
    "some/dir/containing/data",
    "output/dir",
    overwrite_existing=False)
```

# Example Usage

```python
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
from torch import nn, float32, as_tensor

from torch.nn import MSELoss
from torch.jit import script
from torch import randperm
from time import time
from utils import NormImage
from numpy import prod

norm = script(NormImage()).cuda()

from pytorch_h5dataset import H5Dataset

batch_size = 100
epochs = 100
device = 'cuda:0'
dataset = H5Dataset(dataset_name="my_dataset",
                    dataset_root='/abs/path/to/ds',
                    transforms=Resize((244, 244)),
                    loading_crop_size=(0.73, 1.33),  # cropped aspect ratio 
                    loading_crop_area_ratio_range=244 * 244)  # number of cropped px read more at definition of random_located_sized_crop_function
dataloader = DataLoader(dataset, batch_size=4 * batch_size, num_workers=0)

model = nn.Linear(3 * 244 * 244, 1000).to(device)
criterion = MSELoss()
sum_loss = 0
t0 = time()
num_out = 0
for e in range(epochs):
    for sample, (meta_class, meta_indices) in dataloader:
        other_meta_data = dataset.get_meta_data_from_indices(meta_indices)
        sample, meta = sample.view(-1, *sample.shape[2:]).to(device), as_tensor(meta_class.view(-1), dtype=float32,
                                                                                device=device).requires_grad_(True)
        sample = norm(sample)
        perm = randperm(len(sample))
        for i in range(0, perm.size(0), batch_size):
            rand_indexes = perm[i:i + batch_size]
            x = sample[rand_indexes].view(-1, 3 * 244 * 244)
            y = meta[rand_indexes]
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
