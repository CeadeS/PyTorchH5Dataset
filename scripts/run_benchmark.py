if __name__ == '__main__':

    from pytorch_h5dataset.benchmark import BenchmarkDataset
    from pytorch_h5dataset import H5DataLoader
    from time import time
    import psutil
    import platform
    cpu_count = 0 if platform.system() == 'Windows' else psutil.cpu_count()
    from torch.utils.data import DataLoader

    benchmarkdataset = BenchmarkDataset()
    print(len(benchmarkdataset.h5dataset))
    print(len(benchmarkdataset.imageFolderDataset))


    print('H5')
    for num_workers in [0,0,1,2,3,4,5,6,7,8]:
        benchmarkdataset = BenchmarkDataset()
        batch_size = 100
        dataLoader = H5DataLoader(dataset=benchmarkdataset.h5dataset, device='cpu:0', batch_size=batch_size,
                                            num_workers=num_workers, return_meta_indices=True, pin_memory=True)

        l = 0
        to = time()
        for idx, (sample, (meta_class, sample_indices)) in enumerate(dataLoader):
            l+=sample.sum()
        print(f"num workers: {num_workers:3d}  time elapsed: {time() - to:3.2f} sample sum {l}")


    print('Folder')
    for num_workers in [0,1,2,3,4,5,6,7,8]:
        dataloader = DataLoader(benchmarkdataset.imageFolderDataset, batch_size=batch_size,
                                num_workers=num_workers, pin_memory=True, shuffle=True)
        l = 0
        to = time()
        for idx, (sample, (meta_class, sample_indices)) in enumerate(dataLoader):
            l+=sample.sum()
        print(f"num workers: {num_workers:3d}  time elapsed: {time() - to:3.2f} sample sum {l}")
