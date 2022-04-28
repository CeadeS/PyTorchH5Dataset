class Transform:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        result = sample

        for transform in self.transforms:
            result = transform(result)
        return result


