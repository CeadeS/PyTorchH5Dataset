import unittest
import numpy as np
from pytorch_h5dataset.benchmark import Benchmarker

class MyTestCase(unittest.TestCase):
    def test_decorate_iterator_class(self):
        test_benchmarker = Benchmarker()

        @test_benchmarker.decorate_iterator_class
        class TestIterator:

            def __init__(self, max=100):
                self.max = max

            def __iter__(self):
                self.n = 0
                self.l = []
                return self

            def __next__(self):
                if self.n < self.max:
                    self.l.append(np.random.uniform(0,1,(100,100)))
                    self.n += 1
                    return self.l
                else:
                    raise StopIteration

        iterator = TestIterator()

        for _,i in  enumerate(iterator):
            pass
        self.assertEqual(len(test_benchmarker .get_stats_df()),100)

    def test_decorator_iterator_func(self):
        test_benchmarker  = Benchmarker()

        @test_benchmarker.decorate_iterator_func
        def iterator():
            l = []
            for a in range(100):
                l.append(np.random.uniform(0,1,(100,100)))
                yield l

        for _ in  enumerate(iterator()):
            pass
        self.assertEqual(len(test_benchmarker .get_stats_df()),100)

