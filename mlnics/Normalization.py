import torch

"""
Normalization functions to operate on snapshot matrix
"""

class Normalization:
    def __init__(self, input_normalization=False):
        super(Normalization, self).__init__()
        self.initialized = False
        if input_normalization:
            self.axis = 0
        else:
            self.axis = 1

    def __call__(self, matrix, normalize=True):
        raise NotImplementedError("Calling method of abstract class Normalization")

class IdentityNormalization(Normalization):
    def __init__(self, input_normalization=False):
        super(IdentityNormalization, self).__init__(input_normalization)
        self.initialized = True

    def __call__(self, matrix, normalize=True):
        return matrix

class MeanNormalization(Normalization):
    def __init__(self, input_normalization=False):
        super(MeanNormalization, self).__init__(input_normalization)
        self.mean = None

    def __call__(self, matrix, normalize=True):
        if normalize:
            if self.mean is None:
                self.initialized = True
                self.mean = torch.mean(matrix, axis=self.axis, keepdims=True)
            return matrix - self.mean
        else:
            return matrix + self.mean

class StandardNormalization(Normalization):
    def __init__(self, input_normalization=False):
        super(StandardNormalization, self).__init__(input_normalization)
        self.mean = None
        self.std = None

    def __call__(self, matrix, normalize=True):
        if normalize:
            if self.mean is None and self.std is None:
                self.initialized = True
                self.mean = torch.mean(matrix, axis=self.axis, keepdims=True)
                self.std = torch.std(matrix, axis=self.axis, keepdims=True)
                self.std[torch.abs(self.std) <= 1e-6] = 1.
                #self.mean = torch.mean(matrix)
                #self.std = torch.std(matrix)
            return (matrix - self.mean) / self.std
        else:
            return matrix * self.std + self.mean

class MinMaxNormalization(Normalization):
    def __init__(self, input_normalization=False):
        super(MinMaxNormalization, self).__init__(input_normalization)
        self.min = None
        self.max = None

    def __call__(self, matrix, normalize=True):
        if normalize:
            if self.min is None and self.max is None:
                self.initialized = True
                self.min, _ = torch.min(matrix, axis=self.axis, keepdims=True)
                self.max, _ = torch.max(matrix, axis=self.axis, keepdims=True)

            return 2 * (matrix - self.min) / (self.max - self.min) - 1
        else:
            return (self.max - self.min) * (matrix + 1) / 2 + self.min
