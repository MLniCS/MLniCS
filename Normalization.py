import torch

"""
Normalization functions to operate on snapshot matrix
"""
class Normalization:
    def __init__(self):
        super(Normalization, self).__init__()

    def __call__(self, matrix, normalize=True):
        raise NotImplementedError("Calling method of abstract class Normalization")

class IdentityNormalization(Normalization):
    def __init__(self):
        super(IdentityNormalization, self).__init__()

    def __call__(self, matrix, normalize=True, axis=1):
        return matrix

class MeanNormalization(Normalization):
    def __init__(self):
        super(MeanNormalization, self).__init__()
        self.mean = None

    def __call__(self, matrix, normalize=True, axis=1):
        if normalize:
            if self.mean is None:
                self.mean = torch.mean(matrix, axis=axis, keepdims=True)
            return matrix - self.mean
        else:
            return matrix + self.mean

class StandardNormalization(Normalization):
    def __init__(self):
        super(StandardNormalization, self).__init__()
        self.mean = None
        self.std = None

    def __call__(self, matrix, normalize=True, axis=1):
        if normalize:
            if self.mean is None and self.std is None:
                self.mean = torch.mean(matrix, axis=axis, keepdims=True)
                self.std = torch.std(matrix, axis=axis, keepdims=True)
                self.std[torch.abs(self.std) <= 1e-6] = 1.
            return (matrix - self.mean) / self.std
        else:
            return matrix * self.std + self.mean
