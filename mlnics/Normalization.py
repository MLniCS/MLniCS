import torch

"""
Normalization functions to operate on snapshot matrix
"""

class Normalization:
    """
    Abstract class for normalization operations in matrix data.

    Attributes:
    initialized (bool): Whether the normalization parameters have been initialized.
    axis (int): Axis to perform normalization along, either 0 for input normalization or 1 for channel normalization.

    Methods:
    call (matrix, normalize=True): Raises a NotImplementedError, must be overridden in subclass.
    """
    
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
    """
    Identity normalization, returns the input matrix unchanged.

    Inherits from:
    Normalization

    Attributes:
    initialized (bool): Whether the normalization parameters have been initialized.
    axis (int): Axis to perform normalization along, either 0 for input normalization or 1 for channel normalization.

    Methods:
    call (matrix, normalize=True): Returns the input matrix, normalize argument has no effect.
    """

    def __init__(self, input_normalization=False):
        super(IdentityNormalization, self).__init__(input_normalization)
        self.initialized = True

    def __call__(self, matrix, normalize=True):
        return matrix


class MeanNormalization(Normalization):
    """
    Mean normalization, subtracts the mean of the matrix along the specified axis.

    Inherits from:
    Normalization

    Attributes:
    initialized (bool): Whether the normalization parameters have been initialized.
    axis (int): Axis to perform normalization along, either 0 for input normalization or 1 for channel normalization.
    mean (torch.Tensor): Mean of the matrix along the specified axis, None if not yet initialized.

    Methods:
    call (matrix, normalize=True): Returns the input matrix with the mean subtracted if normalize is True, or adds the mean if normalize is False.
    """

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
    """
    Standard normalization, subtracts the mean and divides by the standard deviation of the matrix along the specified axis.

    Inherits from:
    Normalization

    Attributes:
    initialized (bool): Whether the normalization parameters have been initialized.
    axis (int): Axis to perform normalization along, either 0 for input normalization or 1 for channel normalization.
    mean (torch.Tensor): Mean of the matrix along the specified axis, None if not yet initialized.
    std (torch.Tensor): Standard deviation of the matrix along the specified axis, None if not yet initialized.

    Methods:
    call (matrix, normalize=True): Returns the input matrix with the mean subtracted and divided by the standard deviation if normalize is True, or multiplies by the standard deviation and adds the mean if normalize is False.
    """

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
    """
    Min-Max normalization, normalizes the matrix to the range [-1, 1] along the specified axis.

    Inherits from:
    Normalization

    Attributes:
    initialized (bool): Whether the normalization parameters have been initialized.
    axis (int): Axis to perform normalization along, either 0 for input normalization or 1 for channel normalization.
    min (torch.Tensor): Minimum of the matrix along the specified axis, None if not yet initialized.
    max (torch.Tensor): Maximum of the matrix along the specified axis, None if not yet initialized.

    Methods:
    call (matrix, normalize=True): Returns the input matrix normalized to the range [-1, 1] if normalize is True, or returns the original values if normalize is False.
    """

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
